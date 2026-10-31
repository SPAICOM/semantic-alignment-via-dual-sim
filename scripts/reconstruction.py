"""Unified SIM reconstruction script — the image-reconstruction analogue of ``classification.py``.

One job per (seed, L, M_int, snr, strategy, …) combination. Loads the TX/RX
autoencoder pair, embeds the raw images, computes the alignment and the no-SIM
baselines, wraps them in a :class:`ReconstructionTask`, then dispatches to the
same task-agnostic SIM runner as classification (the ``sim`` config's
``_target_``, via Hydra ``instantiate``). The runner trains the SIM(s) and the
task scores the RX decoder's reconstructions (MSE/PSNR) and writes a
comparison PNG next to the Parquet row.

Supported architectures: every ``sim`` config (joint, mono, mono_rx,
disjoint_sym, disjoint_asym, disjoint_4sim). Ready-made configs (Linear + PPFE):
  - Joint Dual-SIM               config_reconstruction_joint
  - Mono-SIM TX (constructive)   config_reconstruction_mono_constructive
  - Disjoint sym (constructive)  config_reconstruction_disjoint_sym_constructive

Results go to ``results/reconstruction/{result_subdir}/{tipology}/``.

Examples
--------
    just rec-all
    just rec-joint datasets/reconstruction=fashion_mnist
    just rec sim.L=5 sim.snr=10 alignment/reconstruction=linear
    uv run python scripts/reconstruction.py sim=mono training=mono sim.constructive=true
"""

import os
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from semantic_alignment.data.reconstruction import (
    compute_alignment,
    ppfe_cluster_data,
    prepare_reconstruction,
)
from semantic_alignment.evaluation.reconstruction import (
    psnr_from_mse,
    reconstruction_mse,
    run_original_reconstruction,
)
from semantic_alignment.evaluation.tasks import ReconstructionTask
from semantic_alignment.models.autoencoder import (
    load_autoencoder,
    load_autoencoder_training_meta,
    regularization_tag,
)
from semantic_alignment.sim.original._utils import make_fname, tipology_tag
from semantic_alignment.utils import channel as _channel_fn
from semantic_alignment.utils import standardize_channel

_CSI_SIM_TYPES = ("disjoint_sym", "disjoint_4sim")


def _tags(cfg: DictConfig) -> tuple[str, str, str]:
    """Return ``(tx_enc, rx_enc, reg_tag)`` identifying the autoencoder pair in filenames.

    ``tx_enc``/``rx_enc`` play the role of the classification encoder names
    (``{variant}seed{seed}``); ``reg_tag`` (``AE``/``betaVAE``/``sigmaVAE``/
    ``MMD-InfoVAE``) prefixes the filename so runs with differently-regularised
    checkpoints don't overwrite each other.
    """
    rc = cfg.reconstruction
    dataset = cfg.datasets.name
    tx_enc = f"{rc.tx_variant}seed{rc.tx_seed}"
    rx_enc = f"{rc.rx_variant}seed{rc.rx_seed}"
    tx_reg = regularization_tag(load_autoencoder_training_meta(dataset, rc.tx_variant, rc.tx_seed))
    rx_reg = regularization_tag(load_autoencoder_training_meta(dataset, rc.rx_variant, rc.rx_seed))
    reg_tag = tx_reg if tx_reg == rx_reg else f"tx-{tx_reg}_rx-{rx_reg}"
    return tx_enc, rx_enc, reg_tag


def _n_clusters(cfg: DictConfig) -> int | None:
    """PPFE cluster count actually used: ``channel_dim`` for full-CSI architectures."""
    if cfg.alignment.type != "PPFE":
        return None
    ch_dim_cfg = cfg.sim.get("channel_dim", None)
    if cfg.sim.sim_type in _CSI_SIM_TYPES and ch_dim_cfg is not None:
        # F_T and F_R_H must live in the ch_dim space — a physical constraint.
        return int(ch_dim_cfg)
    return cfg.alignment.n_clusters


def _ch_dim(cfg: DictConfig, A_shape: tuple[int, int]) -> int:
    """Channel dimension the runner will use (``disjoint_asym`` follows the aligned side)."""
    sim = cfg.sim
    if sim.get("sim_type", "") in ("disjoint_asym", "mono"):
        return A_shape[1] if sim.get("alignment_side", "tx") == "rx" else A_shape[0]
    ch_dim = sim.get("channel_dim", None)
    return int(ch_dim) if ch_dim is not None else int(max(A_shape))


def _out_path(cfg: DictConfig, A_shape: tuple[int, int], tags: tuple[str, str, str]) -> Path:
    """Return the output Parquet path for this job — no embedding needed.

    Mirrors the runners' naming so finished jobs are skipped before loading data;
    the runner re-checks the exact path itself.
    """
    sim = cfg.sim
    tx_enc, rx_enc, reg_tag = tags
    snr_raw = sim.get("snr", None)
    snr = None if str(snr_raw) == "null" else snr_raw
    result_subdir = sim.get("result_subdir", "")
    fname = make_fname(
        cfg.alignment.type,
        cfg.training.seed,
        sim.L,
        sim.M_int,
        _ch_dim(cfg, A_shape),
        snr,
        tx_enc,
        rx_enc,
        sim.get("s_tx", None),
        sim.get("s_rx", None),
        _n_clusters(cfg),
        dataset=cfg.datasets.name,
        channel_type=sim.get("channel_type", "rayleigh"),
        rice_factor_db=sim.get("rice_factor_db", None),
        cluster_on=cfg.alignment.get("cluster_on", "pilots"),
    )
    topo = tipology_tag(
        sim.get("sim_type", ""), sim.get("constructive", False), sim.get("alignment_side", None)
    )
    out_dir = ReconstructionTask.results_dir / result_subdir
    return (out_dir / topo if topo else out_dir) / f"{reg_tag}_{fname}"


def _lrs(cfg: DictConfig) -> tuple[float | None, float | None]:
    """``(lr_tx, lr_rx)``: alignment-specific overrides first, then the training config."""
    return (
        cfg.alignment.get("lr_tx", cfg.training.get("lr_tx", None)),
        cfg.alignment.get("lr_rx", cfg.training.get("lr_rx", None)),
    )


def _setup(cfg: DictConfig, tx_model, rx_model, reg_tag: str, device: torch.device) -> dict:
    """Embed, align, and evaluate the no-SIM baselines — every step shared by all architectures.

    Returns
    -------
    dict
        Runner context: the :class:`ReconstructionTask` (test data, decoder,
        whitening, *No Mismatch* and *Original* baselines), ``A_target``, the
        MIMO channel ``H_mimo``, and the full-CSI PPFE frames ``F_T``/``F_R_H``.
    """
    rc = cfg.reconstruction
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    prep = prepare_reconstruction(
        tx_model,
        rx_model,
        cfg.datasets.hf_repo,
        cfg.datasets.hf_config,
        rc.train_label_size,
        image_col=cfg.datasets.image_col,
        label_col=cfg.datasets.label_col,
        image_size=tuple(cfg.datasets.image_size),
        channels=cfg.datasets.channels,
        method=rc.method,
        grouping=rc.grouping,
        train_split=cfg.datasets.train_split,
        test_split=cfg.datasets.test_split,
        save_to_disk=rc.save_to_disk,
        seed=seed,
        device=device,
    )

    n = rc.num_preview
    images_test = prep.images_test
    with torch.no_grad():
        recon_nm = torch.cat(
            [
                rx_model.decode(prep.data.test.z_rx[i : i + rc.batch_size].to(device)).cpu()
                for i in range(0, len(images_test), rc.batch_size)
            ]
        )
    no_mismatch_mse = reconstruction_mse(recon_nm, images_test)

    strategy = cfg.alignment.type
    cluster_data = None
    if strategy == "PPFE" and cfg.alignment.get("cluster_on", "pilots") == "train_tx":
        cluster_data = ppfe_cluster_data(
            prep.data.train, prep.L_in, prep.mu_in, prep.L_out, prep.mu_out, device
        )
    A_target, F_T, F_R_H = compute_alignment(
        prep.input_w,
        prep.output_w,
        prep.pilots.z_rx,
        strategy,
        lmb=cfg.alignment.get("lmb", 1e-3),
        n_clusters=_n_clusters(cfg),
        n_proto=cfg.alignment.get("n_proto", None),
        seed=seed,
        cluster_data=cluster_data,
    )
    original_mse, original_psnr, align_preview = run_original_reconstruction(
        prep.data.test.z_tx,
        images_test,
        A_target,
        prep.L_in,
        prep.mu_in,
        prep.L_out,
        prep.mu_out,
        rx_model,
        device,
        batch_size=rc.batch_size,
        num_preview=n,
    )
    print(f" [{strategy}] original: MSE={original_mse:.4f} | PSNR={original_psnr:.2f} dB")

    task = ReconstructionTask(
        prep.data.test.z_tx,
        images_test,
        prep.L_in,
        prep.mu_in,
        prep.L_out,
        prep.mu_out,
        rx_model,
        device,
        strategy_name=strategy,
        reg_tag=reg_tag,
        no_mismatch_mse=no_mismatch_mse,
        no_mismatch_psnr=psnr_from_mse(no_mismatch_mse),
        original_mse=original_mse,
        original_psnr=original_psnr,
        no_mismatch_preview=recon_nm[:n],
        align_preview=align_preview,
        batch_size=rc.batch_size,
        num_preview=n,
    )

    # Same channel draw as classification; Mono-SIM draws its own and ignores it.
    h_dim = _ch_dim(cfg, tuple(A_target.shape))
    ch_type = cfg.sim.get("channel_type", "rayleigh")
    H_mimo = standardize_channel(
        _channel_fn(
            h_dim, h_dim, channel_type=ch_type, rice_factor_db=cfg.sim.get("rice_factor_db", None)
        ).to(device),
        channel_type=ch_type,
    )

    is_csi = cfg.sim.sim_type in _CSI_SIM_TYPES
    return dict(
        task=task,
        A_target=A_target,
        H_mimo=H_mimo,
        F_T=F_T if is_csi else None,
        F_R_H=F_R_H if is_csi else None,
    )


@hydra.main(version_base=None, config_path="../config", config_name="config_reconstruction")
def main(cfg: DictConfig) -> None:
    """Hydra entry point — load the AE pair, align, and run the SIM experiment.

    Skips the job before any embedding work when its output Parquet already exists.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration composed from the selected config name and any
        command-line overrides.
    """
    gpu = cfg.training.get("gpu", None)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rc = cfg.reconstruction
    dataset = cfg.datasets.name
    tags = _tags(cfg)
    tx_enc, rx_enc, reg_tag = tags

    tx_model = load_autoencoder(dataset, rc.tx_variant, rc.tx_seed, device)
    rx_model = load_autoencoder(dataset, rc.rx_variant, rc.rx_seed, device)
    # A_target is (D_rx/2, D_tx/2): known from the latent sizes before embedding anything.
    A_shape = (rx_model.latent_dim // 2, tx_model.latent_dim // 2)
    out = _out_path(cfg, A_shape, tags)
    if out.exists():
        print(f" Already done: {out.name} — skipping.")
        return

    print(
        f" seed={cfg.training.seed} | sim={cfg.sim.name} | dataset={dataset} | "
        f"align={cfg.alignment.type} | tx={tx_enc} | rx={rx_enc} | reg={reg_tag}"
    )

    ctx = _setup(cfg, tx_model, rx_model, reg_tag, device)
    lr_tx, lr_rx = _lrs(cfg)
    instantiate(
        cfg.sim,
        _recursive_=False,
        slayer=cfg.sim.wavelength * cfg.sim.slayer_multiplier,
        dx=cfg.sim.wavelength / cfg.sim.spacing_divisor,
        seed=cfg.training.seed,
        device=device,
        strategy_name=cfg.alignment.type,
        n_clusters=_n_clusters(cfg),
        cluster_on=cfg.alignment.get("cluster_on", "pilots"),
        dataset=dataset,
        tx_enc=tx_enc,
        rx_enc=rx_enc,
        max_iters=cfg.training.max_iters,
        lr=cfg.training.get("lr", None),
        lr_tx=lr_tx,
        lr_rx=lr_rx,
        k_T=int(cfg.training.get("k_T", 100)),
        k_R=int(cfg.training.get("k_R", 100)),
        optimizer=cfg.training.get("optimizer", "adam"),
        scheduler=cfg.training.get("scheduler", None),
        warmup_frac=float(cfg.training.get("warmup_frac", 0.05)),
        clip_value=cfg.training.get("clip_value", None),
        save_curve=bool(cfg.training.get("save_curve", False)),
        phase_init=str(cfg.training.get("phase_init", "zeros")),
        **ctx,
    )


if __name__ == "__main__":
    main()

"""Unified SIM classification script — all architectures in one entry point.

One job per (seed, L, M_int, snr, …) combination.  The experiment YAML's
``_target_`` field determines which runner is called; Hydra ``instantiate``
dispatches automatically with no match statement here.

Supported architectures (select via ``--config-name``):
  - Joint Dual-SIM            config_joint_layers / config_joint_snr / config_joint_slayer
  - Disjoint asym             config_disjoint_asym_layers / _snr
  - Disjoint sym (full CSI)   config_disjoint_sym_layers / _snr
  - Disjoint 4-SIM (full CSI) config_disjoint_4sim_layers / _snr
  - Mono-SIM (TX-only)        config_mono_layers / config_mono_snr

*No Mismatch* accuracy is computed once per job as the linear-probe score
on the raw RX test embeddings.

Examples
--------
    just run-joint
    just run-mono
    just run-disjoint-asym
    just run-joint datasets=cifar100 alignment=linear
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
from torch.utils.data import DataLoader, TensorDataset

from semantic_alignment.data.prepare import prepare_semasia
from semantic_alignment.evaluation.linear_probe import clf_accuracy, train_linear_probe
from semantic_alignment.evaluation.neural_classifier import load_neural_clf
from semantic_alignment.evaluation.original_accuracy import run_original_accuracy
from semantic_alignment.sim.original._utils import make_fname, tipology_tag
from semantic_alignment.utils import cn_matrix

_RESULTS = Path("results")


def _snr_tag(snr) -> str:
    """Return ``'inf'`` for null SNR, otherwise the string representation."""
    return "inf" if str(snr) == "null" else str(snr)


def _out_path(cfg: DictConfig) -> Path | None:
    """Return the output Parquet path for this job from cfg alone — no data needed."""
    sim = cfg.sim
    result_subdir = sim.get("result_subdir", None)
    if result_subdir is None:
        return None

    seed = cfg.training.seed
    strategy = cfg.alignment.type
    tx = cfg.data.tx_enc
    rx = cfg.data.rx_enc
    M = sim.M_int
    ch_dim = sim.get("channel_dim", None)
    if ch_dim is None:
        # ch_dim is computed from A_target at runtime — can't determine path statically
        return None
    snr_raw = sim.get("snr", None)
    snr = None if str(snr_raw) == "null" else snr_raw
    s_tx = sim.get("s_tx", None)
    s_rx = sim.get("s_rx", None)
    n_clusters = cfg.alignment.get("n_clusters", None) if strategy == "PPFE" else None
    dataset = cfg.datasets.hf_name

    topo = tipology_tag(
        sim.get("sim_type", ""),
        sim.get("constructive", False),
        sim.get("alignment_side", None),
    )
    fname = make_fname(
        strategy,
        seed,
        sim.L,
        M,
        ch_dim,
        snr,
        tx,
        rx,
        s_tx,
        s_rx,
        n_clusters,
        dataset=dataset,
    )
    return _RESULTS / result_subdir / topo / fname if topo else _RESULTS / result_subdir / fname


def _setup(cfg: DictConfig) -> dict:
    """Load data, compute the alignment target, and collect all shared experiment context.

    Runs every step that is common to all architecture variants: loads
    embeddings from Semasia, selects pilot pairs, trains a linear probe, optionally
    loads a neural classifier, computes whitening statistics, computes the
    alignment target (Ridge regression or PPFE), evaluates the *No Mismatch*
    and *Original* (no-SIM, no-channel) accuracy baselines, and draws a random
    MIMO channel matrix.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration composed from the selected config name and any
        command-line overrides.

    Returns
    -------
    dict
        Shared context dictionary with the following keys:

        ``seed`` : int
            Global random seed used for this run.
        ``device`` : torch.device
            CUDA device if available, otherwise CPU.
        ``A_target`` : np.ndarray
            Alignment matrix of shape ``(D_rx/2, D_tx/2)``, complex64.
        ``H_mimo`` : torch.Tensor
            Complex Gaussian MIMO channel of shape ``(ch_dim, ch_dim)``.
        ``test_loader`` : DataLoader
            Test set yielding ``(z_tx, labels)`` batches.
        ``clf`` : torch.Tensor
            Linear probe weight matrix.
        ``L_in``, ``mu_in`` : torch.Tensor
            Cholesky factor and mean of the TX whitening transform.
        ``L_out``, ``mu_out`` : torch.Tensor
            Cholesky factor and mean of the RX whitening transform.
        ``no_mismatch_probe`` : float
            No-mismatch linear-probe accuracy (percent).
        ``no_mismatch_neural`` : float or None
            No-mismatch neural-classifier accuracy (percent); ``None`` if no
            neural classifier was found.
        ``original_probe`` : float
            Oracle accuracy with alignment applied but no SIM/channel (percent).
        ``original_neural`` : float or None
            Same for the neural classifier; ``None`` if not available.
        ``clf_neural``
            Loaded neural classifier, or ``None``.
        ``strategy_name`` : str
            Alignment strategy label (``"Linear"`` or ``"PPFE"``).
        ``max_iters`` : int
            Number of optimization iterations.
        ``lr`` : float
            Learning rate.
        ``tx_enc``, ``rx_enc`` : str
            Encoder names embedded in output filenames.
        ``optimizer`` : str
            Optimizer type (``"adam"`` or ``"sgd"``).
        ``scheduler`` : str
            LR scheduler type (``"none"`` or ``"cosine"``).
        ``warmup_frac`` : float
            Warm-up fraction for the cosine scheduler.
        ``clip_value`` : float or None
            Gradient clipping norm; ``None`` disables clipping.
        ``tx_steps_per_rx``, ``rx_steps_per_tx`` : int
            Alternating update ratios for joint optimization.
        ``F_T``, ``F_R_H`` : np.ndarray or None
            PPFE TX synthesis and RX analysis frames; ``None`` unless the
            strategy is PPFE and the sim type requires full-CSI frames.
    """
    seed = cfg.training.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    is_csi = cfg.sim.sim_type in ("disjoint_sym", "disjoint_4sim")
    if cfg.alignment.type == "PPFE":
        if is_csi:
            ch_dim_cfg = cfg.sim.get("channel_dim", None)
            # n_clusters must equal channel_dim so F_T and F_R_H are dimensionally
            # consistent with B_T and B_R (both live in the ch_dim space).
            n_k = int(ch_dim_cfg) if ch_dim_cfg is not None else None
        else:
            n_k = cfg.alignment.n_clusters
    else:
        n_k = None

    prep = prepare_semasia(
        hf_name=cfg.datasets.hf_name,
        tx_enc=cfg.data.tx_enc,
        rx_enc=cfg.data.rx_enc,
        train_label_size=cfg.data.train_label_size,
        alignment_type=cfg.alignment.type,
        lmb=cfg.alignment.get("lmb", 1e-3),
        n_clusters=n_k,
        n_proto=cfg.alignment.get("n_proto", None),
        keep_csi_frames=is_csi,
        method=cfg.data.method,
        grouping=cfg.data.grouping,
        label_col=cfg.datasets.label_col,
        train_split=cfg.datasets.train_split,
        test_split=cfg.datasets.test_split,
        save_to_disk=cfg.data.save_to_disk,
        seed=seed,
        device=device,
    )
    data = prep.data
    L_in, mu_in, L_out, mu_out = prep.L_in, prep.mu_in, prep.L_out, prep.mu_out
    A_target, F_T, F_R_H = prep.A_target, prep.F_T, prep.F_R_H

    # Linear probe — always trained
    clf_probe = train_linear_probe(
        data.train.z_rx.to(device),
        data.train.labels.to(device),
        num_classes=cfg.datasets.num_classes,
    )

    # Neural classifier — loaded if available, silently skipped otherwise
    try:
        clf_neural = load_neural_clf(cfg.datasets.hf_name, cfg.data.rx_enc, seed, device)
        print(" Neural classifier loaded.")
    except FileNotFoundError:
        clf_neural = None
        print(" Neural classifier not found — run 'just train-clf' to enable neural accuracy.")

    no_mismatch_probe = clf_accuracy(
        clf_probe,
        data.test.z_rx.to(device),
        data.test.labels.to(device),
    )
    no_mismatch_neural = (
        clf_accuracy(
            clf_neural,
            data.test.z_rx.to(device),
            data.test.labels.to(device),
        )
        if clf_neural is not None
        else None
    )
    test_loader = DataLoader(
        TensorDataset(data.test.z_tx, data.test.labels),
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    # Original: A_target applied to TX embeddings, no SIM, no channel
    original_probe = run_original_accuracy(
        test_loader,
        A_target,
        L_in,
        mu_in,
        L_out,
        mu_out,
        clf_probe,
        device,
    )
    original_neural = (
        run_original_accuracy(
            test_loader,
            A_target,
            L_in,
            mu_in,
            L_out,
            mu_out,
            clf_neural,
            device,
        )
        if clf_neural is not None
        else None
    )
    if cfg.sim.get("sim_type", "") == "disjoint_asym":
        alignment_side = cfg.sim.get("alignment_side", "tx")
        h_dim = int(A_target.shape[0]) if alignment_side == "tx" else int(A_target.shape[1])
    else:
        ch_dim_cfg = cfg.sim.get("channel_dim", None)
        h_dim = ch_dim_cfg if ch_dim_cfg is not None else int(max(A_target.shape))
    H_mimo = cn_matrix(0, 1, size=(h_dim, h_dim)).to(device)

    return dict(
        seed=seed,
        device=device,
        A_target=A_target,
        H_mimo=H_mimo,
        test_loader=test_loader,
        clf=clf_probe,
        L_in=L_in,
        mu_in=mu_in,
        L_out=L_out,
        mu_out=mu_out,
        no_mismatch_probe=no_mismatch_probe,
        no_mismatch_neural=no_mismatch_neural,
        original_probe=original_probe,
        original_neural=original_neural,
        clf_neural=clf_neural,
        strategy_name=cfg.alignment.type,
        max_iters=cfg.training.max_iters,
        lr=cfg.training.get("lr", None),
        lr_tx=cfg.training.get("lr_tx", None),
        lr_rx=cfg.training.get("lr_rx", None),
        tx_enc=cfg.data.tx_enc,
        rx_enc=cfg.data.rx_enc,
        optimizer=cfg.training.get("optimizer", "adam"),
        scheduler=cfg.training.get("scheduler", None),
        warmup_frac=float(cfg.training.get("warmup_frac", 0.05)),
        clip_value=cfg.training.get("clip_value", None),
        tx_steps_per_rx=int(cfg.training.get("tx_steps_per_rx", 1)),
        rx_steps_per_tx=int(cfg.training.get("rx_steps_per_tx", 1)),
        F_T=F_T,
        F_R_H=F_R_H,
    )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point — load data, build the alignment, run the experiment.

    Dispatches to the correct experiment runner via ``hydra.utils.instantiate``
    using the ``_target_`` field in the experiment YAML.  Skips the job when
    the output Parquet file already exists.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration composed from the selected config name and any
        command-line overrides.
    """
    gpu = cfg.training.get("gpu", None)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    out = _out_path(cfg)
    if out is not None and out.exists():
        print(f" Already done: {out.name} — skipping.")
        return

    print(
        f" seed={cfg.training.seed} | sim={cfg.sim.name} | "
        f"dataset={cfg.datasets.hf_name} | align={cfg.alignment.type}"
    )

    ctx = _setup(cfg)
    F_T = ctx.pop("F_T")
    F_R_H = ctx.pop("F_R_H")
    # joint-specific params — absent in mono/disjoint training configs
    clip_value = ctx.pop("clip_value", None)
    tx_steps_per_rx = ctx.pop("tx_steps_per_rx", 1)
    rx_steps_per_tx = ctx.pop("rx_steps_per_tx", 1)
    n_clusters = cfg.alignment.get("n_clusters", None) if cfg.alignment.type == "PPFE" else None

    instantiate(
        cfg.sim,
        _recursive_=False,
        slayer=cfg.sim.wavelength * cfg.sim.slayer_multiplier,
        dx=cfg.sim.wavelength / cfg.sim.spacing_divisor,
        clip_value=clip_value,
        tx_steps_per_rx=tx_steps_per_rx,
        rx_steps_per_tx=rx_steps_per_tx,
        F_T=F_T,
        F_R_H=F_R_H,
        n_clusters=n_clusters,
        dataset=cfg.datasets.hf_name,
        **ctx,
    )


if __name__ == "__main__":
    main()

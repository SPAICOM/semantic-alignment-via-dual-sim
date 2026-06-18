"""Dual-SIM experiment — No Mismatch + SIM accuracy in one run.

One job per (seed, L, M_int, snr, …) combination.

*No Mismatch* accuracy is computed once per job as the linear-probe score
on the raw RX test embeddings — the upper bound when there is no domain gap.
It is saved alongside the SIM result in every Parquet row, mirroring the
reference repo (semantic-alignment-via-sim) layout.

Sweeps are defined in ``config/experiment/*.yaml`` via ``hydra.sweeper.params``;
activate with ``-m``.

Examples
--------
    just run-joint                              # layers sweep
    just run-joint-snr
    just run-joint-slayer
    just run-disjoint
    just run-joint datasets=cifar100
    just run-joint alignment=linear
"""

import os
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from semantic_alignment.data.semasia import load_semasia, select_pilots
from semantic_alignment.evaluation.linear_probe import clf_accuracy, train_linear_probe
from semantic_alignment.evaluation.neural_classifier import load_neural_clf
from semantic_alignment.evaluation.original_accuracy import run_original_accuracy
from semantic_alignment.experiments.experiment_runner import (
    run_single_disjoint_4sim_layers,
    run_single_disjoint_4sim_snr,
    run_single_disjoint_csi_layers,
    run_single_disjoint_csi_snr,
    run_single_disjoint_layers,
    run_single_disjoint_snr,
    run_single_layers_config,
    run_single_slayer_config,
    run_single_snr_config,
)
from semantic_alignment.utils.alignment_utils import ppfe, ridge_regression
from semantic_alignment.utils.utils import (
    complex_compressed_tensor,
    complex_gaussian_matrix,
    prewhiten,
)

_RESULTS = Path("results")


def _snr_tag(snr) -> str:
    return "inf" if str(snr) == "null" else str(snr)


def _out_path(cfg: DictConfig) -> Path | None:
    """Return the output Parquet path for this job from cfg alone — no data needed."""
    exp = cfg.experiment
    name = exp.name
    seed = cfg.training.seed
    strategy = cfg.alignment.type
    tx = cfg.data.tx_enc
    rx = cfg.data.rx_enc
    M = exp.M_int
    ch_dim = exp.get("channel_dim", None)
    ch_dim = ch_dim if ch_dim is not None else 384
    ch_tag = f"ch{ch_dim}"

    if name == "layers":
        return (
            _RESULTS
            / "dual_layers"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{ch_tag}_{tx}_{rx}.parquet"
        )
    if name == "snr":
        return (
            _RESULTS
            / "dual_snr"
            / f"{strategy}_seed{seed}_M{M}x{M}_{ch_tag}_snr{_snr_tag(exp.snr)}_{tx}_{rx}.parquet"
        )
    if name == "slayer":
        return (
            _RESULTS
            / "dual_slayer"
            / f"{strategy}_seed{seed}_stx{exp.s_tx}_srx{exp.s_rx}_{ch_tag}_{tx}_{rx}.parquet"
        )
    if name == "disjoint_layers":
        return (
            _RESULTS
            / "disjoint_layers"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{ch_tag}_{tx}_{rx}.parquet"
        )
    if name == "disjoint_csi_layers":
        return (
            _RESULTS
            / "disjoint_csi_layers"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{ch_tag}_{tx}_{rx}.parquet"
        )
    if name == "disjoint_4sim_layers":
        return (
            _RESULTS
            / "disjoint_4sim_layers"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{ch_tag}_{tx}_{rx}.parquet"
        )
    if name == "disjoint_csi_snr":
        return (
            _RESULTS
            / "disjoint_csi_snr"
            / f"{strategy}_seed{seed}_M{M}x{M}_{ch_tag}_snr{_snr_tag(exp.snr)}_{tx}_{rx}.parquet"
        )
    if name == "disjoint_4sim_snr":
        return (
            _RESULTS
            / "disjoint_4sim_snr"
            / f"{strategy}_seed{seed}_M{M}x{M}_{ch_tag}_snr{_snr_tag(exp.snr)}_{tx}_{rx}.parquet"
        )
    if name == "disjoint_snr":
        return (
            _RESULTS
            / "disjoint_snr"
            / f"{strategy}_seed{seed}_M{M}x{M}_{ch_tag}_snr{_snr_tag(exp.snr)}_{tx}_{rx}.parquet"
        )
    if name == "channel_dim_joint":
        return (
            _RESULTS
            / "dual_channel_dim"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{ch_tag}_{tx}_{rx}.parquet"
        )
    if name == "channel_dim_disjoint":
        return (
            _RESULTS
            / "disjoint_channel_dim"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{ch_tag}_{tx}_{rx}.parquet"
        )
    if name == "channel_dim_disjoint_csi":
        return (
            _RESULTS
            / "disjoint_csi_channel_dim"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{ch_tag}_{tx}_{rx}.parquet"
        )
    if name == "channel_dim_disjoint_4sim":
        return (
            _RESULTS
            / "disjoint_4sim_channel_dim"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{ch_tag}_{tx}_{rx}.parquet"
        )
    return None


def _setup(cfg: DictConfig):
    """Load data, compute A_target and No Mismatch accuracy. Returns shared context."""
    seed = cfg.training.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = load_semasia(
        hf_name=cfg.datasets.hf_name,
        tx_enc=cfg.data.tx_enc,
        rx_enc=cfg.data.rx_enc,
        label_col=cfg.datasets.label_col,
        train_split=cfg.datasets.train_split,
        test_split=cfg.datasets.test_split,
    )
    pilots = select_pilots(
        data.train,
        train_label_size=cfg.data.train_label_size,
        method=cfg.data.method,
        grouping=cfg.data.grouping,
        seed=seed,
    )
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

    no_mismatch_probe = (
        clf_accuracy(
            clf_probe,
            data.test.z_rx.to(device),
            data.test.labels.to(device),
        )
        * 100
    )
    no_mismatch_neural = (
        clf_accuracy(
            clf_neural,
            data.test.z_rx.to(device),
            data.test.labels.to(device),
        )
        * 100
        if clf_neural is not None
        else None
    )
    test_loader = DataLoader(
        TensorDataset(data.test.z_tx, data.test.labels),
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    input_c = complex_compressed_tensor(pilots.z_tx.T, device=device)
    output_c = complex_compressed_tensor(pilots.z_rx.T, device=device)
    input_w, L_in, mu_in = prewhiten(input_c, device=device)
    output_w, L_out, mu_out = prewhiten(output_c, device=device)

    _CSI_EXPERIMENTS = {
        "disjoint_csi_layers",
        "disjoint_csi_snr",
        "disjoint_4sim_layers",
        "disjoint_4sim_snr",
    }

    if cfg.alignment.type == "Linear":
        A_target = ridge_regression(input_w, output_w, lmb=cfg.alignment.lmb)
        F_T, F_R_H = None, None
    else:
        is_csi = cfg.experiment.name in _CSI_EXPERIMENTS
        if is_csi:
            # n_clusters must equal channel_dim so F_T and F_R_H are dimensionally
            # consistent with B_T and B_R (both live in the ch_dim space).
            ch_dim_cfg = cfg.experiment.get("channel_dim", None)
            n_k = (
                int(ch_dim_cfg)
                if ch_dim_cfg is not None
                else int(max(input_w.shape[0], output_w.shape[0]))
            )
        else:
            n_k = cfg.alignment.n_clusters

        F_T, F_R_H = ppfe(
            input_w,
            output_w,
            output_real=pilots.z_rx,
            n_clusters=n_k,
            n_proto=cfg.alignment.n_proto,
            seed=seed,
        )
        A_target = F_R_H @ F_T

        if not is_csi:
            F_T, F_R_H = None, None

    # Original: A_target applied to TX embeddings, no SIM, no channel
    original_probe = (
        run_original_accuracy(
            test_loader,
            A_target,
            L_in,
            mu_in,
            L_out,
            mu_out,
            clf_probe,
            device,
        )
        * 100
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
        * 100
        if clf_neural is not None
        else None
    )
    ch_dim_cfg = cfg.experiment.get("channel_dim", None)
    h_dim = ch_dim_cfg if ch_dim_cfg is not None else int(max(A_target.shape))
    H_mimo = complex_gaussian_matrix(0, 1, size=(h_dim, h_dim)).to(device)

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
        lr=cfg.training.lr,
        tx_enc=cfg.data.tx_enc,
        rx_enc=cfg.data.rx_enc,
        optimizer=cfg.training.get("optimizer", "adam"),
        scheduler=cfg.training.get("scheduler", "none"),
        warmup_frac=float(cfg.training.get("warmup_frac", 0.05)),
        F_T=F_T,
        F_R_H=F_R_H,
    )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    gpu = cfg.training.get("gpu", None)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    exp = cfg.experiment
    exp_name = exp.name

    out = _out_path(cfg)
    if out is not None and out.exists():
        print(f" Already done: {out.name} — skipping.")
        return

    print(
        f" seed={cfg.training.seed} | exp={exp_name} | "
        f"dataset={cfg.datasets.hf_name} | align={cfg.alignment.type}"
    )

    ctx = _setup(cfg)
    # F_T / F_R_H are only used by CSI-disjoint runners; pop them so the generic
    # **ctx spread doesn't break other runner signatures.
    F_T = ctx.pop("F_T")
    F_R_H = ctx.pop("F_R_H")
    snr = None if str(exp.get("snr", "null")) == "null" else exp.snr
    ch_dim = exp.get("channel_dim", None)

    match exp_name:
        case "layers":
            run_single_layers_config(L=exp.L, M_int=exp.M_int, channel_dim=ch_dim, **ctx)
        case "snr":
            run_single_snr_config(L=exp.L, M_int=exp.M_int, snr=snr, channel_dim=ch_dim, **ctx)
        case "slayer":
            run_single_slayer_config(
                L=exp.L, M_int=exp.M_int, s_tx=exp.s_tx, s_rx=exp.s_rx, channel_dim=ch_dim, **ctx
            )
        case "disjoint_layers":
            run_single_disjoint_layers(L=exp.L, M_int=exp.M_int, channel_dim=ch_dim, **ctx)
        case "disjoint_snr":
            run_single_disjoint_snr(L=exp.L, M_int=exp.M_int, snr=snr, channel_dim=ch_dim, **ctx)
        case "disjoint_csi_layers":
            run_single_disjoint_csi_layers(
                L=exp.L, M_int=exp.M_int, channel_dim=ch_dim, F_T=F_T, F_R_H=F_R_H, **ctx
            )
        case "disjoint_csi_snr":
            run_single_disjoint_csi_snr(
                L=exp.L, M_int=exp.M_int, snr=snr, channel_dim=ch_dim, F_T=F_T, F_R_H=F_R_H, **ctx
            )
        case "disjoint_4sim_layers":
            run_single_disjoint_4sim_layers(
                L=exp.L, M_int=exp.M_int, channel_dim=ch_dim, F_T=F_T, F_R_H=F_R_H, **ctx
            )
        case "disjoint_4sim_snr":
            run_single_disjoint_4sim_snr(
                L=exp.L, M_int=exp.M_int, snr=snr, channel_dim=ch_dim, F_T=F_T, F_R_H=F_R_H, **ctx
            )
        case "channel_dim_joint":
            run_single_layers_config(
                L=exp.L,
                M_int=exp.M_int,
                channel_dim=ch_dim,
                result_subdir="dual_channel_dim",
                **ctx,
            )
        case "channel_dim_disjoint":
            run_single_disjoint_layers(
                L=exp.L,
                M_int=exp.M_int,
                channel_dim=ch_dim,
                result_subdir="disjoint_channel_dim",
                **ctx,
            )
        case "channel_dim_disjoint_csi":
            run_single_disjoint_csi_layers(
                L=exp.L,
                M_int=exp.M_int,
                channel_dim=ch_dim,
                result_subdir="disjoint_csi_channel_dim",
                F_T=F_T,
                F_R_H=F_R_H,
                **ctx,
            )
        case "channel_dim_disjoint_4sim":
            run_single_disjoint_4sim_layers(
                L=exp.L,
                M_int=exp.M_int,
                channel_dim=ch_dim,
                result_subdir="disjoint_4sim_channel_dim",
                F_T=F_T,
                F_R_H=F_R_H,
                **ctx,
            )
        case _:
            print(f"Unknown experiment '{exp_name}'.")


if __name__ == "__main__":
    main()

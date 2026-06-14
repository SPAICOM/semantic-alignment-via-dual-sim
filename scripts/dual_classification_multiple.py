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
    just run-multi                              # layers sweep
    just run-multi experiment=snr
    just run-multi experiment=slayer
    just run-multi experiment=disjoint_layers
    just run-multi datasets=cifar100
    just run-multi alignment=linear
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

    if name == "layers":
        return (
            _RESULTS / "dual_layers" / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{tx}_{rx}.parquet"
        )
    if name == "snr":
        return (
            _RESULTS
            / "dual_snr"
            / f"{strategy}_seed{seed}_M{M}x{M}_snr{_snr_tag(exp.snr)}_{tx}_{rx}.parquet"
        )
    if name == "slayer":
        return (
            _RESULTS
            / "dual_slayer"
            / f"{strategy}_seed{seed}_stx{exp.s_tx}_srx{exp.s_rx}_{tx}_{rx}.parquet"
        )
    if name == "disjoint_layers":
        return (
            _RESULTS
            / "disjoint_layers"
            / f"{strategy}_seed{seed}_L{exp.L}_M{M}x{M}_{tx}_{rx}.parquet"
        )
    if name == "disjoint_snr":
        return (
            _RESULTS
            / "disjoint_snr"
            / f"{strategy}_seed{seed}_M{M}x{M}_snr{_snr_tag(exp.snr)}_{tx}_{rx}.parquet"
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

    if cfg.alignment.type == "Linear":
        A_target = ridge_regression(input_w, output_w, lmb=cfg.alignment.lmb)
    else:
        A_target = ppfe(
            input_w,
            output_w,
            output_real=pilots.z_rx,
            n_clusters=cfg.alignment.n_clusters,
            n_proto=cfg.alignment.n_proto,
            seed=seed,
        )

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
    H_mimo = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device)

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

    if exp_name == "layers":
        run_single_layers_config(L=exp.L, M_int=exp.M_int, **ctx)

    elif exp_name == "snr":
        run_single_snr_config(
            L=exp.L,
            M_int=exp.M_int,
            snr=None if str(exp.snr) == "null" else exp.snr,
            **ctx,
        )
    elif exp_name == "slayer":
        run_single_slayer_config(
            L=exp.L,
            M_int=exp.M_int,
            s_tx=exp.s_tx,
            s_rx=exp.s_rx,
            **ctx,
        )
    elif exp_name == "disjoint_layers":
        run_single_disjoint_layers(L=exp.L, M_int=exp.M_int, **ctx)

    elif exp_name == "disjoint_snr":
        run_single_disjoint_snr(
            L=exp.L,
            M_int=exp.M_int,
            snr=None if str(exp.snr) == "null" else exp.snr,
            **ctx,
        )
    else:
        print(f"Unknown experiment '{exp_name}'.")


if __name__ == "__main__":
    main()

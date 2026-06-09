"""Mono-SIM ablation — No Mismatch + TX-only SIM accuracy in one run.

Mirrors ``dual_classification_multiple.py`` but uses only the TX metasurface.
Supports two modes via ``experiment.use_channel``:

  False (default): y = β G_T x_w  — no channel, no noise
  True:            y = G_eq(H) · (H · β G_T · x_w + noise)  — MMSE equalisation

Always evaluates with the linear probe.  If a pre-trained neural classifier is
available (run ``just train-clf`` first), it is also evaluated and the results
are stored in the *_neural Parquet columns.

Examples
--------
    just run-mono                                      # no-channel sweep
    just run-mono experiment.use_channel=true          # with-channel sweep
    just run-mono alignment=linear
    uv run python scripts/mono_sim.py -m training.seed=42 experiment.L=10
"""

import os

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
from semantic_alignment.experiments.experiment_runner import run_single_mono_layers_config
from semantic_alignment.utils.alignment_utils import ppfe, ridge_regression
from semantic_alignment.utils.utils import (
    complex_compressed_tensor,
    complex_gaussian_matrix,
    prewhiten,
)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    exp = cfg.experiment
    use_channel = bool(exp.get("use_channel", False))
    snr = exp.get("snr", None)
    snr = None if str(snr) == "null" else snr

    print(
        f" seed={cfg.training.seed} | Mono-SIM{'(+H)' if use_channel else ''} | "
        f"L={exp.L} M_int={exp.M_int} | "
        f"dataset={cfg.datasets.hf_name} | align={cfg.alignment.type}"
    )

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
    H_mimo = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device) if use_channel else None

    run_single_mono_layers_config(
        L=exp.L,
        M_int=exp.M_int,
        no_mismatch_probe=no_mismatch_probe,
        original_probe=original_probe,
        A_target=A_target,
        test_loader=test_loader,
        clf=clf_probe,
        L_in=L_in,
        mu_in=mu_in,
        L_out=L_out,
        mu_out=mu_out,
        device=device,
        strategy_name=cfg.alignment.type,
        seed=seed,
        max_iters=cfg.training.max_iters,
        lr=cfg.training.lr,
        use_channel=use_channel,
        H_mimo=H_mimo,
        snr=snr,
        clf_neural=clf_neural,
        no_mismatch_neural=no_mismatch_neural,
        original_neural=original_neural,
        tx_enc=cfg.data.tx_enc,
        rx_enc=cfg.data.rx_enc,
        use_scheduler=bool(cfg.training.get("use_scheduler", False)),
    )


if __name__ == "__main__":
    main()

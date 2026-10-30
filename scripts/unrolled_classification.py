"""Deep-unrolling entry point for semantic classification.

Sets up the Semasia embeddings, alignment, linear probe and (optional) neural
classifier, wraps them in a :class:`ClassificationTask`, and runs the shared
task-agnostic pipeline (:func:`unrolled_pipeline.run_unrolled`), which trains
the unrolled step sizes on batches of random MIMO channels via
``UnrolledLightning`` and evaluates probe/neural accuracy on test channels.
Classic Adam baselines: ``compute_unrolled_baseline_classification.py``.
The reconstruction analogue is ``unrolled_reconstruction.py``.
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import hydra
import lightning as lg
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
from unrolled_pipeline import _draw_channel, run_unrolled

from semantic_alignment.data.prepare import prepare_semasia
from semantic_alignment.evaluation.linear_probe import train_linear_probe
from semantic_alignment.evaluation.neural_classifier import load_neural_clf
from semantic_alignment.evaluation.tasks import ClassificationTask

torch.set_float32_matmul_precision("high")


def _setup(cfg: DictConfig) -> dict:
    """Load data, set seed/device, return shared context."""
    seed = cfg.training.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lg.seed_everything(seed, workers=True)

    is_csi = cfg.sim.get("sim_type", "") in ("disjoint_sym", "disjoint_4sim")
    if cfg.alignment.type == "PPFE":
        ch_dim_cfg = cfg.sim.get("channel_dim", None)
        n_k = int(ch_dim_cfg) if ch_dim_cfg is not None and is_csi else cfg.alignment.n_clusters
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
        cluster_on=cfg.alignment.get("cluster_on", "pilots"),
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

    clf_probe = train_linear_probe(
        data.train.z_rx.to(device),
        data.train.labels.to(device),
        num_classes=cfg.datasets.num_classes,
    )

    try:
        clf_neural = load_neural_clf(cfg.datasets.hf_name, cfg.data.rx_enc, seed, device)
        print(" Neural classifier loaded.")
    except FileNotFoundError:
        clf_neural = None
        print(" Neural classifier not found — run 'just train-clf' to enable neural accuracy.")

    test_loader = DataLoader(
        TensorDataset(data.test.z_tx, data.test.labels),
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )
    task = ClassificationTask(
        test_loader, L_in, mu_in, L_out, mu_out, clf_probe, device, clf_neural=clf_neural
    )
    Z_test, labels = task.precompute_cascade_inputs()

    return dict(
        task=task,
        dataset=cfg.datasets.hf_name,
        seed=seed,
        device=device,
        A_target=A_target,
        H_mimo=_draw_channel(cfg, A_target, device),
        test_loader=test_loader,
        Z_test=Z_test,
        labels=labels,
        clf=clf_probe,
        L_in=L_in,
        mu_in=mu_in,
        L_out=L_out,
        mu_out=mu_out,
        clf_neural=clf_neural,
        strategy_name=cfg.alignment.type,
        lr=cfg.training.get("lr", 0.1),
        lr_tx=cfg.alignment.get("lr_tx", cfg.training.get("lr_tx", 0.1)),
        lr_rx=cfg.alignment.get("lr_rx", cfg.training.get("lr_rx", 0.2)),
        tx_enc=cfg.data.tx_enc,
        rx_enc=cfg.data.rx_enc,
        F_T=F_T,
        F_R_H=F_R_H,
    )


@hydra.main(version_base=None, config_path="../config", config_name="config_unrolled_joint")
def main(cfg: DictConfig) -> None:
    gpu = cfg.training.get("gpu", None)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    lg.seed_everything(cfg.training.seed, workers=True)
    ctx = _setup(cfg)
    run_unrolled(cfg, ctx, ctx["task"])


if __name__ == "__main__":
    main()

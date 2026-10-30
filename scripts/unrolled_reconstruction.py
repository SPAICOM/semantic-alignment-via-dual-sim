"""Deep-unrolling entry point for image reconstruction — the reconstruction analogue of ``unrolled_classification.py``.

Same pipeline as ``unrolled_classification.py`` (:func:`unrolled_pipeline.run_unrolled`: build the
SIM(s), train the unrolled step sizes on batches of random MIMO channels,
evaluate every unrolled step on the test channels); only the downstream task
changes. The TX/RX autoencoder pair, the alignment and the no-SIM baselines are
set up exactly as in ``reconstruction.py``, and a :class:`ReconstructionTask`
scores each step's cascade by the RX decoder's reconstruction MSE / PSNR.

Every unrolled step decodes the evaluation images once per test channel, so the
task scores the first ``reconstruction.unrolled_eval_images`` test images
(default 500) rather than the whole split.

Results go to ``results/reconstruction/unrolled/{mode}/``, filenames prefixed
with the autoencoder regularisation tag; the warm-start cache lives in
``results/reconstruction/unrolled/joint/warmstart/``.

Examples
--------
    just rec-unrolled-joint
    just rec-unrolled-linear warmstart=none channel_dm.h0_var=0
    uv run python scripts/unrolled_reconstruction.py --config-name config_unrolled_reconstruction \\
        hydra.mode=RUN training.K_list=[1,5] training.lr_w=[1e-3,5e-4]
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import hydra
import lightning as lg
import torch
from omegaconf import DictConfig
from reconstruction import _lrs, _tags
from reconstruction import _setup as _setup_reconstruction
from unrolled_pipeline import _draw_channel, run_unrolled

from semantic_alignment.models.autoencoder import load_autoencoder

torch.set_float32_matmul_precision("high")


def _setup(cfg: DictConfig, device: torch.device) -> dict:
    """Load the AE pair, align, and build the :class:`ReconstructionTask` + unrolled context."""
    rc = cfg.reconstruction
    dataset = cfg.datasets.name
    tx_enc, rx_enc, reg_tag = _tags(cfg)
    tx_model = load_autoencoder(dataset, rc.tx_variant, rc.tx_seed, device)
    rx_model = load_autoencoder(dataset, rc.rx_variant, rc.rx_seed, device)

    base = _setup_reconstruction(cfg, tx_model, rx_model, reg_tag, device)
    task = base["task"]
    task.cascade_images = rc.get("unrolled_eval_images", None)
    A_target = base["A_target"]
    strategy = cfg.alignment.type
    lr_tx, lr_rx = _lrs(cfg)
    return dict(
        task=task,
        dataset=dataset,
        seed=cfg.training.seed,
        device=device,
        A_target=A_target,
        H_mimo=_draw_channel(cfg, A_target, device),
        strategy_name=strategy,
        lr=cfg.training.get("lr", 0.1),
        lr_tx=lr_tx,
        lr_rx=lr_rx,
        tx_enc=tx_enc,
        rx_enc=rx_enc,
        F_T=base["F_T"],
        F_R_H=base["F_R_H"],
        # Warm-start caches are keyed on dataset + strategy (+ the reg_tag file prefix).
        ws_extra_key={"reg_tag": reg_tag},
    )


@hydra.main(
    version_base=None, config_path="../config", config_name="config_unrolled_reconstruction"
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point — set up the reconstruction task and run the unrolled pipeline."""
    gpu = cfg.training.get("gpu", None)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lg.seed_everything(cfg.training.seed, workers=True)
    ctx = _setup(cfg, device)
    run_unrolled(cfg, ctx, ctx["task"])


if __name__ == "__main__":
    main()

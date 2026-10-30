"""Train and save a ConvVAE autoencoder on raw images.

One checkpoint per (dataset, variant, seed) — saved to models/autoencoders/.
Training metrics are streamed to Weights & Biases.

The saved model's encoder half produces the z_tx/z_rx embeddings for
image-reconstruction alignment; train both variants (base, small) per dataset
to get a TX/RX pair, mirroring vit_base/vit_small in the classification
pipeline.

Examples
--------
    just train-ae                                     # full sweep from config
    just dataset=mnist train-ae
    uv run python scripts/train_autoencoder.py autoencoder=small training.seed=42
"""

import random

import hydra
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from semantic_alignment.data.reconstruction import load_split
from semantic_alignment.evaluation.reconstruction import psnr, reconstruction_mse
from semantic_alignment.models.autoencoder import (
    load_autoencoder,
    save_autoencoder,
    train_autoencoder,
)


@hydra.main(version_base=None, config_path="../config", config_name="config_train_autoencoder")
def main(cfg: DictConfig) -> None:
    """Train and persist a ConvVAE autoencoder for one (dataset, variant, seed) triple.

    Skips training if a checkpoint already exists for this configuration.
    Logs training metrics to Weights & Biases and writes the final checkpoint
    to ``models/autoencoders/``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration composed from ``config_train_autoencoder``.
    """
    seed = cfg.training.seed
    dataset = cfg.datasets.name
    variant = cfg.autoencoder.variant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f" seed={seed} | dataset={dataset} | variant={variant}")

    # Skip if already trained
    try:
        load_autoencoder(dataset, variant, seed, device)
        print(" Already trained — skipping.")
        return
    except FileNotFoundError:
        pass

    image_size = tuple(cfg.datasets.image_size)
    channels = cfg.datasets.channels

    print(f" Loading images: {dataset}")
    images_train, _ = load_split(
        cfg.datasets.hf_repo,
        cfg.datasets.hf_config,
        cfg.datasets.train_split,
        cfg.datasets.image_col,
        cfg.datasets.label_col,
        image_size,
        channels,
    )
    images_test, _ = load_split(
        cfg.datasets.hf_repo,
        cfg.datasets.hf_config,
        cfg.datasets.test_split,
        cfg.datasets.image_col,
        cfg.datasets.label_col,
        image_size,
        channels,
    )

    ae_cfg = cfg.autoencoder
    logger = WandbLogger(
        project="dual-sim-project-autoencoder",
        name=f"{dataset}_{variant}_seed{seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )

    print(
        f" Training ConvVAE(widths={list(ae_cfg.channel_widths)}, latent={ae_cfg.latent_dim}, "
        f"use_residual={ae_cfg.get('use_residual', False)}, "
        f"beta={ae_cfg.beta} (warmup={ae_cfg.beta_warmup_epochs}ep), l1_weight={ae_cfg.l1_weight}, "
        f"ssim_weight={ae_cfg.get('ssim_weight', 0.0)}, free_bits={ae_cfg.get('free_bits', 0.0)}, "
        f"sigma_vae={ae_cfg.get('sigma_vae', False)}, "
        f"mmd_weight={ae_cfg.get('mmd_weight', 0.0)} (warmup={ae_cfg.get('mmd_warmup_epochs', 10)}ep), "
        f"perceptual_weight={ae_cfg.get('perceptual_weight', 0.0)} "
        f"(start={ae_cfg.get('perceptual_start_epoch', 0)}ep, warmup={ae_cfg.get('perceptual_warmup_epochs', 10)}ep), "
        f"use_cosine_schedule={ae_cfg.get('use_cosine_schedule', False)}, "
        f"lr={ae_cfg.lr}, epochs={ae_cfg.max_epochs}, patience={ae_cfg.patience}) ..."
    )

    model = train_autoencoder(
        images=images_train,
        in_channels=channels,
        image_size=image_size,
        latent_dim=ae_cfg.latent_dim,
        channel_widths=list(ae_cfg.channel_widths),
        beta=ae_cfg.beta,
        beta_warmup_epochs=ae_cfg.beta_warmup_epochs,
        l1_weight=ae_cfg.l1_weight,
        ssim_weight=ae_cfg.get("ssim_weight", 0.0),
        ssim_warmup_epochs=ae_cfg.get("ssim_warmup_epochs", 10),
        free_bits=ae_cfg.get("free_bits", 0.0),
        sigma_vae=ae_cfg.get("sigma_vae", False),
        mmd_weight=ae_cfg.get("mmd_weight", 0.0),
        mmd_warmup_epochs=ae_cfg.get("mmd_warmup_epochs", 10),
        use_residual=ae_cfg.get("use_residual", False),
        perceptual_weight=ae_cfg.get("perceptual_weight", 0.0),
        perceptual_warmup_epochs=ae_cfg.get("perceptual_warmup_epochs", 10),
        perceptual_start_epoch=ae_cfg.get("perceptual_start_epoch", 0),
        use_cosine_schedule=ae_cfg.get("use_cosine_schedule", False),
        lr=ae_cfg.lr,
        max_epochs=ae_cfg.max_epochs,
        patience=ae_cfg.patience,
        batch_size=ae_cfg.batch_size,
        val_fraction=ae_cfg.val_fraction,
        device=device,
        seed=seed,
        logger=logger,
        gradient_clip_val=ae_cfg.get("gradient_clip_val", 1.0),
    )

    with torch.no_grad():
        x = images_test.to(device)
        x_hat, _, _ = model(x)
        test_mse = reconstruction_mse(x_hat, x)
        test_psnr = psnr(x_hat, x)

    print(f" Test MSE: {test_mse:.4f} | PSNR: {test_psnr:.2f} dB")
    logger.log_metrics({"test_mse": test_mse, "test_psnr": test_psnr})
    logger.experiment.finish()

    training_meta = {
        "beta": ae_cfg.beta,
        "sigma_vae": ae_cfg.get("sigma_vae", False),
        "mmd_weight": ae_cfg.get("mmd_weight", 0.0),
        "free_bits": ae_cfg.get("free_bits", 0.0),
        "use_residual": ae_cfg.get("use_residual", False),
        "perceptual_weight": ae_cfg.get("perceptual_weight", 0.0),
        "use_cosine_schedule": ae_cfg.get("use_cosine_schedule", False),
    }
    path = save_autoencoder(model, dataset, variant, seed, training_meta=training_meta)
    print(f" Saved to {path}")


if __name__ == "__main__":
    main()

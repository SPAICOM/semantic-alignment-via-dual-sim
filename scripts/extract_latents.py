"""Extract paired TX/RX latent embeddings from two trained ConvVAE checkpoints.

No precomputed embedding collection exists for reconstruction (classification
has Semasia) — this script is the local stand-in: loads the
``reconstruction.{tx,rx}_{variant,seed}`` autoencoders (see ``just train-ae``),
runs both encoders over the same raw images, and saves the paired
``(z_tx, z_rx, labels)`` train/test tensors to
``data/latents_reconstruction/{dataset}/``.

Examples
--------
    just extract-latents
    uv run python scripts/extract_latents.py reconstruction.tx_seed=27 reconstruction.rx_seed=27
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from semantic_alignment.data.reconstruction import load_reconstruction_pair
from semantic_alignment.models.autoencoder import load_autoencoder

LATENTS_DIR = Path("data/latents_reconstruction")


@hydra.main(version_base=None, config_path="../config", config_name="config_extract_latents")
def main(cfg: DictConfig) -> None:
    """Embed a dataset's images with two trained autoencoders and save the paired latents.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration composed from ``config_extract_latents``.
    """
    rc = cfg.reconstruction
    dataset = cfg.datasets.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f" dataset={dataset} | tx={rc.tx_variant}(seed{rc.tx_seed}) | "
        f"rx={rc.rx_variant}(seed{rc.rx_seed})"
    )

    tx_model = load_autoencoder(dataset, rc.tx_variant, rc.tx_seed, device)
    rx_model = load_autoencoder(dataset, rc.rx_variant, rc.rx_seed, device)

    data, _ = load_reconstruction_pair(
        hf_repo=cfg.datasets.hf_repo,
        hf_config=cfg.datasets.hf_config,
        tx_model=tx_model,
        rx_model=rx_model,
        image_col=cfg.datasets.image_col,
        label_col=cfg.datasets.label_col,
        image_size=tuple(cfg.datasets.image_size),
        channels=cfg.datasets.channels,
        train_split=cfg.datasets.train_split,
        test_split=cfg.datasets.test_split,
        device=device,
        batch_size=rc.batch_size,
        save_to_disk=rc.save_to_disk,
    )

    print(f" train: z_tx={tuple(data.train.z_tx.shape)} z_rx={tuple(data.train.z_rx.shape)}")
    print(f" test : z_tx={tuple(data.test.z_tx.shape)} z_rx={tuple(data.test.z_rx.shape)}")

    tag = f"tx_{rc.tx_variant}_seed{rc.tx_seed}__rx_{rc.rx_variant}_seed{rc.rx_seed}"
    out_path = LATENTS_DIR / dataset / f"{tag}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            split: {"z_tx": d.z_tx, "z_rx": d.z_rx, "labels": d.labels}
            for split, d in (("train", data.train), ("test", data.test))
        },
        out_path,
    )
    print(f" Saved to {out_path}")


if __name__ == "__main__":
    main()

"""Linear semantic alignment between two independently-trained ConvVAE autoencoders.

No SIM, no MIMO channel — the reconstruction-task analogue of the
*Original No MIMO* baseline (see ``evaluation/original_accuracy.py``): a
small pilot set is used to estimate a linear map A between the TX and RX
latent spaces, and reconstruction quality is measured by decoding the
aligned embedding with the RX decoder and comparing to the ground-truth
image. Represents the upper bound on what linear alignment alone can
achieve, before adding a physical metasurface/channel.

Reports three points of reference (logged to W&B):
  - oracle   : RX decodes its own true embedding (upper bound, no alignment needed)
  - aligned  : RX decodes A @ (whitened TX embedding) (the actual result)
  - mismatch : RX decodes a different sample's true embedding (lower-bound floor)

Examples
--------
    just align-ae
    uv run python scripts/align_reconstruction.py reconstruction.tx_seed=27 reconstruction.rx_seed=27
"""

import random

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from semantic_alignment.data.reconstruction import compute_alignment, prepare_reconstruction
from semantic_alignment.evaluation.reconstruction import (
    psnr,
    reconstruction_mse,
    run_original_reconstruction,
)
from semantic_alignment.models.autoencoder import load_autoencoder


def _to_wandb_image(x: torch.Tensor, caption: str) -> wandb.Image:
    """Convert one ``(C, H, W)`` image in ``[0, 1]`` into a ``wandb.Image``."""
    arr = x.detach().clamp(0, 1).cpu().numpy()
    arr = arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
    return wandb.Image(arr, caption=caption)


@hydra.main(version_base=None, config_path="../config", config_name="config_align_reconstruction")
def main(cfg: DictConfig) -> None:
    """Estimate a linear TX->RX alignment and evaluate reconstruction quality.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration composed from ``config_align_reconstruction``.
    """
    rc = cfg.reconstruction
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = cfg.datasets.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f" dataset={dataset} | tx={rc.tx_variant}(seed{rc.tx_seed}) | "
        f"rx={rc.rx_variant}(seed{rc.rx_seed}) | pilots/class={rc.train_label_size}"
    )

    tx_model = load_autoencoder(dataset, rc.tx_variant, rc.tx_seed, device)
    rx_model = load_autoencoder(dataset, rc.rx_variant, rc.rx_seed, device)

    print(" Embedding train/test splits with both encoders ...")
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
    A_target, _, _ = compute_alignment(
        prep.input_w, prep.output_w, prep.pilots.z_rx, "Linear", lmb=cfg.lmb
    )

    n = rc.num_preview
    z_rx_test = prep.data.test.z_rx
    images_test = prep.images_test.to(device)

    mse_aligned, psnr_aligned, recon_aligned_prev = run_original_reconstruction(
        prep.data.test.z_tx,
        prep.images_test,
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
    with torch.no_grad():
        recon_oracle = rx_model.decode(z_rx_test.to(device))
        recon_mismatch = rx_model.decode(z_rx_test[torch.randperm(len(z_rx_test))].to(device))
    mse_oracle, psnr_oracle = (
        reconstruction_mse(recon_oracle, images_test),
        psnr(recon_oracle, images_test),
    )
    mse_mismatch, psnr_mismatch = (
        reconstruction_mse(recon_mismatch, images_test),
        psnr(recon_mismatch, images_test),
    )

    print(f" oracle   (true z_rx)        : MSE={mse_oracle:.4f} | PSNR={psnr_oracle:.2f} dB")
    print(f" aligned  (A @ whitened z_tx): MSE={mse_aligned:.4f} | PSNR={psnr_aligned:.2f} dB")
    print(f" mismatch (shuffled z_rx)    : MSE={mse_mismatch:.4f} | PSNR={psnr_mismatch:.2f} dB")

    run = wandb.init(
        project="dual-sim-project-reconstruction-alignment",
        name=f"{dataset}_tx-{rc.tx_variant}{rc.tx_seed}_rx-{rc.rx_variant}{rc.rx_seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )
    run.log(
        {
            "mse_oracle": mse_oracle,
            "psnr_oracle": psnr_oracle,
            "mse_aligned": mse_aligned,
            "psnr_aligned": psnr_aligned,
            "mse_mismatch": mse_mismatch,
            "psnr_mismatch": psnr_mismatch,
        }
    )

    images = []
    for i in range(n):
        images.append(_to_wandb_image(prep.images_test[i], f"{i}_original"))
        images.append(_to_wandb_image(recon_oracle[i], f"{i}_oracle"))
        images.append(_to_wandb_image(recon_aligned_prev[i], f"{i}_aligned"))
    run.log({"comparison": images})
    run.finish()

    print(f" W&B run: {run.url}")


if __name__ == "__main__":
    main()

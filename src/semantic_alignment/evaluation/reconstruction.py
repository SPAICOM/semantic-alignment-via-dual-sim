"""Reconstruction-quality evaluation — the reconstruction-task analogue of :mod:`inference`.

Where the classification pipeline scores aligned RX embeddings by decoding
them into a class index, the reconstruction pipeline decodes them into an
image with the RX autoencoder and compares to the ground-truth pixels.

:func:`evaluate_reconstruction` takes any
:data:`~semantic_alignment.sim.link.Transmit` (whitened TX embeddings →
received signal, see :mod:`semantic_alignment.sim.link`) and wraps it with the
shared whitening convention

    x_white = L_in⁻¹ (x_complex − μ_in)
    ẑ       = L_out · y + μ_out

followed by the decode + metric pass.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_alignment.sim.link import Transmit, alignment_transmit
from semantic_alignment.utils import pack_complex, solve, unpack_complex

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def reconstruction_mse(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """Mean squared pixel error between reconstructed and ground-truth images.

    Parameters
    ----------
    x_hat : torch.Tensor
        Reconstructed images, shape ``(N, C, H, W)``.
    x : torch.Tensor
        Ground-truth images, same shape, values in ``[0, 1]``.

    Returns
    -------
    float
        Mean squared error.
    """
    return F.mse_loss(x_hat, x, reduction="mean").item()


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    """Convert an MSE into PSNR (dB); ``inf`` for a pixel-perfect reconstruction."""
    return 10.0 * math.log10(max_val**2 / mse) if mse > 0 else float("inf")


def psnr(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB between reconstructed and ground-truth images.

    Parameters
    ----------
    x_hat : torch.Tensor
        Reconstructed images, shape ``(N, C, H, W)``.
    x : torch.Tensor
        Ground-truth images, same shape.
    max_val : float, optional
        Maximum possible pixel value (``1.0`` for images normalised to ``[0, 1]``).

    Returns
    -------
    float
        PSNR in dB; ``inf`` if the reconstruction is pixel-perfect.
    """
    return psnr_from_mse(reconstruction_mse(x_hat, x), max_val)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_reconstruction(
    transmit: Transmit,
    z_tx_test: torch.Tensor,
    images_test: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    rx_model: nn.Module,
    device: torch.device,
    batch_size: int = 512,
    num_preview: int = 0,
) -> tuple[float, float, torch.Tensor | None]:
    """Push the test set through ``transmit`` and score the RX decoder's reconstructions.

    Parameters
    ----------
    transmit : Transmit
        Maps whitened TX embeddings ``(D_tx/2, B)`` to the received signal
        ``(D_rx/2, B)`` (see :mod:`semantic_alignment.sim.link`).
    z_tx_test : torch.Tensor
        TX embeddings of the test split, shape ``(N, D_tx)``.
    images_test : torch.Tensor
        Ground-truth test images, shape ``(N, C, H, W)``, values in ``[0, 1]``.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Pilot whitening statistics.
    rx_model : ConvVAE
        Trained RX autoencoder; only its decoder half is used.
    device : torch.device
        Inference device.
    batch_size : int, optional
        Mini-batch size for the decode + metric pass.
    num_preview : int, optional
        Number of leading test reconstructions to return for a comparison grid.

    Returns
    -------
    mse, psnr_db : float
        Reconstruction MSE and PSNR (dB) over the whole test split.
    preview : torch.Tensor or None
        First ``num_preview`` reconstructions ``(n, C, H, W)``; ``None`` if
        ``num_preview == 0``.
    """
    rx_model = rx_model.to(device).eval()
    se_sum, n_pixels = 0.0, 0
    preview = []

    with torch.no_grad():
        for i in range(0, len(z_tx_test), batch_size):
            x_batch = z_tx_test[i : i + batch_size].to(device)
            img_batch = images_test[i : i + batch_size].to(device)

            x_white = solve(L_in, pack_complex(x_batch.T, device=device) - mu_in)
            y = transmit(x_white)
            z_hat = (L_out.to(y.dtype) @ y) + mu_out.to(y.dtype)
            recon = rx_model.decode(unpack_complex(z_hat, device=device).T)

            se_sum += F.mse_loss(recon, img_batch, reduction="sum").item()
            n_pixels += img_batch.numel()
            if sum(len(p) for p in preview) < num_preview:
                preview.append(recon)

    mse = se_sum / n_pixels
    preview_t = torch.cat(preview)[:num_preview] if num_preview > 0 else None
    return mse, psnr_from_mse(mse), preview_t


def run_original_reconstruction(
    z_tx_test: torch.Tensor,
    images_test: torch.Tensor,
    A_target: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    rx_model: nn.Module,
    device: torch.device,
    batch_size: int = 512,
    num_preview: int = 0,
) -> tuple[float, float, torch.Tensor | None]:
    """Apply ``A_target`` to TX embeddings (no SIM, no channel) and score the reconstruction.

    The reconstruction-task analogue of
    :func:`semantic_alignment.evaluation.original_accuracy.run_original_accuracy`:
    an upper bound on what the alignment strategy alone can achieve before any
    hardware or channel impairments are introduced.

    Returns
    -------
    mse, psnr_db, preview
        See :func:`evaluate_reconstruction`.
    """
    return evaluate_reconstruction(
        alignment_transmit(A_target),
        z_tx_test,
        images_test,
        L_in,
        mu_in,
        L_out,
        mu_out,
        rx_model,
        device,
        batch_size=batch_size,
        num_preview=num_preview,
    )


def save_comparison_grid(
    columns: dict[str, torch.Tensor], out_path: Path, pad: int = 6, scale: int = 4
) -> Path:
    """Save a PNG grid comparing several reconstructions against the ground truth.

    Tiles are upscaled (nearest-neighbour) by ``scale`` before compositing —
    source images are typically tiny (28x28/32x32), too small to both inspect
    and fit a readable column header otherwise.

    Parameters
    ----------
    columns : dict of str -> torch.Tensor
        Ordered ``{label: images}`` pairs, each of shape ``(N, C, H, W)`` in
        ``[0, 1]``; all must share the same ``N``. Typical keys: ``"original"``,
        ``"no_mismatch"``, ``"{strategy}_align"`` (A-only, no channel), ``"sim_{strategy}"``.
    out_path : Path
        Where to write the PNG (parent directories are created as needed).
    pad : int, optional
        Pixel padding between tiles and around the column-label header.
    scale : int, optional
        Integer upscale factor applied to each tile before compositing.

    Returns
    -------
    Path
        ``out_path``, for chaining.
    """
    from PIL import Image, ImageDraw

    labels = list(columns.keys())
    first = next(iter(columns.values()))
    n, c, h, w = first.shape
    h, w = h * scale, w * scale

    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_widths = [draw_probe.textlength(label) for label in labels]
    col_w = int(max(w, *text_widths) + pad) if labels else w
    label_height = 20

    grid_w = len(labels) * col_w + pad
    grid_h = label_height + pad + n * h + (n + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)

    for col, label in enumerate(labels):
        x = pad + col * col_w
        draw.text((x, 2), label, fill="black")
        imgs = columns[label]
        for row in range(n):
            arr = imgs[row].detach().clamp(0, 1).cpu().numpy()
            arr = arr[0] if c == 1 else arr.transpose(1, 2, 0)
            arr = (arr * 255).astype("uint8")
            tile = Image.fromarray(arr, mode="L" if c == 1 else "RGB").convert("RGB")
            tile = tile.resize((w, h), resample=Image.NEAREST)
            y = label_height + pad + row * (h + pad)
            canvas.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


if __name__ == "__main__":
    x = torch.rand(4, 3, 32, 32)
    x_hat = x + 0.05 * torch.randn_like(x)
    print(f"reconstruction_mse: {reconstruction_mse(x_hat, x):.4f}")
    print(f"psnr              : {psnr(x_hat, x):.2f} dB")

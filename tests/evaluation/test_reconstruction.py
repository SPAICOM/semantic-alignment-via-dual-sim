"""Tests for evaluation/reconstruction.py: metrics and the evaluation loop."""

import math

import pytest
import torch
import torch.nn as nn
from PIL import Image

from semantic_alignment.evaluation.reconstruction import (
    evaluate_reconstruction,
    psnr,
    psnr_from_mse,
    reconstruction_mse,
    run_original_reconstruction,
    save_comparison_grid,
)
from semantic_alignment.sim.link import alignment_transmit
from semantic_alignment.utils import pack_complex, unpack_complex


def _cplx(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.complex64, generator=g)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_psnr_matches_definition():
    x = torch.zeros(1, 1, 2, 2)
    x_hat = torch.full_like(x, 0.1)
    assert reconstruction_mse(x_hat, x) == pytest.approx(0.01)
    assert psnr(x_hat, x) == pytest.approx(20.0)
    assert psnr(x, x) == math.inf
    assert psnr_from_mse(0.0) == math.inf


# ---------------------------------------------------------------------------
# evaluate_reconstruction
# ---------------------------------------------------------------------------


class _Decoder(nn.Module):
    """Maps a real latent ``(N, 2)`` to a 1×1×2 'image' (identity decode)."""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.view(-1, 1, 1, 2)


def test_evaluate_reconstruction_identity_link_is_perfect():
    """Identity whitening + identity link ⇒ the decoder sees the TX latent exactly."""
    z = torch.rand(7, 2)
    images = z.view(-1, 1, 1, 2).clone()
    eye, zero = torch.eye(1, dtype=torch.complex64), torch.zeros(1, 1, dtype=torch.complex64)
    mse, psnr_db, preview = evaluate_reconstruction(
        alignment_transmit(eye),
        z,
        images,
        eye,
        zero,
        eye,
        zero,
        _Decoder(),
        torch.device("cpu"),
        batch_size=3,
        num_preview=5,
    )
    assert mse == pytest.approx(0.0, abs=1e-12)
    assert preview.shape == (5, 1, 1, 2)
    torch.testing.assert_close(preview, images[:5])


def test_run_original_reconstruction_matches_manual_pipeline():
    z = torch.rand(6, 4)
    A = _cplx(2, 2)
    L = torch.eye(2, dtype=torch.complex64) * 2
    mu = _cplx(2, 1, seed=3)
    dec = type("D", (nn.Module,), {"decode": lambda self, z: z.view(-1, 1, 2, 2)})()

    x_w = torch.linalg.solve(L, pack_complex(z.T) - mu)
    expected = unpack_complex(L @ (A @ x_w) + mu).T.view(-1, 1, 2, 2)
    images = torch.rand(6, 1, 2, 2)

    mse, _, preview = run_original_reconstruction(
        z, images, A, L, mu, L, mu, dec, torch.device("cpu"), num_preview=6
    )
    torch.testing.assert_close(preview, expected)
    assert mse == pytest.approx(reconstruction_mse(expected, images), rel=1e-5)


def test_save_comparison_grid(tmp_path):
    cols = {"original": torch.rand(3, 3, 4, 4), "sim": torch.rand(3, 3, 4, 4)}
    out = save_comparison_grid(cols, tmp_path / "sub" / "grid.png", pad=2, scale=2)
    with Image.open(out) as img:
        assert img.mode == "RGB"
        assert img.height == 20 + 2 + 3 * 8 + 4 * 2  # label + pad + rows + gaps

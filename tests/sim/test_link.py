"""Tests for sim/link.py: the task-agnostic transmit builders."""

import torch

from semantic_alignment.sim.link import (
    alignment_transmit,
    awgn_sigma,
    cascade_transmit,
    dual_transmit,
    equalized_transmit,
)


def _cplx(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.complex64, generator=g)


def test_awgn_sigma():
    assert awgn_sigma(None) is None
    assert awgn_sigma(0.0) == 1.0
    assert abs(awgn_sigma(20.0) - 0.1) < 1e-12


def test_alignment_transmit_is_matrix_product():
    A, x = _cplx(3, 4), _cplx(4, 5, seed=1)
    torch.testing.assert_close(alignment_transmit(A)(x), A @ x)


def test_cascade_transmit_noiseless_and_noisy():
    Z, x = _cplx(3, 4), _cplx(4, 5, seed=1)
    beta = torch.tensor(0.5 - 0.25j)
    torch.testing.assert_close(cascade_transmit(Z, beta, None)(x), beta * (Z @ x))
    y = cascade_transmit(Z, beta, 0.0)(x)  # SNR 0 dB → σ = 1: noise is clearly visible
    assert not torch.allclose(y, beta * (Z @ x))


def test_dual_transmit_matches_cascade_product():
    G_T, H, G_R, x = _cplx(4, 3), _cplx(4, 4, seed=1), _cplx(2, 4, seed=2), _cplx(3, 5, seed=3)
    beta = torch.tensor(1.5 + 0.5j)
    y = dual_transmit(G_T, H, G_R, beta, None)(x)
    torch.testing.assert_close(y, cascade_transmit(G_R @ H @ G_T, beta, None)(x))


def test_dual_transmit_noise_uses_global_rng():
    G_T, H, G_R, x = _cplx(4, 3), _cplx(4, 4, seed=1), _cplx(2, 4, seed=2), _cplx(3, 5, seed=3)
    transmit = dual_transmit(G_T, H, G_R, torch.tensor(1.0 + 0j), 10.0)
    torch.manual_seed(0)
    y1 = transmit(x)
    torch.manual_seed(0)
    torch.testing.assert_close(transmit(x), y1)


def test_equalized_transmit_undoes_channel():
    """With an invertible H and no noise, ZF equalisation leaves β G_T x."""
    G_T, x = _cplx(4, 4), _cplx(4, 5, seed=1)
    H = _cplx(4, 4, seed=2) + 3 * torch.eye(4, dtype=torch.complex64)
    beta = torch.tensor(2.0 + 0j)
    y = equalized_transmit(G_T, H, beta, None)(x)
    torch.testing.assert_close(y, beta * (G_T @ x), rtol=1e-4, atol=1e-4)
    y_raw = equalized_transmit(G_T, H, beta, None, equalize=False)(x)
    torch.testing.assert_close(y_raw, H @ (beta * G_T @ x))

"""Physical-link transmit functions — the task-agnostic output of a trained SIM.

A *transmit* function maps whitened TX embeddings ``x_white`` (complex,
``(D_tx/2, B)``) to the received signal ``y`` (complex, ``(D_rx/2, B)``),
noise included. SIM runners build one from their trained cascades; a
downstream task (:mod:`semantic_alignment.evaluation.tasks`) de-whitens ``y``
and scores it (classification accuracy, reconstruction PSNR, ...).

Noise follows the whitening convention: TX embeddings are whitened
(``P_TX = 1``), so ``σ = 1 / √SNR_linear``. AWGN is drawn from the global
torch RNG on every call.

Builders
--------
alignment_transmit  — ``y = A x`` (no SIM, no channel: the *Original* baseline).
cascade_transmit    — ``y = β Z x + v`` for a precomputed effective cascade
                      (joint curve: ``G_R H G_T``; constructive mono: ``H G_T`` / ``G_R H``).
dual_transmit       — ``y = β G_R H G_T x + v`` (joint / disjoint Dual-SIM).
equalized_transmit  — ``y = G_eq (H β G_T x + v)`` (equalized TX Mono-SIM).
"""

import math
from collections.abc import Callable

import torch

from semantic_alignment.utils import awgn, rx_equalizer

Transmit = Callable[[torch.Tensor], torch.Tensor]


def awgn_sigma(snr_db: float | None) -> float | None:
    """AWGN standard deviation for whitened (unit-power) TX signals; ``None`` if noiseless."""
    return 1.0 / math.sqrt(10 ** (snr_db / 10)) if snr_db is not None else None


def alignment_transmit(A_target: torch.Tensor) -> Transmit:
    """``y = A x`` — alignment applied in software, no SIM, no channel, no noise."""
    A = torch.as_tensor(A_target, dtype=torch.complex64)
    return lambda x: A.to(x.device) @ x.to(A.dtype)


def cascade_transmit(Z: torch.Tensor, beta: torch.Tensor, snr_db: float | None) -> Transmit:
    """``y = β Z x + v`` for a precomputed effective cascade ``Z``."""
    sigma = awgn_sigma(snr_db)

    def transmit(x: torch.Tensor) -> torch.Tensor:
        y = beta * (Z @ x.to(Z.dtype))
        if sigma is not None:
            y = y + awgn(sigma, y.shape, device=y.device)
        return y

    return transmit


def dual_transmit(
    G_T: torch.Tensor,
    H_mimo: torch.Tensor,
    G_R: torch.Tensor,
    beta: torch.Tensor,
    snr_db: float | None,
) -> Transmit:
    """``y = β G_R H G_T x + v`` — TX SIM, channel, RX SIM applied in sequence.

    β is applied at the transmitter; AWGN enters after the cascade and is
    *not* scaled by β.
    """
    sigma = awgn_sigma(snr_db)

    def transmit(x: torch.Tensor) -> torch.Tensor:
        y = beta * (G_R @ (H_mimo @ (G_T @ x.to(G_T.dtype))))
        if sigma is not None:
            y = y + awgn(sigma, y.shape, device=y.device)
        return y

    return transmit


def equalized_transmit(
    G_T: torch.Tensor,
    H_mimo: torch.Tensor,
    beta: torch.Tensor,
    snr_db: float | None,
    mmse: bool = False,
    equalize: bool = True,
) -> Transmit:
    """``y = G_eq (H β G_T x + v)`` — TX Mono-SIM with a digital RX equalizer.

    ``G_eq`` is ZF (``H†``) unless ``mmse``, in which case it is regularised
    with ``snr_db``. ``equalize=False`` drops ``G_eq`` (``y = H β G_T x + v``).
    """
    G_eq = rx_equalizer(H_mimo, snr_db if mmse else None) if equalize else None
    sigma = awgn_sigma(snr_db)

    def transmit(x: torch.Tensor) -> torch.Tensor:
        y = H_mimo @ (beta * G_T @ x.to(G_T.dtype))
        if sigma is not None:
            y = y + awgn(sigma, y.shape, device=y.device)
        return G_eq @ y if G_eq is not None else y

    return transmit

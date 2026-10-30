"""Inference routines for evaluating semantic classification accuracy over-the-air.

All functions share the same whitening convention:
    x_white = L_in⁻¹ (x_complex − μ_in)
    ŷ       = L_out · y_received + μ_out

**Noise power convention**
AWGN sigma is defined relative to the *transmitted* signal power, which equals 1
because the TX embeddings are whitened.  Therefore:

    σ = 1 / √SNR_linear

This is computed once outside the batch loop and is the same for every batch.

Every SIM architecture is expressed as a :data:`~semantic_alignment.sim.link.Transmit`
(``x_white → y``, noise included); :func:`transmit_accuracy` is the single
classification loop over the test set. The per-architecture functions below are
thin wrappers that build the matching transmit.

Functions
---------
transmit_accuracy           — accuracy through any transmit function.
run_evaluation              — joint Dual-SIM (G_R · H · G_T), optional AWGN.
run_evaluation_mmse         — digital MMSE/ZF baseline (no SIM).
run_evaluation_mono         — TX-only SIM with optional MMSE equalizer at RX.
run_evaluation_disjoint_fixed — disjoint Dual-SIM with frozen TX and RX cascades.
run_evaluation_cascade      — any precomputed effective cascade Z.
cascade_accuracies          — probe/neural accuracy of a noiseless cascade on precomputed tensors.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from semantic_alignment.evaluation.linear_probe import (
    clf_accuracy,
    clf_predict,
    linear_probe_accuracy,
)
from semantic_alignment.sim.link import (
    Transmit,
    cascade_transmit,
    dual_transmit,
    equalized_transmit,
)
from semantic_alignment.utils import (
    awgn,
    pack_complex,
    rx_equalizer,
    solve,
    unpack_complex,
)


def transmit_accuracy(
    transmit: Transmit,
    dataloader: DataLoader,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf: torch.Tensor | nn.Module,
    device: torch.device,
) -> float:
    """Top-1 accuracy of ``clf`` on test embeddings sent through ``transmit``.

    Per batch: ``x_white = L_in⁻¹ (x − μ_in)`` → ``y = transmit(x_white)`` →
    ``ŷ = L_out y + μ_out`` → classify.

    Parameters
    ----------
    transmit : Transmit
        Whitened TX embeddings → received signal (see :mod:`semantic_alignment.sim.link`).
    dataloader : DataLoader
        Test set yielding ``(z_tx_batch, labels)`` pairs.
    L_in, mu_in : torch.Tensor
        TX whitening statistics.
    L_out, mu_out : torch.Tensor
        RX whitening statistics.
    clf : torch.Tensor or nn.Module
        Linear probe or neural classifier.
    device : torch.device
        Inference device.

    Returns
    -------
    float
        Top-1 classification accuracy in ``[0, 1]``.
    """
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_real_batch, labels_batch in dataloader:
            x_real_batch = x_real_batch.to(device)
            labels_batch = labels_batch.to(device)

            x_complex = pack_complex(x_real_batch.T, device=device)
            x_white = solve(L_in, x_complex - mu_in)

            y = transmit(x_white)
            z_hat = (L_out.to(y.dtype) @ y) + mu_out.to(y.dtype)
            y_hat_real = unpack_complex(z_hat, device=device).T

            preds = clf_predict(clf, y_hat_real)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    return (np.array(all_preds) == np.array(all_labels)).mean()


def cascade_accuracies(
    Z_eff: torch.Tensor,
    beta: torch.Tensor,
    Z_test: torch.Tensor,
    labels: torch.Tensor,
    W_probe: torch.Tensor,
    clf_neural: nn.Module | None,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    device: torch.device,
) -> tuple[float, float | None]:
    """Evaluate probe and neural classifier accuracies through the optimized cascade.

    Parameters
    ----------
    Z_eff : Tensor
        End-to-end cascade matrix.
    beta : Tensor
        MMSE scaling factor.
    Z_test : Tensor
        Test input matrix, shape (n_dim, N).
    labels : Tensor
        Test labels, shape (N,).
    W_probe : Tensor
        Linear probe weight matrix.
    clf_neural : nn.Module or None
        Neural classifier (optional).
    L_out : Tensor
        Output whitening lower-triangular factor.
    mu_out : Tensor
        Output whitening mean.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[float, float or None]
        ``(probe_acc, neural_acc)`` — accuracies in [0, 1].
    """
    dev = Z_eff.device
    cdtype = Z_eff.dtype
    Z_test = Z_test.to(device=dev, dtype=cdtype)
    labels = labels.to(dev)
    W_probe = W_probe.to(device=dev, dtype=torch.float32)
    L_out = L_out.to(device=dev, dtype=cdtype)
    mu_out = mu_out.to(device=dev, dtype=cdtype)
    if clf_neural is not None:
        clf_neural = clf_neural.to(dev)
    z_aligned = beta * (Z_eff @ Z_test)
    z_aligned = (L_out @ z_aligned) + mu_out
    z_aligned = unpack_complex(z_aligned, device).T
    probe_acc = linear_probe_accuracy(W_probe, z_aligned.float(), labels)
    neural_acc = None
    if clf_neural is not None:
        neural_acc = clf_accuracy(clf_neural, z_aligned.float(), labels)
    return probe_acc, neural_acc


def run_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    H_mimo: torch.Tensor,
    snr_db: float | None,
    beta_opt: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf: torch.Tensor | nn.Module,
    device: torch.device,
) -> float:
    """Evaluate semantic accuracy for the joint Dual-SIM architecture.

    Signal model: ``ŷ = β G_R H G_T x_white + v`` (:func:`~semantic_alignment.sim.link.dual_transmit`).

    Parameters
    ----------
    model : DualSIMoptimizerTorch
        Trained joint Dual-SIM model.
    dataloader : DataLoader
        Test set yielding ``(z_tx_batch, labels)`` pairs.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    snr_db : float or None
        If given, AWGN noise is added at this SNR level; ``None`` means noiseless.
    beta_opt : torch.Tensor
        Optimal scalar aligning ``Z`` to the target matrix ``A``.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Pilot whitening statistics.
    clf : torch.Tensor or nn.Module
        Linear probe or neural classifier.
    device : torch.device
        Inference device.

    Returns
    -------
    float
        Top-1 classification accuracy in ``[0, 1]``.
    """
    model.eval()
    with torch.no_grad():
        G_T = model.get_TX_cascade()
        G_R = model.get_RX_cascade()
    transmit = dual_transmit(G_T, H_mimo, G_R, beta_opt, snr_db)
    return transmit_accuracy(transmit, dataloader, L_in, mu_in, L_out, mu_out, clf, device)


def run_evaluation_mmse(
    dataloader: DataLoader,
    H_mimo: torch.Tensor,
    snr_db: float,
    A_target: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf: torch.Tensor | nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the digital MMSE baseline (no SIM, SVD-based equalizer).

    Signal model: ``y = H x_pad + n``,  ``x_recovered = G_mmse y``,
    then ``A_target`` is applied digitally.

    Parameters
    ----------
    dataloader : DataLoader
        Test set yielding ``(z_tx_batch, labels)`` pairs.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    snr_db : float
        Operating SNR in dB for AWGN and MMSE regularization.
    A_target : torch.Tensor
        Alignment matrix applied after equalization.
    L_in : torch.Tensor
        Cholesky factor of the TX pilot covariance.
    mu_in : torch.Tensor
        TX pilot mean.
    L_out : torch.Tensor
        Cholesky factor of the RX pilot covariance.
    mu_out : torch.Tensor
        RX pilot mean.
    clf : torch.Tensor or nn.Module
        Linear probe or neural classifier.
    device : torch.device
        Inference device.

    Returns
    -------
    float
        Top-1 classification accuracy in ``[0, 1]``.
    """
    all_preds, all_labels = [], []
    G_mmse = rx_equalizer(H_mimo, snr_db)
    # σ = 1/√SNR — TX signal is whitened, so P_TX = 1
    sigma = 1.0 / math.sqrt(10 ** (snr_db / 10)) if snr_db is not None else None

    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)

        x_complex = pack_complex(x_real_batch.T, device=device)
        x_white = solve(L_in, x_complex - mu_in).to(H_mimo.dtype)

        D_tx = x_white.shape[0]
        D_ch = H_mimo.shape[1]
        padding = torch.zeros((D_ch - D_tx, x_white.shape[1]), device=device, dtype=x_white.dtype)
        x_padded = torch.cat([x_white, padding], dim=0)

        y_signal = H_mimo @ x_padded
        if sigma is not None:
            y_received = y_signal + awgn(sigma, y_signal.shape, device=device)
        else:
            y_received = y_signal

        x_recovered = (G_mmse @ y_received)[:D_tx, :]
        z_hat_complex = A_target @ x_recovered
        z_hat = (L_out @ z_hat_complex) + mu_out
        y_hat_real = unpack_complex(z_hat, device=device).T

        preds = clf_predict(clf, y_hat_real)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    return (np.array(all_preds) == np.array(all_labels)).mean()


def run_evaluation_mono(
    G_T: torch.Tensor,
    dataloader: DataLoader,
    H_mimo: torch.Tensor,
    snr_db: float | None,
    beta_opt: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf: torch.Tensor | nn.Module,
    device: torch.device,
    use_equalizer: bool = True,
    mmse: bool = False,
) -> float:
    """Evaluate a Mono-SIM (TX only) over a MIMO channel.

    Signal model (:func:`~semantic_alignment.sim.link.equalized_transmit`):
    ``ŷ = G_eq (H β G_T x + v)``; ``use_equalizer=False`` drops ``G_eq``.

    Parameters
    ----------
    G_T : torch.Tensor
        TX cascade matrix from the trained Mono-SIM.
    dataloader : DataLoader
        Test set yielding ``(z_tx_batch, labels)`` pairs.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    snr_db : float or None
        AWGN SNR in dB; ``None`` means noiseless.
    beta_opt : torch.Tensor
        Optimal scalar aligning ``G_T`` to the target ``A``.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Pilot whitening statistics.
    clf : torch.Tensor or nn.Module
        Linear probe or neural classifier.
    device : torch.device
        Inference device.
    use_equalizer : bool, optional
        If ``True`` (default) apply the ZF/MMSE equalizer ``G_eq``.
    mmse : bool, optional
        ``False`` (default): ZF — ``G_eq = H†``. ``True``: MMSE regularised with ``snr_db``.

    Returns
    -------
    float
        Top-1 classification accuracy in ``[0, 1]``.
    """
    transmit = equalized_transmit(G_T, H_mimo, beta_opt, snr_db, mmse=mmse, equalize=use_equalizer)
    return transmit_accuracy(transmit, dataloader, L_in, mu_in, L_out, mu_out, clf, device)


def run_evaluation_disjoint_fixed(
    tx_model: nn.Module,
    rx_model: nn.Module,
    dataloader: DataLoader,
    H_mimo: torch.Tensor,
    snr_db: float | None,
    beta_global: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf: torch.Tensor | nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the disjoint Dual-SIM architecture with frozen TX and RX cascades.

    Signal model: ``ŷ = β G_R H G_T x_w + v`` (:func:`~semantic_alignment.sim.link.dual_transmit`).

    Parameters
    ----------
    tx_model, rx_model : MonoSIMoptimizerTorch
        Trained TX-side / RX-side Mono-SIMs (anything with ``get_cascade()``).
    dataloader : DataLoader
        Test set yielding ``(z_tx_batch, labels)`` pairs.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    snr_db : float or None
        AWGN SNR in dB; ``None`` means noiseless.
    beta_global : torch.Tensor
        Global scalar aligning the cascade ``G_R H G_T`` to the target ``A``.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Pilot whitening statistics.
    clf : torch.Tensor or nn.Module
        Linear probe or neural classifier.
    device : torch.device
        Inference device.

    Returns
    -------
    float
        Top-1 classification accuracy in ``[0, 1]``.
    """
    tx_model.eval()
    rx_model.eval()
    with torch.no_grad():
        G_T = tx_model.get_cascade()
        G_R = rx_model.get_cascade()
    transmit = dual_transmit(G_T, H_mimo, G_R, beta_global, snr_db)
    return transmit_accuracy(transmit, dataloader, L_in, mu_in, L_out, mu_out, clf, device)


def run_evaluation_cascade(
    Z: torch.Tensor,
    dataloader: DataLoader,
    snr_db: float | None,
    beta_opt: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf: torch.Tensor | nn.Module,
    device: torch.device,
) -> float:
    """Evaluate from a precomputed effective cascade ``Z``.

    Signal model: ``ŷ = β Z x + v`` (:func:`~semantic_alignment.sim.link.cascade_transmit`) —
    joint: ``Z = G_R H G_T``; constructive mono: ``H G_T`` / ``G_R H``.

    Parameters
    ----------
    Z : torch.Tensor
        Precomputed effective cascade matrix (complex).
    dataloader : DataLoader
        Test set yielding ``(z_tx_batch, labels)`` pairs.
    snr_db : float or None
        AWGN SNR in dB; ``None`` means noiseless.
    beta_opt : torch.Tensor
        Optimal alignment scalar.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Pilot whitening statistics.
    clf : torch.Tensor or nn.Module
        Linear probe or neural classifier.
    device : torch.device
        Inference device.

    Returns
    -------
    float
        Top-1 classification accuracy in ``[0, 1]``.
    """
    transmit = cascade_transmit(Z, beta_opt, snr_db)
    return transmit_accuracy(transmit, dataloader, L_in, mu_in, L_out, mu_out, clf, device)


if __name__ == "__main__":
    print("inference.py — evaluation functions for each SIM architecture:")
    print("  run_evaluation               : Joint Dual-SIM  (β G_R H G_T x + v, β at TX)")
    print("  run_evaluation_mmse          : Digital MMSE baseline (no SIM)")
    print("  run_evaluation_mono          : Mono-SIM TX-only + MMSE equalizer")
    print("  run_evaluation_disjoint_fixed: Disjoint Dual-SIM with frozen cascades")
    print()
    print("Called automatically by experiment runners. Invoke via: just run-joint / run-mono / ...")

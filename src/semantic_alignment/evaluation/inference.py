"""Inference routines for evaluating semantic classification accuracy over-the-air.

All functions share the same whitening convention:
    x_white = L_in⁻¹ (x_complex − μ_in)
    ŷ       = L_out · y_received + μ_out

where L_in, μ_in come from prewhitening the TX pilots and L_out, μ_out from the
RX pilots.  The received signal y_received is then fed to a classifier (linear
probe or neural MLP) to obtain top-1 class predictions.

Functions
---------
run_evaluation              — joint Dual-SIM (G_R · H · G_T), optional AWGN.
run_evaluation_mmse         — digital MMSE/ZF baseline (no SIM).
run_evaluation_mono_channel — TX-only SIM with MMSE equalizer at RX.
run_evaluation_disjoint_fixed — disjoint Dual-SIM with frozen TX and RX cascades.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_alignment.evaluation.linear_probe import clf_predict
from semantic_alignment.utils.utils import (
    a_inv_times_b,
    awgn,
    complex_compressed_tensor,
    decompress_complex_tensor,
    get_rx_equalizer,
    sigma_given_snr,
)


def run_evaluation(
    model,
    dataloader: DataLoader,
    H_mimo: torch.Tensor,
    snr_db: float | None,
    beta_opt: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf,
    device: torch.device,
) -> float:
    """Evaluate semantic accuracy for the joint Dual-SIM architecture.

    Signal model: ``ŷ = β · G_R · (H G_T x_white + noise)``.  The AWGN is injected
    at the receive antennas (between the channel and the RX SIM), so the RX cascade
    ``G_R`` filters it, exactly as in the disjoint architecture.

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
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        G_T = model.get_TX_cascade()
        G_R = model.get_RX_cascade()

    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10)
        H_pinv = torch.linalg.pinv(H_mimo)
        pinv_trace = torch.trace(H_pinv.H @ H_pinv)

    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)

        x_complex = complex_compressed_tensor(x_real_batch.T, device=device)
        x_white = a_inv_times_b(L_in, x_complex - mu_in).to(G_T.dtype)

        # TX SIM + channel → receive antennas.  AWGN enters here, before the RX
        # SIM, so G_R filters the noise as a physical receive metasurface would.
        y_ant = H_mimo @ (G_T @ x_white)
        if snr_db is not None:
            n = y_ant.shape[1]
            sigma = torch.sqrt(
                (torch.trace(y_ant @ y_ant.H) / n) / (snr_linear * pinv_trace)
            ).real.item()
            y_ant = y_ant + awgn(sigma, y_ant.shape, device=device)

        # RX SIM equalizes the noisy antenna signal, then the global scalar β.
        y = beta_opt * (G_R @ y_ant)

        z_hat = (L_out @ y) + mu_out
        y_hat_real = decompress_complex_tensor(z_hat, device=device).T

        preds = clf_predict(clf, y_hat_real)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    return (np.array(all_preds) == np.array(all_labels)).mean()


def run_evaluation_mmse(
    dataloader: DataLoader,
    H_mimo: torch.Tensor,
    snr_db: float,
    A_target: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf,
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
    G_mmse = get_rx_equalizer(H_mimo, snr_db)

    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)

        x_complex = complex_compressed_tensor(x_real_batch.T, device=device)
        x_white = a_inv_times_b(L_in, x_complex - mu_in).to(H_mimo.dtype)

        D_tx = x_white.shape[0]
        D_ch = H_mimo.shape[1]
        padding = torch.zeros((D_ch - D_tx, x_white.shape[1]), device=device, dtype=x_white.dtype)
        x_padded = torch.cat([x_white, padding], dim=0)

        y_signal = H_mimo @ x_padded
        sigma_v = sigma_given_snr(snr_db, y_signal)
        y_received = y_signal + awgn(sigma_v, y_signal.shape, device=device)

        x_recovered = (G_mmse @ y_received)[:D_tx, :]
        z_hat_complex = A_target @ x_recovered
        z_hat = (L_out @ z_hat_complex) + mu_out
        y_hat_real = decompress_complex_tensor(z_hat, device=device).T

        preds = clf_predict(clf, y_hat_real)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    return (np.array(all_preds) == np.array(all_labels)).mean()


def run_evaluation_mono_channel(
    G_T: torch.Tensor,
    dataloader: DataLoader,
    H_mimo: torch.Tensor,
    snr_db: float | None,
    beta_opt: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf,
    device: torch.device,
) -> float:
    """Evaluate a Mono-SIM (TX only) with MIMO channel and ZF/MMSE equalization.

    Signal model: ``ŷ = β · G_eq(H) · (H · G_T · x_w + noise)`` where ``G_eq`` is
    a zero-forcing equalizer when ``snr_db`` is ``None`` (infinite SNR, noiseless)
    and an MMSE equalizer otherwise.  ``G_eq · H ≈ I`` so the channel is undone by
    the equalizer alone; ``β`` only rescales the recovered alignment.

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
        Optimal scalar aligning ``G_mmse H G_T`` to the target ``A``.
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
    # Left-inverse equalizer: ZF when snr_db is None, MMSE otherwise.  Unlike the
    # SVD precoder/equalizer pair, G_eq alone inverts the channel (G_eq · H ≈ I),
    # so no transmit precoding is required by the TX-only Mono-SIM.
    G_eq = get_rx_equalizer(H_mimo, snr_db)

    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10)
        H_pinv = torch.linalg.pinv(H_mimo)
        pinv_trace = torch.trace(H_pinv.H @ H_pinv)

    all_preds, all_labels = [], []
    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)

        x_c = complex_compressed_tensor(x_real_batch.T, device=device)
        x_w = a_inv_times_b(L_in, x_c - mu_in).to(G_T.dtype)

        # TX SIM cascade then propagation through the MIMO channel.
        y = H_mimo @ (G_T @ x_w)

        # AWGN scaled to the received signal power, only when SNR is finite.
        if snr_db is not None:
            n = y.shape[1]
            sigma = torch.sqrt(
                (torch.trace(y @ y.H) / n) / (snr_linear * pinv_trace)
            ).real.item()
            y = y + awgn(sigma, y.shape, device=device)

        # Channel equalization, then the optimal scalar β (applied at the RX).
        y = G_eq @ y
        y = beta_opt * y

        z = (L_out.to(G_T.dtype) @ y) + mu_out.to(G_T.dtype)
        y_hat = decompress_complex_tensor(z, device=device).T

        preds = clf_predict(clf, y_hat)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    return (np.array(all_preds) == np.array(all_labels)).mean()


def run_evaluation_disjoint_fixed(
    tx_model,
    rx_model,
    dataloader: DataLoader,
    H_mimo: torch.Tensor,
    snr_db: float | None,
    beta_global: torch.Tensor,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    clf,
    device: torch.device,
) -> float:
    """Evaluate the disjoint Dual-SIM architecture with frozen TX and RX cascades.

    Signal model: ``ŷ = β · G_R · (H G_T x_w + noise)``.  The AWGN is injected at
    the receive antennas (between the channel and the RX SIM), so the RX cascade
    ``G_R`` filters it.  Both ``G_T`` and ``G_R`` are computed once and held fixed.

    Parameters
    ----------
    tx_model : MonoSIMoptimizerTorch
        Trained TX-side Mono-SIM.
    rx_model : MonoSIMoptimizerTorch
        Trained RX-side Mono-SIM.
    dataloader : DataLoader
        Test set yielding ``(z_tx_batch, labels)`` pairs.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    snr_db : float or None
        AWGN SNR in dB; ``None`` means noiseless.
    beta_global : torch.Tensor
        Global scalar aligning the cascade ``G_R H G_T`` to the target ``A``.
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
    tx_model.eval()
    rx_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        G_T = tx_model.get_cascade()
        G_R = rx_model.get_cascade()

    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10)
        H_pinv = torch.linalg.pinv(H_mimo)
        pinv_trace = torch.trace(H_pinv.H @ H_pinv)

    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)

        x_complex = complex_compressed_tensor(x_real_batch.T, device=device)
        x_white = a_inv_times_b(L_in, x_complex - mu_in).to(G_T.dtype)

        # TX SIM + channel → receive antennas.  AWGN enters here, *before* the RX
        # SIM, so G_R filters the noise as a physical receive metasurface would.
        y_ant = H_mimo @ (G_T @ x_white)
        if snr_db is not None:
            n = y_ant.shape[1]
            sigma = torch.sqrt(
                (torch.trace(y_ant @ y_ant.H) / n) / (snr_linear * pinv_trace)
            ).real.item()
            y_ant = y_ant + awgn(sigma, y_ant.shape, device=device)

        # RX SIM equalizes the noisy antenna signal, then the global scalar β.
        y = beta_global * (G_R @ y_ant)

        z_hat = (L_out @ y) + mu_out
        y_hat_real = decompress_complex_tensor(z_hat, device=device).T

        preds = clf_predict(clf, y_hat_real)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    return (np.array(all_preds) == np.array(all_labels)).mean()

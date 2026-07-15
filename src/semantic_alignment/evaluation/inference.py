"""Inference routines for evaluating semantic classification accuracy over-the-air.

All functions share the same whitening convention:
    x_white = L_in⁻¹ (x_complex − μ_in)
    ŷ       = L_out · y_received + μ_out

**Noise power convention**
AWGN sigma is defined relative to the *transmitted* signal power, which equals 1
because the TX embeddings are whitened.  Therefore:

    σ = 1 / √SNR_linear

This is computed once outside the batch loop and is the same for every batch.

Functions
---------
run_evaluation              — joint Dual-SIM (G_R · H · G_T), optional AWGN.
run_evaluation_mmse         — digital MMSE/ZF baseline (no SIM).
run_evaluation_mono         — TX-only SIM with optional MMSE equalizer at RX.
run_evaluation_disjoint_fixed — disjoint Dual-SIM with frozen TX and RX cascades.
"""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_alignment.evaluation.linear_probe import clf_predict
from semantic_alignment.utils import (
    awgn,
    pack_complex,
    rx_equalizer,
    solve,
    unpack_complex,
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

    Signal model: ``ŷ = β G_R H G_T x_white + v``.  β is applied at the
    transmitter; AWGN enters additively after the cascade so noise is *not*
    scaled by β.  ``σ²`` is normalised to the β-scaled cascade output power.

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

    # σ = 1/√SNR — TX signal is whitened, so P_TX = 1
    sigma = 1.0 / math.sqrt(10 ** (snr_db / 10)) if snr_db is not None else None

    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)

        x_complex = pack_complex(x_real_batch.T, device=device)
        x_white = solve(L_in, x_complex - mu_in).to(G_T.dtype)

        y_sig = beta_opt * (G_R @ (H_mimo @ (G_T @ x_white)))
        if sigma is not None:
            y = y_sig + awgn(sigma, y_sig.shape, device=device)
        else:
            y = y_sig

        z_hat = (L_out @ y) + mu_out
        y_hat_real = unpack_complex(z_hat, device=device).T

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
    clf,
    device: torch.device,
    use_equalizer: bool = True,
    mmse: bool = False,
) -> float:
    """Evaluate a Mono-SIM (TX only) over a MIMO channel.

    Two signal models depending on ``use_equalizer``:

    - ``True`` (standard): ``ŷ = β · G_eq · (H G_T x + v)`` — channel undone
      analytically by ZF/MMSE; ``β`` aligns ``G_T`` to the target ``A``.
    - ``False`` (constructive): ``ŷ = β · (H G_T x + v)`` — no equalizer; the
      TX-SIM was trained with loss ``‖β H G_T − A‖²`` so ``H G_T ≈ A/β``.
      Noise is present but not mitigated.

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
        Optimal scalar (``⟨H G_T, A⟩/‖H G_T‖²`` for constructive,
        ``⟨G_T, A⟩/‖G_T‖²`` for standard).
    L_in, mu_in : torch.Tensor
        TX whitening statistics.
    L_out, mu_out : torch.Tensor
        RX whitening statistics.
    clf : torch.Tensor or nn.Module
        Linear probe or neural classifier.
    device : torch.device
        Inference device.
    use_equalizer : bool, optional
        If ``True`` (default) apply ZF/MMSE G_eq before β.
        If ``False`` skip G_eq (constructive pre-coding mode).
    mmse : bool, optional
        Selects the RX equalizer type when ``use_equalizer=True``.
        ``False`` (default): ZF — ``G_eq = H†`` (pseudoinverse, ignores SNR).
        ``True``: MMSE — ``G_eq`` regularized with ``snr_db``.

    Returns
    -------
    float
        Top-1 classification accuracy in ``[0, 1]``.
    """
    if use_equalizer:
        eq_snr = snr_db if mmse else None
        G_eq = rx_equalizer(H_mimo, eq_snr)

    # σ = 1/√SNR — TX signal is whitened, so P_TX = 1
    sigma = 1.0 / math.sqrt(10 ** (snr_db / 10)) if snr_db is not None else None

    all_preds, all_labels = [], []
    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)

        x_c = pack_complex(x_real_batch.T, device=device)
        x_w = solve(L_in, x_c - mu_in).to(G_T.dtype)

        y = H_mimo @ (beta_opt * G_T @ x_w)

        if sigma is not None:
            y = y + awgn(sigma, y.shape, device=device)

        if use_equalizer:
            y = G_eq @ y

        z = (L_out.to(G_T.dtype) @ y) + mu_out.to(G_T.dtype)
        y_hat = unpack_complex(z, device=device).T

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

    Signal model: ``ŷ = β G_R H G_T x_w + v``.  β is applied at the
    transmitter; AWGN enters additively after the cascade and is *not* scaled
    by β.  ``σ²`` is normalised to the β-scaled cascade output power.
    Both ``G_T`` and ``G_R`` are computed once and held fixed.

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

    # σ = 1/√SNR — TX signal is whitened, so P_TX = 1
    sigma = 1.0 / math.sqrt(10 ** (snr_db / 10)) if snr_db is not None else None

    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)

        x_complex = pack_complex(x_real_batch.T, device=device)
        x_white = solve(L_in, x_complex - mu_in).to(G_T.dtype)

        y_sig = beta_global * (G_R @ (H_mimo @ (G_T @ x_white)))
        if sigma is not None:
            y = y_sig + awgn(sigma, y_sig.shape, device=device)
        else:
            y = y_sig

        z_hat = (L_out @ y) + mu_out
        y_hat_real = unpack_complex(z_hat, device=device).T

        preds = clf_predict(clf, y_hat_real)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    return (np.array(all_preds) == np.array(all_labels)).mean()


if __name__ == "__main__":
    print("inference.py — evaluation functions for each SIM architecture:")
    print("  run_evaluation               : Joint Dual-SIM  (β G_R H G_T x + v, β at TX)")
    print("  run_evaluation_mmse          : Digital MMSE baseline (no SIM)")
    print("  run_evaluation_mono          : Mono-SIM TX-only + MMSE equalizer")
    print("  run_evaluation_disjoint_fixed: Disjoint Dual-SIM with frozen cascades")
    print()
    print("Called automatically by experiment runners. Invoke via: just run-joint / run-mono / ...")

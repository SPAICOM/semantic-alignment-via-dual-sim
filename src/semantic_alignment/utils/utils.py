"""Signal-processing and tensor utilities.

Provides complex tensor helpers, AWGN noise generation, SNR/sigma conversion,
pre-whitening via Cholesky decomposition, and MMSE-SVD channel equalization.
"""

import math
import shutil
from pathlib import Path

import torch


def complex_compressed_tensor(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Compress a real tensor into a half-size complex tensor.

    Splits the feature dimension in half: the first half becomes the real part
    and the second half becomes the imaginary part.  Input is cast to float32
    so the output is always complex64, matching the SIM optimizer precision.

    Parameters
    ----------
    x : torch.Tensor
        Real-valued tensor of shape ``(d, n)``.  If ``d`` is odd a zero row is
        appended before splitting.
    device : str, optional
        Target device for the output tensor.

    Returns
    -------
    torch.Tensor
        Complex64 tensor of shape ``(d//2, n)``.
    """
    d, n = x.shape
    if d % 2 != 0:
        x = torch.cat((x, torch.zeros((1, n), dtype=x.dtype, device=x.device)), dim=0)
        d += 1

    real_part = x[: d // 2, :].float()
    imaginary_part = x[d // 2 :, :].float()
    return torch.view_as_complex(torch.stack((real_part, imaginary_part), dim=-1)).to(device)


def decompress_complex_tensor(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Invert ``complex_compressed_tensor``: complex ``(d/2, n)`` → real ``(d, n)``.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor of shape ``(d/2, n)``.
    device : str, optional
        Target device for the output tensor.

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``(d, n)`` formed by concatenating real and
        imaginary parts along the first dimension.
    """
    return torch.cat((x.real, x.imag), dim=0).to(device)


def complex_tensor(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Cast a real tensor to complex by setting the imaginary part to zero.

    Parameters
    ----------
    x : torch.Tensor
        Real-valued tensor of any shape.
    device : str, optional
        Unused (kept for API consistency); output is placed on ``x.device``.

    Returns
    -------
    torch.Tensor
        Complex tensor with the same shape and dtype promoted to complex.
    """
    device = x.device
    return torch.view_as_complex(torch.stack((x, torch.zeros_like(x)), dim=-1)).to(device)


def complex_gaussian_matrix(mean: float, std: float, size: tuple[int]) -> torch.Tensor:
    """Draw a complex Gaussian random matrix with i.i.d. entries.

    Each complex entry is CN(mean, std²), i.e. real and imaginary parts are
    each N(mean, (std/√2)²).

    Parameters
    ----------
    mean : float
        Mean of each real/imaginary component.
    std : float
        Standard deviation of the *complex* entry (split equally between real
        and imaginary parts).
    size : tuple[int]
        Shape of the output matrix.

    Returns
    -------
    torch.Tensor
        Complex64 random matrix of shape ``size``.
    """
    real_part = torch.normal(mean, std / 2, size=size)
    imag_part = torch.normal(mean, std / 2, size=size)
    return torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))


def awgn(sigma: float, size: torch.Size, device: str = "cpu") -> torch.Tensor:
    """Generate a complex AWGN noise sample.

    The noise power is split equally between the real and imaginary components,
    each with standard deviation ``sigma / √2``, so the total complex power
    equals ``sigma²``.

    Parameters
    ----------
    sigma : float
        Standard deviation of the complex noise (total, not per component).
    size : torch.Size
        Shape of the output noise tensor.
    device : str, optional
        Target device.

    Returns
    -------
    torch.Tensor
        Complex noise tensor of shape ``size``.
    """
    sigma = sigma / math.sqrt(2)
    r = torch.normal(mean=0, std=sigma, size=size)
    i = torch.normal(mean=0, std=sigma, size=size)
    return torch.view_as_complex(torch.stack((r, i), dim=-1)).to(device)


def snr(signal: torch.Tensor, sigma: float) -> float:
    """Compute the Signal-to-Noise Ratio in dB.

    Parameters
    ----------
    signal : torch.Tensor
        Complex signal tensor of any shape.
    sigma : float
        Noise standard deviation (complex).

    Returns
    -------
    float
        SNR in dB: ``10 log10(E[|s|²] / σ²)``.
    """
    return 10 * torch.log10(torch.mean(torch.abs(signal) ** 2) / sigma**2).item()


def sigma_given_snr(snr: float, signal: torch.Tensor) -> float:
    """Return the noise standard deviation that yields a target SNR.

    Parameters
    ----------
    snr : float
        Desired SNR in dB.
    signal : torch.Tensor
        Reference signal used to measure average power.

    Returns
    -------
    float
        Noise standard deviation σ such that ``10 log10(P_signal / σ²) = snr``.
    """
    snr_linear = math.pow(10, snr / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    return math.sqrt(signal_power / snr_linear)


def a_inv_times_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve the linear system ``a @ x = b`` for ``x``, i.e. compute ``A⁻¹ B``.

    Uses ``torch.linalg.solve`` with a fallback to CPU for MPS devices, which
    do not support complex linear solves.

    Parameters
    ----------
    a : torch.Tensor
        Square coefficient matrix of shape ``(d, d)``.
    b : torch.Tensor
        Right-hand side of shape ``(d, n)``.

    Returns
    -------
    torch.Tensor
        Solution ``x`` of shape ``(d, n)``, same device as ``a``.
    """
    try:
        if a.device.type == "mps" and a.is_complex():
            return torch.linalg.solve(a.detach().cpu(), b.detach().cpu()).to(a.device)
        return torch.linalg.solve(a, b)
    except RuntimeError as e:
        if "MPS only supports floats" in str(e) or "linalg.lu_factor" in str(e):
            return torch.linalg.solve(a.detach().cpu(), b.detach().cpu()).to(a.device)
        if "The input tensor A must have at least 2 dimensions" in str(e):
            if len(a.shape) == 2 and a.shape[0] == 1:
                return (1 / a) @ b
        raise


def prewhiten(
    x_train: torch.Tensor,
    x_test: torch.Tensor = None,
    device: str = "cpu",
):
    """Prewhiten data using the Cholesky factor of the training covariance.

    Computes ``L`` from ``C = L Lᴴ`` (Cholesky) and returns
    ``z = L⁻¹ (x − μ)`` where ``μ`` is the training column mean.
    This decorrelates the training set so that whitened pilots have identity
    covariance, which is required by the alignment estimators.

    Parameters
    ----------
    x_train : torch.Tensor
        Training data of shape ``(d, n_train)``, complex64.
    x_test : torch.Tensor, optional
        Test data of shape ``(d, n_test)`` to whiten with training statistics.
    device : str, optional
        Target device for all returned tensors.

    Returns
    -------
    z_train : torch.Tensor
        Whitened training data ``(d, n_train)``.
    L : torch.Tensor
        Lower-triangular Cholesky factor ``(d, d)``.
    mean : torch.Tensor
        Column mean of training data ``(d, 1)``.
    z_test : torch.Tensor
        Whitened test data ``(d, n_test)`` — only returned when ``x_test`` is given.
    """
    # Cast covariance to input dtype to prevent implicit float64 upcasting
    C = torch.cov(x_train).to(x_train.dtype)
    try:
        L, _ = torch.linalg.cholesky_ex(C)
    except RuntimeError as e:
        if "The input tensor A must have at least 2 dimensions" in str(e):
            L = torch.sqrt(C).unsqueeze(0).unsqueeze(1)
        else:
            raise

    mean = x_train.mean(axis=1)[:, None]
    z_train = a_inv_times_b(L, x_train - mean)

    if x_test is not None:
        z_test = a_inv_times_b(L, x_test - mean)
        return z_train.to(device), L.to(device), mean.to(device), z_test.to(device)

    return z_train.to(device), L.to(device), mean.to(device)


def remove_non_empty_dir(path: str) -> None:
    """Recursively remove a directory and all its contents.

    Parameters
    ----------
    path : str
        Path to the directory to remove.

    Raises
    ------
    NotADirectoryError
        If ``path`` exists but is not a directory.
    """
    dir_path = Path(path)
    if not dir_path.exists():
        print(f"The path '{path}' does not exist.")
        return
    if not dir_path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")
    shutil.rmtree(dir_path)
    print(f"Successfully removed directory: {dir_path}")


def mmse_svd_equalizer(channel_matrix: torch.Tensor, snr_db: float = None):
    """Compute a joint MMSE/ZF SVD precoder–equalizer pair for a MIMO channel.

    Decomposes ``H = U Σ Vᴴ`` and constructs:
      - Precoder ``F = Vᴴ.H`` (transmit beamforming).
      - Equalizer ``G`` = MMSE if ``snr_db`` is given, ZF otherwise.

    Both matrices are normalized so ``‖F‖_F = 1``.

    Parameters
    ----------
    channel_matrix : torch.Tensor
        Complex MIMO channel matrix of shape ``(Nr, Nt)``.
    snr_db : float, optional
        Operating SNR in dB used to regularize the MMSE solution.
        If ``None`` a zero-forcing (ZF) equalizer is returned.

    Returns
    -------
    G : torch.Tensor
        Receive equalizer of shape ``(Nt, Nr)``.
    F : torch.Tensor
        Transmit precoder of shape ``(Nt, Nt)``, unit Frobenius norm.
    """
    U, s, Vh = torch.linalg.svd(channel_matrix)
    Nr, _ = channel_matrix.shape
    Sigma = torch.diag(s).to(channel_matrix.dtype)
    F = Vh.H

    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10)
        reg = (1.0 / snr_linear) * torch.eye(Nr, dtype=channel_matrix.dtype)
        inv_term = torch.inverse(Sigma.H @ Sigma + reg)
        G = inv_term @ (U @ Sigma).H
    else:
        inv_Sigma = torch.diag(1.0 / s).to(channel_matrix.dtype)
        G = inv_Sigma @ U.H

    G *= torch.norm(F, p="fro")
    F /= torch.norm(F, p="fro")
    return G, F


def get_rx_equalizer(channel_matrix: torch.Tensor, snr_db: float = None) -> torch.Tensor:
    """Compute a receive-only MMSE/ZF equalizer (no TX precoding).

    Parameters
    ----------
    channel_matrix : torch.Tensor
        Complex MIMO channel matrix of shape ``(Nr, Nt)``.
    snr_db : float, optional
        Operating SNR in dB for MMSE regularization.
        If ``None`` a ZF equalizer is returned.

    Returns
    -------
    torch.Tensor
        Receive equalizer ``Q`` of shape ``(Nt, Nr)``.
    """
    U, s, Vh = torch.linalg.svd(channel_matrix, full_matrices=False)
    Sigma = torch.diag(s).to(channel_matrix.dtype)
    V = Vh.H

    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10)
        reg = (1.0 / snr_linear) * torch.eye(
            Sigma.shape[1], dtype=channel_matrix.dtype, device=channel_matrix.device
        )
        inv_term = torch.inverse(Sigma.H @ Sigma + reg)
        Q = V @ inv_term @ (U @ Sigma).H
    else:
        inv_Sigma = torch.diag(1.0 / s).to(channel_matrix.dtype)
        Q = V @ inv_Sigma @ U.H

    return Q

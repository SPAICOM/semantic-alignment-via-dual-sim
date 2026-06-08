"""Signal-processing and tensor utilities."""

import math
import shutil
from pathlib import Path

import torch


def complex_compressed_tensor(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Compress a real tensor (d, n) into a complex tensor (d/2, n).

    The first half of the feature dimension becomes the real part and the
    second half becomes the imaginary part.
    """
    d, n = x.shape
    if d % 2 != 0:
        x = torch.cat((x, torch.zeros((1, n), dtype=x.dtype, device=x.device)), dim=0)
        d += 1

    real_part = x[: d // 2, :].float()
    imaginary_part = x[d // 2 :, :].float()
    return torch.view_as_complex(torch.stack((real_part, imaginary_part), dim=-1)).to(device)


def decompress_complex_tensor(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Invert complex_compressed_tensor: (d/2, n) complex → (d, n) real."""
    return torch.cat((x.real, x.imag), dim=0).to(device)


def complex_tensor(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Cast a real tensor to complex by setting the imaginary part to zero."""
    device = x.device
    return torch.view_as_complex(torch.stack((x, torch.zeros_like(x)), dim=-1)).to(device)


def complex_gaussian_matrix(mean: float, std: float, size: tuple[int]) -> torch.Tensor:
    """Return a complex Gaussian random matrix."""
    real_part = torch.normal(mean, std / 2, size=size)
    imag_part = torch.normal(mean, std / 2, size=size)
    return torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))


def awgn(sigma: float, size: torch.Size, device: str = "cpu") -> torch.Tensor:
    """Return a complex AWGN noise vector with real standard deviation *sigma*."""
    sigma = sigma / math.sqrt(2)
    r = torch.normal(mean=0, std=sigma, size=size)
    i = torch.normal(mean=0, std=sigma, size=size)
    return torch.view_as_complex(torch.stack((r, i), dim=-1)).to(device)


def snr(signal: torch.Tensor, sigma: float) -> float:
    """Return the Signal-to-Noise Ratio in dB."""
    return 10 * torch.log10(torch.mean(torch.abs(signal) ** 2) / sigma**2).item()


def sigma_given_snr(snr: float, signal: torch.Tensor) -> float:
    """Return the noise standard deviation corresponding to a target SNR (dB)."""
    snr_linear = math.pow(10, snr / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    return math.sqrt(signal_power / snr_linear)


def a_inv_times_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute A^{-1} B efficiently, with MPS fallback for complex tensors."""
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
    """
    Prewhiten training (and optionally test) data using training statistics.

    Returns (z_train, L, mean) or (z_train, L, mean, z_test).
    """
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
    """Remove a non-empty directory. Use with caution."""
    dir_path = Path(path)
    if not dir_path.exists():
        print(f"The path '{path}' does not exist.")
        return
    if not dir_path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")
    shutil.rmtree(dir_path)
    print(f"Successfully removed directory: {dir_path}")


def mmse_svd_equalizer(channel_matrix: torch.Tensor, snr_db: float = None):
    """
    Linear MMSE/ZF precoder and equaliser via SVD.

    Returns (G, F).
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
    """Stand-alone MMSE/ZF receiver equaliser (TX does not precod)."""
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

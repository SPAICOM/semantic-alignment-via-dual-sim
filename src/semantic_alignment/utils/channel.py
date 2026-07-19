"""Channel and signal-processing utilities.

Covers channel matrix generation, AWGN noise, SNR conversion, MIMO
equalization, complex tensor encoding/decoding, whitening, and Cholesky
linear solves.
"""

import math

import torch

# ── Channel generation ────────────────────────────────────────────────────────


def cn_matrix(
    mean: float,
    std: float,
    size: tuple[int],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a complex Gaussian random matrix with i.i.d. CN(mean, std²) entries.

    Each entry is CN(mean, std²): real and imaginary parts are each
    independently drawn from N(mean, (std/√2)²), so E[|x − mean|²] = std².

    Parameters
    ----------
    mean : float
        Mean of each real/imaginary component.
    std : float
        Standard deviation of the *complex* entry (split equally between real
        and imaginary parts).
    size : tuple[int]
        Shape of the output matrix.
    generator : torch.Generator or None, optional
        Optional RNG for reproducibility.

    Returns
    -------
    torch.Tensor
        Complex64 random matrix of shape ``size``.

    Notes
    -----
    The scaling ``* std + mean * (1 + 1j)`` ensures:
    - ``randn`` produces CN(0, 1) entries (real and imag each ~ N(0, 1/2))
    - multiplying by ``std`` scales to CN(0, std²)
    - adding ``mean * (1 + 1j)`` shifts both components by ``mean``
    """
    return torch.randn(size, dtype=torch.complex64, generator=generator) * std + mean * (1 + 1j)


# ── Noise and SNR ─────────────────────────────────────────────────────────────


def awgn(
    sigma: float,
    size: torch.Size,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a complex AWGN noise sample with total power ``sigma²``.

    Each entry is CN(0, sigma²): real and imaginary parts each have variance
    sigma²/2, so E[|n|²] = sigma².

    Parameters
    ----------
    sigma : float
        Standard deviation of the complex noise (total, not per component).
    size : torch.Size
        Shape of the output noise tensor.
    device : str, optional
        Target device (default ``"cpu"``).
    generator : torch.Generator or None, optional
        Optional RNG for reproducibility.

    Returns
    -------
    torch.Tensor
        Complex64 noise tensor of shape ``size``.
    """
    # randn gives CN(0,1); scaling by sigma gives CN(0, sigma²).
    return torch.randn(size, dtype=torch.complex64, generator=generator).to(device) * sigma


def snr(signal: torch.Tensor, sigma: float) -> float:
    """Compute the Signal-to-Noise Ratio in dB.

    Parameters
    ----------
    signal : torch.Tensor
        Complex signal tensor of any shape.
    sigma : float
        Complex noise standard deviation (total power = sigma²).

    Returns
    -------
    float
        SNR in dB: ``10 log10(E[|s|²] / σ²)``.
    """
    # Mean squared modulus of the signal divided by noise power.
    return 10 * torch.log10(torch.mean(torch.abs(signal) ** 2) / sigma**2).item()


def sigma_from_snr(snr_db: float, signal: torch.Tensor) -> float:
    """Return the noise standard deviation that yields a target SNR.

    Parameters
    ----------
    snr_db : float
        Desired SNR in dB.
    signal : torch.Tensor
        Reference signal used to measure average power.

    Returns
    -------
    float
        Noise standard deviation σ such that ``10 log10(P_signal / σ²) = snr_db``.

    Notes
    -----
    From ``snr_db = 10 log10(P / σ²)`` we get
    ``σ = sqrt(P / 10^(snr_db/10))``.
    """
    # Convert dB SNR to linear ratio.
    snr_linear = math.pow(10, snr_db / 10)
    # Average signal power E[|s|²].
    signal_power = torch.mean(torch.abs(signal) ** 2)
    # Solve for σ: σ² = P_signal / SNR_linear.
    return math.sqrt(signal_power / snr_linear)


# ── MIMO equalization ─────────────────────────────────────────────────────────


def svd_equalizer(
    channel_matrix: torch.Tensor, snr_db: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a joint MMSE/ZF SVD precoder–equalizer pair for a MIMO channel.

    Decomposes ``H = U Σ Vᴴ`` and constructs:
      - Precoder ``F = Vᴴ.H`` (transmit beamforming).
      - Equalizer ``G`` — MMSE if ``snr_db`` is given, ZF otherwise.

    Both matrices are normalized so ``‖F‖_F = 1``.

    Parameters
    ----------
    channel_matrix : torch.Tensor
        Complex MIMO channel matrix of shape ``(Nr, Nt)``.  Must be square;
        for non-square channels use :func:`rx_equalizer`.
    snr_db : float or None, optional
        Operating SNR in dB used to regularize the MMSE solution.
        If ``None`` a zero-forcing (ZF) equalizer is returned (default).

    Returns
    -------
    G : torch.Tensor
        Receive equalizer, shape ``(Nr, Nr)``.
    F : torch.Tensor
        Transmit precoder, shape ``(Nt, Nr)``, unit Frobenius norm.

    Notes
    -----
    MMSE solution: ``G = (ΣᴴΣ + σ²I)⁻¹ (UΣ)ᴴ``
    ZF solution:   ``G = Σ⁻¹ Uᴴ``
    """
    # SVD: H = U Σ Vᴴ, with Σ = diag(s).
    U, s, Vh = torch.linalg.svd(channel_matrix, full_matrices=False)
    k = s.shape[0]
    Sigma = torch.diag(s).to(channel_matrix.dtype)
    # Precoder: F = Vᴴ.H (column-space beamforming).
    F = Vh.H

    if snr_db is not None:
        # MMSE: regularise with noise variance σ² = 1/SNR_linear.
        snr_linear = 10 ** (snr_db / 10)
        reg = (1.0 / snr_linear) * torch.eye(
            k, dtype=channel_matrix.dtype, device=channel_matrix.device
        )
        inv_term = torch.inverse(Sigma.H @ Sigma + reg)
        G = inv_term @ (U @ Sigma).H
    else:
        # ZF: invert the singular values directly.
        inv_Sigma = torch.diag(1.0 / s).to(channel_matrix.dtype)
        G = inv_Sigma @ U.H

    # Normalise so that ‖F‖_F = 1 and transfer the scale to G.
    G *= torch.norm(F, p="fro")
    F /= torch.norm(F, p="fro")
    return G, F


def rx_equalizer(
    channel_matrix: torch.Tensor,
    snr_db: float | None = None,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Compute a receive-only MMSE/ZF equalizer (no TX precoding).

    Parameters
    ----------
    channel_matrix : torch.Tensor
        Complex MIMO channel matrix of shape ``(Nr, Nt)``.
    snr_db : float or None, optional
        Operating SNR in dB for MMSE regularization.
        If ``None`` a ZF equalizer is returned (default).
    epsilon : float, optional
        Tikhonov regularization added to singular values for the ZF path:
        ``Q = V diag(σ/(σ²+ε)) Uᴴ``.  Ignored when ``snr_db`` is set
        (default ``0.0``).

    Returns
    -------
    torch.Tensor
        Receive equalizer ``Q`` of shape ``(Nt, Nr)``.

    Notes
    -----
    MMSE solution: ``Q = V (ΣᴴΣ + σ²I)⁻¹ (UΣ)ᴴ``
    ZF solution:   ``Q = V diag(σ/(σ²+ε)) Uᴴ``
    """
    # SVD: H = U Σ Vᴴ.
    U, s, Vh = torch.linalg.svd(channel_matrix, full_matrices=False)
    V = Vh.H
    k = s.shape[0]

    if snr_db is not None:
        # MMSE: regularise with noise variance.
        Sigma = torch.diag(s).to(channel_matrix.dtype)
        snr_linear = 10 ** (snr_db / 10)
        reg = (1.0 / snr_linear) * torch.eye(
            k, dtype=channel_matrix.dtype, device=channel_matrix.device
        )
        inv_term = torch.inverse(Sigma.H @ Sigma + reg)
        Q = V @ inv_term @ (U @ Sigma).H
    else:
        # ZF with optional Tikhonov stabilisation on small singular values.
        reg_inv_s = (s / (s**2 + epsilon)).to(channel_matrix.dtype)
        Q = V @ torch.diag(reg_inv_s) @ U.H

    return Q


# ── Complex tensor encoding / decoding ───────────────────────────────────────


def pack_complex(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Pack a real tensor into a half-size complex tensor.

    Splits the feature dimension in half: the first half becomes the real part
    and the second half becomes the imaginary part.  Input is cast to float32
    so the output is always complex64.

    Parameters
    ----------
    x : torch.Tensor
        Real-valued tensor of shape ``(d, n)``.  If ``d`` is odd a zero row is
        appended before splitting.
    device : str, optional
        Target device for the output tensor (default ``"cpu"``).

    Returns
    -------
    torch.Tensor
        Complex64 tensor of shape ``(d//2, n)``.

    Notes
    -----
    Example: ``x = [a, b, c, d]`` (d=4) → ``[a+bi, c+di]`` (d//2=2).
    """
    d = x.shape[-2]
    # Pad with a zero row if the feature dimension is odd.
    if d % 2 != 0:
        pad = list(x.shape)
        pad[-2] = 1
        x = torch.cat((x, torch.zeros(pad, dtype=x.dtype, device=x.device)), dim=-2)
        d += 1
    # Split into real and imaginary halves, then interleave into complex.
    real_part = x[..., : d // 2, :].float()
    imaginary_part = x[..., d // 2 :, :].float()
    return torch.view_as_complex(torch.stack((real_part, imaginary_part), dim=-1)).to(device)


def unpack_complex(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Invert :func:`pack_complex`: complex ``(d/2, n)`` → real ``(d, n)``.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor of shape ``(d/2, n)``.
    device : str, optional
        Target device for the output tensor (default ``"cpu"``).

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``(d, n)`` formed by concatenating real and
        imaginary parts along the first dimension.
    """
    return torch.cat((x.real, x.imag), dim=-2).to(device)


def to_complex(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
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
        Complex tensor with the same shape, dtype promoted to complex.
    """
    # Stack real part with a zero imaginary part, then view as complex.
    return torch.view_as_complex(torch.stack((x, torch.zeros_like(x)), dim=-1)).to(x.device)


# ── Linear algebra helpers ────────────────────────────────────────────────────


def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve ``a @ x = b`` for ``x``, i.e. compute ``A⁻¹ B``.

    Uses ``torch.linalg.solve`` with a CPU fallback for MPS devices that do
    not support complex linear solves.

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

    Raises
    ------
    RuntimeError
        If the linear solve fails for reasons other than MPS compatibility
        or dimensionality issues.
    """
    try:
        # Standard path: solve on the tensor's native device.
        if a.device.type == "mps" and a.is_complex():
            # MPS does not support complex linalg; fall back to CPU.
            return torch.linalg.solve(a.detach().cpu(), b.detach().cpu()).to(a.device)
        return torch.linalg.solve(a, b)
    except RuntimeError as e:
        if "MPS only supports floats" in str(e) or "linalg.lu_factor" in str(e):
            # MPS fallback for complex tensors.
            return torch.linalg.solve(a.detach().cpu(), b.detach().cpu()).to(a.device)
        if "The input tensor A must have at least 2 dimensions" in str(e):
            # Scalar or 1×1 case: direct division.
            if len(a.shape) == 2 and a.shape[0] == 1:
                return (1 / a) @ b
        raise


def prewhiten(
    x_train: torch.Tensor,
    x_test: torch.Tensor | None = None,
    device: str = "cpu",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Prewhiten data using the Cholesky factor of the training covariance.

    Computes ``L`` from ``C = L Lᴴ`` and returns ``z = L⁻¹ (x − μ)``
    where ``μ`` is the training column mean.

    Parameters
    ----------
    x_train : torch.Tensor
        Training data of shape ``(d, n_train)``, complex64.
    x_test : torch.Tensor or None, optional
        Test data of shape ``(d, n_test)`` to whiten with training statistics.
        If ``None`` only the training result is returned (default).
    device : str, optional
        Target device for all returned tensors (default ``"cpu"``).

    Returns
    -------
    z_train : torch.Tensor
        Whitened training data ``(d, n_train)``.
    L : torch.Tensor
        Lower-triangular Cholesky factor ``(d, d)``.
    mean : torch.Tensor
        Column mean of training data ``(d, 1)``.
    z_test : torch.Tensor
        Whitened test data ``(d, n_test)`` — only returned when ``x_test``
        is given.

    Notes
    -----
    The Cholesky factor ``L`` satisfies ``C = L Lᴴ`` where ``C`` is the
    sample covariance of ``x_train``.  Whitening transforms data to
    zero-mean, unit-covariance: ``E[z zᴴ] = I``.
    """
    # Sample covariance matrix C = (1/(n-1)) X Xᴴ (centered).
    C = torch.cov(x_train).to(x_train.dtype)

    # Cholesky decomposition: C = L Lᴴ, L lower-triangular.
    try:
        L, _ = torch.linalg.cholesky_ex(C)
    except RuntimeError as e:
        if "The input tensor A must have at least 2 dimensions" in str(e):
            # Degenerate case: scalar covariance → use sqrt.
            L = torch.sqrt(C).unsqueeze(0).unsqueeze(1)
        else:
            raise

    # Column mean for centering: shape (d, 1).
    mean = x_train.mean(axis=1)[:, None]
    # Whitening: z = L⁻¹ (x - μ) via the Cholesky factor.
    z_train = solve(L, x_train - mean)

    if x_test is not None:
        # Apply the same whitening transform to the test set.
        z_test = solve(L, x_test - mean)
        return z_train.to(device), L.to(device), mean.to(device), z_test.to(device)

    return z_train.to(device), L.to(device), mean.to(device)


if __name__ == "__main__":
    H = cn_matrix(0, 1, (4, 4))
    X = cn_matrix(0, 1, (4, 200))
    sig = sigma_from_snr(20.0, X)
    noise = awgn(sig, X.shape)
    print(f"cn_matrix      : {H.shape} {H.dtype}")
    print(f"SNR            : {snr(X, sig):.1f} dB  (target 20 dB)")
    Xw, L_chol, mu = prewhiten(X)
    print(f"prewhiten      : {Xw.shape}, mean ≈ {Xw.mean().abs().item():.2e}")
    Q_mmse, P_mmse = svd_equalizer(H, snr_db=10.0)
    print(f"svd_equalizer  : Q={Q_mmse.shape}, P={P_mmse.shape}")
    Q_zf = rx_equalizer(H, snr_db=None)
    print(f"rx_equalizer   : Q={Q_zf.shape}")

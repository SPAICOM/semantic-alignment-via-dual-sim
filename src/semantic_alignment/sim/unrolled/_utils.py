"""Lightweight utility functions for unrolled SIM optimizers.

Provides phase computation, NMSE evaluation, gradient normalization, and
tensor-splitting helpers used across the unrolled backbone classes.
"""

import torch


def _phase_from_logit(
    logit: torch.Tensor, dtype: torch.dtype = torch.complex64, device: torch.device | None = None
) -> torch.Tensor:
    """Map unconstrained phase logit to a unit-magnitude complex phasor.

    Computes ``exp(j·2π·sigmoid(ξ))`` in float64 for numerical stability,
    then casts to the requested ``dtype`` and ``device``.

    Parameters
    ----------
    logit : torch.Tensor
        Unconstrained real-valued phase logits, shape (...,).
    dtype : torch.dtype
        Target complex dtype for the output.
    device : torch.device or None
        Target device; if None, follows the input tensor.

    Returns
    -------
    torch.Tensor
        Complex phasor with magnitude ≈ 1 and phase in [0, 2π).
    """
    s = torch.sigmoid(logit.to(torch.float64))
    return torch.exp(1j * 2 * torch.pi * s).to(dtype=dtype, device=device)


def _nmse(Z: torch.Tensor, A: torch.Tensor, A2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute normalized mean-squared error between received signal Z and target A.

    NMSE = 1 - |⟨Z, A⟩|² / (‖Z‖² · ‖A‖²), where inner products are over
    the spatial dimensions (last two axes).

    Parameters
    ----------
    Z : torch.Tensor
        Received signal matrix, shape (..., M, N).
    A : torch.Tensor
        Target signal matrix, shape (..., M, N).
    A2 : torch.Tensor
        Precomputed ‖A‖² per batch element, shape (...,).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor
        NMSE value in [0, 1]; 0 means perfect alignment.
    """
    # Inner product |⟨Z, A⟩|² over spatial dims (last two axes)
    ip = (Z.conj() * A).sum((-2, -1)).abs() ** 2
    # ‖Z‖² over spatial dims
    zz = (Z.conj() * Z).sum((-2, -1)).real
    return 1.0 - ip / (zz * A2 + eps)


def _normalize(
    g_list: list[torch.Tensor], norm: str = "rms", eps: float = 1e-12
) -> list[torch.Tensor]:
    """Normalize a list of gradient tensors for stable optimizer steps.

    Parameters
    ----------
    g_list : list[torch.Tensor]
        Gradient tensors, each shape (..., param_dim).
    norm : str
        Normalization strategy:

        - ``"none"`` — pass through unchanged.
        - ``"sign"`` — unit-sign: ``g / (|g| + ε)`` element-wise.
        - ``"rms"`` — root-mean-square: divide by RMS of the flattened
          concatenation of all gradient tensors.
        - ``"unit"`` — unit-norm: divide by the L2 norm of the flattened
          concatenation.
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    list[torch.Tensor]
        Normalized gradients, same shapes as input.
    """
    if norm == "none":
        return g_list
    if norm == "sign":
        return [g / (g.abs().detach() + eps) for g in g_list]
    # Flatten all gradients into a single vector per batch element
    flat = torch.cat([g.reshape(g.shape[0], -1) for g in g_list], dim=1)
    if norm == "rms":
        # RMS = sqrt(mean(g²))
        s = flat.pow(2).mean(dim=1, keepdim=True).sqrt()
    elif norm == "unit":
        # L2 norm
        s = flat.norm(dim=1, keepdim=True)
    else:
        raise ValueError(norm)
    # Detach scaling factor to avoid backprop through normalization
    s = (s + eps).detach()
    return [g / s.view(-1, *([1] * (g.dim() - 1))) for g in g_list]


def _split_sides(vals: list, n_left: int) -> tuple[list, list]:
    """Split a list into TX-side and RX-side components.

    Parameters
    ----------
    vals : list
        Combined list of TX + RX values.
    n_left : int
        Number of elements belonging to the TX side.

    Returns
    -------
    tuple[list, list]
        ``(tx_vals, rx_vals)``.
    """
    return list(vals[:n_left]), list(vals[n_left:])

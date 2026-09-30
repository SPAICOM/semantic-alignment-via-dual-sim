"""Phasor cascade builders and backward passes for differentiable SIM simulation.

A *cascade* is the composition of alternating fixed linear transforms (``W``)
and element-wise phasor modulations (``v = exp(j·2π·sigmoid(ξ))``) that model
how a semantic signal propagates through a stack of reconfigurable metasurfaces.

Two multiplexing conventions are supported:

- **TX cascade** (``Y·W`` scheme): modulate *after* each linear transform.
- **RX cascade** (``W·Y`` scheme): modulate *before* each linear transform.

The backward functions compute analytic gradients through these cascades
for use in end-to-end gradient descent (the "unrolled" optimizers).
"""

import torch


def xi_to_phasor(xi: torch.Tensor, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """Map unconstrained phase parameter ξ → complex phasor e^(j·2π·sigmoid(ξ)).

    Keeps sigmoid in float64 for numerical stability, then casts to ``dtype``.
    Output shape matches input shape.

    Parameters
    ----------
    xi : torch.Tensor
        Unconstrained real-valued phase logits, shape (...,).
    dtype : torch.dtype
        Target complex dtype for the output phasor.

    Returns
    -------
    torch.Tensor
        Complex phasor with magnitude ≈ 1 and phase in [0, 2π).
    """
    # Compute sigmoid in float64 to avoid half-precision saturation near ±∞
    s = torch.sigmoid(xi.to(torch.float64))
    # Phase = 2π·sigmoid(ξ) maps ℝ → [0, 2π); exp gives unit-magnitude phasor
    return torch.exp(1j * 2 * torch.pi * s).to(dtype)


def build_cascade_TX(
    W_list: list[torch.Tensor],
    phases: list[torch.Tensor],
    dtype: torch.dtype = torch.complex64,
    save_inter: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list]:
    """Build a TX cascade: G = diag(v_L) W_L ... diag(v_1) W_1.

    In the TX (Y·W) convention the phasor modulates *after* each linear
    transform:  G_i = diag(v_i) · (W_i · G_{i-1}).

    Parameters
    ----------
    W_list : list[Tensor]
        Linear transforms for each layer, shape (..., d_out, d_in).
    phases : list[Tensor]
        Phase logits for each layer, shape (..., d_out).
    dtype : torch.dtype
        Complex dtype for the cascade matrix.
    save_inter : bool
        If True, return ``(G, inter)`` where ``inter`` stores intermediate
        values needed by the backward pass.

    Returns
    -------
    Tensor or tuple[Tensor, list]
        The full cascade matrix ``G``, or ``(G, inter)`` if ``save_inter``.
    """
    B = phases[0].shape[0]  # batch dimension
    d_in = W_list[0].shape[1]  # input dimension (rows of first W)
    dev = phases[0].device
    # Start with identity: G_0 = I
    G = torch.eye(d_in, dtype=dtype, device=dev).unsqueeze(0).expand(B, -1, -1).clone()
    inter = [] if save_inter else None
    for i, W in enumerate(W_list):
        # Apply linear transform: U_i = W_i · G_{i-1}
        U = (W.to(dtype) if W.dtype != dtype else W) @ G
        if save_inter:
            # Recompute phasor + store pre-activation s for backward
            s = torch.sigmoid(phases[i].to(torch.float64))
            v = torch.exp(1j * 2 * torch.pi * s).to(dtype=dtype, device=dev)
            inter.append((U, v, s))
        else:
            v = xi_to_phasor(phases[i], dtype).to(device=dev)
        # Apply element-wise phasor modulation: G_i = v_i ⊙ U_i
        G = v.unsqueeze(-1) * U
    return (G, inter) if save_inter else G


def build_cascade_RX(
    W_list: list[torch.Tensor],
    phases: list[torch.Tensor],
    dtype: torch.dtype = torch.complex64,
    save_inter: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list]:
    """Build an RX cascade: G = W_{L-1} diag(v_{L-1}) ... W_0 diag(v_0).

    In the RX (W·Y) convention the phasor modulates *before* each linear
    transform:  G_i = W_i · (v_i ⊙ G_{i-1}).

    Parameters
    ----------
    W_list : list[Tensor]
        Linear transforms for each layer, shape (..., d_out, d_in).
    phases : list[Tensor]
        Phase logits for each layer, shape (..., d_in).
    dtype : torch.dtype
        Complex dtype for the cascade matrix.
    save_inter : bool
        If True, return ``(G, inter)`` where ``inter`` stores intermediate
        values needed by the backward pass.

    Returns
    -------
    Tensor or tuple[Tensor, list]
        The full cascade matrix ``G``, or ``(G, inter)`` if ``save_inter``.
    """
    dev = phases[0].device
    G = None
    inter = [] if save_inter else None
    for i, W in enumerate(W_list):
        # Compute phasor from phase logits
        if save_inter:
            # Recompute sigmoid for backward; store pre-activation s
            s = torch.sigmoid(phases[i].to(torch.float64))
            v = torch.exp(1j * 2 * torch.pi * s).to(dtype=dtype, device=dev)
        else:
            v = xi_to_phasor(phases[i], dtype).to(device=dev)
        if G is None:
            # First layer: G_prev = I
            m = v.shape[1]
            G_prev = torch.eye(m, dtype=dtype, device=dev)
            G = v.unsqueeze(-1) * G_prev
        else:
            G_prev = G
            # Apply element-wise phasor modulation before linear transform
            G = v.unsqueeze(-1) * G
        Wi = W.to(dtype) if W.dtype != dtype else W
        if save_inter:
            inter.append((Wi, v, G_prev, s))
        # Apply linear transform after phasor modulation
        G = Wi @ G
    return (G, inter) if save_inter else G


def cascade_backward_TX(
    W_list: list[torch.Tensor],
    inter: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    Gbar: torch.Tensor,
) -> list[torch.Tensor]:
    """Compute analytic gradients through a TX cascade via reverse-mode autodiff.

    Backpropagates ``Gbar = ∂L/∂G`` through each layer to obtain
    ``g_list[i] = ∂L/∂ξ_i`` for the phase logits.

    Parameters
    ----------
    W_list : list[Tensor]
        Linear transforms (not differentiated; treated as constants).
    inter : list[tuple]
        Intermediate values from ``build_cascade_TX(..., save_inter=True)``:
        each element is ``(U, v, s)`` where ``U`` is the pre-phasor matrix,
        ``v`` is the phasor, and ``s`` is the pre-activation sigmoid input.
    Gbar : Tensor
        Upstream gradient ``∂L/∂G``, shape (..., d_out).

    Returns
    -------
    list[Tensor]
        Per-layer phase gradients ``∂L/∂ξ_i``, one tensor per layer.
    """
    g_list = [None] * len(W_list)
    for i in reversed(range(len(W_list))):
        U, v, s = inter[i]
        # Gradient through phasor modulation: ∂G_i/∂v_i = U_i
        vbar = (Gbar.conj() * U).sum(-1)
        # Gradient through imaginary part (phase): ∂|v|/∂ξ = -Im(v·v̄)
        dphase = -(v * vbar).imag
        # Chain rule through sigmoid: d/dξ sigmoid(ξ) = σ(1-σ)
        chain = 2 * torch.pi * s * (1.0 - s)
        g_list[i] = dphase * chain
        # Propagate gradient through linear transform: Ubar = W_i^H · Gbar
        Ubar = v.conj().unsqueeze(-1) * Gbar
        Gbar = W_list[i].to(torch.complex128).conj().mT @ Ubar
    return g_list


def cascade_backward_RX(
    W_list: list[torch.Tensor],
    inter: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    Gbar: torch.Tensor,
) -> list[torch.Tensor]:
    """Compute analytic gradients through an RX cascade via reverse-mode autodiff.

    Backpropagates ``Gbar = ∂L/∂G`` through each layer to obtain
    ``g_list[i] = ∂L/∂ξ_i`` for the phase logits.

    Parameters
    ----------
    W_list : list[Tensor]
        Linear transforms (not differentiated; treated as constants).
    inter : list[tuple]
        Intermediate values from ``build_cascade_RX(..., save_inter=True)``:
        each element is ``(W, v, G_prev, s)`` where ``W`` is the linear
        transform, ``v`` is the phasor, ``G_prev`` is the previous cascade
        matrix, and ``s`` is the pre-activation sigmoid input.
    Gbar : Tensor
        Upstream gradient ``∂L/∂G``, shape (..., d_out).

    Returns
    -------
    list[Tensor]
        Per-layer phase gradients ``∂L/∂ξ_i``, one tensor per layer.
    """
    L = len(W_list)
    g_list = [None] * L
    for i in reversed(range(L)):
        W, v, G_prev, s = inter[i]
        # Gradient through linear transform: Z_bar = W_i^H · Gbar
        Z_bar = W.conj().mT @ Gbar
        # Gradient through phasor modulation: ∂G_i/∂v_i = G_prev
        vbar = (Z_bar.conj() * G_prev).sum(-1)
        # Phase gradient: ∂|v|/∂ξ = -Im(v·v̄)
        dphase = -(v * vbar).imag
        # Chain rule through sigmoid: d/dξ sigmoid(ξ) = σ(1-σ)
        chain = 2 * torch.pi * s * (1.0 - s)
        g_list[i] = dphase * chain
        if i > 0:
            # Propagate to previous layer: Gbar = v_i^* ⊙ Z_bar
            Gbar = v.conj().unsqueeze(-1) * Z_bar
    return g_list

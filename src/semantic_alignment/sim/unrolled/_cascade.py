import torch


def build_cascade_TX(W_list, phases, dtype=torch.complex64, save_inter=False):
    """TX cascade: G = diag(v_L) W_L ... diag(v_1) W_1 (Y·W scheme).

    Returns (G, inter) if save_inter else G.
    """
    B = phases[0].shape[0]
    d_in = W_list[0].shape[1]
    dev = phases[0].device
    G = torch.eye(d_in, dtype=dtype, device=dev).unsqueeze(0).expand(B, -1, -1).clone()
    inter = [] if save_inter else None
    for i, W in enumerate(W_list):
        U = (W.to(dtype) if W.dtype != dtype else W) @ G
        s = torch.sigmoid(phases[i].to(torch.float64))
        v = torch.exp(1j * 2 * torch.pi * s).to(dtype=dtype, device=dev)
        if save_inter:
            inter.append((U, v, s))
        G = v.unsqueeze(-1) * U
    return (G, inter) if save_inter else G


def build_cascade_RX(W_list, phases, dtype=torch.complex64, save_inter=False):
    """RX cascade: G = W_{L-1} diag(v_{L-1}) ... W_0 diag(v_0) (W·Y scheme).

    Returns (G, inter) if save_inter else G.
    """
    dev = phases[0].device
    G = None
    inter = [] if save_inter else None
    for i, W in enumerate(W_list):
        s = torch.sigmoid(phases[i].to(torch.float64))
        v = torch.exp(1j * 2 * torch.pi * s).to(dtype=dtype, device=dev)
        if G is None:
            m = v.shape[1]
            G_prev = torch.eye(m, dtype=dtype, device=dev)
            G = v.unsqueeze(-1) * G_prev
        else:
            G_prev = G
            G = v.unsqueeze(-1) * G
        Wi = W.to(dtype) if W.dtype != dtype else W
        if save_inter:
            inter.append((Wi, v, G_prev, s))
        G = Wi @ G
    return (G, inter) if save_inter else G


def cascade_backward_TX(W_list, inter, Gbar):
    """Reverse sweep through a TX cascade."""
    g_list = [None] * len(W_list)
    for i in reversed(range(len(W_list))):
        U, v, s = inter[i]
        vbar = (Gbar.conj() * U).sum(-1)
        dphase = -(v * vbar).imag
        chain = 2 * torch.pi * s * (1.0 - s)
        g_list[i] = dphase * chain
        Ubar = v.conj().unsqueeze(-1) * Gbar
        Gbar = W_list[i].to(torch.complex128).conj().mT @ Ubar
    return g_list


def cascade_backward_RX(W_list, inter, Gbar):
    """Reverse sweep through an RX cascade."""
    L = len(W_list)
    g_list = [None] * L
    for i in reversed(range(L)):
        W, v, G_prev, s = inter[i]
        Z_bar = W.conj().mT @ Gbar
        vbar = (Z_bar.conj() * G_prev).sum(-1)
        dphase = -(v * vbar).imag
        chain = 2 * torch.pi * s * (1.0 - s)
        g_list[i] = dphase * chain
        if i > 0:
            Gbar = v.conj().unsqueeze(-1) * Z_bar
    return g_list

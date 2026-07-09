import torch


def _phase_from_logit(logit, dtype=torch.complex64, device=None):
    s = torch.sigmoid(logit.to(torch.float64))
    return torch.exp(1j * 2 * torch.pi * s).to(dtype=dtype, device=device)


def _nmse(Z, A, A2, eps=1e-12):
    ip = (Z.conj() * A).sum((-2, -1)).abs() ** 2
    zz = (Z.conj() * Z).sum((-2, -1)).real
    return 1.0 - ip / (zz * A2 + eps)


def _normalize(g_list, norm="rms", eps=1e-12):
    if norm == "none":
        return g_list
    if norm == "sign":
        return [g / (g.abs().detach() + eps) for g in g_list]
    flat = torch.cat([g.reshape(g.shape[0], -1) for g in g_list], dim=1)
    if norm == "rms":
        s = flat.pow(2).mean(dim=1, keepdim=True).sqrt()
    elif norm == "unit":
        s = flat.norm(dim=1, keepdim=True)
    else:
        raise ValueError(norm)
    s = (s + eps).detach()
    return [g / s.view(-1, *([1] * (g.dim() - 1))) for g in g_list]


def _split_sides(vals, n_left):
    return list(vals[:n_left]), list(vals[n_left:])

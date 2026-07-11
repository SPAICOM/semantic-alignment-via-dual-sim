import torch
import torch.nn as nn

from ._cascade import (
    build_cascade_RX,
    build_cascade_TX,
    cascade_backward_RX,
    cascade_backward_TX,
)
from ._step import build_step
from ._utils import _nmse, _normalize


class MonoSIMUnrolledSCA(nn.Module):
    """Deep unrolled SCA for a single SIM (TX or RX).

    Parameters
    ----------
    sim : MonoSIMoptimizerTorch
        ``sim.side`` selects TX (Y·W) or RX (W·Y) cascade.
    K : int  number of unrolled layers
    init_w : float  initial step size
    coupling : str  ``"scalar"``, ``"diagonal"``, or ``"conv"``
    innovation_norm : str  ``"rms"``, ``"unit"``, ``"sign"``, ``"none"``
    momentum : bool  Adam-style EMA on the innovation
    init_mu : float  initial momentum coefficient
    innovation_mode : str  ``"autograd"`` or ``"analytic"``
    learn_S : bool  learnable per-layer phase decay
    shapes : list[tuple[int,int]] | None  (H,W) per SIM layer (required for conv)
    k_spatial : int  spatial kernel size for conv coupling
    k_layer : int  depth kernel size for conv3d
    eps : float
    """

    def __init__(
        self,
        sim,
        K: int,
        init_w: float = 0.1,
        coupling: str = "diagonal",
        innovation_norm: str = "rms",
        momentum: bool = True,
        init_mu: float = 0.9,
        innovation_mode: str = "autograd",
        learn_S: bool = False,
        shapes=None,
        k_spatial: int = 3,
        k_layer: int = 1,
        k_spatial_s: int | None = None,
        k_layer_s: int | None = None,
        mu_mode: str = "scalar",
        s_mode: str = "scalar",
        reg_s_mode: str = "none",
        w_mode: str = "kernel",
        eps: float = 1e-12,
    ):
        super().__init__()
        if coupling not in ("scalar", "diagonal", "conv", "conv2d"):
            raise ValueError(
                f"coupling must be 'scalar', 'diagonal', 'conv', or 'conv2d'; got {coupling!r}"
            )
        if innovation_mode not in ("autograd", "analytic"):
            raise ValueError(
                f"innovation_mode must be 'autograd' or 'analytic'; got {innovation_mode!r}"
            )
        self.side = sim.side
        if self.side == "tx":
            self._build = build_cascade_TX
            self._backward = cascade_backward_TX
        elif self.side == "rx":
            self._build = build_cascade_RX
            self._backward = cascade_backward_RX
        else:
            raise ValueError(f"sim.side must be 'tx' or 'rx'; got {self.side!r}")

        object.__setattr__(self, "_sim_ref", sim)

        self.K = int(K)
        self.coupling = coupling
        self.innovation_norm = innovation_norm
        self.innovation_mode = innovation_mode
        self.learn_S = learn_S
        self.eps = eps

        self.sizes = [p.shape[0] for p in sim.xi]
        self.n_phases = sum(self.sizes)

        for i, W in enumerate([getattr(sim, f"W_{i}") for i in range(sim.L)]):
            self.register_buffer(f"W_{i}", W)
        self._n_W = sim.L

        step_kwargs = dict(
            momentum=momentum,
            init_mu=init_mu,
            learn_S=learn_S,
            init_w=init_w,
            eps=eps,
            k_spatial=k_spatial,
            k_layer=k_layer,
            k_spatial_s=k_spatial_s,
            k_layer_s=k_layer_s,
            mu_mode=mu_mode,
            s_mode=s_mode,
            reg_s_mode=reg_s_mode,
            w_mode=w_mode,
        )
        self.step = build_step(coupling, K, self.sizes, shapes, **step_kwargs)

    # ---- helpers -----------------------------------------------------------

    def _get_W(self):
        return [getattr(self, f"W_{i}") for i in range(self._n_W)]

    # ---- physics -----------------------------------------------------------

    def _cascade(self, phases, left_mat=None, right_mat=None):
        G = self._build(self._get_W(), phases)
        if left_mat is not None:
            G = left_mat @ G
        if right_mat is not None:
            G = G @ right_mat
        return G

    # ---- innovation --------------------------------------------------------

    def _innovation_autograd(self, A, A2, phases, left_mat, right_mat, create_graph):
        with torch.inference_mode(False), torch.enable_grad():
            flat = torch.cat(phases, dim=1).clone().requires_grad_(True)
            leaf = list(flat.split(self.sizes, dim=1))
            Z = self._cascade(leaf, left_mat, right_mat)
            nmse = _nmse(Z, A, A2, self.eps)
            (grad,) = torch.autograd.grad(nmse.sum(), (flat,), create_graph=create_graph)
        return nmse.detach(), [-g.detach() for g in grad.split(self.sizes, dim=1)]

    def _innovation_analytic(self, A, A2, phases, left_mat, right_mat):
        phases = [p.detach() for p in phases]
        G_fwd, inter = self._build(self._get_W(), phases, torch.complex128, True)
        Z = G_fwd
        left = left_mat.to(torch.complex128) if left_mat is not None else None
        right = right_mat.to(torch.complex128) if right_mat is not None else None
        if left is not None:
            Z = left @ Z
        if right is not None:
            Z = Z @ right
        A64 = A.to(torch.complex128)
        ip = (Z.conj() * A64).sum((-2, -1))
        s = (Z.conj() * Z).sum((-2, -1)).real
        beta = ip / (s + self.eps)
        nmse = (1.0 - (ip.abs() ** 2) / (s * A2 + self.eps)).to(torch.float32)
        C = (beta.conj()[..., None, None] * (A64 - beta[..., None, None] * Z)).to(Z.dtype)
        scale = (2.0 / A2).unsqueeze(-1) if A2.dim() >= 1 else 2.0 / A2
        cot = C
        if self.side == "tx":
            if right is not None:
                cot = cot @ right.conj().mT
            if left is not None:
                cot = left.conj().mT @ cot
        else:
            if left is not None:
                cot = left.conj().mT @ cot
            if right is not None:
                cot = cot @ right.conj().mT
        return nmse, [g * scale for g in self._backward(self._get_W(), inter, cot)]

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        A,
        left_mat=None,
        right_mat=None,
        B: int = 1,
        record: bool = False,
        create_graph=None,
        return_phases: bool = False,
        step_callback=None,
        k_max: int | None = None,
        supervision: str = "final",
        weight_ratio: float = 1.0,
    ):
        if create_graph is None:
            create_graph = self.training
        k_run = k_max if k_max is not None else self.K
        analytic = self.innovation_mode == "analytic"
        deep = supervision == "all"
        if deep and not (0.0 < weight_ratio <= 1.0):
            raise ValueError(
                f"weight_ratio must be in (0, 1] when supervision='all'; got {weight_ratio}. Use supervision='final' to train on the last step only."
            )
        gamma = weight_ratio ** (1.0 / (k_run - 1)) if (weight_ratio < 1.0 and k_run > 1) else 1.0

        dev = next(self.buffers()).device
        A = A.to(dev)
        A2 = (A.conj() * A).sum((-2, -1)).real
        if left_mat is not None:
            left_mat = left_mat.to(dev)
        if right_mat is not None:
            right_mat = right_mat.to(dev)

        phases = [torch.zeros(B, m, device=dev) for m in self.sizes]
        v_mom = self.step.init_momentum(B, self.sizes, dev)

        need_full_curves = deep
        curves = []

        for k in range(k_run):
            if analytic:
                _, g_list = self._innovation_analytic(A, A2, phases, left_mat, right_mat)
            else:
                _, g_list = self._innovation_autograd(
                    A, A2, phases, left_mat, right_mat, create_graph
                )

            g_list = _normalize(g_list, self.innovation_norm, self.eps)
            phases, v_mom = self.step.apply(phases, g_list, v_mom, k)

            if need_full_curves or k == k_run - 1:
                curves.append(_nmse(self._cascade(phases, left_mat, right_mat), A, A2, self.eps))

            if step_callback is not None and not self.training:
                step_callback(k, phases)

        if need_full_curves:
            if gamma < 1.0:
                w = torch.tensor(
                    [gamma ** (k_run - 1 - k) for k in range(k_run)],
                    device=dev,
                    dtype=torch.float32,
                )
                loss = (w.unsqueeze(-1) * torch.stack(curves)).sum(0).mean() / w.sum()
            else:
                loss = torch.stack(curves).mean()
        else:
            loss = curves[-1].mean() if curves else torch.tensor(0.0, device=dev)

        if return_phases:
            return loss, curves, [p.detach() for p in phases]
        return loss, curves

    def reg_S(self):
        return self.step.reg_S()

    @torch.no_grad()
    def eval_curve(
        self, A, left_mat=None, right_mat=None, k_max=None, supervision="final", weight_ratio=1.0
    ):
        self.eval()
        _, curves = self.forward(
            A,
            left_mat=left_mat,
            right_mat=right_mat,
            B=A.shape[0],
            create_graph=False,
            k_max=k_max,
            supervision=supervision,
            weight_ratio=weight_ratio,
        )
        return [c.mean().item() for c in curves]

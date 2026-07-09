import torch
import torch.nn as nn

from ._cascade import (
    build_cascade_RX,
    build_cascade_TX,
    cascade_backward_RX,
    cascade_backward_TX,
)
from ._utils import _nmse, _normalize


class MonoSIMUnrolledSCA(nn.Module):
    """Deep unrolled SCA for a single SIM (TX or RX).

    Parameters
    ----------
    sim : MonoSIMoptimizerTorch
        Built mono-SIM.  ``sim.side`` selects TX (Y·W) or RX (W·Y) cascade.
    K : int
        Number of unrolled layers.
    init_w : float
        Initial step size.
    coupling : str
        ``"scalar"`` or ``"diagonal"``.
    innovation_norm : str
        ``"rms"``, ``"unit"``, ``"sign"``, ``"none"``.
    momentum : bool
        Learnable EMA on the innovation.
    init_mu : float
        Initial momentum coefficient (before logit).
    first_order : bool
        Detach innovation from graph (no 2nd-order gradients).
    innovation_mode : str
        ``"autograd"`` or ``"analytic"``.
    learn_S : bool
        If ``True``, learn a per-coordinate phase momentum ``S``.
    eps : float
        Numerical stability.
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
        first_order: bool = True,
        innovation_mode: str = "autograd",
        learn_S: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()
        if coupling not in ("scalar", "diagonal"):
            raise ValueError(f"coupling must be 'scalar' or 'diagonal', got {coupling!r}")
        if innovation_mode not in ("autograd", "analytic"):
            raise ValueError(
                f"innovation_mode must be 'autograd' or 'analytic', got {innovation_mode!r}"
            )
        if innovation_mode == "analytic" and not first_order:
            raise ValueError("analytic innovation_mode requires first_order=True")

        self.side = sim.side
        if self.side == "tx":
            self._build = build_cascade_TX
            self._backward = cascade_backward_TX
        elif self.side == "rx":
            self._build = build_cascade_RX
            self._backward = cascade_backward_RX
        else:
            raise ValueError(f"sim.side must be 'tx' or 'rx', got {self.side!r}")

        object.__setattr__(self, "_sim_ref", sim)

        self.K = int(K)
        self.coupling = coupling
        self.innovation_norm = innovation_norm
        self.momentum = momentum
        self.first_order = first_order
        self.innovation_mode = innovation_mode
        self.learn_S = learn_S
        self.eps = eps

        self.sizes = [p.shape[0] for p in sim.xi]
        self.n_phases = sum(self.sizes)

        W_list = [getattr(sim, f"W_{i}") for i in range(sim.L)]
        for i, W in enumerate(W_list):
            self.register_buffer(f"W_{i}", W)
        self._n_W = sim.L

        shp = (self.K,) if coupling == "scalar" else (self.K, self.n_phases)
        self.w = nn.Parameter(torch.full(shp, float(init_w)))

        if learn_S:
            s_shp = (self.K,) if coupling == "scalar" else (self.K, self.n_phases)
            self.s = nn.Parameter(torch.ones(s_shp))

        if momentum:
            mu0 = float(torch.logit(torch.tensor(init_mu)))
            self.mu = nn.Parameter(torch.full(shp, mu0))

    # ---- helpers -----------------------------------------------------------

    def _get_W(self):
        return [getattr(self, f"W_{i}") for i in range(self._n_W)]

    def _step_tensors(self, k):
        if self.coupling == "scalar":
            return [self.w[k]] * len(self.sizes)
        chunks = torch.split(self.w[k], self.sizes)
        return [c[None, :] for c in chunks]

    def _mu_tensors(self, k):
        mu_k = torch.sigmoid(self.mu[k]).clamp(max=0.999)
        if self.coupling == "scalar":
            return [mu_k] * len(self.sizes)
        chunks = torch.split(mu_k, self.sizes)
        return [c[None, :] for c in chunks]

    def _s_tensors(self, k):
        if not self.learn_S:
            return None
        if self.coupling == "scalar":
            return [self.s[k]] * len(self.sizes)
        chunks = torch.split(self.s[k], self.sizes)
        return [c[None, :] for c in chunks]

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
            (grad,) = torch.autograd.grad(
                nmse.sum(), (flat,), create_graph=create_graph and not self.first_order
            )
        return nmse.detach(), [-g.detach() for g in grad.split(self.sizes, dim=1)]

    def _innovation_analytic(self, A, A2, phases, left_mat, right_mat):
        phases = [p.detach() for p in phases]

        # Forward in fp64
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

        g_list = [g * scale for g in self._backward(self._get_W(), inter, cot)]
        return nmse, g_list

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
        gamma: float = 0.9,
    ):
        if create_graph is None:
            create_graph = self.training
        k_run = k_max if k_max is not None else self.K
        analytic = self.innovation_mode == "analytic"
        deep = supervision == "all" and (analytic or not self.first_order)

        dev = next(self.buffers()).device
        A = A.to(dev)
        A2 = (A.conj() * A).sum((-2, -1)).real
        if left_mat is not None:
            left_mat = left_mat.to(dev)
        if right_mat is not None:
            right_mat = right_mat.to(dev)

        phases = [torch.zeros(B, m, device=dev) for m in self.sizes]
        v_mom = [torch.zeros(B, m, device=dev) for m in self.sizes] if self.momentum else None

        need_full_curves = deep and gamma < 1.0
        curves = []

        for k in range(k_run):
            if self.innovation_mode == "autograd":
                _, g_list = self._innovation_autograd(
                    A, A2, phases, left_mat, right_mat, create_graph and not self.first_order
                )
            else:
                _, g_list = self._innovation_analytic(A, A2, phases, left_mat, right_mat)

            g_list = _normalize(g_list, self.innovation_norm, self.eps)

            if self.momentum:
                mu_k = self._mu_tensors(k)
                v_mom = [mu * v + (1 - mu) * g for mu, v, g in zip(mu_k, v_mom, g_list)]
                denom = [(1.0 - mu.detach() ** (k + 1)).clamp(min=1e-4) for mu in mu_k]
                g_list = [v / d for v, d in zip(v_mom, denom)]

            step = self._step_tensors(k)
            ss = self._s_tensors(k)
            if ss is not None:
                phases = [s * x + w * g for x, s, w, g in zip(phases, ss, step, g_list)]
            else:
                phases = [x + w * g for x, w, g in zip(phases, step, g_list)]

            should_compute_nmse = need_full_curves or k == k_run - 1
            if should_compute_nmse:
                Z = self._cascade(phases, left_mat, right_mat)
                curves.append(_nmse(Z, A, A2, self.eps))

            if step_callback is not None and not self.training:
                step_callback(k, phases)

        if need_full_curves:
            w = torch.tensor(
                [gamma ** (k_run - 1 - k) for k in range(k_run)], device=dev, dtype=torch.float32
            )
            loss = (w.unsqueeze(-1) * torch.stack(curves)).sum() / w.sum()
        else:
            loss = curves[-1].mean() if curves else torch.tensor(0.0, device=dev)

        if return_phases:
            return loss, curves, [p.detach() for p in phases]
        return loss, curves

    def reg_S(self):
        if not self.learn_S:
            return torch.tensor(0.0, device=self.w.device)
        return (self.s - 1.0).pow(2).sum()

    @torch.no_grad()
    def eval_curve(
        self,
        A,
        left_mat=None,
        right_mat=None,
        k_max: int | None = None,
        supervision: str = "final",
        gamma: float = 0.9,
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
            gamma=gamma,
        )
        return [c.mean().item() for c in curves]

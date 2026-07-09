import torch
import torch.nn as nn

from ._cascade import (
    build_cascade_RX,
    build_cascade_TX,
    cascade_backward_RX,
    cascade_backward_TX,
)
from ._utils import _nmse, _normalize


class DualSIMUnrolledSCA(nn.Module):
    """Deep unrolled SCA for dual-SIM (joint TX + RX).

    Parameters
    ----------
    sim : DualSIMoptimizerTorch
        Built dual-SIM with W_T, W_R buffers and phase sizes.
    K : int
        Number of unrolled layers.
    init_w : float
        Initial step size (per-layer scalar or per-coordinate diagonal).
    coupling : str
        ``"scalar"`` (one step per SIM layer) or ``"diagonal"`` (per-coordinate step).
    innovation_norm : str
        Gradient normalisation: ``"rms"``, ``"unit"``, ``"sign"``, or ``"none"``.
    momentum : bool
        Learnable EMA (Adam-style) on the innovation.
    init_mu : float
        Initial momentum coefficient (before logit).
    first_order : bool
        Detach innovation from the graph (no 2nd-order gradients).
    innovation_mode : str
        ``"autograd"`` (standard autograd), ``"analytic"`` (hand-coded adjoint, fp64).
    learn_S : bool
        If ``True``, learn a per-coordinate phase momentum ``S``
        (``xi^{k+1} = diag(s) xi^k + W^k g^k``) with a single vector shared
        across all layers.
    eps : float
        Small constant for numerical stability.
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

        object.__setattr__(self, "_sim_ref", sim)

        self.K = int(K)
        self.coupling = coupling
        self.innovation_norm = innovation_norm
        self.momentum = momentum
        self.first_order = first_order
        self.innovation_mode = innovation_mode
        self.learn_S = learn_S
        self.eps = eps

        self.sizes_T = [p.shape[0] for p in sim.xi_T]
        self.sizes_R = [p.shape[0] for p in sim.xi_R]
        self.n_T = sum(self.sizes_T)
        self.n_R = sum(self.sizes_R)

        W_T = [getattr(sim, f"W_T_{i}") for i in range(sim.L_T)]
        W_R = [getattr(sim, f"W_R_{i}") for i in range(sim.L_R)]
        for i, W in enumerate(W_T):
            self.register_buffer(f"W_T_{i}", W)
        for i, W in enumerate(W_R):
            self.register_buffer(f"W_R_{i}", W)
        self._n_W_T = sim.L_T
        self._n_W_R = sim.L_R

        shp_T = (self.K,) if coupling == "scalar" else (self.K, self.n_T)
        shp_R = (self.K,) if coupling == "scalar" else (self.K, self.n_R)
        self.w_T = nn.Parameter(torch.full(shp_T, float(init_w)))
        self.w_R = nn.Parameter(torch.full(shp_R, float(init_w)))

        if learn_S:
            s_shp_T = (self.K,) if coupling == "scalar" else (self.K, self.n_T)
            s_shp_R = (self.K,) if coupling == "scalar" else (self.K, self.n_R)
            self.s_T = nn.Parameter(torch.ones(s_shp_T))
            self.s_R = nn.Parameter(torch.ones(s_shp_R))

        if momentum:
            mu0 = float(torch.logit(torch.tensor(init_mu)))
            self.mu_T = nn.Parameter(torch.full(shp_T, mu0))
            self.mu_R = nn.Parameter(torch.full(shp_R, mu0))

    # ---- helpers -----------------------------------------------------------

    def _get_W_T(self):
        return [getattr(self, f"W_T_{i}") for i in range(self._n_W_T)]

    def _get_W_R(self):
        return [getattr(self, f"W_R_{i}") for i in range(self._n_W_R)]

    def _mu_tensors(self, mu: nn.Parameter, sizes, k):
        mu_k = torch.sigmoid(mu[k]).clamp(max=0.999)
        if self.coupling == "scalar":
            return [mu_k] * len(sizes)
        chunks = torch.split(mu_k, sizes)
        return [c[None, :] for c in chunks]

    def _step_tensors(self, w: nn.Parameter, sizes, k):
        if self.coupling == "scalar":
            return [w[k]] * len(sizes)
        chunks = torch.split(w[k], sizes)
        return [c[None, :] for c in chunks]

    def _s_tensors(self, s_param: nn.Parameter, sizes, k):
        if self.coupling == "scalar":
            return [s_param[k]] * len(sizes)
        chunks = torch.split(s_param[k], sizes)
        return [c[None, :] for c in chunks]

    def _apply_step(self, phases, g_list, step_tensors, s_tensors=None):
        if s_tensors is not None:
            return [s * x + w * g for x, s, w, g in zip(phases, s_tensors, step_tensors, g_list)]
        return [x + w * g for x, w, g in zip(phases, step_tensors, g_list)]

    # ---- physics -----------------------------------------------------------

    def _cascade(self, H, xT, xR):
        GT = build_cascade_TX(self._get_W_T(), xT)
        GR = build_cascade_RX(self._get_W_R(), xR)
        return GR @ H @ GT

    # ---- innovation --------------------------------------------------------

    def _innovation_autograd(self, H, A, A2, xT, xR, create_graph):
        with torch.inference_mode(False), torch.enable_grad():
            flat_T = torch.cat(xT, dim=1).clone().requires_grad_(True)
            flat_R = torch.cat(xR, dim=1).clone().requires_grad_(True)
            leaf_T = list(flat_T.split(self.sizes_T, dim=1))
            leaf_R = list(flat_R.split(self.sizes_R, dim=1))
            Z = self._cascade(H, leaf_T, leaf_R)
            nmse = _nmse(Z, A, A2, self.eps)
            grad_T, grad_R = torch.autograd.grad(
                nmse.sum(), (flat_T, flat_R), create_graph=create_graph and not self.first_order
            )
        gT = [-g.detach() for g in grad_T.split(self.sizes_T, dim=1)]
        gR = [-g.detach() for g in grad_R.split(self.sizes_R, dim=1)]
        return nmse.detach(), gT, gR

    def _innovation_analytic(self, H, A, A2, xT, xR):
        xT = [p.detach() for p in xT]
        xR = [p.detach() for p in xR]

        GT, interT = build_cascade_TX(self._get_W_T(), xT, torch.complex128, True)
        GR, interR = build_cascade_RX(self._get_W_R(), xR, torch.complex128, True)
        H64 = H.to(torch.complex128)
        Z = GR @ H64 @ GT

        A64 = A.to(torch.complex128)
        ip = (Z.conj() * A64).sum((-2, -1))
        s = (Z.conj() * Z).sum((-2, -1)).real
        beta = ip / (s + self.eps)
        nmse = (1.0 - (ip.abs() ** 2) / (s * A2 + self.eps)).to(torch.float32)
        C = (beta.conj()[..., None, None] * (A64 - beta[..., None, None] * Z)).to(Z.dtype)

        scale = (2.0 / A2).unsqueeze(-1) if A2.dim() >= 1 else 2.0 / A2
        GT_bar = (GR @ H64).conj().mT @ C
        GR_bar = C @ (H64 @ GT).conj().mT
        gT = [g * scale for g in cascade_backward_TX(self._get_W_T(), interT, GT_bar)]
        gR = [g * scale for g in cascade_backward_RX(self._get_W_R(), interR, GR_bar)]
        return nmse, gT, gR

    # ---- unrolled step logic -----------------------------------------------

    def _run_segment(self, ph_flat, vm_flat, A, A2, H, k_start, k_end):
        phT = list(ph_flat[: self.n_T].split(self.sizes_T, dim=1))
        phR = list(ph_flat[self.n_T :].split(self.sizes_R, dim=1))
        vmT = list(vm_flat[: self.n_T].split(self.sizes_T, dim=1)) if self.momentum else None
        vmR = list(vm_flat[self.n_T :].split(self.sizes_R, dim=1)) if self.momentum else None

        for k in range(k_start, k_end):
            _, gT, gR = self._innovation_analytic(
                H, A, A2, [p.detach() for p in phT], [p.detach() for p in phR]
            )
            gT = _normalize(gT, self.innovation_norm, self.eps)
            gR = _normalize(gR, self.innovation_norm, self.eps)

            if self.momentum:
                muT = self._mu_tensors(self.mu_T, self.sizes_T, k)
                muR = self._mu_tensors(self.mu_R, self.sizes_R, k)
                vmT = [mu * v + (1 - mu) * g for mu, v, g in zip(muT, vmT, gT)]
                vmR = [mu * v + (1 - mu) * g for mu, v, g in zip(muR, vmR, gR)]
                denomT = [(1.0 - mu.detach() ** (k + 1)).clamp(min=1e-4) for mu in muT]
                denomR = [(1.0 - mu.detach() ** (k + 1)).clamp(min=1e-4) for mu in muR]
                gT = [v / d for v, d in zip(vmT, denomT)]
                gR = [v / d for v, d in zip(vmR, denomR)]

            wT = self._step_tensors(self.w_T, self.sizes_T, k)
            wR = self._step_tensors(self.w_R, self.sizes_R, k)
            phT = self._apply_step(phT, gT, wT)
            phR = self._apply_step(phR, gR, wR)

        ph_flat = torch.cat(phT + phR, dim=1)
        vm_flat = torch.cat(vmT + vmR, dim=1) if self.momentum else vm_flat
        return ph_flat, vm_flat

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        H,
        A,
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
        H = H.to(dev)

        phT = [torch.zeros(B, m, device=dev) for m in self.sizes_T]
        phR = [torch.zeros(B, m, device=dev) for m in self.sizes_R]
        v_mom_T = [torch.zeros(B, m, device=dev) for m in self.sizes_T] if self.momentum else None
        v_mom_R = [torch.zeros(B, m, device=dev) for m in self.sizes_R] if self.momentum else None

        need_full_curves = deep and gamma < 1.0
        curves = []

        for k in range(k_run):
            if self.innovation_mode == "autograd":
                _, gT, gR = self._innovation_autograd(
                    H, A, A2, phT, phR, create_graph and not self.first_order
                )
            else:
                _, gT, gR = self._innovation_analytic(H, A, A2, phT, phR)

            gT = _normalize(gT, self.innovation_norm, self.eps)
            gR = _normalize(gR, self.innovation_norm, self.eps)

            if self.momentum:
                muT = self._mu_tensors(self.mu_T, self.sizes_T, k)
                muR = self._mu_tensors(self.mu_R, self.sizes_R, k)
                v_mom_T = [mu * v + (1 - mu) * g for mu, v, g in zip(muT, v_mom_T, gT)]
                v_mom_R = [mu * v + (1 - mu) * g for mu, v, g in zip(muR, v_mom_R, gR)]
                denomT = [(1.0 - mu.detach() ** (k + 1)).clamp(min=1e-4) for mu in muT]
                denomR = [(1.0 - mu.detach() ** (k + 1)).clamp(min=1e-4) for mu in muR]
                gT = [v / d for v, d in zip(v_mom_T, denomT)]
                gR = [v / d for v, d in zip(v_mom_R, denomR)]

            wT = self._step_tensors(self.w_T, self.sizes_T, k)
            wR = self._step_tensors(self.w_R, self.sizes_R, k)
            sT = self._s_tensors(self.s_T, self.sizes_T, k) if self.learn_S else None
            sR = self._s_tensors(self.s_R, self.sizes_R, k) if self.learn_S else None
            phT = self._apply_step(phT, gT, wT, sT)
            phR = self._apply_step(phR, gR, wR, sR)

            should_compute_nmse = need_full_curves or k == k_run - 1
            if should_compute_nmse:
                Z = self._cascade(H, phT, phR)
                curves.append(_nmse(Z, A, A2, self.eps))

            if step_callback is not None and not self.training:
                step_callback(k, phT + phR)

        if need_full_curves:
            w = torch.tensor(
                [gamma ** (k_run - 1 - k) for k in range(k_run)], device=dev, dtype=torch.float32
            )
            loss = (w.unsqueeze(-1) * torch.stack(curves)).sum() / w.sum()
        else:
            loss = curves[-1].mean() if curves else torch.tensor(0.0, device=dev)

        if return_phases:
            return loss, curves, [p.detach() for p in phT + phR]
        return loss, curves

    def reg_S(self):
        if not self.learn_S:
            return torch.tensor(0.0, device=self.w_T.device)
        return (self.s_T - 1.0).pow(2).sum() + (self.s_R - 1.0).pow(2).sum()

    @torch.no_grad()
    def eval_curve(
        self,
        H,
        A,
        k_max: int | None = None,
        supervision: str = "final",
        gamma: float = 0.9,
    ):
        self.eval()
        _, curves = self.forward(
            H,
            A,
            B=A.shape[0],
            create_graph=False,
            k_max=k_max,
            supervision=supervision,
            gamma=gamma,
        )
        return [c.mean().item() for c in curves]

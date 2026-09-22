"""Dual-SIM unrolled optimizers — joint and backward-compat alias."""

from collections.abc import Callable

import torch
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from .._cascade import (
    build_cascade_RX,
    build_cascade_TX,
    cascade_backward_RX,
    cascade_backward_TX,
)
from ..original.optimizers.dualsim import DualSIMoptimizerTorch
from ._base import DualUnrolledSCA
from ._step import build_step
from ._utils import _nmse, _normalize


class JointSIMUnrolledSCA(DualUnrolledSCA):
    """Deep unrolled SCA for dual-SIM with joint TX + RX cascade.

    Both TX and RX phases are updated simultaneously in each step using
    the gradient of the joint NMSE ``‖β G_R H G_T − A‖²``.

    Parameters
    ----------
    sim : DualSIMoptimizerTorch
    K : int  number of unrolled layers
    init_w : float  initial step size
    coupling : str  ``"scalar"``, ``"diagonal"``, or ``"conv"``
    innovation_norm : str  ``"rms"``, ``"unit"``, ``"sign"``, or ``"none"``
    momentum : bool  Adam-style EMA on the innovation
    init_mu : float  initial momentum coefficient
    innovation_mode : str  ``"autograd"`` or ``"analytic"``
    learn_S : bool  learnable per-layer phase decay ``xi = s*xi + w*g``
    shapes_T : list[tuple[int,int]] | None  (H,W) per TX SIM layer (required for conv)
    shapes_R : list[tuple[int,int]] | None  (H,W) per RX SIM layer (required for conv)
    k_spatial : int  spatial kernel size for conv coupling
    k_layer : int  depth kernel size for conv3d (intermediate layers)
    eps : float
    """

    def __init__(
        self,
        sim: DualSIMoptimizerTorch,
        K: int,
        init_w: float = 0.1,
        coupling: str = "diagonal",
        innovation_norm: str = "rms",
        momentum: bool = True,
        init_mu: float = 0.9,
        innovation_mode: str = "autograd",
        learn_S: bool = False,
        shapes_T: list[tuple[int, int]] | None = None,
        shapes_R: list[tuple[int, int]] | None = None,
        k_spatial: int = 3,
        k_layer: int = 1,
        k_spatial_s: int | None = None,
        k_layer_s: int | None = None,
        mu_mode: str = "scalar",
        s_mode: str = "scalar",
        reg_s_mode: str = "none",
        w_mode: str = "kernel",
        eps: float = 1e-12,
        phases_T_init: list | None = None,
        phases_R_init: list | None = None,
        gradient_checkpointing: bool = False,
    ):
        """Initialize joint dual-SIM unrolled optimizer.

        Parameters
        ----------
        sim : DualSIMoptimizerTorch
            Pre-built dual-SIM model with propagation matrices.
        K : int
            Number of unrolled optimization steps.
        init_w : float
            Initial step-size for the learnable step module.
        coupling : str
            Step module type: ``"scalar"``, ``"diagonal"``, ``"conv"``, or ``"conv2d"``.
        innovation_norm : str
            Gradient normalization: ``"rms"``, ``"unit"``, ``"sign"``, or ``"none"``.
        momentum : bool
            Enable Adam-style EMA on the innovation.
        init_mu : float
            Initial momentum coefficient.
        innovation_mode : str
            ``"autograd"`` for PyTorch autograd, ``"analytic"`` for closed-form.
        learn_S : bool
            Learn per-iteration phase scaling factors.
        shapes_T, shapes_R : list[tuple[int, int]] or None
            Spatial grid shapes per TX/RX layer (required for conv coupling).
        k_spatial : int
            Spatial kernel size for conv coupling.
        k_layer : int
            Depth kernel size for conv3d.
        eps : float
            Numerical stability constant.
        phases_T_init, phases_R_init : list[Tensor] or None
            Initial phase values for TX/RX.
        """
        super().__init__()
        if coupling not in ("scalar", "diagonal", "conv", "conv2d"):
            raise ValueError(
                f"coupling must be 'scalar', 'diagonal', 'conv', or 'conv2d'; got {coupling!r}"
            )
        if innovation_mode not in ("autograd", "analytic"):
            raise ValueError(
                f"innovation_mode must be 'autograd' or 'analytic'; got {innovation_mode!r}"
            )
        # Store sim reference without registering as submodule (it holds its own params)
        object.__setattr__(self, "_sim_ref", sim)

        self.K = int(K)
        self.coupling = coupling
        self.innovation_norm = innovation_norm
        self.innovation_mode = innovation_mode
        self.learn_S = learn_S
        self.eps = eps

        # Extract phase parameter counts from the NumPy optimizer
        self.sizes_T = [p.shape[0] for p in sim.xi_T]
        self.sizes_R = [p.shape[0] for p in sim.xi_R]
        self.n_T = sum(self.sizes_T)
        self.n_R = sum(self.sizes_R)

        # Register TX and RX propagation matrices as buffers
        for i, W in enumerate([getattr(sim, f"W_T_{i}") for i in range(sim.L_T)]):
            self.register_buffer(f"W_T_{i}", W)
        for i, W in enumerate([getattr(sim, f"W_R_{i}") for i in range(sim.L_R)]):
            self.register_buffer(f"W_R_{i}", W)
        self._n_W_T = sim.L_T
        self._n_W_R = sim.L_R

        # Build separate step modules for TX and RX
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
        self.step_T = build_step(coupling, K, self.sizes_T, shapes_T, **step_kwargs)
        self.step_R = build_step(coupling, K, self.sizes_R, shapes_R, **step_kwargs)

        self._phT_warm: list | None = phases_T_init
        self._phR_warm: list | None = phases_R_init
        self.gradient_checkpointing = gradient_checkpointing

    # ---- helpers -----------------------------------------------------------

    def _get_W_T(self) -> list[torch.Tensor]:
        """Retrieve TX propagation matrices.

        Returns
        -------
        list[Tensor]
            One propagation matrix per TX SIM layer.
        """
        return [getattr(self, f"W_T_{i}") for i in range(self._n_W_T)]

    def _get_W_R(self) -> list[torch.Tensor]:
        """Retrieve RX propagation matrices.

        Returns
        -------
        list[Tensor]
            One propagation matrix per RX SIM layer.
        """
        return [getattr(self, f"W_R_{i}") for i in range(self._n_W_R)]

    # ---- physics -----------------------------------------------------------

    def _cascade(
        self, H: torch.Tensor, xT: list[torch.Tensor], xR: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the joint end-to-end cascade Z = G_R · H · G_T.

        Parameters
        ----------
        H : Tensor
            MIMO channel matrix, shape (B, d_rx, d_tx).
        xT : list[Tensor]
            TX phase logits.
        xR : list[Tensor]
            RX phase logits.

        Returns
        -------
        Tensor
            End-to-end cascade matrix Z.
        """
        return build_cascade_RX(self._get_W_R(), xR) @ H @ build_cascade_TX(self._get_W_T(), xT)

    # ---- innovation --------------------------------------------------------

    def _innovation_autograd(
        self,
        H: torch.Tensor,
        A: torch.Tensor,
        A2: torch.Tensor,
        xT: list[torch.Tensor],
        xR: list[torch.Tensor],
        create_graph: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Compute NMSE and TX/RX gradients via PyTorch autograd.

        Parameters
        ----------
        H : Tensor
            Channel matrix.
        A : Tensor
            Target alignment matrix.
        A2 : Tensor
            Precomputed ‖A‖² per batch element.
        xT, xR : list[Tensor]
            Current TX/RX phase logits.
        create_graph : bool
            If True, keep computation graph for higher-order gradients.

        Returns
        -------
        tuple[Tensor, list[Tensor], list[Tensor]]
            ``(nmse, gT, gR)`` — NMSE and per-layer gradients for TX and RX.
        """
        with torch.inference_mode(False), torch.enable_grad():
            # Clone inputs that may be inference tensors (e.g. when called from test_step)
            H, A = H.clone(), A.clone()
            A2 = (A.conj() * A).sum((-2, -1)).real
            # Flatten TX and RX phases for joint differentiation
            flat_T = torch.cat(xT, dim=1).clone().requires_grad_(True)
            flat_R = torch.cat(xR, dim=1).clone().requires_grad_(True)
            leaf_T = list(flat_T.split(self.sizes_T, dim=1))
            leaf_R = list(flat_R.split(self.sizes_R, dim=1))
            Z = self._cascade(H, leaf_T, leaf_R)
            nmse = _nmse(Z, A, A2, self.eps)
            grad_T, grad_R = torch.autograd.grad(
                nmse.sum(),
                (flat_T, flat_R),
                create_graph=create_graph,
            )
        gT = [-g.detach() for g in grad_T.split(self.sizes_T, dim=1)]
        gR = [-g.detach() for g in grad_R.split(self.sizes_R, dim=1)]
        return nmse.detach(), gT, gR

    def _innovation_analytic(
        self,
        H: torch.Tensor,
        A: torch.Tensor,
        A2: torch.Tensor,
        xT: list[torch.Tensor],
        xR: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Compute NMSE and TX/RX gradients via analytic backward pass.

        Uses closed-form derivatives through the phasor cascades for speed
        and memory efficiency.

        Parameters
        ----------
        H : Tensor
            Channel matrix.
        A : Tensor
            Target alignment matrix.
        A2 : Tensor
            Precomputed ‖A‖² per batch element.
        xT, xR : list[Tensor]
            Current TX/RX phase logits.

        Returns
        -------
        tuple[Tensor, list[Tensor], list[Tensor]]
            ``(nmse, gT, gR)`` — NMSE and per-layer gradients for TX and RX.
        """
        xT = [p.detach() for p in xT]
        xR = [p.detach() for p in xR]
        # Build TX/RX cascades with intermediates (float128 for precision)
        GT, interT = build_cascade_TX(self._get_W_T(), xT, torch.complex128, True)
        GR, interR = build_cascade_RX(self._get_W_R(), xR, torch.complex128, True)
        H64 = H.to(torch.complex128)
        Z = GR @ H64 @ GT
        A64 = A.to(torch.complex128)
        # Compute β and NMSE
        ip = (Z.conj() * A64).sum((-2, -1))
        s = (Z.conj() * Z).sum((-2, -1)).real
        beta = ip / (s + self.eps)
        nmse = (1.0 - (ip.abs() ** 2) / (s * A2 + self.eps)).to(torch.float32)
        # Cotangent: C = β* · (A − βZ)
        C = (beta.conj()[..., None, None] * (A64 - beta[..., None, None] * Z)).to(Z.dtype)
        scale = (2.0 / A2).unsqueeze(-1) if A2.dim() >= 1 else 2.0 / A2
        # Backward through TX cascade: cotangent flows through H @ G_T
        gT = [
            g * scale
            for g in cascade_backward_TX(self._get_W_T(), interT, (GR @ H64).conj().mT @ C)
        ]
        # Backward through RX cascade: cotangent flows through G_R @ H
        gR = [
            g * scale
            for g in cascade_backward_RX(self._get_W_R(), interR, C @ (H64 @ GT).conj().mT)
        ]
        return nmse, gT, gR

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        H: torch.Tensor,
        A: torch.Tensor,
        B: int = 1,
        record: bool = False,
        create_graph: bool | None = None,
        return_phases: bool = False,
        step_callback: Callable | None = None,
        k_max: int | None = None,
        supervision: str = "final",
        weight_ratio: float = 1.0,
    ) -> (
        tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        """Run K unrolled steps on the joint dual-SIM cascade.

        Parameters
        ----------
        H : Tensor
            Channel matrix, shape (d_rx, d_tx) or (B, d_rx, d_tx).
        A : Tensor
            Target alignment matrix, shape (d_rx, d_in).
        B : int
            Batch size (number of parallel phase profiles to optimize).
        record : bool
            If True, record per-step phases (currently unused).
        create_graph : bool or None
            If None, defaults to ``self.training``.
        return_phases : bool
            If True, also return the final phase logits.
        step_callback : callable or None
            Optional callback ``callback(k, phT + phR)`` after each step.
        k_max : int or None
            Override the number of steps (default ``self.K``).
        supervision : str
            ``"final"`` for last-step loss, ``"all"`` for deep supervision.
        weight_ratio : float
            Geometric decay ratio for deep supervision weights.

        Returns
        -------
        tuple[Tensor, list[Tensor]] or tuple[Tensor, list[Tensor], list[Tensor]]
            ``(loss, curves)`` or ``(loss, curves, phases)`` if ``return_phases``.
        """
        if create_graph is None:
            create_graph = self.training
        k_run = k_max if k_max is not None else self.K
        analytic = self.innovation_mode == "analytic"
        deep = supervision == "all"
        if deep and not (0.0 < weight_ratio <= 1.0):
            raise ValueError(
                f"weight_ratio must be in (0, 1] when supervision='all'; got {weight_ratio}."
            )
        gamma = weight_ratio ** (1.0 / (k_run - 1)) if (weight_ratio < 1.0 and k_run > 1) else 1.0

        dev = next(self.buffers()).device
        A, H = A.to(dev), H.to(dev)
        A2 = (A.conj() * A).sum((-2, -1)).real

        if self._phT_warm is not None:
            phT = [p.unsqueeze(0).expand(B, -1).clone() for p in self._phT_warm]
            phR = [p.unsqueeze(0).expand(B, -1).clone() for p in self._phR_warm]
        else:
            phT = [torch.zeros(B, m, device=dev) for m in self.sizes_T]
            phR = [torch.zeros(B, m, device=dev) for m in self.sizes_R]
        vT = self.step_T.init_momentum(B, self.sizes_T, dev)
        vR = self.step_R.init_momentum(B, self.sizes_R, dev)

        curves = []

        n_T = len(self.sizes_T)
        n_R = len(self.sizes_R)
        use_ckpt = self.gradient_checkpointing and self.training and not analytic
        has_mom = vT is not None

        for k in range(k_run):
            if use_ckpt:

                def _make_step(k_):
                    def _step(*flat):
                        _phT = list(flat[:n_T])
                        _phR = list(flat[n_T : n_T + n_R])
                        _vT = list(flat[n_T + n_R : 2 * n_T + n_R]) if has_mom else None
                        _vR = list(flat[2 * n_T + n_R :]) if has_mom else None
                        _, _gT, _gR = self._innovation_autograd(H, A, A2, _phT, _phR, create_graph)
                        _gT = _normalize(_gT, self.innovation_norm, self.eps)
                        _gR = _normalize(_gR, self.innovation_norm, self.eps)
                        _phT_new, _vT_new = self.step_T.apply(_phT, _gT, _vT, k_)
                        _phR_new, _vR_new = self.step_R.apply(_phR, _gR, _vR, k_)
                        out = _phT_new + _phR_new
                        if has_mom:
                            out = out + _vT_new + _vR_new
                        return tuple(out)

                    return _step

                flat_in = tuple(phT + phR + (vT if has_mom else []) + (vR if has_mom else []))
                flat_out = _grad_ckpt(_make_step(k), *flat_in, use_reentrant=False)
                phT = list(flat_out[:n_T])
                phR = list(flat_out[n_T : n_T + n_R])
                if has_mom:
                    vT = list(flat_out[n_T + n_R : 2 * n_T + n_R])
                    vR = list(flat_out[2 * n_T + n_R :])
            else:
                if analytic:
                    _, gT, gR = self._innovation_analytic(H, A, A2, phT, phR)
                else:
                    _, gT, gR = self._innovation_autograd(H, A, A2, phT, phR, create_graph)
                gT = _normalize(gT, self.innovation_norm, self.eps)
                gR = _normalize(gR, self.innovation_norm, self.eps)
                phT, vT = self.step_T.apply(phT, gT, vT, k)
                phR, vR = self.step_R.apply(phR, gR, vR, k)

            if deep or k == k_run - 1:
                if use_ckpt:
                    _phT_cat = torch.cat(phT, dim=1)
                    _phR_cat = torch.cat(phR, dim=1)

                    def _nmse_fn(_phT_cat_, _phR_cat_):
                        _phT_ = list(_phT_cat_.split(self.sizes_T, dim=1))
                        _phR_ = list(_phR_cat_.split(self.sizes_R, dim=1))
                        return _nmse(self._cascade(H, _phT_, _phR_), A, A2, self.eps)

                    curves.append(_grad_ckpt(_nmse_fn, _phT_cat, _phR_cat, use_reentrant=False))
                else:
                    curves.append(_nmse(self._cascade(H, phT, phR), A, A2, self.eps))

            if step_callback is not None and not self.training:
                step_callback(k, phT + phR)

        if deep:
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
            return loss, curves, [p.detach() for p in phT + phR]
        return loss, curves

    def reg_S(self) -> torch.Tensor:
        """Return combined S-regularization loss from TX and RX step modules.

        Returns
        -------
        torch.Tensor
            Sum of TX and RX regularization losses.
        """
        return self.step_T.reg_S() + self.step_R.reg_S()

    @torch.no_grad()
    def eval_curve(
        self,
        H: torch.Tensor,
        A: torch.Tensor,
        k_max: int | None = None,
        supervision: str = "final",
        weight_ratio: float = 1.0,
    ) -> list[float]:
        """Evaluate the NMSE convergence curve without gradients.

        Parameters
        ----------
        H : Tensor
            Channel matrix.
        A : Tensor
            Target alignment matrix.
        k_max : int or None
            Override number of steps.
        supervision : str
            ``"final"`` or ``"all"``.
        weight_ratio : float
            Deep supervision weight ratio.

        Returns
        -------
        list[float]
            Per-step mean NMSE values.
        """
        self.eval()
        _, curves = self.forward(
            H,
            A,
            B=H.shape[0],
            create_graph=False,
            k_max=k_max,
            supervision=supervision,
            weight_ratio=weight_ratio,
        )
        return [c.mean().item() for c in curves]


# Backward-compatibility alias
DualSIMUnrolledSCA = JointSIMUnrolledSCA

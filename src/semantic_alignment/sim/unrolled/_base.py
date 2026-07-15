"""Backbone base classes for unrolled SCA optimizers."""

import torch
import torch.nn as nn

from .._cascade import (
    build_cascade_RX,
    build_cascade_TX,
    cascade_backward_RX,
    cascade_backward_TX,
)
from ..core import optimize_mono
from ._step import build_step
from ._utils import _nmse


class MonoUnrolledSCA(nn.Module):
    """Per-SIM backbone: W buffers, learnable step module, and cascade/innovation helpers.

    Owns everything needed to run one SIM's unrolled optimization loop —
    propagation matrices, step-size parameters, and the gradient computation
    primitives — but does **not** define ``forward``.  Subclasses add the
    concrete forward loop (e.g. :class:`MonoSIMUnrolledSCA`), or this class is
    used as a building block inside multi-SIM containers (e.g.
    :class:`DisjointUnrolledSCA`).

    Parameters
    ----------
    sim : MonoSIMoptimizerTorch
    K : int  number of unrolled steps
    init_w, coupling, innovation_norm, momentum, init_mu, innovation_mode,
    learn_S, shapes, k_spatial, k_layer, k_spatial_s, k_layer_s,
    mu_mode, s_mode, reg_s_mode, w_mode, eps,
    warm_start_t, warm_start_lr, warm_start_H_seed
        All forwarded to :func:`~._step.build_step` and stored for subclasses.
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
        warm_start_t: int = 0,
        warm_start_lr: float = 0.1,
        warm_start_H_seed: int | None = None,
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
        self.warm_start_t = int(warm_start_t)
        self.warm_start_lr = warm_start_lr
        self.warm_start_H_seed = warm_start_H_seed

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

    # ---- helpers -------------------------------------------------------

    def _get_W(self):
        return [getattr(self, f"W_{i}") for i in range(self._n_W)]

    # ---- physics -------------------------------------------------------

    def _cascade(self, phases, left_mat=None, right_mat=None):
        G = self._build(self._get_W(), phases)
        if left_mat is not None:
            G = left_mat @ G
        if right_mat is not None:
            G = G @ right_mat
        return G

    # ---- innovation ----------------------------------------------------

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

    # ---- warm-start ----------------------------------------------------

    def _make_H_start(self, A):
        dev = A.device
        n_R, n_T = A.shape[-2], A.shape[-1]
        if self.warm_start_H_seed is None:
            return torch.eye(n_R, n_T, dtype=A.dtype, device=dev)
        gen = torch.Generator().manual_seed(self.warm_start_H_seed)
        H = (
            torch.randn(n_R, n_T, generator=gen) + 1j * torch.randn(n_R, n_T, generator=gen)
        ) / 2**0.5
        return H.to(dtype=A.dtype, device=dev)

    def _warm_start(self, A, left_mat, B):
        """t classic Adam steps on H_start; expand result to batch size B."""
        H_start = self._make_H_start(A) if left_mat is not None else None
        dev = A.device
        phases_1, _, _ = optimize_mono(
            self._get_W(),
            self.sizes,
            A,
            self.warm_start_t,
            self.warm_start_lr,
            dev,
            left_matrix=H_start,
            best_tracking=False,
        )
        return [p.unsqueeze(0).expand(B, -1).clone() for p in phases_1]

    # ---- regularization ------------------------------------------------

    def reg_S(self):
        return self.step.reg_S()

    # ---- forward (implemented by subclasses) ---------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not define forward(). "
            "Use MonoSIMUnrolledSCA or a DisjointUnrolledSCA subclass."
        )


class DualUnrolledSCA(nn.Module):
    """Abstract base class for two-SIM unrolled optimizers.

    Serves as a common type for :class:`JointSIMUnrolledSCA` and the
    :class:`DisjointUnrolledSCA` family.  Subclasses implement
    ``forward``, ``reg_S``, and ``eval_curve``.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def reg_S(self):
        raise NotImplementedError

    def eval_curve(self, *args, **kwargs):
        raise NotImplementedError

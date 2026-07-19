"""Symmetric (full-CSI) disjoint dual-SIM unrolled optimizer."""

from collections.abc import Callable

import torch

from ..original.optimizers.monosim import MonoSIMoptimizerTorch
from ._disjoint import DisjointUnrolledSCA


class DisjointSymUnrolledSCA(DisjointUnrolledSCA):
    """Deep unrolled SCA for symmetric (full-CSI) disjoint Dual-SIM.

    Two independent SIMs (TX + RX) optimized in K interleaved steps matching
    the classic disjoint Adam baseline.  Requires batched SVD components
    ``(U, s, Vh)`` from ``HChannelDataModule(precompute_svd=True)``.

    constructive=False (default — equalized):
        TX loss: ``‖β G_T − tx_ch_inv @ tx_target‖²``
        RX loss: ``‖β G_R − rx_target @ rx_ch_inv‖²``
    constructive=True:
        TX loss: ``‖β (Σ^½ Vᴴ) G_T − tx_target‖²``
        RX loss: ``‖β G_R (U Σ^½) − rx_target‖²``

    Parameters
    ----------
    sim_tx : MonoSIMoptimizerTorch  ``side='tx'``
    sim_rx : MonoSIMoptimizerTorch  ``side='rx'``
    K : int  number of interleaved steps
    constructive : bool
    epsilon : float  Tikhonov regularization for SVD inversion
    **shared_kwargs  forwarded to both :class:`MonoUnrolledSCA` constructors
    """

    def __init__(
        self,
        sim_tx: MonoSIMoptimizerTorch,
        sim_rx: MonoSIMoptimizerTorch,
        K: int,
        constructive: bool = False,
        epsilon: float = 0.0,
        **shared_kwargs,
    ):
        """Initialize symmetric disjoint dual-SIM.

        Parameters
        ----------
        sim_tx : MonoSIMoptimizerTorch
            TX-side SIM (``side='tx'``).
        sim_rx : MonoSIMoptimizerTorch
            RX-side SIM (``side='rx'``).
        K : int
            Number of interleaved steps.
        constructive : bool
            If True, use constructive (full-CSI) targets.
        epsilon : float
            Tikhonov regularization for SVD inversion.
        **shared_kwargs
            Forwarded to both :class:`MonoUnrolledSCA` constructors.
        """
        super().__init__(sim_tx, sim_rx, K, **shared_kwargs)
        self.constructive = constructive
        self.epsilon = epsilon

    def _make_inputs(
        self,
        U: torch.Tensor,
        s: torch.Tensor,
        Vh: torch.Tensor,
        tx_target: torch.Tensor,
        rx_target: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Convert batched SVD to ``(tx_target_eff, tx_left, tx_right, rx_target_eff, rx_left, rx_right)``."""
        eps = self.epsilon
        dtype = U.dtype

        if self.constructive:
            # reg_sqrt_s = sqrt(σ² / (σ + ε)) ≈ σ^½  for large σ, → 0 for σ → 0
            reg_sqrt_s = torch.sqrt(s**2 / (s + eps)).to(dtype)  # (B, k)
            tx_left = reg_sqrt_s.unsqueeze(-1) * Vh  # (B, k, n_T) = Σ^½ Vᴴ
            rx_right = U * reg_sqrt_s.unsqueeze(-2)  # (B, n_R, k) = U Σ^½
            return tx_target, tx_left, None, rx_target, None, rx_right
        else:
            # reg_inv_sqrt_s = sqrt(σ / (σ² + ε)) ≈ σ^{-½}  for large σ, → 0 for σ → 0
            reg_inv_sqrt_s = torch.sqrt(s / (s**2 + eps)).to(dtype)  # (B, k)
            tx_ch_inv = Vh.mH * reg_inv_sqrt_s.unsqueeze(-2)  # (B, n_T, k)
            rx_ch_inv = reg_inv_sqrt_s.unsqueeze(-1) * U.mH  # (B, k, n_R)
            return tx_ch_inv @ tx_target, None, None, rx_target @ rx_ch_inv, None, None

    def forward(
        self,
        U: torch.Tensor,
        s: torch.Tensor,
        Vh: torch.Tensor,
        tx_target: torch.Tensor,
        rx_target: torch.Tensor,
        create_graph: bool | None = None,
        k_max: int | None = None,
        supervision: str = "final",
        weight_ratio: float = 1.0,
        step_callback: Callable | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """K interleaved steps on batched SVD components.

        Parameters
        ----------
        U, s, Vh : (B, n_R, k), (B, k), (B, k, n_T)  batched SVD of H
        tx_target : (k, n_tx_in) or (ch_dim, n_tx_in)  TX alignment target
        rx_target : (n_rx_out, k) or (n_rx_out, ch_dim)  RX alignment target

        Returns
        -------
        loss, curves_tx, curves_rx
        """
        B = U.shape[0]
        dev = next(self.sim_tx.buffers()).device
        U, s, Vh = U.to(dev), s.to(dev), Vh.to(dev)
        tx_target, rx_target = tx_target.to(dev), rx_target.to(dev)

        if create_graph is None:
            create_graph = self.training
        k_run = k_max if k_max is not None else self.sim_tx.K

        A_tx, tx_left, tx_right, A_rx, rx_left, rx_right = self._make_inputs(
            U, s, Vh, tx_target, rx_target
        )
        return self._run_loop(
            A_tx,
            tx_left,
            tx_right,
            A_rx,
            rx_left,
            rx_right,
            B,
            create_graph,
            k_run,
            supervision,
            weight_ratio,
            step_callback,
        )

    @torch.no_grad()
    def eval_curve(
        self,
        U: torch.Tensor,
        s: torch.Tensor,
        Vh: torch.Tensor,
        tx_target: torch.Tensor,
        rx_target: torch.Tensor,
        k_max: int | None = None,
        supervision: str = "final",
        weight_ratio: float = 1.0,
    ) -> tuple[list[float], list[float]]:
        """Evaluate TX and RX NMSE convergence curves without gradients.

        Parameters
        ----------
        U, s, Vh : Tensor
            Batched SVD components of H.
        tx_target, rx_target : Tensor
            TX and RX alignment targets.
        k_max : int or None
            Override number of steps.
        supervision : str
            ``"final"`` or ``"all"``.
        weight_ratio : float
            Deep supervision weight ratio.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(curves_tx, curves_rx)`` — per-step mean NMSE for each SIM.
        """
        self.eval()
        _, curves_tx, curves_rx = self.forward(
            U,
            s,
            Vh,
            tx_target,
            rx_target,
            create_graph=False,
            k_max=k_max,
            supervision=supervision,
            weight_ratio=weight_ratio,
        )
        return (
            [c.mean().item() for c in curves_tx],
            [c.mean().item() for c in curves_rx],
        )

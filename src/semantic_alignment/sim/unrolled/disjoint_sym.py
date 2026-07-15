import torch

from ._disjoint import DisjointUnrolledSCA


class DisjointSymUnrolledSCA(DisjointUnrolledSCA):
    """Deep unrolled SCA for symmetric (full-CSI) disjoint Dual-SIM.

    Two independent SIMs (TX + RX) optimized in K interleaved steps matching
    the classic disjoint Adam baseline.  Requires batched SVD components
    ``(U, s, Vh)`` from ``HChannelDataModule(precompute_svd=True)``.

    constructive=False (default — equalized):
        TX loss: ``‖β G_T − B_T T_T‖²``
        RX loss: ``‖β G_R − T_R B_R‖²``
    constructive=True:
        TX loss: ``‖β (Σ^½ Vᴴ) G_T − T_T‖²``
        RX loss: ``‖β G_R (U Σ^½) − T_R‖²``

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
        sim_tx,
        sim_rx,
        K: int,
        constructive: bool = False,
        epsilon: float = 0.0,
        **shared_kwargs,
    ):
        super().__init__(sim_tx, sim_rx, K, **shared_kwargs)
        self.constructive = constructive
        self.epsilon = epsilon

    def _make_inputs(self, U, s, Vh, T_T, T_R):
        """Convert batched SVD to ``(A_tx, tx_left, tx_right, A_rx, rx_left, rx_right)``."""
        eps = self.epsilon
        dtype = U.dtype

        if self.constructive:
            # reg_sqrt_s = sqrt(σ² / (σ + ε)) ≈ σ^½  for large σ, → 0 for σ → 0
            reg_sqrt_s = torch.sqrt(s**2 / (s + eps)).to(dtype)  # (B, k)
            tx_left = reg_sqrt_s.unsqueeze(-1) * Vh  # (B, k, n_T) = Σ^½ Vᴴ
            rx_right = U * reg_sqrt_s.unsqueeze(-2)  # (B, n_R, k) = U Σ^½
            return T_T, tx_left, None, T_R, None, rx_right
        else:
            # reg_inv_sqrt_s = sqrt(σ / (σ² + ε)) ≈ σ^{-½}  for large σ, → 0 for σ → 0
            reg_inv_sqrt_s = torch.sqrt(s / (s**2 + eps)).to(dtype)  # (B, k)
            B_T = Vh.mH * reg_inv_sqrt_s.unsqueeze(-2)  # (B, n_T, k)
            B_R = reg_inv_sqrt_s.unsqueeze(-1) * U.mH  # (B, k, n_R)
            return B_T @ T_T, None, None, T_R @ B_R, None, None

    def forward(
        self,
        U,
        s,
        Vh,
        T_T,
        T_R,
        create_graph=None,
        k_max=None,
        supervision: str = "final",
        weight_ratio: float = 1.0,
        step_callback=None,
    ):
        """K interleaved steps on batched SVD components.

        Parameters
        ----------
        U, s, Vh : (B, n_R, k), (B, k), (B, k, n_T)  batched SVD of H
        T_T : (k, n_tx_in) or (ch_dim, n_tx_in)  TX alignment target
        T_R : (n_rx_out, k) or (n_rx_out, ch_dim)  RX alignment target

        Returns
        -------
        loss, curves_tx, curves_rx
        """
        B = U.shape[0]
        dev = next(self.sim_tx.buffers()).device
        U, s, Vh = U.to(dev), s.to(dev), Vh.to(dev)
        T_T, T_R = T_T.to(dev), T_R.to(dev)

        if create_graph is None:
            create_graph = self.training
        k_run = k_max if k_max is not None else self.sim_tx.K

        A_tx, tx_left, tx_right, A_rx, rx_left, rx_right = self._make_inputs(U, s, Vh, T_T, T_R)
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
    def eval_curve(self, U, s, Vh, T_T, T_R, k_max=None, supervision="final", weight_ratio=1.0):
        self.eval()
        _, curves_tx, curves_rx = self.forward(
            U,
            s,
            Vh,
            T_T,
            T_R,
            create_graph=False,
            k_max=k_max,
            supervision=supervision,
            weight_ratio=weight_ratio,
        )
        return (
            [c.mean().item() for c in curves_tx],
            [c.mean().item() for c in curves_rx],
        )

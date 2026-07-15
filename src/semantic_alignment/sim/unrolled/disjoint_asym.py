import torch

from ._disjoint import DisjointUnrolledSCA


class DisjointAsymUnrolledSCA(DisjointUnrolledSCA):
    """Deep unrolled SCA for asymmetric disjoint Dual-SIM.

    Two independent SIMs (TX + RX) optimized in K interleaved steps matching
    the classic disjoint Adam baseline.  Channel matrices H are provided per
    batch and converted on-the-fly to per-sample targets / side matrices.

    constructive=False, alignment_side='tx'  (default):
        TX: ``‖β G_T − A‖²``       (target = A, H-independent)
        RX: ``‖β G_R − H†‖²``      (target = H†_batch, per-sample)
    constructive=True, alignment_side='tx':
        TX: ``‖β G_T − A‖²``
        RX: ``‖β G_R H − I‖²``     (rx_right = H_batch)
    constructive=True, alignment_side='rx':
        TX: ``‖β H G_T − I‖²``     (tx_left = H_batch)
        RX: ``‖β G_R − A‖²``

    Parameters
    ----------
    sim_tx : MonoSIMoptimizerTorch  ``side='tx'``
    sim_rx : MonoSIMoptimizerTorch  ``side='rx'``
    K : int  number of interleaved steps
    constructive : bool
    alignment_side : str  ``'tx'`` or ``'rx'``
    epsilon : float  Tikhonov regularization for H† (non-constructive only)
    **shared_kwargs  forwarded to both :class:`MonoUnrolledSCA` constructors
    """

    def __init__(
        self,
        sim_tx,
        sim_rx,
        K: int,
        constructive: bool = False,
        alignment_side: str = "tx",
        epsilon: float = 0.0,
        **shared_kwargs,
    ):
        super().__init__(sim_tx, sim_rx, K, **shared_kwargs)
        if not constructive and alignment_side == "rx":
            raise ValueError(
                "Non-constructive asymmetric with alignment_side='rx' is not supported."
            )
        self.constructive = constructive
        self.alignment_side = alignment_side
        self.epsilon = epsilon

    def _make_inputs(self, H_batch, A_target):
        """Compute ``(A_tx, tx_left, tx_right, A_rx, rx_left, rx_right)`` from a batch of H."""
        eps = self.epsilon
        dev = H_batch.device
        dtype = H_batch.dtype
        A = A_target.to(dev)

        if not self.constructive:
            # H† via regularized SVD:  H† = V diag(σ/(σ²+ε)) Uᴴ
            U, s, Vh = torch.linalg.svd(H_batch, full_matrices=False)
            reg_s = (s / (s**2 + eps)).to(dtype)  # (B, k)
            Q = (Vh.mH * reg_s.unsqueeze(-2)) @ U.mH  # (B, n_T, n_R)
            return A, None, None, Q, None, None

        if self.alignment_side == "tx":
            ch_dim = A.shape[0]
            I_ch = torch.eye(ch_dim, dtype=dtype, device=dev)
            return A, None, None, I_ch, None, H_batch

        # alignment_side == 'rx'
        ch_dim = A.shape[1]
        I_ch = torch.eye(ch_dim, dtype=dtype, device=dev)
        return I_ch, H_batch, None, A, None, None

    def forward(
        self,
        H_batch,
        A_target,
        create_graph=None,
        k_max=None,
        supervision: str = "final",
        weight_ratio: float = 1.0,
        step_callback=None,
    ):
        """K interleaved steps on a batch of channel matrices.

        Parameters
        ----------
        H_batch : (B, n_R, n_T)  batch of channel matrices
        A_target : (n_rx_out, n_tx_in)  fixed alignment target

        Returns
        -------
        loss, curves_tx, curves_rx
        """
        B = H_batch.shape[0]
        dev = next(self.sim_tx.buffers()).device
        H_batch = H_batch.to(dev)

        if create_graph is None:
            create_graph = self.training
        k_run = k_max if k_max is not None else self.sim_tx.K

        A_tx, tx_left, tx_right, A_rx, rx_left, rx_right = self._make_inputs(H_batch, A_target)
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
    def eval_curve(self, H_batch, A_target, k_max=None, supervision="final", weight_ratio=1.0):
        self.eval()
        _, curves_tx, curves_rx = self.forward(
            H_batch,
            A_target,
            create_graph=False,
            k_max=k_max,
            supervision=supervision,
            weight_ratio=weight_ratio,
        )
        return (
            [c.mean().item() for c in curves_tx],
            [c.mean().item() for c in curves_rx],
        )

"""Shared interleaved-loop backbone for disjoint (independent TX+RX) unrolled SCA."""

from collections.abc import Callable

import torch

from ..original.optimizers.monosim import MonoSIMoptimizerTorch
from ._base import DualUnrolledSCA, MonoUnrolledSCA
from ._utils import _nmse, _normalize


class DisjointUnrolledSCA(DualUnrolledSCA):
    """Backbone for disjoint Dual-SIM unrolled SCA.

    Holds two independent :class:`MonoUnrolledSCA` instances (TX + RX) and
    runs them in K interleaved steps (1 TX step + 1 RX step per iteration),
    matching the classic disjoint Adam baseline.

    Subclasses override :meth:`_make_inputs` to compute the six-tuple
    ``(A_tx, tx_left, tx_right, A_rx, rx_left, rx_right)`` from whatever
    batch inputs their scenario requires, and define a thin ``forward``
    that moves data to device, calls :meth:`_make_inputs`, then calls
    :meth:`_run_loop`.

    Parameters
    ----------
    sim_tx : MonoSIMoptimizerTorch  ``side='tx'``
    sim_rx : MonoSIMoptimizerTorch  ``side='rx'``
    K : int  number of interleaved steps
    **shared_kwargs  forwarded to both :class:`MonoUnrolledSCA` constructors
    """

    def __init__(
        self, sim_tx: MonoSIMoptimizerTorch, sim_rx: MonoSIMoptimizerTorch, K: int, **shared_kwargs
    ):
        """Initialize disjoint dual-SIM with independent TX and RX backbones.

        Parameters
        ----------
        sim_tx : MonoSIMoptimizerTorch
            TX-side SIM optimizer (``side='tx'``).
        sim_rx : MonoSIMoptimizerTorch
            RX-side SIM optimizer (``side='rx'``).
        K : int
            Number of interleaved optimization steps.
        **shared_kwargs
            Forwarded to both :class:`MonoUnrolledSCA` constructors.
        """
        super().__init__()
        self.sim_tx = MonoUnrolledSCA(sim_tx, K, **shared_kwargs)
        self.sim_rx = MonoUnrolledSCA(sim_rx, K, **shared_kwargs)

    # ---- interface ---------------------------------------------------------

    def _make_inputs(
        self, *args
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return (A_tx, tx_left, tx_right, A_rx, rx_left, rx_right)."""
        raise NotImplementedError

    # ---- shared loop -------------------------------------------------------

    def _run_loop(
        self,
        A_tx: torch.Tensor,
        tx_left: torch.Tensor | None,
        tx_right: torch.Tensor | None,
        A_rx: torch.Tensor,
        rx_left: torch.Tensor | None,
        rx_right: torch.Tensor | None,
        B: int,
        create_graph: bool,
        k_run: int,
        supervision: str = "final",
        weight_ratio: float = 1.0,
        step_callback: Callable | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """K interleaved steps (1 TX + 1 RX each).

        Returns
        -------
        loss : scalar tensor
        curves_tx : list[(B,) tensors] — per-step TX NMSE
        curves_rx : list[(B,) tensors] — per-step RX NMSE
        """
        dev = next(self.sim_tx.buffers()).device
        deep = supervision == "all"
        gamma = weight_ratio ** (1.0 / (k_run - 1)) if (weight_ratio < 1.0 and k_run > 1) else 1.0

        A2_tx = (A_tx.conj() * A_tx).sum((-2, -1)).real
        A2_rx = (A_rx.conj() * A_rx).sum((-2, -1)).real

        tx_phases = [torch.zeros(B, m, device=dev) for m in self.sim_tx.sizes]
        rx_phases = [torch.zeros(B, m, device=dev) for m in self.sim_rx.sizes]
        v_tx = self.sim_tx.step.init_momentum(B, self.sim_tx.sizes, dev)
        v_rx = self.sim_rx.step.init_momentum(B, self.sim_rx.sizes, dev)

        curves_tx, curves_rx = [], []

        for k in range(k_run):
            _, g_tx = self.sim_tx._innovation_autograd(
                A_tx, A2_tx, tx_phases, tx_left, tx_right, create_graph
            )
            g_tx = _normalize(g_tx, self.sim_tx.innovation_norm, self.sim_tx.eps)
            tx_phases, v_tx = self.sim_tx.step.apply(tx_phases, g_tx, v_tx, k)

            _, g_rx = self.sim_rx._innovation_autograd(
                A_rx, A2_rx, rx_phases, rx_left, rx_right, create_graph
            )
            g_rx = _normalize(g_rx, self.sim_rx.innovation_norm, self.sim_rx.eps)
            rx_phases, v_rx = self.sim_rx.step.apply(rx_phases, g_rx, v_rx, k)

            if step_callback is not None:
                step_callback(k, tx_phases, rx_phases)

            if deep or k == k_run - 1:
                curves_tx.append(
                    _nmse(
                        self.sim_tx._cascade(tx_phases, tx_left, tx_right),
                        A_tx,
                        A2_tx,
                        self.sim_tx.eps,
                    )
                )
                curves_rx.append(
                    _nmse(
                        self.sim_rx._cascade(rx_phases, rx_left, rx_right),
                        A_rx,
                        A2_rx,
                        self.sim_rx.eps,
                    )
                )

        if deep:
            if gamma < 1.0:
                w = torch.tensor(
                    [gamma ** (k_run - 1 - k) for k in range(k_run)],
                    device=dev,
                    dtype=torch.float32,
                )
                loss_tx = (w.unsqueeze(-1) * torch.stack(curves_tx)).sum(0).mean() / w.sum()
                loss_rx = (w.unsqueeze(-1) * torch.stack(curves_rx)).sum(0).mean() / w.sum()
            else:
                loss_tx = torch.stack(curves_tx).mean()
                loss_rx = torch.stack(curves_rx).mean()
        else:
            loss_tx = curves_tx[-1].mean()
            loss_rx = curves_rx[-1].mean()

        return loss_tx + loss_rx, curves_tx, curves_rx

    # ---- common accessors --------------------------------------------------

    def reg_S(self) -> torch.Tensor:
        """Return combined regularization loss from TX and RX step modules.

        Returns
        -------
        torch.Tensor
            Sum of TX and RX S-regularization losses.
        """
        return self.sim_tx.reg_S() + self.sim_rx.reg_S()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Run the disjoint optimization loop (implemented by subclasses).

        Raises
        ------
        NotImplementedError
            Always — subclasses must override this method.
        """
        raise NotImplementedError

    def eval_curve(self, *args, **kwargs) -> list[float]:
        """Evaluate NMSE convergence curves (implemented by subclasses).

        Raises
        ------
        NotImplementedError
            Always — subclasses must override this method.
        """
        raise NotImplementedError

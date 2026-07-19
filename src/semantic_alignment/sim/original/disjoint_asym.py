"""Asymmetric disjoint Dual-SIM runner."""

import numpy as np
import torch

from semantic_alignment.utils import rx_equalizer

from ._utils import tipology_tag
from .disjoint_2sim import Disjoint2SIM, DisjointTargets


class DisjointAsymSIM(Disjoint2SIM):
    """Asymmetric disjoint Dual-SIM runner."""

    @staticmethod
    def build_targets(
        A_target: torch.Tensor,
        H_mimo: torch.Tensor,
        ch_dim: int,
        device: torch.device,
        epsilon: float = 0.0,
        constructive: bool = False,
        alignment_side: str = "tx",
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute the 6 matrices for the asymmetric disjoint strategy.

        Returns ``(tx_target, tx_left, tx_right, rx_target, rx_left, rx_right)``.
        """
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

        with torch.no_grad():
            I_ch = torch.eye(ch_dim, dtype=torch.complex64, device=device)

            if not constructive:
                if alignment_side == "rx":
                    raise NotImplementedError(
                        "Disjoint Asym RX equalized (non-constructive) is not supported. "
                        "Use alignment_side='rx' with constructive=True instead."
                    )
                Q = rx_equalizer(H_torch, snr_db=None, epsilon=epsilon)
                return A_torch, None, None, Q, None, None

            if alignment_side == "tx":
                return A_torch, None, None, I_ch, None, H_torch
            else:
                return I_ch, H_torch, None, A_torch, None, None

    def compute_targets(
        self,
        A_target: torch.Tensor | np.ndarray,
        H_mimo: torch.Tensor | np.ndarray,
        ch_dim: int,
        device: torch.device,
        **kwargs,
    ) -> DisjointTargets:
        """Compute the 6 training matrices for the asymmetric disjoint strategy.

        Parameters
        ----------
        A_target : Tensor
            Target alignment matrix.
        H_mimo : Tensor
            MIMO channel matrix.
        ch_dim : int
            Channel dimension.
        device : torch.device
            Target device.
        **kwargs
            Must include ``constructive`` (bool) and ``alignment_side`` (str).

        Returns
        -------
        DisjointTargets
            Training matrices for TX and RX SIMs.
        """
        tx_target, tx_left, tx_right, rx_target, rx_left, rx_right = self.build_targets(
            A_target,
            H_mimo,
            ch_dim,
            device,
            epsilon=kwargs.get("epsilon", 0.0),
            constructive=kwargs.get("constructive", False),
            alignment_side=kwargs.get("alignment_side", "tx"),
        )
        return DisjointTargets(tx_target, tx_left, tx_right, rx_target, rx_left, rx_right)

    def _ch_dim(self, A_target: torch.Tensor | np.ndarray, **kwargs) -> int:
        """Return channel dimension based on alignment side.

        For ``alignment_side="rx"``, uses the number of TX antennas;
        otherwise uses the number of RX antennas.
        """
        side = kwargs.get("alignment_side", "tx")
        return int(A_target.shape[1]) if side == "rx" else int(A_target.shape[0])

    def _tipology_subdir(self, **kwargs) -> str:
        """Return tipology subdirectory: ``constructive/{tx|rx}`` or ``equalized/{tx|rx}``."""
        return tipology_tag(
            "disjoint_asym",
            kwargs.get("constructive", False),
            kwargs.get("alignment_side", "tx"),
        )

    @classmethod
    def run_sim(cls, **kwargs) -> None:
        """Entry point for Hydra ``_target_``."""
        cls().run(**kwargs)

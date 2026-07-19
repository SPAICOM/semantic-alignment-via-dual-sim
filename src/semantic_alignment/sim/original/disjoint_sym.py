"""Symmetric (full-CSI) disjoint Dual-SIM runner."""

import numpy as np
import torch

from ._utils import tipology_tag
from .disjoint_2sim import Disjoint2SIM, DisjointTargets


class DisjointSymSIM(Disjoint2SIM):
    """Symmetric (full-CSI) disjoint Dual-SIM runner."""

    @staticmethod
    def build_alignment_targets(
        A_target: torch.Tensor | np.ndarray,
        ch_dim: int,
        strategy_name: str,
        device: torch.device,
        F_T: torch.Tensor | np.ndarray | None = None,
        F_R_H: torch.Tensor | np.ndarray | None = None,
    ):
        """Return the H-independent alignment targets (tx_target, rx_target) for sym disjoint."""
        A = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
        if strategy_name == "PPFE":
            if F_T is None or F_R_H is None:
                raise ValueError("PPFE requires F_T and F_R_H.")
            return (
                torch.as_tensor(F_T, dtype=torch.complex64, device=device),
                torch.as_tensor(F_R_H, dtype=torch.complex64, device=device),
            )
        elif strategy_name == "Linear":
            U_A, s_A, Vh_A = torch.linalg.svd(A, full_matrices=False)
            sqrt_sA = torch.sqrt(s_A[:ch_dim]).to(A.dtype)
            tx_target = torch.diag(sqrt_sA) @ Vh_A[:ch_dim, :]
            rx_target = U_A[:, :ch_dim] @ torch.diag(sqrt_sA)
            return tx_target.detach(), rx_target.detach()
        else:
            raise NotImplementedError(
                f"Sym alignment targets not implemented for '{strategy_name}'."
            )

    @staticmethod
    def build_targets(
        A_target: torch.Tensor,
        H_mimo: torch.Tensor,
        ch_dim: int,
        strategy_name: str,
        device: torch.device,
        F_T: torch.Tensor | np.ndarray | None = None,
        F_R_H: torch.Tensor | np.ndarray | None = None,
        epsilon: float = 0.0,
        constructive: bool = False,
        **_,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute the 6 matrices for the symmetric disjoint strategy.

        Returns ``(tx_target, tx_left, tx_right, rx_target, rx_left, rx_right)``.
        """
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

        with torch.no_grad():
            U, s, Vh = torch.linalg.svd(H_torch, full_matrices=False)

            if strategy_name == "PPFE":
                if F_T is None or F_R_H is None:
                    raise ValueError("PPFE requires F_T and F_R_H (n_clusters=ch_dim).")
                tx_target = torch.as_tensor(F_T, dtype=torch.complex64, device=device)
                rx_target = torch.as_tensor(F_R_H, dtype=torch.complex64, device=device)
            elif strategy_name == "Linear":
                if ch_dim > min(A_torch.shape):
                    raise ValueError(
                        f"Linear sym requires channel_dim <= {min(A_torch.shape)}, got {ch_dim}"
                    )
                U_A, s_A, Vh_A = torch.linalg.svd(A_torch, full_matrices=False)
                sqrt_sA = torch.sqrt(s_A[:ch_dim]).to(A_torch.dtype)
                tx_target = torch.diag(sqrt_sA) @ Vh_A[:ch_dim, :]
                rx_target = U_A[:, :ch_dim] @ torch.diag(sqrt_sA)
            else:
                raise NotImplementedError(f"Sym disjoint not implemented for '{strategy_name}'.")

            reg_inv_sqrt_s = torch.sqrt(s / (s**2 + epsilon)).to(H_torch.dtype)

            if not constructive:
                tx_ch_inv = Vh.H @ torch.diag(reg_inv_sqrt_s)
                rx_ch_inv = torch.diag(reg_inv_sqrt_s) @ U.H
                if ch_dim < tx_ch_inv.shape[0]:
                    tx_ch_inv = tx_ch_inv[:ch_dim, :ch_dim]
                    rx_ch_inv = rx_ch_inv[:ch_dim, :ch_dim]
                return (
                    (tx_ch_inv @ tx_target).detach(),
                    None,
                    None,
                    (rx_target @ rx_ch_inv).detach(),
                    None,
                    None,
                )

            reg_sqrt_s = torch.sqrt(s**2 / (s + epsilon)).to(H_torch.dtype)
            tx_left = torch.diag(reg_sqrt_s) @ Vh
            rx_right = U @ torch.diag(reg_sqrt_s)
            return tx_target.detach(), tx_left, None, rx_target.detach(), None, rx_right

    def compute_targets(
        self,
        A_target: torch.Tensor | np.ndarray,
        H_mimo: torch.Tensor | np.ndarray,
        ch_dim: int,
        device: torch.device,
        **kwargs,
    ) -> DisjointTargets:
        """Compute the 6 training matrices for the symmetric disjoint strategy.

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
            Must include ``strategy_name`` (str). Optional: ``constructive`` (bool),
            ``F_T``, ``F_R_H``, ``epsilon`` (float).

        Returns
        -------
        DisjointTargets
            Training matrices for TX and RX SIMs.
        """
        tx_target, tx_left, tx_right, rx_target, rx_left, rx_right = self.build_targets(
            A_target,
            H_mimo,
            ch_dim,
            kwargs.get("strategy_name", "PPFE"),
            device,
            F_T=kwargs.get("F_T"),
            F_R_H=kwargs.get("F_R_H"),
            epsilon=kwargs.get("epsilon", 0.0),
            constructive=kwargs.get("constructive", False),
        )
        return DisjointTargets(tx_target, tx_left, tx_right, rx_target, rx_left, rx_right)

    def _tipology_subdir(self, **kwargs) -> str:
        """Return tipology subdirectory: ``"constructive"`` or ``"equalized"``."""
        return tipology_tag("disjoint_sym", kwargs.get("constructive", False))

    @classmethod
    def run_sim(cls, **kwargs) -> None:
        """Entry point for Hydra ``_target_``."""
        cls().run(**kwargs)

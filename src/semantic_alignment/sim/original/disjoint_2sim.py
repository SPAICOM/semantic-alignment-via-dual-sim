"""Abstract base for all 2-SIM disjoint runners.

Subclasses implement only ``compute_targets()`` (and optionally ``_ch_dim()``
and ``_tipology_subdir()``); the shared build / optimize / save logic lives
here, and the evaluation is delegated to the downstream task.
"""

import gc
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.optim as optim

from semantic_alignment.sim.link import Transmit, dual_transmit

from ._base import SIMRunner
from ._utils import ch_dims, make_fname, phase_state, snr_tag
from .mono import MonoSIM
from .optimizers.monosim import MonoSIMoptimizerTorch

if TYPE_CHECKING:
    from semantic_alignment.evaluation.tasks import Task


@dataclass
class DisjointTargets:
    """The six matrices that fully define a 2-SIM disjoint training problem.

    Each SIM minimises ``‖β (tx_left @ G @ tx_right) − tx_target‖²``;
    ``None`` means identity (no matrix on that side).
    """

    tx_target: torch.Tensor
    tx_left: torch.Tensor | None
    tx_right: torch.Tensor | None
    rx_target: torch.Tensor
    rx_left: torch.Tensor | None
    rx_right: torch.Tensor | None


class Disjoint2SIM(SIMRunner):
    """Abstract base class for 2-SIM disjoint runners.

    Subclasses must implement :meth:`compute_targets`.
    Optionally override :meth:`_ch_dim` and :meth:`_tipology_subdir`.
    """

    @abstractmethod
    def compute_targets(
        self,
        A_target: torch.Tensor | np.ndarray,
        H_mimo: torch.Tensor | np.ndarray,
        ch_dim: int,
        device: torch.device,
        **kwargs,
    ) -> DisjointTargets:
        """Compute the 6 training matrices for TX and RX SIMs."""
        ...

    def _ch_dim(self, A_target: torch.Tensor | np.ndarray, **kwargs) -> int:
        """Return the channel dimension; defaults to the max of A_target shape."""
        return kwargs.get("channel_dim") or int(max(A_target.shape))

    def _tipology_subdir(self, **kwargs) -> str:
        """Return the tipology subdirectory name; empty by default."""
        return ""

    @staticmethod
    def build_sims(
        L: int,
        M_int: int,
        A_target: torch.Tensor | np.ndarray,
        ch_dim: int,
        slayer_tx: float,
        slayer_rx: float,
        wavelength: float,
        dx: float,
        device: torch.device,
    ) -> tuple[MonoSIMoptimizerTorch, MonoSIMoptimizerTorch]:
        """Build the independent TX-side (``tx_in → ch``) and RX-side (``ch → rx_out``) Mono-SIMs."""
        ch_x, ch_y = ch_dims(ch_dim)
        tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
        rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))
        tx_model = MonoSIM.build_model(
            L,
            M_int,
            tx_in_x,
            tx_in_y,
            ch_x,
            ch_y,
            "tx",
            slayer_tx,
            wavelength,
            dx,
            device,
        )
        rx_model = MonoSIM.build_model(
            L,
            M_int,
            ch_x,
            ch_y,
            rx_out_x,
            rx_out_y,
            "rx",
            slayer_rx,
            wavelength,
            dx,
            device,
        )
        return tx_model, rx_model

    @staticmethod
    def optimize_sims(
        tx_model: MonoSIMoptimizerTorch,
        rx_model: MonoSIMoptimizerTorch,
        targets: DisjointTargets,
        lr_tx: float,
        lr_rx: float,
        max_iters: int,
        optimizer: str = "adam",
        scheduler: str = "none",
        warmup_frac: float = 0.05,
    ) -> tuple[list[float], torch.Tensor, list[float], torch.Tensor]:
        """Train TX-SIM and RX-SIM independently against their targets.

        Returns
        -------
        tuple
            ``(tx_loss_history, beta_T, rx_loss_history, beta_R)``.
        """
        sim_kw = dict(
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            max_iters=max_iters,
        )
        tx_loss, beta_T = tx_model.optimize_sim(
            target_matrix=targets.tx_target,
            lr=lr_tx,
            left_matrix=targets.tx_left,
            right_matrix=targets.tx_right,
            **sim_kw,
        )
        rx_loss, beta_R = rx_model.optimize_sim(
            target_matrix=targets.rx_target,
            lr=lr_rx,
            left_matrix=targets.rx_left,
            right_matrix=targets.rx_right,
            **sim_kw,
        )
        return tx_loss, beta_T, rx_loss, beta_R

    @staticmethod
    def combine_betas(
        beta_T: torch.Tensor | complex, beta_R: torch.Tensor | complex, device: torch.device
    ) -> tuple[complex, complex, torch.Tensor]:
        """Return ``(β_T, β_R, β̄ = β_T · β_R)`` — β̄ scales the full cascade ``G_R H G_T``."""
        beta_T_c = complex(beta_T.item() if isinstance(beta_T, torch.Tensor) else beta_T)
        beta_R_c = complex(beta_R.item() if isinstance(beta_R, torch.Tensor) else beta_R)
        beta_g = torch.tensor(beta_T_c * beta_R_c, dtype=torch.complex64, device=device)
        return beta_T_c, beta_R_c, beta_g

    def run(
        self,
        task: "Task",
        L: int,
        M_int: int,
        A_target: torch.Tensor | np.ndarray,
        H_mimo: torch.Tensor | np.ndarray,
        device: torch.device,
        result_subdir: str,
        strategy_name: str = "Linear",
        seed: int = 42,
        max_iters: int = 3000,
        tx_enc: str = "",
        rx_enc: str = "",
        optimizer: str = "adam",
        scheduler: str = "none",
        warmup_frac: float = 0.05,
        channel_dim: int | None = None,
        wavelength: float = 0.005,
        slayer: float = 0.025,
        dx: float = 0.0025,
        snr: float | None = None,
        n_clusters: int | None = None,
        lr_tx: float = 0.1,
        lr_rx: float = 0.1,
        s_tx: int | None = None,
        s_rx: int | None = None,
        dataset: str = "cifar10",
        save_curve: bool = False,
        **kwargs,
    ) -> None:
        """Run one disjoint 2-SIM experiment.

        Builds separate TX and RX mono-SIM models, optimizes each independently
        using ``compute_targets`` to define per-SIM training objectives, then
        lets ``task`` score the combined link ``β̄ G_R H G_T x + v`` and saves
        the result row.

        Parameters
        ----------
        task : Task
            Downstream task scoring the trained link.
        L : int
            Number of SIM layers per side.
        M_int : int
            Intermediate dimension.
        A_target : Tensor
            Target alignment matrix.
        H_mimo : Tensor
            MIMO channel matrix.
        device : torch.device
            Computation device.
        result_subdir : str
            Subdirectory under ``task.results_dir``.
        strategy_name : str
            Alignment strategy name.
        seed : int
            Random seed.
        max_iters : int
            Optimization budget per SIM.
        snr : float or None
            Signal-to-noise ratio.
        lr_tx, lr_rx : float
            Learning rates for TX and RX.
        """
        ch_dim = self._ch_dim(A_target, channel_dim=channel_dim, **kwargs)
        tipology_subdir = self._tipology_subdir(**kwargs)

        is_slayer = s_tx is not None and s_rx is not None
        fname = make_fname(
            strategy_name,
            seed,
            L,
            M_int,
            ch_dim,
            snr,
            tx_enc,
            rx_enc,
            n_clusters=n_clusters,
            dataset=dataset,
            s_tx=s_tx,
            s_rx=s_rx,
            channel_type=kwargs.get("channel_type", "rayleigh"),
            rice_factor_db=kwargs.get("rice_factor_db", None),
            cluster_on=kwargs.get("cluster_on", "pilots"),
        )
        out_path = self.out_path(task, result_subdir, tipology_subdir, fname)
        if out_path.exists():
            print(f"Skipping {out_path.name}.")
            return

        self.seed_everything(seed)

        targets = self.compute_targets(
            A_target,
            H_mimo,
            ch_dim,
            device,
            strategy_name=strategy_name,
            **kwargs,
        )

        slayer_tx = s_tx * wavelength if is_slayer else slayer
        slayer_rx = s_rx * wavelength if is_slayer else slayer
        tx_model, rx_model = self.build_sims(
            L, M_int, A_target, ch_dim, slayer_tx, slayer_rx, wavelength, dx, device
        )

        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)
        curve: list[dict] = []

        def _link(
            tx: MonoSIMoptimizerTorch, rx: MonoSIMoptimizerTorch, beta_g: torch.Tensor
        ) -> Transmit:
            tx.eval()
            rx.eval()
            with torch.no_grad():
                G_T, G_R = tx.get_cascade(), rx.get_cascade()
            return dual_transmit(G_T, H_torch, G_R, beta_g, snr)

        if save_curve:
            opt_tx = optim.Adam(list(tx_model.xi), lr=lr_tx)
            opt_rx = optim.Adam(list(rx_model.xi), lr=lr_rx)
            tx_loss_history: list[float] = []
            rx_loss_history: list[float] = []
            beta_T_last = torch.tensor(1.0, dtype=torch.complex64, device=device)
            beta_R_last = torch.tensor(1.0, dtype=torch.complex64, device=device)

            def _beta(Z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
                return torch.sum(torch.conj(Z) * A) / (torch.sum(torch.conj(Z) * Z) + 1e-12)

            def _step(model, opt_m, tgt, left, right):
                model.train()
                with torch.no_grad():
                    G = model.get_cascade()
                    Z = left @ G if left is not None else G
                    Z = Z @ right if right is not None else Z
                    beta_k = _beta(Z, tgt).detach()
                opt_m.zero_grad()
                G = model.get_cascade()
                Z = left @ G if left is not None else G
                Z = Z @ right if right is not None else Z
                loss = torch.norm(beta_k * Z - tgt, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
                loss.backward()
                opt_m.step()
                return loss.item(), beta_k

            for _ in range(max_iters):
                t_loss, beta_T_last = _step(
                    tx_model, opt_tx, targets.tx_target, targets.tx_left, targets.tx_right
                )
                r_loss, beta_R_last = _step(
                    rx_model, opt_rx, targets.rx_target, targets.rx_left, targets.rx_right
                )
                tx_loss_history.append(t_loss)
                rx_loss_history.append(r_loss)

                beta_bar = complex(beta_T_last.item()) * complex(beta_R_last.item())
                beta_g = torch.tensor(beta_bar, dtype=torch.complex64, device=device)
                curve.append(task.evaluate_step(_link(tx_model, rx_model, beta_g)))

            tx_loss = tx_loss_history
            rx_loss = rx_loss_history
            beta_T = beta_T_last
            beta_R = beta_R_last
        else:
            tx_loss, beta_T, rx_loss, beta_R = self.optimize_sims(
                tx_model,
                rx_model,
                targets,
                lr_tx,
                lr_rx,
                max_iters,
                optimizer=optimizer,
                scheduler=scheduler,
                warmup_frac=warmup_frac,
            )

        beta_T_c, beta_R_c, beta_g = self.combine_betas(beta_T, beta_R, device)

        sim = task.evaluate(_link(tx_model, rx_model, beta_g))

        print(
            f" Disjoint L={L} M={M_int} snr={snr_tag(snr)} seed={seed}"
            f" | {task.summary(sim)}"
            f" | β_T={abs(beta_T_c):.3e} β_R={abs(beta_R_c):.3e}"
        )

        self.save(
            task,
            out_path,
            head={
                "result_subdir": result_subdir,
                "seed": seed,
                "strategy": strategy_name,
                "M_int": f"{M_int}x{M_int}",
                "L": L,
                "snr": snr,
                "s_tx": s_tx,
                "s_rx": s_rx,
                "tx_enc": tx_enc,
                "rx_enc": rx_enc,
            },
            sim=sim,
            tail={
                "lr_tx": lr_tx,
                "lr_rx": lr_rx,
                "beta_T": abs(beta_T_c),
                "beta_R": abs(beta_R_c),
                "loss": tx_loss[-1] if tx_loss else 0.0,
                "loss_history": tx_loss,
                "rx_loss": rx_loss[-1] if rx_loss else 0.0,
                "rx_loss_history": rx_loss,
                "channel_dim": ch_dim,
                "n_clusters": n_clusters,
                "tipology": tipology_subdir,
                "constructive": kwargs.get("constructive", False),
                "alignment_side": kwargs.get("alignment_side"),
                "dataset": dataset,
                **task.curve_columns(curve if save_curve else None),
            },
            params={
                "phases": {"tx": phase_state(tx_model), "rx": phase_state(rx_model)},
                "G_T": tx_model.get_cascade(),
                "G_R": rx_model.get_cascade(),
                "H": H_torch,
                "beta": beta_g,
                "A_target": torch.as_tensor(A_target),
            },
        )
        del tx_model, rx_model
        gc.collect()
        torch.cuda.empty_cache()

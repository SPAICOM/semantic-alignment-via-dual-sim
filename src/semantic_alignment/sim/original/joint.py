"""Joint Dual-SIM runner — covers layers, SNR, and s-layer sweeps for any downstream task."""

import gc
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from semantic_alignment.sim.link import cascade_transmit, dual_transmit

from ._base import SIMRunner
from ._utils import ch_dims, make_fname, phase_state, snr_tag
from .optimizers.dualsim import DualSIMoptimizer, DualSIMoptimizerTorch

if TYPE_CHECKING:
    from semantic_alignment.evaluation.tasks import Task


class JointSIM(SIMRunner):
    """Joint Dual-SIM runner.

    Optimizes TX and RX phase profiles jointly using alternating Adam steps
    on the end-to-end cascade Z = G_R · H · G_T.
    """

    @staticmethod
    def build_model(
        L: int,
        M_int: int,
        wavelength: float,
        dx: float,
        thickness_tx: float,
        thickness_rx: float,
        device: torch.device,
        tx_in_x: int,
        tx_in_y: int,
        ch_x: int,
        ch_y: int,
        rx_out_x: int,
        rx_out_y: int,
    ) -> DualSIMoptimizerTorch:
        """Instantiate and return a joint Dual-SIM model on the target device."""
        sim_cpu = DualSIMoptimizer(
            num_layers_TX=L,
            num_meta_atoms_TX_in_x=tx_in_x,
            num_meta_atoms_TX_in_y=tx_in_y,
            num_meta_atoms_TX_out_x=ch_x,
            num_meta_atoms_TX_out_y=ch_y,
            num_meta_atoms_TX_int_x=M_int,
            num_meta_atoms_TX_int_y=M_int,
            thickness_TX=thickness_tx,
            num_layers_RX=L,
            num_meta_atoms_RX_in_x=ch_x,
            num_meta_atoms_RX_in_y=ch_y,
            num_meta_atoms_RX_out_x=rx_out_x,
            num_meta_atoms_RX_out_y=rx_out_y,
            num_meta_atoms_RX_int_x=M_int,
            num_meta_atoms_RX_int_y=M_int,
            thickness_RX=thickness_rx,
            wavelength=wavelength,
            spacings={k: dx for k in ("tx_in", "tx_out", "tx_int", "rx_in", "rx_out", "rx_int")},
            verbose=False,
        )
        return DualSIMoptimizerTorch(sim_cpu).to(device)

    @classmethod
    def solve(
        cls,
        L: int,
        M_int: int,
        A_target: torch.Tensor | np.ndarray,
        H_mimo: torch.Tensor | np.ndarray,
        device: torch.device,
        ch_dim: int,
        thickness_tx: float,
        thickness_rx: float,
        wavelength: float = 0.005,
        dx: float = 0.0025,
        max_iters: int = 3000,
        lr_tx: float = 0.1,
        lr_rx: float = 0.2,
        optimizer: str = "adam",
        scheduler: str = "none",
        warmup_frac: float = 0.05,
        clip_value: float | None = None,
        k_T: int = 100,
        k_R: int = 100,
        phase_init: str = "zeros",
        eval_fn: Callable | None = None,
    ) -> tuple[DualSIMoptimizerTorch, torch.Tensor, torch.Tensor, list[float]]:
        """Build and train a joint Dual-SIM against ``A_target``.

        The physical-layer step of :meth:`run`, also usable on its own.
        The caller is responsible for seeding.

        Returns
        -------
        model : DualSIMoptimizerTorch
            Trained joint Dual-SIM.
        Z_eff : torch.Tensor
            Effective cascade ``G_R H G_T``.
        beta : torch.Tensor
            MMSE scalar aligning ``Z_eff`` to ``A_target``.
        loss_history : list[float]
        """
        ch_x, ch_y = ch_dims(ch_dim)
        tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
        rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))

        model = cls.build_model(
            L,
            M_int,
            wavelength,
            dx,
            thickness_tx,
            thickness_rx,
            device,
            tx_in_x,
            tx_in_y,
            ch_x,
            ch_y,
            rx_out_x,
            rx_out_y,
        )
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

        model.init_phases(phase_init)

        loss_history = model.optimize_alternating(
            A_target=A_target,
            H_mimo=H_mimo,
            max_iters=max_iters,
            lr_tx=lr_tx,
            lr_rx=lr_rx,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            clip_value=clip_value,
            k_T=k_T,
            k_R=k_R,
            eval_fn=eval_fn,
        )

        with torch.no_grad():
            Z_f, _ = model.get_effective_cascade(H_torch)
            beta = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)
        return model, Z_f, beta, loss_history

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
        clip_value: float | None = None,
        k_T: int = 100,
        k_R: int = 100,
        channel_dim: int | None = None,
        wavelength: float = 0.005,
        slayer: float = 0.025,
        dx: float = 0.0025,
        snr: float | None = None,
        s_tx: int | None = None,
        s_rx: int | None = None,
        n_clusters: int | None = None,
        lr_tx: float = 0.1,
        lr_rx: float = 0.2,
        dataset: str = "cifar10",
        save_curve: bool = False,
        phase_init: str = "zeros",
        channel_type: str = "rayleigh",
        rice_factor_db: float | None = None,
        cluster_on: str = "pilots",
        **_,
    ) -> None:
        """Run one joint Dual-SIM experiment.

        Builds a dual-SIM model, optimizes TX+RX phases jointly via alternating
        Adam, lets ``task`` score the link ``β G_R H G_T x + v``, and saves the
        result row under ``task.results_dir``.

        Parameters
        ----------
        task : Task
            Downstream task scoring the trained link.
        L : int
            Number of SIM layers (same for TX and RX).
        M_int : int
            Intermediate dimension (meta-atoms per side).
        A_target : np.ndarray or Tensor
            Target alignment matrix.
        H_mimo : np.ndarray or Tensor
            MIMO channel matrix.
        device : torch.device
            Computation device.
        result_subdir : str
            Subdirectory under ``task.results_dir`` for saving.
        strategy_name : str
            Alignment strategy name (for filename).
        seed : int
            Random seed.
        max_iters : int
            Total optimization budget.
        optimizer : str
            ``"adam"`` or ``"sgd"``.
        k_T, k_R : int
            Consecutive TX/RX steps per alternating cycle.
        snr : float or None
            Signal-to-noise ratio.
        s_tx, s_rx : int or None
            S-layer thicknesses (in wavelengths).
        lr_tx, lr_rx : float
            Learning rates for TX and RX.
        save_curve : bool
            If true, score the link after every alternating step.
        """
        ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))

        is_slayer = s_tx is not None and s_rx is not None
        lr_tag = f"_lrtx{lr_tx}_lrrx{lr_rx}" if "lr_search" in result_subdir else ""
        fname = make_fname(
            strategy_name,
            seed,
            L,
            M_int,
            ch_dim,
            snr,
            tx_enc,
            rx_enc,
            s_tx,
            s_rx,
            n_clusters,
            dataset=dataset,
            extra_tag=lr_tag,
            channel_type=channel_type,
            rice_factor_db=rice_factor_db,
            cluster_on=cluster_on,
        )
        log_prefix = (
            f"s_tx={s_tx}λ s_rx={s_rx}λ seed={seed}"
            if is_slayer
            else f"L={L} M={M_int} snr={snr_tag(snr)} seed={seed}"
        )
        out_path = self.out_path(task, result_subdir, "", fname)
        if out_path.exists():
            print(f"Skipping {out_path.name}.")
            return

        self.seed_everything(seed)

        thickness_tx = (s_tx * wavelength * L) if is_slayer else (slayer * L)
        thickness_rx = (s_rx * wavelength * L) if is_slayer else (slayer * L)

        curve: list[dict] = []

        def _eval_fn(Z_eff: torch.Tensor, beta: torch.Tensor) -> None:
            curve.append(task.evaluate_step(cascade_transmit(Z_eff, beta, snr)))

        model, _, beta, loss_history = self.solve(
            L,
            M_int,
            A_target,
            H_mimo,
            device,
            ch_dim,
            thickness_tx,
            thickness_rx,
            wavelength=wavelength,
            dx=dx,
            max_iters=max_iters,
            lr_tx=lr_tx,
            lr_rx=lr_rx,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            clip_value=clip_value,
            k_T=k_T,
            k_R=k_R,
            phase_init=phase_init,
            eval_fn=_eval_fn if save_curve else None,
        )

        model.eval()
        with torch.no_grad():
            G_T, G_R = model.get_TX_cascade(), model.get_RX_cascade()
        sim = task.evaluate(dual_transmit(G_T, H_mimo, G_R, beta, snr))

        print(f" {log_prefix} | {task.summary(sim)}")

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
                "beta_T": None,
                "beta_R": None,
                "loss": loss_history[-1] if loss_history else 0.0,
                "loss_history": loss_history,
                "rx_loss": None,
                "rx_loss_history": None,
                "channel_dim": ch_dim,
                "n_clusters": n_clusters,
                "dataset": dataset,
                **task.curve_columns(curve if save_curve else None),
            },
            params={
                "phases": {"sim": phase_state(model)},
                "G_T": G_T,
                "G_R": G_R,
                "H": torch.as_tensor(H_mimo),
                "beta": beta,
                "A_target": torch.as_tensor(A_target),
            },
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def run_sim(cls, **kwargs) -> None:
        """Entry point for Hydra ``_target_``."""
        cls().run(**kwargs)

"""Mono-SIM runner (TX or RX side) — covers layers and SNR sweeps for any downstream task."""

import gc
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from semantic_alignment.sim.link import Transmit, cascade_transmit, equalized_transmit
from semantic_alignment.utils import channel, cn_matrix, standardize_channel

from ._base import SIMRunner
from ._utils import ch_dims, make_fname, phase_state, tipology_tag
from .optimizers.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch

if TYPE_CHECKING:
    from semantic_alignment.evaluation.tasks import Task


class MonoSIM(SIMRunner):
    """Class-based entry point for the Mono-SIM (TX- or RX-side) experiment."""

    @classmethod
    def build_model(
        cls,
        L: int,
        M_int: int,
        in_x: int,
        in_y: int,
        out_x: int,
        out_y: int,
        side: str,
        slayer: float,
        wavelength: float,
        dx: float,
        device: torch.device,
    ) -> MonoSIMoptimizerTorch:
        """Build and return a MonoSIMoptimizerTorch on the target device."""
        cpu = MonoSIMoptimizer(
            num_layers=L,
            num_meta_atoms_in_x=in_x,
            num_meta_atoms_in_y=in_y,
            num_meta_atoms_out_x=out_x,
            num_meta_atoms_out_y=out_y,
            num_meta_atoms_int_x=M_int,
            num_meta_atoms_int_y=M_int,
            thickness=slayer * L,
            wavelength=wavelength,
            spacings={"in": dx, "out": dx, "int": dx},
            side=side,
        )
        return MonoSIMoptimizerTorch(cpu).to(device)

    @staticmethod
    def draw_channel(
        A_target: torch.Tensor | np.ndarray,
        alignment_side: str,
        device: torch.device,
        channel_type: str = "rayleigh",
        rice_factor_db: float | None = None,
    ) -> torch.Tensor:
        """Draw the square channel the Mono-SIM sees (Rayleigh unless ``channel_type="rician"``).

        TX side: ``H ∈ C^{Rdimc×Rdimc}``; RX side: ``H ∈ C^{Tdimc×Tdimc}``.
        """
        h_dim = int(A_target.shape[1]) if alignment_side == "rx" else int(A_target.shape[0])
        if channel_type == "rician":
            H = channel(h_dim, h_dim, channel_type=channel_type, rice_factor_db=rice_factor_db)
            return standardize_channel(H, channel_type=channel_type).to(device)
        H_mimo = cn_matrix(0, 1, size=(h_dim, h_dim)).to(device)
        return torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    @classmethod
    def solve(
        cls,
        L: int,
        M_int: int,
        A_target: torch.Tensor | np.ndarray,
        H_torch: torch.Tensor,
        device: torch.device,
        slayer: float = 0.025,
        wavelength: float = 0.005,
        dx: float = 0.0025,
        max_iters: int = 3000,
        lr: float = 0.1,
        optimizer: str = "adam",
        scheduler: str = "none",
        warmup_frac: float = 0.05,
        constructive: bool = False,
        alignment_side: str = "tx",
        eval_fn: Callable | None = None,
    ) -> tuple[MonoSIMoptimizerTorch, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
        """Build and train a Mono-SIM against ``A_target``.

        The physical-layer step of :meth:`run`, also usable on its own. The
        caller is responsible for seeding and for drawing ``H_torch``
        (see :meth:`draw_channel`).

        Returns
        -------
        sim_model : MonoSIMoptimizerTorch
            Trained Mono-SIM.
        G_f : torch.Tensor
            Trained SIM cascade (``G_T`` or ``G_R``).
        Z_eff : torch.Tensor
            Effective cascade β is fitted to: ``H G_T`` (tx) / ``G_R H`` (rx)
            when ``constructive``, else ``G_f``.
        beta : torch.Tensor
            MMSE scalar aligning ``Z_eff`` to ``A_target``.
        loss_history : list[float]
        """
        tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
        rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))

        sim_model = cls.build_model(
            L,
            M_int,
            tx_in_x,
            tx_in_y,
            rx_out_x,
            rx_out_y,
            alignment_side,
            slayer,
            wavelength,
            dx,
            device,
        )
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)

        loss_history, _ = sim_model.optimize_sim(
            target_matrix=A_torch,
            max_iters=max_iters,
            lr=lr,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            left_matrix=H_torch if (constructive and alignment_side == "tx") else None,
            right_matrix=H_torch if (constructive and alignment_side == "rx") else None,
            eval_fn=eval_fn,
        )

        with torch.no_grad():
            G_f = sim_model.get_cascade()
            if constructive:
                # TX: H·G_T;  RX: G_R·H
                Z_eff = H_torch @ G_f if alignment_side == "tx" else G_f @ H_torch
            else:
                Z_eff = G_f
            beta = torch.sum(torch.conj(Z_eff) * A_torch) / (
                torch.sum(torch.conj(Z_eff) * Z_eff) + 1e-12
            )
        return sim_model, G_f, Z_eff, beta, loss_history

    def run(
        self,
        task: "Task",
        L: int,
        M_int: int,
        A_target: torch.Tensor | np.ndarray,
        device: torch.device,
        strategy_name: str = "Linear",
        seed: int = 42,
        max_iters: int = 3000,
        lr: float = 0.1,
        snr: float | None = None,
        tx_enc: str = "",
        rx_enc: str = "",
        optimizer: str = "adam",
        scheduler: str = "none",
        warmup_frac: float = 0.05,
        channel_dim: int | None = None,
        wavelength: float = 0.005,
        slayer: float = 0.025,
        dx: float = 0.0025,
        n_clusters: int | None = None,
        constructive: bool = False,
        alignment_side: str = "tx",
        use_mmse: bool = False,
        dataset: str = "cifar10",
        result_subdir: str = "",
        save_curve: bool = False,
        channel_type: str = "rayleigh",
        rice_factor_db: float | None = None,
        cluster_on: str = "pilots",
        **_,
    ) -> None:
        """Run one Mono-SIM job (TX or RX side).

        ``alignment_side='tx'``: SIM on the transmitter, loss = NMSE(H·G_T, A)
        when constructive else NMSE(G_T, A).
        ``alignment_side='rx'``: SIM on the receiver, loss = NMSE(G_R·H, A)
        (only constructive=True is supported for the RX side).

        The channel is drawn here (:meth:`draw_channel`), after seeding; the
        trained link is scored by ``task``.
        """
        if alignment_side == "rx" and not constructive:
            raise NotImplementedError("Mono RX equalized is not implemented.")
        subdir = (
            result_subdir if result_subdir else ("mono_snr" if snr is not None else "mono_layers")
        )
        # Actual channel size (see draw_channel); ``channel_dim`` is ignored by the Mono-SIM.
        ch_dim = int(A_target.shape[1]) if alignment_side == "rx" else int(A_target.shape[0])

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
            channel_type=channel_type,
            rice_factor_db=rice_factor_db,
            cluster_on=cluster_on,
        )
        tipo = tipology_tag("mono", constructive, alignment_side=alignment_side)
        out_path = self.out_path(task, subdir, tipo, fname)

        if out_path.exists():
            print(f"Skipping {out_path.name}.")
            return

        self.seed_everything(seed)

        H_torch = self.draw_channel(A_target, alignment_side, device, channel_type, rice_factor_db)

        def _link(G: torch.Tensor, Z_eff: torch.Tensor, beta: torch.Tensor) -> Transmit:
            # Constructive: Z_eff = H·G_T (tx) or G_R·H (rx) already includes the channel.
            if constructive:
                return cascade_transmit(Z_eff, beta, snr)
            return equalized_transmit(G, H_torch, beta, snr, mmse=use_mmse)

        curve: list[dict] = []

        def _eval_fn(Z_eff: torch.Tensor, beta: torch.Tensor) -> None:
            curve.append(task.evaluate_step(_link(Z_eff, Z_eff, beta)))

        sim_model, G_f, Z_eff, beta, loss_history = self.solve(
            L,
            M_int,
            A_target,
            H_torch,
            device,
            slayer=slayer,
            wavelength=wavelength,
            dx=dx,
            max_iters=max_iters,
            lr=lr,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            constructive=constructive,
            alignment_side=alignment_side,
            eval_fn=_eval_fn if save_curve else None,
        )

        with torch.no_grad():
            sim = task.evaluate(_link(G_f, Z_eff, beta))

        print(
            f" Mono-{alignment_side.upper()}{'(snr=' + str(snr) + ')' if snr is not None else ''}"
            f" L={L} M={M_int} seed={seed} constructive={constructive}"
            f" | {task.summary(sim)}"
        )

        self.save(
            task,
            out_path,
            head={
                "result_subdir": subdir,
                "seed": seed,
                "strategy": strategy_name,
                "M_int": f"{M_int}x{M_int}",
                "L": L,
                "snr": snr,
                "s_tx": None,
                "s_rx": None,
                "tx_enc": tx_enc,
                "rx_enc": rx_enc,
            },
            sim=sim,
            tail={
                "lr": lr,
                "beta_T": None,
                "beta_R": None,
                "loss": loss_history[-1] if loss_history else 0.0,
                "loss_history": loss_history,
                "rx_loss": None,
                "rx_loss_history": None,
                "channel_dim": ch_dim,
                "n_clusters": n_clusters,
                "tipology": tipo,
                "constructive": constructive,
                "alignment_side": alignment_side,
                "dataset": dataset,
                **task.curve_columns(curve if save_curve else None),
            },
            params={
                "phases": {"sim": phase_state(sim_model)},
                "G": G_f,
                "Z_eff": Z_eff,
                "H": H_torch,
                "beta": beta,
                "A_target": torch.as_tensor(A_target),
            },
        )
        del sim_model
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def run_sim(cls, **kwargs) -> None:
        """Entry point for Hydra ``_target_``."""
        cls().run(**kwargs)

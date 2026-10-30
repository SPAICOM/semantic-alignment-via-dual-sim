"""Disjoint 4-SIM runner (full CSI, 2 TX + 2 RX) — covers layers and SNR sweeps."""

import gc
from typing import TYPE_CHECKING

import numpy as np
import torch

from semantic_alignment.sim.link import dual_transmit

from ._base import SIMRunner
from ._utils import (
    ch_dims,
    inter_sim_propagation,
    make_fname,
    phase_state,
    snr_tag,
    tipology_tag,
)
from .optimizers.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch

if TYPE_CHECKING:
    from semantic_alignment.evaluation.tasks import Task


class Disjoint4SIM(SIMRunner):
    """4-SIM disjoint runner (full CSI, 2 TX + 2 RX)."""

    def run(
        self,
        task: "Task",
        L: int,
        M_int: int,
        A_target: torch.Tensor | np.ndarray,
        H_mimo: torch.Tensor | np.ndarray,
        device: torch.device,
        result_subdir: str,
        strategy_name: str = "PPFE",
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
        F_T: torch.Tensor | np.ndarray | None = None,
        F_R_H: torch.Tensor | np.ndarray | None = None,
        lr_tx: float = 0.1,
        lr_rx: float = 0.1,
        epsilon: float = 0.0,
        constructive: bool = False,
        dataset: str = "cifar10",
        n_clusters: int | None = None,
        channel_type: str = "rayleigh",
        rice_factor_db: float | None = None,
        cluster_on: str = "pilots",
        **_,
    ) -> None:
        """Run one 4-SIM disjoint experiment (2 TX + 2 RX).

        Builds four independent mono-SIM models (t1, t2, r2, r1), optimizes each
        with separate training objectives derived from SVD decomposition of H,
        then composes the effective cascades via inter-SIM propagation and lets
        ``task`` score the combined link.

        Parameters
        ----------
        task : Task
            Downstream task scoring the trained link.
        L : int
            Number of SIM layers per mono-SIM.
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
            ``"PPFE"`` or ``"Linear"``.
        seed : int
            Random seed.
        max_iters : int
            Optimization budget per mono-SIM.
        snr : float or None
            Signal-to-noise ratio.
        F_T, F_R_H : Tensor or None
            PPFE frame matrices (required for PPFE strategy).
        lr_tx, lr_rx : float
            Learning rates.
        epsilon : float
            Regularization for SVD pseudo-inverse.
        constructive : bool
            If True, use constructive (full-CSI) targets.
        """
        if strategy_name not in ("PPFE", "Linear"):
            raise NotImplementedError(f"4-SIM disjoint not implemented for '{strategy_name}'.")

        ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
        ch_x, ch_y = ch_dims(ch_dim)
        tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
        rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))

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
        log_prefix = f"Disjoint4SIM L={L} M={M_int} snr={snr_tag(snr)} seed={seed}"
        tipo = tipology_tag("disjoint_4sim", constructive)
        out_path = self.out_path(task, result_subdir, tipo, fname)
        if out_path.exists():
            print(f"Skipping {out_path.name}.")
            return

        self.seed_everything(seed)

        def _make_sim(in_x: int, in_y: int, out_x: int, out_y: int, side: str):
            """Factory closure: build a NumPy MonoSIMoptimizer for one of the 4 SIMs."""
            return MonoSIMoptimizer(
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

        t1_model = MonoSIMoptimizerTorch(_make_sim(tx_in_x, tx_in_y, ch_x, ch_y, "tx")).to(device)
        t2_model = MonoSIMoptimizerTorch(_make_sim(ch_x, ch_y, ch_x, ch_y, "tx")).to(device)
        r2_model = MonoSIMoptimizerTorch(_make_sim(ch_x, ch_y, ch_x, ch_y, "rx")).to(device)
        r1_model = MonoSIMoptimizerTorch(_make_sim(ch_x, ch_y, rx_out_x, rx_out_y, "rx")).to(device)

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

        with torch.no_grad():
            U, s, Vh = torch.linalg.svd(H_torch, full_matrices=False)
            reg_inv_sqrt_s = torch.sqrt(s / (s**2 + epsilon)).to(H_torch.dtype)
            tx_ch_inv = Vh.H @ torch.diag(reg_inv_sqrt_s)
            rx_ch_inv = torch.diag(reg_inv_sqrt_s) @ U.H

            match strategy_name:
                case "PPFE":
                    if F_T is None or F_R_H is None:
                        raise ValueError(
                            "PPFE 4-SIM requires F_T and F_R_H (call ppfe with "
                            "n_clusters=ch_dim and pass the frames to this function)."
                        )
                    tx_target = torch.as_tensor(F_T, dtype=torch.complex64, device=device)
                    rx_target = torch.as_tensor(F_R_H, dtype=torch.complex64, device=device)
                case "Linear":
                    if ch_dim > min(A_torch.shape):
                        raise ValueError(
                            f"Linear 4-SIM requires channel_dim ≤ {min(A_torch.shape)}, got {ch_dim}"
                        )
                    U_A, s_A, Vh_A = torch.linalg.svd(A_torch, full_matrices=False)
                    sqrt_sA = torch.sqrt(s_A[:ch_dim]).to(A_torch.dtype)
                    tx_target = torch.diag(sqrt_sA) @ Vh_A[:ch_dim, :]
                    rx_target = U_A[:, :ch_dim] @ torch.diag(sqrt_sA)

        sim_kw_base = dict(
            optimizer=optimizer, scheduler=scheduler, warmup_frac=warmup_frac, max_iters=max_iters
        )

        I_ch = torch.eye(ch_dim, dtype=torch.complex64, device=device)
        if constructive:
            reg_inv_sqrt_s = torch.sqrt(s / (s**2 + epsilon)).to(H_torch.dtype)
            t2_left = torch.diag(reg_inv_sqrt_s) @ Vh
            r2_right = U @ torch.diag(reg_inv_sqrt_s)
            t2_tgt, r2_tgt = I_ch, I_ch
        else:
            t2_left, r2_right = None, None
            t2_tgt, r2_tgt = tx_ch_inv, rx_ch_inv

        t1_loss, beta_T1 = t1_model.optimize_sim(target_matrix=tx_target, lr=lr_tx, **sim_kw_base)
        t2_loss, beta_T2 = t2_model.optimize_sim(
            target_matrix=t2_tgt, lr=lr_tx, left_matrix=t2_left, **sim_kw_base
        )
        r2_loss, beta_R2 = r2_model.optimize_sim(
            target_matrix=r2_tgt, lr=lr_rx, right_matrix=r2_right, **sim_kw_base
        )
        r1_loss, beta_R1 = r1_model.optimize_sim(target_matrix=rx_target, lr=lr_rx, **sim_kw_base)

        beta_bar = complex(beta_T1) * complex(beta_T2) * complex(beta_R2) * complex(beta_R1)
        beta_g = torch.tensor(beta_bar, dtype=torch.complex64, device=device)

        W_inter = inter_sim_propagation(ch_x, ch_y, slayer, dx, wavelength, device)

        with torch.no_grad():
            G_T_eff = t2_model.get_cascade() @ W_inter @ t1_model.get_cascade()
            G_R_eff = r1_model.get_cascade() @ W_inter @ r2_model.get_cascade()

        sim = task.evaluate(dual_transmit(G_T_eff, H_mimo, G_R_eff, beta_g, snr))

        print(
            f" {log_prefix}"
            f" | {task.summary(sim)}"
            f" | β_T1={abs(beta_T1):.2e} β_T2={abs(beta_T2):.2e}"
            f" β_R2={abs(beta_R2):.2e} β_R1={abs(beta_R1):.2e}"
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
                "s_tx": None,
                "s_rx": None,
                "tx_enc": tx_enc,
                "rx_enc": rx_enc,
            },
            sim=sim,
            tail={
                "lr_tx": lr_tx,
                "lr_rx": lr_rx,
                "beta_T": abs(beta_T1) * abs(beta_T2),
                "beta_R": abs(beta_R2) * abs(beta_R1),
                "loss": t1_loss[-1] if t1_loss else 0.0,
                "loss_history": t1_loss,
                "rx_loss": r1_loss[-1] if r1_loss else 0.0,
                "rx_loss_history": r1_loss,
                "beta_T1": abs(beta_T1),
                "beta_T2": abs(beta_T2),
                "beta_R2": abs(beta_R2),
                "beta_R1": abs(beta_R1),
                "t2_loss": t2_loss[-1] if t2_loss else 0.0,
                "t2_loss_history": t2_loss,
                "r2_loss": r2_loss[-1] if r2_loss else 0.0,
                "r2_loss_history": r2_loss,
                "channel_dim": ch_dim,
                "tipology": tipo,
                "constructive": constructive,
                "dataset": dataset,
            },
            params={
                "phases": {
                    name: phase_state(m)
                    for name, m in (
                        ("t1", t1_model),
                        ("t2", t2_model),
                        ("r2", r2_model),
                        ("r1", r1_model),
                    )
                },
                "G_T": G_T_eff,
                "G_R": G_R_eff,
                "H": H_torch,
                "beta": beta_g,
                "A_target": A_torch,
            },
        )
        del t1_model, t2_model, r2_model, r1_model
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def run_sim(cls, **kwargs) -> None:
        """Entry point for Hydra ``_target_``."""
        cls().run(**kwargs)

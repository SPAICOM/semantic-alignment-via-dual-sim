"""Joint Dual-SIM runner — covers layers, SNR, and s-layer sweeps in one function."""

import gc
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from semantic_alignment.evaluation.inference import run_evaluation

from ._base import SIMRunner
from ._utils import RESULTS_DIR, ch_dims, make_fname, neural_str, pct, save_result, snr_tag
from .optimizers.dualsim import DualSIMoptimizer, DualSIMoptimizerTorch


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

    def run(
        self,
        L: int,
        M_int: int,
        no_mismatch_probe: float,
        original_probe: float,
        A_target: torch.Tensor | np.ndarray,
        H_mimo: torch.Tensor | np.ndarray,
        test_loader: DataLoader,
        clf: torch.Tensor | nn.Module,
        L_in: torch.Tensor,
        mu_in: torch.Tensor,
        L_out: torch.Tensor,
        mu_out: torch.Tensor,
        device: torch.device,
        result_subdir: str,
        strategy_name: str = "Linear",
        seed: int = 42,
        max_iters: int = 3000,
        clf_neural: torch.Tensor | nn.Module | None = None,
        no_mismatch_neural: float | None = None,
        original_neural: float | None = None,
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
        **_,
    ) -> None:
        """Run one joint Dual-SIM experiment.

        Builds a dual-SIM model, optimizes TX+RX phases jointly via alternating
        Adam, evaluates classification accuracy through the optimized cascade,
        and saves results to a Parquet file.

        Parameters
        ----------
        L : int
            Number of SIM layers (same for TX and RX).
        M_int : int
            Intermediate dimension (meta-atoms per side).
        no_mismatch_probe : float
            Baseline accuracy without mismatch (for comparison).
        original_probe : float
            Baseline accuracy with original (unoptimized) SIM.
        A_target : np.ndarray or Tensor
            Target alignment matrix.
        H_mimo : np.ndarray or Tensor
            MIMO channel matrix.
        test_loader : DataLoader
            Test dataset loader.
        clf : classifier
            Probe classifier for accuracy evaluation.
        L_in, mu_in : float
            RX-side equalizer parameters (input L matrix and mu).
        L_out, mu_out : float
            RX-side equalizer parameters (output L matrix and mu).
        device : torch.device
            Computation device.
        result_subdir : str
            Subdirectory under ``results/`` for saving.
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
        """
        ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
        ch_x, ch_y = ch_dims(ch_dim)
        tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
        rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))

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
            s_tx,
            s_rx,
            n_clusters,
            dataset=dataset,
        )
        log_prefix = (
            f"s_tx={s_tx}λ s_rx={s_rx}λ seed={seed}"
            if is_slayer
            else f"L={L} M={M_int} snr={snr_tag(snr)} seed={seed}"
        )
        out_path = RESULTS_DIR / result_subdir / fname
        if out_path.exists():
            print(f"Skipping {out_path.name}.")
            return

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        thickness_tx = (s_tx * wavelength * L) if is_slayer else (slayer * L)
        thickness_rx = (s_rx * wavelength * L) if is_slayer else (slayer * L)

        model = self.build_model(
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
        )

        with torch.no_grad():
            Z_f, _ = model.get_effective_cascade(H_torch)
            beta = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

        sim_probe = run_evaluation(
            model, test_loader, H_mimo, snr, beta, L_in, mu_in, L_out, mu_out, clf, device
        )
        sim_neural = (
            run_evaluation(
                model,
                test_loader,
                H_mimo,
                snr,
                beta,
                L_in,
                mu_in,
                L_out,
                mu_out,
                clf_neural,
                device,
            )
            if clf_neural is not None
            else None
        )

        print(
            f" {log_prefix}"
            f" | nm_linear={no_mismatch_probe * 100:.2f}%{neural_str('nm_neural', no_mismatch_neural)}"
            f" | orig_linear={original_probe * 100:.2f}%{neural_str('orig_neural', original_neural)}"
            f" | sim_linear={sim_probe * 100:.2f}%{neural_str('sim_neural', pct(sim_neural))}"
        )

        save_result(
            out_path,
            {
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
                "no_mismatch_probe": no_mismatch_probe,
                "no_mismatch_neural": no_mismatch_neural,
                "original_probe": original_probe,
                "original_neural": original_neural,
                "sim_accuracy_probe": sim_probe,
                "sim_accuracy_neural": pct(sim_neural),
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
            },
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def run_sim(cls, **kwargs) -> None:
        """Entry point for Hydra ``_target_``."""
        cls().run(**kwargs)

"""Mono-SIM (TX-only) runner — covers layers and SNR sweeps in one function."""

import gc
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from semantic_alignment.evaluation.inference import run_evaluation_cascade, run_evaluation_mono
from semantic_alignment.utils import cn_matrix

from ._base import SIMRunner
from ._utils import RESULTS_DIR, ch_dims, make_fname, neural_str, pct, save_result, tipology_tag
from .optimizers.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch


class MonoSIM(SIMRunner):
    """Class-based entry point for the Mono-SIM (TX-only) experiment."""

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

    def run(
        self,
        L: int,
        M_int: int,
        no_mismatch_probe: float,
        original_probe: float,
        A_target: torch.Tensor | np.ndarray,
        test_loader: DataLoader,
        clf: torch.Tensor | nn.Module,
        L_in: torch.Tensor,
        mu_in: torch.Tensor,
        L_out: torch.Tensor,
        mu_out: torch.Tensor,
        device: torch.device,
        strategy_name: str = "Linear",
        seed: int = 42,
        max_iters: int = 3000,
        lr: float = 0.1,
        H_mimo: torch.Tensor | np.ndarray | None = None,
        snr: float | None = None,
        clf_neural: torch.Tensor | nn.Module | None = None,
        no_mismatch_neural: float | None = None,
        original_neural: float | None = None,
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
        use_mmse: bool = False,
        dataset: str = "cifar10",
        result_subdir: str = "",
        save_curve: bool = False,
        **_,
    ) -> None:
        """Run one Mono-SIM (TX-only) job."""
        subdir = (
            result_subdir if result_subdir else ("mono_snr" if snr is not None else "mono_layers")
        )
        ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
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
        )
        tipo = tipology_tag("mono", constructive)
        out_path = RESULTS_DIR / subdir / tipo / fname if tipo else RESULTS_DIR / subdir / fname

        if out_path.exists():
            print(f"Skipping {out_path.name}.")
            return

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        h_dim = int(A_target.shape[0])
        H_mimo = cn_matrix(0, 1, size=(h_dim, h_dim)).to(device)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

        tx_model = self.build_model(
            L,
            M_int,
            tx_in_x,
            tx_in_y,
            rx_out_x,
            rx_out_y,
            "tx",
            slayer,
            wavelength,
            dx,
            device,
        )
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)

        curve_probe: list[float] = []
        curve_neural: list[float] = []

        def _eval_fn(Z_eff: torch.Tensor, beta: torch.Tensor) -> None:
            if constructive:
                acc = run_evaluation_cascade(
                    Z_eff, test_loader, snr, beta, L_in, mu_in, L_out, mu_out, clf, device
                )
            else:
                acc = run_evaluation_mono(
                    G_T=Z_eff,
                    dataloader=test_loader,
                    H_mimo=H_torch,
                    snr_db=snr,
                    beta_opt=beta,
                    L_in=L_in,
                    mu_in=mu_in,
                    L_out=L_out,
                    mu_out=mu_out,
                    clf=clf,
                    device=device,
                    use_equalizer=True,
                    mmse=use_mmse,
                )
            curve_probe.append(acc)
            if clf_neural is not None:
                if constructive:
                    acc_n = run_evaluation_cascade(
                        Z_eff,
                        test_loader,
                        snr,
                        beta,
                        L_in,
                        mu_in,
                        L_out,
                        mu_out,
                        clf_neural,
                        device,
                    )
                else:
                    acc_n = run_evaluation_mono(
                        G_T=Z_eff,
                        dataloader=test_loader,
                        H_mimo=H_torch,
                        snr_db=snr,
                        beta_opt=beta,
                        L_in=L_in,
                        mu_in=mu_in,
                        L_out=L_out,
                        mu_out=mu_out,
                        clf=clf_neural,
                        device=device,
                        use_equalizer=True,
                        mmse=use_mmse,
                    )
                curve_neural.append(acc_n)

        loss_history, _ = tx_model.optimize_sim(
            target_matrix=A_torch,
            max_iters=max_iters,
            lr=lr,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            left_matrix=H_torch if constructive else None,
            eval_fn=_eval_fn if save_curve else None,
        )

        with torch.no_grad():
            G_T_f = tx_model.get_cascade()
            G_eff = H_torch @ G_T_f if constructive else G_T_f
            beta = torch.sum(torch.conj(G_eff) * A_torch) / (
                torch.sum(torch.conj(G_eff) * G_eff) + 1e-12
            )

            eval_kw = dict(
                G_T=G_T_f,
                dataloader=test_loader,
                H_mimo=H_torch,
                snr_db=snr,
                beta_opt=beta,
                L_in=L_in,
                mu_in=mu_in,
                L_out=L_out,
                mu_out=mu_out,
                device=device,
                use_equalizer=not constructive,
                mmse=use_mmse,
            )
            mono_probe = run_evaluation_mono(clf=clf, **eval_kw)
            mono_neural = (
                run_evaluation_mono(clf=clf_neural, **eval_kw) if clf_neural is not None else None
            )

        print(
            f" Mono{'(snr=' + str(snr) + ')' if snr is not None else ''} L={L} M={M_int} seed={seed}"
            f" constructive={constructive}"
            f" | nm_linear={no_mismatch_probe * 100:.2f}%{neural_str('nm_neural', no_mismatch_neural)}"
            f" | orig_linear={original_probe * 100:.2f}%{neural_str('orig_neural', original_neural)}"
            f" | sim_linear={mono_probe * 100:.2f}%{neural_str('sim_neural', pct(mono_neural))}"
        )

        save_result(
            out_path,
            {
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
                "no_mismatch_probe": no_mismatch_probe,
                "no_mismatch_neural": no_mismatch_neural,
                "original_probe": original_probe,
                "original_neural": original_neural,
                "sim_accuracy_probe": mono_probe,
                "sim_accuracy_neural": pct(mono_neural),
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
                "dataset": dataset,
                "sim_accuracy_probe_curve": curve_probe if save_curve else None,
                "sim_accuracy_neural_curve": curve_neural if save_curve else None,
            },
        )
        del tx_model
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def run_sim(cls, **kwargs) -> None:
        """Entry point for Hydra ``_target_``."""
        cls().run(**kwargs)

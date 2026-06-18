"""Core 2-SIM disjoint runner — receives pre-computed TX/RX targets.

This module is not called directly from YAML configs; instead it is invoked
by :func:`~semantic_alignment.experiments.disjoint_asym.run_disjoint_asym`
and :func:`~semantic_alignment.experiments.disjoint_sym.run_disjoint_sym`
after they have computed the strategy-specific TX/RX target matrices.

The training procedure trains TX-SIM and RX-SIM independently (each with
:meth:`MonoSIMoptimizerTorch.optimize_sim`), then evaluates the combined
cascade ``G_R H G_T`` with :func:`run_evaluation_disjoint_fixed`.
"""

import gc
import random

import numpy as np
import torch

from semantic_alignment.evaluation.inference import run_evaluation_disjoint_fixed
from semantic_alignment.sim.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch

from ._utils import RESULTS_DIR, ch_dims, make_fname, neural_str, pct, save_result, snr_tag


def run_disjoint_2sim(
    target_tx: torch.Tensor,
    target_rx: torch.Tensor,
    L: int,
    M_int: int,
    A_target,
    H_mimo,
    test_loader,
    clf,
    L_in,
    mu_in,
    L_out,
    mu_out,
    device,
    result_subdir: str,
    strategy_name: str = "Linear",
    seed: int = 42,
    max_iters: int = 3000,
    lr: float = 0.1,
    clf_neural=None,
    no_mismatch_neural: float | None = None,
    no_mismatch_probe: float = 0.0,
    original_neural: float | None = None,
    original_probe: float = 0.0,
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
    **_,
) -> None:
    """Run one disjoint 2-SIM job given pre-computed TX and RX target matrices.

    TX-SIM is optimised to emulate ``target_tx``, RX-SIM to emulate
    ``target_rx``, each independently.  At inference ``β̄ = β_T · β_R`` is
    applied to the full cascade ``G_R H G_T x``.

    Results are written to ``results/{result_subdir}/`` as a single-row
    Parquet file.  The job is skipped if the output file already exists.

    Parameters
    ----------
    target_tx : torch.Tensor
        Pre-computed target matrix for the TX-SIM.
    target_rx : torch.Tensor
        Pre-computed target matrix for the RX-SIM.
    L : int
        Number of metasurface layers per SIM.
    M_int : int
        Side length of the square intermediate aperture.
    A_target : array-like
        Full alignment matrix (used only to determine aperture sizes and for
        the result row).
    H_mimo : array-like
        MIMO channel matrix.
    test_loader : DataLoader
        Test set yielding ``(z_tx, labels)`` pairs.
    clf :
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics.
    device : torch.device
        Inference and optimization device.
    result_subdir : str
        Subdirectory under ``results/`` where the Parquet file is written.
    strategy_name : str, optional
        Alignment strategy label.
    seed : int, optional
        Global random seed.
    max_iters : int, optional
        Optimization iterations per SIM.
    lr : float, optional
        Learning rate.
    clf_neural : optional
        Optional neural classifier.
    no_mismatch_neural, no_mismatch_probe : float or None, optional
        No-mismatch baselines.
    original_neural, original_probe : float or None, optional
        Oracle linear-alignment baselines.
    tx_enc, rx_enc : str, optional
        Encoder names embedded in the output filename.
    optimizer : str, optional
        ``"adam"`` or ``"sgd"``.
    scheduler : str, optional
        ``"none"`` or ``"cosine"``.
    warmup_frac : float, optional
        Cosine-scheduler warm-up fraction.
    channel_dim : int or None, optional
        Physical channel dimension override.
    wavelength : float, optional
        Operating wavelength λ (metres).
    slayer : float, optional
        Default inter-layer spacing.
    dx : float, optional
        Meta-atom spacing (metres).
    snr : float or None, optional
        AWGN SNR in dB; ``None`` means noiseless.
    """
    ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
    ch_x, ch_y = ch_dims(ch_dim)
    tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
    rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))

    fname = make_fname(strategy_name, seed, L, M_int, ch_dim, snr, tx_enc, rx_enc)
    out_path = RESULTS_DIR / result_subdir / fname
    if out_path.exists():
        print(f"Skipping {out_path.name}.")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=tx_in_x,
        num_meta_atoms_in_y=tx_in_y,
        num_meta_atoms_out_x=ch_x,
        num_meta_atoms_out_y=ch_y,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
        side="tx",
    )
    rx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=ch_x,
        num_meta_atoms_in_y=ch_y,
        num_meta_atoms_out_x=rx_out_x,
        num_meta_atoms_out_y=rx_out_y,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
        side="rx",
    )
    tx_model = MonoSIMoptimizerTorch(tx_cpu).to(device)
    rx_model = MonoSIMoptimizerTorch(rx_cpu).to(device)

    sim_kw = dict(
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )
    tx_loss_history, beta_T = tx_model.optimize_sim(target_matrix=target_tx, **sim_kw)
    rx_loss_history, beta_R = rx_model.optimize_sim(target_matrix=target_rx, **sim_kw)

    beta_bar = complex(beta_T) * complex(beta_R)
    beta_g = torch.tensor(beta_bar, dtype=torch.complex64, device=device)

    sim_probe = run_evaluation_disjoint_fixed(
        tx_model,
        rx_model,
        test_loader,
        H_mimo,
        snr,
        beta_g,
        L_in,
        mu_in,
        L_out,
        mu_out,
        clf,
        device,
    )
    sim_neural = (
        run_evaluation_disjoint_fixed(
            tx_model,
            rx_model,
            test_loader,
            H_mimo,
            snr,
            beta_g,
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
        f" Disjoint L={L} M={M_int} snr={snr_tag(snr)} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{neural_str('sim_neural', pct(sim_neural))}"
        f" | β_T={abs(beta_T):.3e} β_R={abs(beta_R):.3e}"
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
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": pct(sim_neural),
            "lr": lr,
            "beta_T": abs(beta_T),
            "beta_R": abs(beta_R),
            "loss": tx_loss_history[-1] if tx_loss_history else 0.0,
            "loss_history": tx_loss_history,
            "rx_loss": rx_loss_history[-1] if rx_loss_history else 0.0,
            "rx_loss_history": rx_loss_history,
            "channel_dim": ch_dim,
        },
    )
    del tx_model, rx_model
    gc.collect()
    torch.cuda.empty_cache()

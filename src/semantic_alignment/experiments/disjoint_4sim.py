"""Disjoint 4-SIM runner (full CSI, 2 TX + 2 RX) — covers layers and SNR sweeps."""

import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__" and __package__ is None:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    __package__ = "semantic_alignment.experiments"

import gc
import random

import numpy as np
import torch

from semantic_alignment.evaluation.inference import run_evaluation_disjoint_fixed
from semantic_alignment.sim.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch

from ._utils import (
    RESULTS_DIR,
    _FixedCascade,
    ch_dims,
    inter_sim_propagation,
    make_fname,
    neural_str,
    pct,
    save_result,
    snr_tag,
)


def run_disjoint_4sim(
    L: int,
    M_int: int,
    no_mismatch_probe: float,
    original_probe: float,
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
    strategy_name: str = "PPFE",
    seed: int = 42,
    max_iters: int = 3000,
    clf_neural=None,
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
    snr: float | None = None,
    F_T=None,
    F_R_H=None,
    lr_tx: float = 0.1,
    lr_rx: float = 0.1,
    **_,
) -> None:
    """Run one disjoint 4-SIM (full CSI) job.

    Four independent SIMs are trained disjointly and chained at inference:

        G_T1 (D_tx→ch) → W → G_T2 (ch→ch) → H → G_R2 (ch→ch) → W → G_R1 (ch→D_rx)

    where W is the free-space propagation matrix between co-located pairs.
    G_T2 emulates the SVD-based channel-inversion factor B_T; G_R2 emulates
    B_R; G_T1 and G_R1 emulate the strategy-specific TX/RX targets.

    Sweep mode is selected by ``snr`` (``None`` → layers sweep, finite → SNR
    sweep).  Supports ``strategy_name`` in ``("PPFE", "Linear")``.
    Results are written to ``results/{result_subdir}/`` as a single-row
    Parquet file.  The job is skipped if the output file already exists.

    Parameters
    ----------
    L : int
        Number of metasurface layers per SIM (hardware equivalent to 2L in
        the 2-SIM case, since there are two SIMs per side).
    M_int : int
        Side length of the square intermediate aperture.
    no_mismatch_probe : float
        No-mismatch linear-probe accuracy, in percent.
    original_probe : float
        Oracle linear-alignment accuracy (no SIM, no channel), in percent.
    A_target : array-like
        Alignment matrix of shape ``(D_rx/2, D_tx/2)``.
    H_mimo : array-like
        MIMO channel matrix.
    test_loader : DataLoader
        Test set yielding ``(z_tx, labels)`` pairs.
    clf :
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics (Cholesky factor and mean for TX and RX).
    device : torch.device
        Inference and optimization device.
    result_subdir : str
        Subdirectory under ``results/`` where the Parquet file is written.
    strategy_name : str, optional
        ``"PPFE"`` or ``"Linear"``.
    seed : int, optional
        Global random seed.
    max_iters : int, optional
        Optimization iterations per SIM.
    lr_tx : float, optional
        TX-SIM learning rate.
    lr_rx : float, optional
        RX-SIM learning rate.
    clf_neural : optional
        Optional neural classifier.
    no_mismatch_neural, original_neural : float or None, optional
        Neural-classifier baselines; ``None`` when not available.
    tx_enc, rx_enc : str, optional
        Encoder names embedded in the output filename.
    optimizer : str, optional
        ``"adam"`` or ``"sgd"``.
    scheduler : str, optional
        ``"none"`` or ``"cosine"``.
    warmup_frac : float, optional
        Cosine-scheduler warm-up fraction.
    channel_dim : int or None, optional
        Physical channel dimension override; defaults to ``max(A_target.shape)``.
    wavelength : float, optional
        Operating wavelength λ (metres).
    slayer : float, optional
        Default inter-layer spacing in wavelengths.
    dx : float, optional
        Meta-atom spacing (metres).
    snr : float or None, optional
        AWGN SNR in dB; ``None`` means noiseless.
    F_T : array-like or None, optional
        PPFE TX synthesis frame (required when ``strategy_name="PPFE"``).
    F_R_H : array-like or None, optional
        PPFE RX analysis frame (required when ``strategy_name="PPFE"``).
    """
    if strategy_name not in ("PPFE", "Linear"):
        raise NotImplementedError(f"4-SIM disjoint not implemented for '{strategy_name}'.")

    ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
    ch_x, ch_y = ch_dims(ch_dim)
    tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
    rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))

    fname = make_fname(strategy_name, seed, L, M_int, ch_dim, snr, tx_enc, rx_enc)
    log_prefix = f"Disjoint4SIM L={L} M={M_int} snr={snr_tag(snr)} seed={seed}"
    out_path = RESULTS_DIR / result_subdir / fname
    if out_path.exists():
        print(f"Skipping {out_path.name}.")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def _make_sim(in_x, in_y, out_x, out_y, side):
        """Build a MonoSIMoptimizer for the given input/output aperture and side."""
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
        inv_sqrt_s = (1.0 / torch.sqrt(s)).to(H_torch.dtype)
        B_T = Vh.H @ torch.diag(inv_sqrt_s)
        B_R = torch.diag(inv_sqrt_s) @ U.H
        I_approx = B_R @ H_torch @ B_T
        I_ref = torch.eye(H_torch.shape[0], dtype=H_torch.dtype, device=H_torch.device)
        assert torch.allclose(I_approx, I_ref, atol=1e-4), (
            f"B_R @ H @ B_T ≉ I, max err={(I_approx - I_ref).abs().max():.2e}"
        )

        match strategy_name:
            case "PPFE":
                if F_T is None or F_R_H is None:
                    raise ValueError(
                        "PPFE 4-SIM requires F_T and F_R_H (call ppfe with "
                        "n_clusters=ch_dim and pass the frames to this function)."
                    )
                T_T1 = torch.as_tensor(F_T, dtype=torch.complex64, device=device)
                T_R1 = torch.as_tensor(F_R_H, dtype=torch.complex64, device=device)
            case "Linear":
                if ch_dim > min(A_torch.shape):
                    raise ValueError(
                        f"Linear 4-SIM requires channel_dim ≤ {min(A_torch.shape)}, got {ch_dim}"
                    )
                U_A, s_A, Vh_A = torch.linalg.svd(A_torch, full_matrices=False)
                sqrt_sA = torch.sqrt(s_A[:ch_dim]).to(A_torch.dtype)
                A_T = torch.diag(sqrt_sA) @ Vh_A[:ch_dim, :]
                A_R = U_A[:, :ch_dim] @ torch.diag(sqrt_sA)
                T_T1, T_R1 = A_T, A_R

    sim_kw_base = dict(
        optimizer=optimizer, scheduler=scheduler, warmup_frac=warmup_frac, max_iters=max_iters
    )
    t1_loss, beta_T1 = t1_model.optimize_sim(target_matrix=T_T1, lr=lr_tx, **sim_kw_base)
    t2_loss, beta_T2 = t2_model.optimize_sim(target_matrix=B_T, lr=lr_tx, **sim_kw_base)
    r2_loss, beta_R2 = r2_model.optimize_sim(target_matrix=B_R, lr=lr_rx, **sim_kw_base)
    r1_loss, beta_R1 = r1_model.optimize_sim(target_matrix=T_R1, lr=lr_rx, **sim_kw_base)

    beta_bar = complex(beta_T1) * complex(beta_T2) * complex(beta_R2) * complex(beta_R1)
    beta_g = torch.tensor(beta_bar, dtype=torch.complex64, device=device)

    W_inter = inter_sim_propagation(ch_x, ch_y, slayer, dx, wavelength, device)

    with torch.no_grad():
        G_T_eff = t2_model.get_cascade() @ W_inter @ t1_model.get_cascade()
        G_R_eff = r1_model.get_cascade() @ W_inter @ r2_model.get_cascade()

    tx_wrap = _FixedCascade(G_T_eff)
    rx_wrap = _FixedCascade(G_R_eff)

    sim_probe = run_evaluation_disjoint_fixed(
        tx_wrap,
        rx_wrap,
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
            tx_wrap,
            rx_wrap,
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
        f" {log_prefix}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{neural_str('sim_neural', pct(sim_neural))}"
        f" | β_T1={abs(beta_T1):.2e} β_T2={abs(beta_T2):.2e}"
        f" β_R2={abs(beta_R2):.2e} β_R1={abs(beta_R1):.2e}"
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
        },
    )
    del t1_model, t2_model, r2_model, r1_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("run_disjoint_4sim: disjoint 4-SIM (full CSI, 2 TX + 2 RX) runner.")
    print("Invoke via: just run-disjoint-4sim")
    print(
        "         or: uv run python scripts/classification.py --config-name config_disjoint_4sim_layers"
    )

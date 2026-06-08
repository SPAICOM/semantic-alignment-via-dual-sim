"""
Single-configuration experiment runners for Dual-SIM and Mono-SIM.

Each function handles ONE (seed, L, M_int, snr, …) combination and writes
its result to a dedicated Parquet file inside a per-experiment subdirectory:

    results/
        dual_layers/    {strategy}_seed{seed}_L{L}_M{M}x{M}.parquet
        dual_snr/       {strategy}_seed{seed}_M{M}x{M}_snr{snr}.parquet
        dual_slayer/    {strategy}_seed{seed}_stx{s_tx}_srx{s_rx}.parquet
        mono/           {strategy}_seed{seed}_L{L}_M{M}x{M}.parquet
        mono_channel/   {strategy}_seed{seed}_L{L}_M{M}x{M}_snr{snr}.parquet
        disjoint_layers/{strategy}_seed{seed}_L{L}_M{M}x{M}.parquet
        disjoint_snr/   {strategy}_seed{seed}_M{M}x{M}_snr{snr}.parquet
        rx_depth/       seed{seed}_LTX{L_TX}_LRX{L_RX}_M{M}x{M}.parquet

If the target file already exists the run is skipped (idempotent sweeps).
"""

import gc
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch

from semantic_alignment.evaluation.inference import (
    run_evaluation,
    run_evaluation_disjoint_fixed,
    run_evaluation_mono_channel,
)
from semantic_alignment.evaluation.linear_probe import clf_predict
from semantic_alignment.sim.dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from semantic_alignment.sim.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch
from semantic_alignment.utils.utils import (
    a_inv_times_b,
    complex_compressed_tensor,
    decompress_complex_tensor,
    mmse_svd_equalizer,
)

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _save_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_parquet(path)


def _snr_tag(snr: float | None) -> str:
    return "inf" if snr is None else str(snr)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _build_joint_sim(L, M_int, wavelength, slayer, dx, device):
    sim_cpu = DualSIMoptimizer(
        num_layers_TX=L,
        num_meta_atoms_TX_in_x=16,
        num_meta_atoms_TX_in_y=12,
        num_meta_atoms_TX_out_x=24,
        num_meta_atoms_TX_out_y=16,
        num_meta_atoms_TX_int_x=M_int,
        num_meta_atoms_TX_int_y=M_int,
        thickness_TX=slayer * L,
        num_layers_RX=L,
        num_meta_atoms_RX_in_x=24,
        num_meta_atoms_RX_in_y=16,
        num_meta_atoms_RX_out_x=24,
        num_meta_atoms_RX_out_y=16,
        num_meta_atoms_RX_int_x=M_int,
        num_meta_atoms_RX_int_y=M_int,
        thickness_RX=slayer * L,
        wavelength=wavelength,
        spacings={k: dx for k in ("tx_in", "tx_out", "tx_int", "rx_in", "rx_out", "rx_int")},
        verbose=False,
    )
    return DualSIMoptimizerTorch(sim_cpu).to(device)


def _eval_mono(G_T, beta, test_loader, L_in, mu_in, L_out, mu_out, clf, device):
    """Evaluate TX-only SIM (no channel, no noise) with a single classifier."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_real, labels in test_loader:
            x_c = complex_compressed_tensor(x_real.T.to(device), device=device)
            x_w = a_inv_times_b(L_in, x_c - mu_in).to(G_T.dtype)
            z = (L_out.to(G_T.dtype) @ (beta * (G_T @ x_w))) + mu_out.to(G_T.dtype)
            y_r = decompress_complex_tensor(z, device=device).T
            all_preds.extend(clf_predict(clf, y_r).cpu().numpy())
            all_labels.extend(labels.numpy())
    return (np.array(all_preds) == np.array(all_labels)).mean()


def _pct(v: float | None) -> float | None:
    return v * 100 if v is not None else None


def _neural_str(label: str, acc_pct: float | None) -> str:
    return f" {label}={acc_pct:.2f}%" if acc_pct is not None else ""


# ---------------------------------------------------------------------------
# Joint Dual-SIM
# ---------------------------------------------------------------------------


def run_single_layers_config(
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
    strategy_name: str = "Linear",
    seed: int = 42,
    max_iters: int = 3000,
    lr: float = 0.1,
    clf_neural=None,
    no_mismatch_neural: float | None = None,
    original_neural: float | None = None,
    tx_enc: str = "",
    rx_enc: str = "",
) -> None:
    """Layers experiment — one (L, M_int, seed) job."""
    out_path = (
        RESULTS_DIR
        / "dual_layers"
        / f"{strategy_name}_seed{seed}_L{L}_M{M_int}x{M_int}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping L={L}, M_int={M_int}, seed={seed}.")
        return

    wavelength = 0.005
    slayer = 5 * wavelength
    dx = wavelength / 2
    actual_lr = 0.1 / (L**1.3)  # scale lr by depth instead of using the config value

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = _build_joint_sim(L, M_int, wavelength, slayer, dx, device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    loss_history = model.optimize_alternating(
        A_target=A_target,
        H_mimo=H_mimo,
        max_iters=max_iters,
        lr=actual_lr,
    )
    with torch.no_grad():
        Z_f, _ = model.get_effective_cascade(H_torch)
        beta = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

    sim_probe = run_evaluation(
        model, test_loader, H_mimo, None, beta, L_in, mu_in, L_out, mu_out, clf, device
    )
    sim_neural = (
        run_evaluation(
            model, test_loader, H_mimo, None, beta, L_in, mu_in, L_out, mu_out, clf_neural, device
        )
        if clf_neural is not None
        else None
    )

    print(
        f" L={L} M={M_int} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{_neural_str('sim_neural', _pct(sim_neural))}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "loss": loss_history[-1] if loss_history else 0.0,
            "lr": actual_lr,
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_single_snr_config(
    L: int,
    M_int: int,
    snr: float | None,
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
    strategy_name: str = "Linear",
    seed: int = 42,
    max_iters: int = 3000,
    lr: float = 0.1,
    clf_neural=None,
    no_mismatch_neural: float | None = None,
    original_neural: float | None = None,
    tx_enc: str = "",
    rx_enc: str = "",
) -> None:
    """SNR experiment — one (M_int, snr, seed) job."""
    out_path = (
        RESULTS_DIR
        / "dual_snr"
        / f"{strategy_name}_seed{seed}_M{M_int}x{M_int}_snr{_snr_tag(snr)}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping M_int={M_int}, snr={snr}, seed={seed}.")
        return

    wavelength = 0.005
    slayer = 5 * wavelength
    dx = wavelength / 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = _build_joint_sim(L, M_int, wavelength, slayer, dx, device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    loss_history = model.optimize_alternating(
        A_target=A_target,
        H_mimo=H_mimo,
        max_iters=max_iters,
        lr=lr,
    )
    with torch.no_grad():
        Z_f, _ = model.get_effective_cascade(H_torch)
        beta = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

    sim_probe = run_evaluation(
        model, test_loader, H_mimo, snr, beta, L_in, mu_in, L_out, mu_out, clf, device
    )
    sim_neural = (
        run_evaluation(
            model, test_loader, H_mimo, snr, beta, L_in, mu_in, L_out, mu_out, clf_neural, device
        )
        if clf_neural is not None
        else None
    )

    print(
        f" M={M_int} snr={snr} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{_neural_str('sim_neural', _pct(sim_neural))}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "snr": snr,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "loss": loss_history[-1] if loss_history else 0.0,
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_single_slayer_config(
    L: int,
    M_int: int,
    s_tx: int,
    s_rx: int,
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
    strategy_name: str = "PPFE",
    seed: int = 42,
    max_iters: int = 3000,
    lr: float = 0.1,
    clf_neural=None,
    no_mismatch_neural: float | None = None,
    original_neural: float | None = None,
    tx_enc: str = "",
    rx_enc: str = "",
) -> None:
    """Layer-spacing experiment — one (s_tx, s_rx, seed) job."""
    out_path = (
        RESULTS_DIR
        / "dual_slayer"
        / f"{strategy_name}_seed{seed}_stx{s_tx}_srx{s_rx}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping s_tx={s_tx}, s_rx={s_rx}, seed={seed}.")
        return

    wavelength = 0.005
    dx = wavelength / 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    sim_cpu = DualSIMoptimizer(
        num_layers_TX=L,
        num_meta_atoms_TX_in_x=16,
        num_meta_atoms_TX_in_y=12,
        num_meta_atoms_TX_out_x=24,
        num_meta_atoms_TX_out_y=16,
        num_meta_atoms_TX_int_x=M_int,
        num_meta_atoms_TX_int_y=M_int,
        thickness_TX=(s_tx * wavelength) * L,
        num_layers_RX=L,
        num_meta_atoms_RX_in_x=24,
        num_meta_atoms_RX_in_y=16,
        num_meta_atoms_RX_out_x=24,
        num_meta_atoms_RX_out_y=16,
        num_meta_atoms_RX_int_x=M_int,
        num_meta_atoms_RX_int_y=M_int,
        thickness_RX=(s_rx * wavelength) * L,
        wavelength=wavelength,
        spacings={k: dx for k in ("tx_in", "tx_out", "tx_int", "rx_in", "rx_out", "rx_int")},
        verbose=False,
    )
    model = DualSIMoptimizerTorch(sim_cpu).to(device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    model.optimize_alternating(A_target=A_target, H_mimo=H_mimo, max_iters=max_iters, lr=lr)

    with torch.no_grad():
        Z_f, _ = model.get_effective_cascade(H_torch)
        beta = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

    sim_probe = run_evaluation(
        model, test_loader, H_mimo, None, beta, L_in, mu_in, L_out, mu_out, clf, device
    )
    sim_neural = (
        run_evaluation(
            model, test_loader, H_mimo, None, beta, L_in, mu_in, L_out, mu_out, clf_neural, device
        )
        if clf_neural is not None
        else None
    )

    print(
        f" s_tx={s_tx}λ s_rx={s_rx}λ seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{_neural_str('sim_neural', _pct(sim_neural))}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "s_tx": s_tx,
            "s_rx": s_rx,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
        },
    )
    del model, sim_cpu
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Mono-SIM (TX-only ablation)
# ---------------------------------------------------------------------------


def run_single_mono_layers_config(
    L: int,
    M_int: int,
    no_mismatch_probe: float,
    original_probe: float,
    A_target,
    test_loader,
    clf,
    L_in,
    mu_in,
    L_out,
    mu_out,
    device,
    strategy_name: str = "Linear",
    seed: int = 42,
    max_iters: int = 3000,
    lr: float = 0.1,
    use_channel: bool = False,
    H_mimo=None,
    snr: float | None = None,
    clf_neural=None,
    no_mismatch_neural: float | None = None,
    original_neural: float | None = None,
    tx_enc: str = "",
    rx_enc: str = "",
) -> None:
    """Mono-SIM ablation — TX-only SIM with optional MIMO channel.

    Without channel (use_channel=False): y = β G_T x_w
    With channel    (use_channel=True):  y = G_eq(H) · (H · β G_T · x_w + noise)
    """
    subdir = "mono_channel" if use_channel else "mono"
    filename = f"{strategy_name}_seed{seed}_L{L}_M{M_int}x{M_int}"
    if use_channel:
        filename += f"_snr{_snr_tag(snr)}"
    filename += f"_{tx_enc}_{rx_enc}"
    out_path = RESULTS_DIR / subdir / f"{filename}.parquet"

    if out_path.exists():
        print(
            f"Skipping mono{'_channel' if use_channel else ''} L={L}, M_int={M_int}, seed={seed}."
        )
        return

    wavelength = 0.005
    slayer = 5 * wavelength
    dx = wavelength / 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=16,
        num_meta_atoms_in_y=12,
        num_meta_atoms_out_x=24,
        num_meta_atoms_out_y=16,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
        spacings={"in": dx, "out": dx, "int": dx},
    )
    tx_model = MonoSIMoptimizerTorch(tx_cpu).to(device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)

    loss_history, _ = tx_model.optimize_sim(target_matrix=A_torch, max_iters=max_iters, lr=lr)

    with torch.no_grad():
        G_T_f = tx_model.get_cascade()
        if use_channel:
            H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)
            G_mmse, _ = mmse_svd_equalizer(H_torch, snr)
            G_eff = G_mmse @ H_torch @ G_T_f
            beta = torch.sum(torch.conj(G_eff) * A_torch) / (
                torch.sum(torch.conj(G_eff) * G_eff) + 1e-12
            )
            mono_probe = run_evaluation_mono_channel(
                G_T_f,
                test_loader,
                H_torch,
                snr,
                beta,
                L_in,
                mu_in,
                L_out,
                mu_out,
                clf,
                device,
            )
            mono_neural = (
                run_evaluation_mono_channel(
                    G_T_f,
                    test_loader,
                    H_torch,
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
        else:
            beta = torch.sum(torch.conj(G_T_f) * A_torch) / (
                torch.sum(torch.conj(G_T_f) * G_T_f) + 1e-12
            )
            mono_probe = _eval_mono(
                G_T_f, beta, test_loader, L_in, mu_in, L_out, mu_out, clf, device
            )
            mono_neural = (
                _eval_mono(G_T_f, beta, test_loader, L_in, mu_in, L_out, mu_out, clf_neural, device)
                if clf_neural is not None
                else None
            )

    print(
        f" Mono{'(+H)' if use_channel else ''} L={L} M={M_int} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={mono_probe * 100:.2f}%{_neural_str('sim_neural', _pct(mono_neural))}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "snr": snr,
            "use_channel": use_channel,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": mono_probe * 100,
            "sim_accuracy_neural": _pct(mono_neural),
            "loss": loss_history[-1] if loss_history else 0.0,
            "lr": lr,
        },
    )
    del tx_model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Disjoint Dual-SIM
# ---------------------------------------------------------------------------


def run_single_disjoint_layers(
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
    strategy_name: str = "Linear",
    seed: int = 42,
    max_iters: int = 5000,
    lr: float = 0.1,
    clf_neural=None,
    no_mismatch_neural: float | None = None,
    original_neural: float | None = None,
    tx_enc: str = "",
    rx_enc: str = "",
) -> None:
    """Disjoint layers experiment — one (L, M_int, seed) job."""
    out_path = (
        RESULTS_DIR
        / "disjoint_layers"
        / f"{strategy_name}_seed{seed}_L{L}_M{M_int}x{M_int}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping disjoint L={L}, M_int={M_int}, seed={seed}.")
        return

    wavelength = 0.005
    slayer = 5 * wavelength

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=16,
        num_meta_atoms_in_y=12,
        num_meta_atoms_out_x=24,
        num_meta_atoms_out_y=16,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
    )
    rx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=24,
        num_meta_atoms_in_y=16,
        num_meta_atoms_out_x=24,
        num_meta_atoms_out_y=16,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
    )
    tx_model = MonoSIMoptimizerTorch(tx_cpu).to(device)
    rx_model = MonoSIMoptimizerTorch(rx_cpu).to(device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    tx_model.optimize_sim(target_matrix=A_torch, max_iters=max_iters, lr=lr)
    with torch.no_grad():
        H_eff = H_torch @ tx_model.get_cascade()
    _, beta_g = rx_model.optimize_rx_sequential(
        H_eff=H_eff, A_target=A_torch, max_iters=max_iters, lr=lr
    )

    sim_probe = run_evaluation_disjoint_fixed(
        tx_model,
        rx_model,
        test_loader,
        H_mimo,
        None,
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
            None,
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
        f" Disjoint L={L} M={M_int} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{_neural_str('sim_neural', _pct(sim_neural))}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
        },
    )
    del tx_model, rx_model
    gc.collect()
    torch.cuda.empty_cache()


def run_single_disjoint_snr(
    L: int,
    M_int: int,
    snr: float | None,
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
    strategy_name: str = "Linear",
    seed: int = 42,
    max_iters: int = 3000,
    lr: float = 0.1,
    clf_neural=None,
    no_mismatch_neural: float | None = None,
    original_neural: float | None = None,
    tx_enc: str = "",
    rx_enc: str = "",
) -> None:
    """Disjoint SNR experiment — one (M_int, snr, seed) job."""
    out_path = (
        RESULTS_DIR
        / "disjoint_snr"
        / f"{strategy_name}_seed{seed}_M{M_int}x{M_int}_snr{_snr_tag(snr)}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping disjoint M={M_int}, snr={snr}, seed={seed}.")
        return

    wavelength = 0.005
    slayer = 5 * wavelength

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=16,
        num_meta_atoms_in_y=12,
        num_meta_atoms_out_x=24,
        num_meta_atoms_out_y=16,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
    )
    rx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=24,
        num_meta_atoms_in_y=16,
        num_meta_atoms_out_x=24,
        num_meta_atoms_out_y=16,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
    )
    tx_model = MonoSIMoptimizerTorch(tx_cpu).to(device)
    rx_model = MonoSIMoptimizerTorch(rx_cpu).to(device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    tx_model.optimize_sim(target_matrix=A_torch, max_iters=max_iters, lr=lr)
    with torch.no_grad():
        H_eff = H_torch @ tx_model.get_cascade()
    _, beta_g = rx_model.optimize_rx_sequential(
        H_eff=H_eff, A_target=A_torch, max_iters=max_iters, lr=lr
    )

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
        f" Disjoint M={M_int} snr={snr} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{_neural_str('sim_neural', _pct(sim_neural))}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "snr": snr,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
        },
    )
    del tx_model, rx_model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Asymmetric helper (kept for manual use)
# ---------------------------------------------------------------------------


def run_single_rx_depth(
    L_TX: int,
    L_RX: int,
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
    seed: int = 42,
    clf_neural=None,
    tx_enc: str = "",
    rx_enc: str = "",
) -> None:
    out_path = (
        RESULTS_DIR
        / "rx_depth"
        / f"seed{seed}_LTX{L_TX}_LRX{L_RX}_M{M_int}x{M_int}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping rx_depth L_TX={L_TX}, L_RX={L_RX}, M_int={M_int}, seed={seed}.")
        return

    wavelength = 0.005
    slayer = 5 * wavelength
    dx = wavelength / 2

    sim_cpu = DualSIMoptimizer(
        num_layers_TX=L_TX,
        num_meta_atoms_TX_in_x=16,
        num_meta_atoms_TX_in_y=12,
        num_meta_atoms_TX_out_x=24,
        num_meta_atoms_TX_out_y=16,
        num_meta_atoms_TX_int_x=M_int,
        num_meta_atoms_TX_int_y=M_int,
        thickness_TX=slayer * L_TX,
        num_layers_RX=L_RX,
        num_meta_atoms_RX_in_x=24,
        num_meta_atoms_RX_in_y=16,
        num_meta_atoms_RX_out_x=24,
        num_meta_atoms_RX_out_y=16,
        num_meta_atoms_RX_int_x=M_int,
        num_meta_atoms_RX_int_y=M_int,
        thickness_RX=slayer * L_RX,
        wavelength=wavelength,
        spacings={k: dx for k in ("tx_in", "tx_out", "tx_int", "rx_in", "rx_out", "rx_int")},
        verbose=False,
    )
    model = DualSIMoptimizerTorch(sim_cpu).to(device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    iters = 500 if L_RX <= 10 else 1000
    model.optimize_alternating(A_target=A_target, H_mimo=H_mimo, max_iters=iters, lr=0.05)

    with torch.no_grad():
        Z_f, _ = model.get_effective_cascade(H_torch)
        beta = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

    sim_probe = run_evaluation(
        model, test_loader, H_mimo, None, beta, L_in, mu_in, L_out, mu_out, clf, device
    )
    sim_neural = (
        run_evaluation(
            model, test_loader, H_mimo, None, beta, L_in, mu_in, L_out, mu_out, clf_neural, device
        )
        if clf_neural is not None
        else None
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "L_tx": L_TX,
            "L_rx": L_RX,
            "M_int": f"{M_int}x{M_int}",
            "no_mismatch_probe": no_mismatch_probe,
            "original_probe": original_probe,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
        },
    )
    del model, sim_cpu
    gc.collect()
    torch.cuda.empty_cache()

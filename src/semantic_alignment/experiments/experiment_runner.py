"""Single-configuration experiment runners for Dual-SIM and Mono-SIM.

Each function handles ONE (seed, L, M_int, snr, …) combination and writes
its result to a dedicated Parquet file inside a per-experiment subdirectory:

    results/
        dual_layers/    {strategy}_seed{seed}_L{L}_M{M}x{M}_{tx}_{rx}.parquet
        dual_snr/       {strategy}_seed{seed}_M{M}x{M}_snr{snr}_{tx}_{rx}.parquet
        dual_slayer/    {strategy}_seed{seed}_stx{s_tx}_srx{s_rx}_{tx}_{rx}.parquet
        mono/           {strategy}_seed{seed}_L{L}_M{M}x{M}_{tx}_{rx}.parquet
        mono_channel/   {strategy}_seed{seed}_L{L}_M{M}x{M}_snr{snr}_{tx}_{rx}.parquet
        disjoint_layers/{strategy}_seed{seed}_L{L}_M{M}x{M}_{tx}_{rx}.parquet
        disjoint_snr/   {strategy}_seed{seed}_M{M}x{M}_snr{snr}_{tx}_{rx}.parquet
        rx_depth/       seed{seed}_LTX{L_TX}_LRX{L_RX}_M{M}x{M}_{tx}_{rx}.parquet

If the target file already exists the run is skipped (idempotent Hydra sweeps).

Every saved row contains both ``*_probe`` (linear probe) and ``*_neural`` (MLP
classifier) accuracy columns so downstream plotting can compare both without
re-running experiments.  Neural columns are ``None`` when no checkpoint is found.
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
    """Write a single-row result dict to a Parquet file, creating parent dirs.

    Parameters
    ----------
    path : Path
        Destination file path.
    row : dict
        Result columns for one experiment configuration.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_parquet(path)


def _snr_tag(snr: float | None) -> str:
    """Return a filesystem-safe string for an SNR value.

    Parameters
    ----------
    snr : float or None
        SNR in dB, or ``None`` for the noiseless case.

    Returns
    -------
    str
        ``"inf"`` when ``snr`` is ``None``, otherwise ``str(snr)``.
    """
    return "inf" if snr is None else str(snr)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _build_joint_sim(L, M_int, wavelength, slayer, dx, device):
    """Instantiate a :class:`DualSIMoptimizerTorch` with fixed aperture sizes.

    Aperture layout (meta-atoms): TX-in 16×12, TX/RX-out 24×16, intermediate M_int×M_int.

    Parameters
    ----------
    L : int
        Number of layers per SIM (same for TX and RX).
    M_int : int
        Number of intermediate meta-atoms along each axis.
    wavelength : float
        Operating wavelength (metres).
    slayer : float
        Inter-layer spacing (metres).
    dx : float
        Meta-atom pitch (metres).
    device : torch.device
        Target device for the returned model.

    Returns
    -------
    DualSIMoptimizerTorch
        Initialized (untrained) joint Dual-SIM model on ``device``.
    """
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
    """Evaluate a TX-only SIM cascade with no channel and no noise.

    Signal model: ``ŷ = L_out · (β G_T x_white) + μ_out``.

    Parameters
    ----------
    G_T : torch.Tensor
        TX cascade matrix.
    beta : torch.Tensor
        Optimal scalar alignment coefficient.
    test_loader : DataLoader
        Yields ``(z_tx_batch, labels)`` pairs.
    L_in : torch.Tensor
        Cholesky factor of TX pilot covariance.
    mu_in : torch.Tensor
        TX pilot mean.
    L_out : torch.Tensor
        Cholesky factor of RX pilot covariance.
    mu_out : torch.Tensor
        RX pilot mean.
    clf : torch.Tensor or nn.Module
        Linear probe or neural classifier.
    device : torch.device
        Inference device.

    Returns
    -------
    float
        Top-1 accuracy in ``[0, 1]``.
    """
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
    """Multiply by 100 to convert a fraction to a percentage, passing ``None`` through.

    Parameters
    ----------
    v : float or None
        Value in ``[0, 1]``, or ``None``.

    Returns
    -------
    float or None
        ``v * 100`` or ``None``.
    """
    return v * 100 if v is not None else None


def _neural_str(label: str, acc_pct: float | None) -> str:
    """Format an optional accuracy percentage as a log string.

    Parameters
    ----------
    label : str
        Metric label to display.
    acc_pct : float or None
        Accuracy already in percentage (0–100), or ``None`` to suppress output.

    Returns
    -------
    str
        ``" label=XX.XX%"`` or empty string when ``acc_pct`` is ``None``.
    """
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
    optimizer: str = "adam",
    scheduler: str = "none",
    warmup_frac: float = 0.05,
) -> None:
    """Run one (L, M_int, seed) job of the layers sweep for the joint Dual-SIM.

    Trains a joint Dual-SIM with ``L`` layers per metasurface and ``M_int``
    intermediate meta-atoms, then evaluates with both the linear probe and
    (optionally) the neural classifier.  Results are written to
    ``results/dual_layers/``.

    Parameters
    ----------
    L : int
        Number of metasurface layers per SIM (TX and RX share the same depth).
    M_int : int
        Number of intermediate meta-atoms along each axis.
    no_mismatch_probe : float
        No-mismatch linear probe accuracy in % (pre-computed, logged only).
    original_probe : float
        Original alignment linear probe accuracy in % (pre-computed, logged only).
    A_target : torch.Tensor
        Target alignment matrix.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    test_loader : DataLoader
        Test split yielding ``(z_tx, labels)`` batches.
    clf : torch.Tensor
        Linear probe weight matrix.
    L_in : torch.Tensor
        Cholesky factor of TX pilot covariance.
    mu_in : torch.Tensor
        TX pilot mean.
    L_out : torch.Tensor
        Cholesky factor of RX pilot covariance.
    mu_out : torch.Tensor
        RX pilot mean.
    device : torch.device
        Training and inference device.
    strategy_name : str, optional
        Alignment strategy label saved in the Parquet row.
    seed : int, optional
        Random seed for reproducibility.
    max_iters : int, optional
        Number of alternating optimization iterations.
    lr : float, optional
        Initial Adam learning rate (overridden internally by depth scaling).
    clf_neural : nn.Module or None, optional
        Neural classifier; accuracy columns are ``None`` when not provided.
    no_mismatch_neural : float or None, optional
        No-mismatch neural accuracy in % (logged only).
    original_neural : float or None, optional
        Original alignment neural accuracy in % (logged only).
    tx_enc : str, optional
        TX encoder name appended to the output filename.
    rx_enc : str, optional
        RX encoder name appended to the output filename.
    """
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
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
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
            "loss_history": loss_history,
            "lr": lr,
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
    optimizer: str = "adam",
    scheduler: str = "none",
    warmup_frac: float = 0.05,
) -> None:
    """Run one (M_int, snr, seed) job of the SNR sweep for the joint Dual-SIM.

    Same architecture as :func:`run_single_layers_config` but evaluates over a
    range of AWGN SNR levels.  Results are written to ``results/dual_snr/``.

    Parameters
    ----------
    L : int
        Number of metasurface layers per SIM.
    M_int : int
        Number of intermediate meta-atoms along each axis.
    snr : float or None
        AWGN SNR in dB for evaluation; ``None`` means noiseless.
    no_mismatch_probe : float
        No-mismatch linear probe accuracy in % (logged only).
    original_probe : float
        Original alignment linear probe accuracy in % (logged only).
    A_target : torch.Tensor
        Target alignment matrix.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    test_loader : DataLoader
        Test split yielding ``(z_tx, labels)`` batches.
    clf : torch.Tensor
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics (Cholesky factors and means).
    device : torch.device
        Training and inference device.
    strategy_name : str, optional
        Alignment strategy label.
    seed : int, optional
        Random seed.
    max_iters : int, optional
        Optimization iterations.
    lr : float, optional
        Initial Adam learning rate.
    clf_neural : nn.Module or None, optional
        Neural classifier.
    no_mismatch_neural, original_neural : float or None, optional
        Pre-computed accuracy values logged to the Parquet row.
    tx_enc, rx_enc : str, optional
        Encoder names appended to the output filename.
    """
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
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
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
            "loss_history": loss_history,
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
    optimizer: str = "adam",
    scheduler: str = "none",
    warmup_frac: float = 0.05,
) -> None:
    """Run one (s_tx, s_rx, seed) job of the layer-spacing sweep.

    Studies how the inter-layer spacing of the TX and RX metasurfaces affects
    alignment quality.  Results are written to ``results/dual_slayer/``.

    Parameters
    ----------
    L : int
        Number of layers per SIM.
    M_int : int
        Number of intermediate meta-atoms along each axis.
    s_tx : int
        TX inter-layer spacing in units of wavelength.
    s_rx : int
        RX inter-layer spacing in units of wavelength.
    no_mismatch_probe : float
        No-mismatch linear probe accuracy in % (logged only).
    original_probe : float
        Original alignment linear probe accuracy in % (logged only).
    A_target : torch.Tensor
        Target alignment matrix.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    test_loader : DataLoader
        Test split yielding ``(z_tx, labels)`` batches.
    clf : torch.Tensor
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics.
    device : torch.device
        Training and inference device.
    strategy_name : str, optional
        Alignment strategy label.
    seed : int, optional
        Random seed.
    max_iters : int, optional
        Optimization iterations.
    lr : float, optional
        Initial Adam learning rate.
    clf_neural : nn.Module or None, optional
        Neural classifier.
    no_mismatch_neural, original_neural : float or None, optional
        Pre-computed accuracy values logged to the Parquet row.
    tx_enc, rx_enc : str, optional
        Encoder names appended to the output filename.
    """
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

    loss_history = model.optimize_alternating(
        A_target=A_target,
        H_mimo=H_mimo,
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
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
            "loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
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
    optimizer: str = "adam",
    scheduler: str = "none",
    warmup_frac: float = 0.05,
) -> None:
    """Run one (L, M_int, seed) job of the Mono-SIM (TX-only) ablation.

    Evaluates a TX-only metasurface with two signal models:

    - ``use_channel=False``:  ``y = β G_T x_white``  (no channel, no noise)
    - ``use_channel=True``:   ``y = G_eq(H) · (H · β G_T · x_white + noise)``

    Results are written to ``results/mono/`` or ``results/mono_channel/``.

    Parameters
    ----------
    L : int
        Number of metasurface layers.
    M_int : int
        Number of intermediate meta-atoms along each axis.
    no_mismatch_probe : float
        No-mismatch linear probe accuracy in % (logged only).
    original_probe : float
        Original alignment linear probe accuracy in % (logged only).
    A_target : torch.Tensor
        Target alignment matrix.
    test_loader : DataLoader
        Test split yielding ``(z_tx, labels)`` batches.
    clf : torch.Tensor
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics.
    device : torch.device
        Training and inference device.
    strategy_name : str, optional
        Alignment strategy label.
    seed : int, optional
        Random seed.
    max_iters : int, optional
        Optimization iterations.
    lr : float, optional
        Adam learning rate.
    use_channel : bool, optional
        If ``True``, route signal through ``H_mimo`` with MMSE equalization.
    H_mimo : torch.Tensor or None, optional
        MIMO channel matrix; required when ``use_channel=True``.
    snr : float or None, optional
        AWGN SNR in dB used when ``use_channel=True``.
    clf_neural : nn.Module or None, optional
        Neural classifier.
    no_mismatch_neural, original_neural : float or None, optional
        Pre-computed accuracy values logged to the Parquet row.
    tx_enc, rx_enc : str, optional
        Encoder names appended to the output filename.
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
        side="tx",
    )
    tx_model = MonoSIMoptimizerTorch(tx_cpu).to(device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)

    loss_history, _ = tx_model.optimize_sim(
        target_matrix=A_torch,
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )

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
            "loss_history": loss_history,
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
    optimizer: str = "adam",
    scheduler: str = "none",
    warmup_frac: float = 0.05,
) -> None:
    """Run one (L, M_int, seed) job of the disjoint layers sweep.

    Trains TX and RX metasurfaces independently (sequentially):
    the TX-SIM is optimised first, then the RX-SIM is optimised against the
    effective channel ``H_eff = H · G_T``.  Results are written to
    ``results/disjoint_layers/``.

    Parameters
    ----------
    L : int
        Number of metasurface layers per SIM.
    M_int : int
        Number of intermediate meta-atoms along each axis.
    no_mismatch_probe : float
        No-mismatch linear probe accuracy in % (logged only).
    original_probe : float
        Original alignment linear probe accuracy in % (logged only).
    A_target : torch.Tensor
        Target alignment matrix.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    test_loader : DataLoader
        Test split yielding ``(z_tx, labels)`` batches.
    clf : torch.Tensor
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics.
    device : torch.device
        Training and inference device.
    strategy_name : str, optional
        Alignment strategy label.
    seed : int, optional
        Random seed.
    max_iters : int, optional
        Optimization iterations for each SIM stage.
    lr : float, optional
        Adam learning rate.
    clf_neural : nn.Module or None, optional
        Neural classifier.
    no_mismatch_neural, original_neural : float or None, optional
        Pre-computed accuracy values logged to the Parquet row.
    tx_enc, rx_enc : str, optional
        Encoder names appended to the output filename.
    """
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
        side="tx",
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
        side="rx",
    )
    tx_model = MonoSIMoptimizerTorch(tx_cpu).to(device)
    rx_model = MonoSIMoptimizerTorch(rx_cpu).to(device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    tx_loss_history, _ = tx_model.optimize_sim(
        target_matrix=A_torch,
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )
    with torch.no_grad():
        H_eff = H_torch @ tx_model.get_cascade()
    rx_loss_history, beta_g = rx_model.optimize_rx_sequential(
        H_eff=H_eff,
        A_target=A_torch,
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
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
            "tx_loss": tx_loss_history[-1] if tx_loss_history else 0.0,
            "tx_loss_history": tx_loss_history,
            "rx_loss": rx_loss_history[-1] if rx_loss_history else 0.0,
            "rx_loss_history": rx_loss_history,
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
    optimizer: str = "adam",
    scheduler: str = "none",
    warmup_frac: float = 0.05,
) -> None:
    """Run one (M_int, snr, seed) job of the disjoint SNR sweep.

    Disjoint Dual-SIM trained sequentially (TX first, then RX against
    ``H_eff``), evaluated at a specific AWGN SNR level.  Results are written
    to ``results/disjoint_snr/``.

    Parameters
    ----------
    L : int
        Number of metasurface layers per SIM.
    M_int : int
        Number of intermediate meta-atoms along each axis.
    snr : float or None
        AWGN SNR in dB; ``None`` means noiseless.
    no_mismatch_probe : float
        No-mismatch linear probe accuracy in % (logged only).
    original_probe : float
        Original alignment linear probe accuracy in % (logged only).
    A_target : torch.Tensor
        Target alignment matrix.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    test_loader : DataLoader
        Test split yielding ``(z_tx, labels)`` batches.
    clf : torch.Tensor
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics.
    device : torch.device
        Training and inference device.
    strategy_name : str, optional
        Alignment strategy label.
    seed : int, optional
        Random seed.
    max_iters : int, optional
        Optimization iterations for each SIM stage.
    lr : float, optional
        Adam learning rate.
    clf_neural : nn.Module or None, optional
        Neural classifier.
    no_mismatch_neural, original_neural : float or None, optional
        Pre-computed accuracy values logged to the Parquet row.
    tx_enc, rx_enc : str, optional
        Encoder names appended to the output filename.
    """
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
        side="tx",
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
        side="rx",
    )
    tx_model = MonoSIMoptimizerTorch(tx_cpu).to(device)
    rx_model = MonoSIMoptimizerTorch(rx_cpu).to(device)
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    tx_loss_history, _ = tx_model.optimize_sim(
        target_matrix=A_torch,
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )
    with torch.no_grad():
        H_eff = H_torch @ tx_model.get_cascade()
    rx_loss_history, beta_g = rx_model.optimize_rx_sequential(
        H_eff=H_eff,
        A_target=A_torch,
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
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
            "tx_loss": tx_loss_history[-1] if tx_loss_history else 0.0,
            "tx_loss_history": tx_loss_history,
            "rx_loss": rx_loss_history[-1] if rx_loss_history else 0.0,
            "rx_loss_history": rx_loss_history,
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
    optimizer: str = "adam",
    scheduler: str = "none",
    warmup_frac: float = 0.05,
) -> None:
    """Study the effect of asymmetric TX/RX SIM depth on alignment quality.

    Trains a joint Dual-SIM where TX and RX may have different layer counts,
    allowing analysis of which side benefits more from increased depth.
    Results are written to ``results/rx_depth/``.

    Parameters
    ----------
    L_TX : int
        Number of layers in the TX-SIM.
    L_RX : int
        Number of layers in the RX-SIM.
    M_int : int
        Number of intermediate meta-atoms along each axis.
    no_mismatch_probe : float
        No-mismatch linear probe accuracy in % (logged only).
    original_probe : float
        Original alignment linear probe accuracy in % (logged only).
    A_target : torch.Tensor
        Target alignment matrix.
    H_mimo : torch.Tensor
        Complex MIMO channel matrix.
    test_loader : DataLoader
        Test split yielding ``(z_tx, labels)`` batches.
    clf : torch.Tensor
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics.
    device : torch.device
        Training and inference device.
    seed : int, optional
        Random seed.
    clf_neural : nn.Module or None, optional
        Neural classifier.
    tx_enc, rx_enc : str, optional
        Encoder names appended to the output filename.
    """
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
    loss_history = model.optimize_alternating(
        A_target=A_target,
        H_mimo=H_mimo,
        max_iters=iters,
        lr=0.05,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
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
            "loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
        },
    )
    del model, sim_cpu
    gc.collect()
    torch.cuda.empty_cache()

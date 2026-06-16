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
from semantic_alignment.sim.dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from semantic_alignment.sim.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch
from semantic_alignment.utils.utils import get_rx_equalizer

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


class _FixedCascade:
    """Minimal wrapper so a pre-computed cascade matrix satisfies the model
    interface expected by :func:`run_evaluation_disjoint_fixed`."""

    def __init__(self, G: torch.Tensor) -> None:
        self._G = G

    def eval(self):
        return self

    def get_cascade(self) -> torch.Tensor:
        return self._G


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
            "snr": None,
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "lr": lr,
            "beta_T": None,
            "beta_R": None,
            "loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "rx_loss": None,
            "rx_loss_history": None,
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
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "lr": lr,
            "beta_T": None,
            "beta_R": None,
            "loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "rx_loss": None,
            "rx_loss_history": None,
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
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "snr": None,
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
            "lr": lr,
            "beta_T": None,
            "beta_R": None,
            "loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "rx_loss": None,
            "rx_loss_history": None,
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

    The signal always propagates through the MIMO channel ``H`` and is equalized
    at the receiver with ``G_eq``: ``ŷ = β G_eq(H) · (H · G_T · x_white + noise)``.
    With ``snr=None`` the equalizer is zero-forcing and no noise is added
    (infinite SNR); with a finite ``snr`` it is MMSE and AWGN is added.

    ``use_channel`` only selects the output location: ``False`` (layers sweep,
    noiseless ZF) writes to ``results/mono/``; ``True`` (SNR sweep, MMSE + AWGN)
    writes to ``results/mono_channel/``.

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
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

        # β is the optimal scalar matching the *bare* SIM cascade to the target A
        # — the same least-squares projection the SIM minimises during training.
        # The channel is undone by the RX equalizer, not by β.
        beta = torch.sum(torch.conj(G_T_f) * A_torch) / (
            torch.sum(torch.conj(G_T_f) * G_T_f) + 1e-12
        )

        # Both sweeps run through the channel: the layers sweep uses snr=None
        # (infinite SNR → zero-forcing, no noise), the SNR sweep uses MMSE + AWGN.
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
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": mono_probe * 100,
            "sim_accuracy_neural": _pct(mono_neural),
            "lr": lr,
            "beta_T": None,
            "beta_R": None,
            "loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "rx_loss": None,
            "rx_loss_history": None,
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

    TX-SIM is optimised independently to emulate the alignment matrix A.
    RX-SIM is optimised independently to emulate the ZF equalizer
    ``Q = (Hᴴ H)⁻¹ Hᴴ = H†``.  Each yields its own scalar β_T, β_R;
    at inference ``β̄ = β_T · β_R``.  Results are written to
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

    # Q = H† = (HᴴH)⁻¹Hᴴ — ZF left-inverse (noiseless)
    with torch.no_grad():
        Q = get_rx_equalizer(H_torch, snr_db=None)

    sim_kw = dict(
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )
    tx_loss_history, beta_T = tx_model.optimize_sim(target_matrix=A_torch, **sim_kw)
    rx_loss_history, beta_R = rx_model.optimize_sim(target_matrix=Q, **sim_kw)

    beta_bar = complex(beta_T) * complex(beta_R)
    beta_g = torch.tensor(beta_bar, dtype=torch.complex64, device=device)

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
        f" | β_T={abs(beta_T):.3e} β_R={abs(beta_R):.3e}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "snr": None,
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "lr": lr,
            "beta_T": abs(beta_T),
            "beta_R": abs(beta_R),
            "loss": tx_loss_history[-1] if tx_loss_history else 0.0,
            "loss_history": tx_loss_history,
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

    TX-SIM emulates A, RX-SIM emulates Q = H†, both trained independently.
    ``β̄ = β_T · β_R`` is applied at inference at the given SNR level.
    Results are written to ``results/disjoint_snr/``.

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

    with torch.no_grad():
        Q = get_rx_equalizer(H_torch, snr_db=None)

    sim_kw = dict(
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )
    tx_loss_history, beta_T = tx_model.optimize_sim(target_matrix=A_torch, **sim_kw)
    rx_loss_history, beta_R = rx_model.optimize_sim(target_matrix=Q, **sim_kw)

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
        f" Disjoint M={M_int} snr={snr} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{_neural_str('sim_neural', _pct(sim_neural))}"
        f" | β_T={abs(beta_T):.3e} β_R={abs(beta_R):.3e}"
    )

    _save_result(
        out_path,
        {
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
            "sim_accuracy_neural": _pct(sim_neural),
            "lr": lr,
            "beta_T": abs(beta_T),
            "beta_R": abs(beta_R),
            "loss": tx_loss_history[-1] if tx_loss_history else 0.0,
            "loss_history": tx_loss_history,
            "rx_loss": rx_loss_history[-1] if rx_loss_history else 0.0,
            "rx_loss_history": rx_loss_history,
        },
    )
    del tx_model, rx_model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Disjoint Dual-SIM with full CSI (SVD precoding/decoding)
# ---------------------------------------------------------------------------


def run_single_disjoint_csi_layers(
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
    """Disjoint Dual-SIM with full CSI at both TX and RX (SVD-based precoding).

    Assumes both sides know ``H = U Σ Vᴴ``.  The TX-SIM is trained to emulate
    the composition ``B_T · A`` (SVD precoder ``B_T = V Σ^{1/2}`` followed by
    the alignment operator).  The RX-SIM is trained independently to emulate
    the SVD decoder ``B_R = Σ^{1/2} Uᴴ``.  Each SIM yields its own optimal
    scalar (β_T, β_R); at inference ``β̄ = β_T · β_R`` is applied to the full
    cascade ``G_R H G_T x``.

    Only PPFE alignment is supported; Linear raises ``NotImplementedError``.

    Results are written to ``results/disjoint_csi_layers/``.
    """
    if strategy_name != "PPFE":
        raise NotImplementedError(
            f"Disjoint CSI not implemented for '{strategy_name}' alignment. Use PPFE."
        )

    out_path = (
        RESULTS_DIR
        / "disjoint_csi_layers"
        / f"{strategy_name}_seed{seed}_L{L}_M{M_int}x{M_int}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping disjoint_csi L={L}, M_int={M_int}, seed={seed}.")
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

    # SVD of H and CSI-based precoder/decoder
    with torch.no_grad():
        U, s, Vh = torch.linalg.svd(H_torch, full_matrices=False)
        sqrt_s = torch.sqrt(s).to(H_torch.dtype)
        B_T = Vh.H @ torch.diag(sqrt_s)  # V Σ^{1/2} ∈ C^{N_T × N_T}
        B_R = torch.diag(sqrt_s) @ U.H  # Σ^{1/2} Uᴴ ∈ C^{N_T × N_R}

    # TX target: B_T A  (SVD precoder composed with alignment)
    # RX target: B_R    (SVD decoder alone)
    T_TX = (B_T @ A_torch).detach()  # C^{N_T × θ̄}
    T_RX = B_R.detach()  # C^{N_T × N_R}

    # Train TX and RX independently — no sequential dependency
    tx_loss_history, beta_T = tx_model.optimize_sim(
        target_matrix=T_TX,
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )
    rx_loss_history, beta_R = rx_model.optimize_sim(
        target_matrix=T_RX,
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )

    beta_bar = complex(beta_T) * complex(beta_R)
    beta_g = torch.tensor(beta_bar, dtype=torch.complex64, device=device)

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
        f" DisjointCSI L={L} M={M_int} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{_neural_str('sim_neural', _pct(sim_neural))}"
        f" | β_T={abs(beta_T):.3e} β_R={abs(beta_R):.3e}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "snr": None,
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "lr": lr,
            "beta_T": abs(beta_T),
            "beta_R": abs(beta_R),
            "loss": tx_loss_history[-1] if tx_loss_history else 0.0,
            "loss_history": tx_loss_history,
            "rx_loss": rx_loss_history[-1] if rx_loss_history else 0.0,
            "rx_loss_history": rx_loss_history,
        },
    )
    del tx_model, rx_model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 4-SIM Disjoint with full CSI (2 TX + 2 RX, each depth L)
# ---------------------------------------------------------------------------


def run_single_disjoint_4sim_layers(
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
    """Disjoint 4-SIM architecture with full CSI: two TX SIMs + two RX SIMs.

    Hardware layout (each SIM has L layers):

        G_T1 (192→384)  →  G_T2 (384→384)  →  H  →  G_R2 (384→384)  →  G_R1 (384→384)

    Targets (PPFE only):
        G_T1 emulates  F_T  = A_target    (alignment synthesis, 192→384)
        G_T2 emulates  B_T  = V Σ^{1/2}   (SVD channel precoder, 384→384)
        G_R2 emulates  B_R  = Σ^{1/2} Uᴴ  (SVD channel decoder,  384→384)
        G_R1 emulates  F_R^H = I_{384}     (identity analysis,    384→384)

    Each SIM is optimised independently, yielding β_T1, β_T2, β_R2, β_R1.
    At inference: β̄ = β_T1 · β_T2 · β_R2 · β_R1.

    Hardware equivalence: L layers per SIM here ≈ 2L layers in the 2-SIM
    disjoint case (G_T = G_T2 G_T1 sums L+L depth at the TX side, idem RX).

    Only PPFE alignment is supported; Linear raises ``NotImplementedError``.

    Results are written to ``results/disjoint_4sim_layers/``.
    """
    if strategy_name != "PPFE":
        raise NotImplementedError(
            f"4-SIM disjoint not implemented for '{strategy_name}' alignment. Use PPFE."
        )

    out_path = (
        RESULTS_DIR
        / "disjoint_4sim_layers"
        / f"{strategy_name}_seed{seed}_L{L}_M{M_int}x{M_int}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping disjoint_4sim L={L}, M_int={M_int}, seed={seed}.")
        return

    wavelength = 0.005
    slayer = 5 * wavelength
    dx = wavelength / 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # G_T1: standard TX SIM  (192 → 384)
    t1_cpu = MonoSIMoptimizer(
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
    # G_T2: square TX SIM  (384 → 384)
    t2_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=24,
        num_meta_atoms_in_y=16,
        num_meta_atoms_out_x=24,
        num_meta_atoms_out_y=16,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
        spacings={"in": dx, "out": dx, "int": dx},
        side="tx",
    )
    # G_R2: square RX SIM  (384 → 384)
    r2_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=24,
        num_meta_atoms_in_y=16,
        num_meta_atoms_out_x=24,
        num_meta_atoms_out_y=16,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
        spacings={"in": dx, "out": dx, "int": dx},
        side="rx",
    )
    # G_R1: standard RX SIM  (384 → 384)
    r1_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=24,
        num_meta_atoms_in_y=16,
        num_meta_atoms_out_x=24,
        num_meta_atoms_out_y=16,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
        spacings={"in": dx, "out": dx, "int": dx},
        side="rx",
    )

    t1_model = MonoSIMoptimizerTorch(t1_cpu).to(device)
    t2_model = MonoSIMoptimizerTorch(t2_cpu).to(device)
    r2_model = MonoSIMoptimizerTorch(r2_cpu).to(device)
    r1_model = MonoSIMoptimizerTorch(r1_cpu).to(device)

    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    # SVD-based precoder / decoder from full CSI
    with torch.no_grad():
        U, s, Vh = torch.linalg.svd(H_torch, full_matrices=False)
        sqrt_s = torch.sqrt(s).to(H_torch.dtype)
        B_T = Vh.H @ torch.diag(sqrt_s)  # V Σ^{1/2} ∈ C^{384×384}
        B_R = torch.diag(sqrt_s) @ U.H  # Σ^{1/2} Uᴴ ∈ C^{384×384}
        I_RX = torch.eye(B_R.shape[0], dtype=torch.complex64, device=device)

    # Targets: F_T = A, B_T, B_R, F_R^H = I
    sim_kwargs = dict(
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )

    t1_loss, beta_T1 = t1_model.optimize_sim(target_matrix=A_torch, **sim_kwargs)
    t2_loss, beta_T2 = t2_model.optimize_sim(target_matrix=B_T, **sim_kwargs)
    r2_loss, beta_R2 = r2_model.optimize_sim(target_matrix=B_R, **sim_kwargs)
    r1_loss, beta_R1 = r1_model.optimize_sim(target_matrix=I_RX, **sim_kwargs)

    beta_bar = complex(beta_T1) * complex(beta_T2) * complex(beta_R2) * complex(beta_R1)
    beta_g = torch.tensor(beta_bar, dtype=torch.complex64, device=device)

    # Compose cascades and wrap for run_evaluation_disjoint_fixed
    with torch.no_grad():
        G_T_eff = t2_model.get_cascade() @ t1_model.get_cascade()  # 384×192
        G_R_eff = r1_model.get_cascade() @ r2_model.get_cascade()  # 384×384

    tx_wrap = _FixedCascade(G_T_eff)
    rx_wrap = _FixedCascade(G_R_eff)

    sim_probe = run_evaluation_disjoint_fixed(
        tx_wrap,
        rx_wrap,
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
            tx_wrap,
            rx_wrap,
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
        f" Disjoint4SIM L={L} M={M_int} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{_neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{_neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{_neural_str('sim_neural', _pct(sim_neural))}"
        f" | β_T1={abs(beta_T1):.2e} β_T2={abs(beta_T2):.2e}"
        f" β_R2={abs(beta_R2):.2e} β_R1={abs(beta_R1):.2e}"
    )

    _save_result(
        out_path,
        {
            "seed": seed,
            "strategy": strategy_name,
            "M_int": f"{M_int}x{M_int}",
            "L": L,
            "snr": None,
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "lr": lr,
            "beta_T": abs(beta_T1) * abs(beta_T2),
            "beta_R": abs(beta_R2) * abs(beta_R1),
            "loss": t1_loss[-1] if t1_loss else 0.0,
            "loss_history": t1_loss,
            "rx_loss": r1_loss[-1] if r1_loss else 0.0,
            "rx_loss_history": r1_loss,
            # per-SIM detail
            "beta_T1": abs(beta_T1),
            "beta_T2": abs(beta_T2),
            "beta_R2": abs(beta_R2),
            "beta_R1": abs(beta_R1),
            "t2_loss": t2_loss[-1] if t2_loss else 0.0,
            "t2_loss_history": t2_loss,
            "r2_loss": r2_loss[-1] if r2_loss else 0.0,
            "r2_loss_history": r2_loss,
        },
    )
    del t1_model, t2_model, r2_model, r1_model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Disjoint CSI — SNR sweep
# ---------------------------------------------------------------------------


def run_single_disjoint_csi_snr(
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
    """SNR sweep variant of :func:`run_single_disjoint_csi_layers`.

    Same CSI-assisted disjoint training (noiseless); evaluation at ``snr`` dB.
    Results are written to ``results/disjoint_csi_snr/``.
    Only PPFE alignment is supported.
    """
    if strategy_name != "PPFE":
        raise NotImplementedError(
            f"Disjoint CSI not implemented for '{strategy_name}' alignment. Use PPFE."
        )

    out_path = (
        RESULTS_DIR
        / "disjoint_csi_snr"
        / f"{strategy_name}_seed{seed}_M{M_int}x{M_int}_snr{_snr_tag(snr)}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping disjoint_csi snr={snr}, M_int={M_int}, seed={seed}.")
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

    with torch.no_grad():
        U, s, Vh = torch.linalg.svd(H_torch, full_matrices=False)
        sqrt_s = torch.sqrt(s).to(H_torch.dtype)
        B_T = Vh.H @ torch.diag(sqrt_s)
        B_R = torch.diag(sqrt_s) @ U.H

    T_TX = (B_T @ A_torch).detach()
    T_RX = B_R.detach()

    sim_kw = dict(
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )
    tx_loss_history, beta_T = tx_model.optimize_sim(target_matrix=T_TX, **sim_kw)
    rx_loss_history, beta_R = rx_model.optimize_sim(target_matrix=T_RX, **sim_kw)

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
        f" DisjointCSI snr={snr} M={M_int} seed={seed}"
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
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "lr": lr,
            "beta_T": abs(beta_T),
            "beta_R": abs(beta_R),
            "loss": tx_loss_history[-1] if tx_loss_history else 0.0,
            "loss_history": tx_loss_history,
            "rx_loss": rx_loss_history[-1] if rx_loss_history else 0.0,
            "rx_loss_history": rx_loss_history,
        },
    )
    del tx_model, rx_model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 4-SIM Disjoint — SNR sweep
# ---------------------------------------------------------------------------


def run_single_disjoint_4sim_snr(
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
    """SNR sweep variant of :func:`run_single_disjoint_4sim_layers`.

    Same 4-SIM CSI-assisted training (noiseless); evaluation at ``snr`` dB.
    Results are written to ``results/disjoint_4sim_snr/``.
    Only PPFE alignment is supported.
    """
    if strategy_name != "PPFE":
        raise NotImplementedError(
            f"4-SIM disjoint not implemented for '{strategy_name}' alignment. Use PPFE."
        )

    out_path = (
        RESULTS_DIR
        / "disjoint_4sim_snr"
        / f"{strategy_name}_seed{seed}_M{M_int}x{M_int}_snr{_snr_tag(snr)}_{tx_enc}_{rx_enc}.parquet"
    )
    if out_path.exists():
        print(f"Skipping disjoint_4sim snr={snr}, M_int={M_int}, seed={seed}.")
        return

    wavelength = 0.005
    slayer = 5 * wavelength
    dx = wavelength / 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def _make_sim(in_x, in_y, out_x, out_y, s):
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
            side=s,
        )

    t1_model = MonoSIMoptimizerTorch(_make_sim(16, 12, 24, 16, "tx")).to(device)
    t2_model = MonoSIMoptimizerTorch(_make_sim(24, 16, 24, 16, "tx")).to(device)
    r2_model = MonoSIMoptimizerTorch(_make_sim(24, 16, 24, 16, "rx")).to(device)
    r1_model = MonoSIMoptimizerTorch(_make_sim(24, 16, 24, 16, "rx")).to(device)

    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    with torch.no_grad():
        U, s, Vh = torch.linalg.svd(H_torch, full_matrices=False)
        sqrt_s = torch.sqrt(s).to(H_torch.dtype)
        B_T = Vh.H @ torch.diag(sqrt_s)
        B_R = torch.diag(sqrt_s) @ U.H
        I_RX = torch.eye(B_R.shape[0], dtype=torch.complex64, device=device)

    sim_kw = dict(
        max_iters=max_iters,
        lr=lr,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
    )
    t1_loss, beta_T1 = t1_model.optimize_sim(target_matrix=A_torch, **sim_kw)
    t2_loss, beta_T2 = t2_model.optimize_sim(target_matrix=B_T, **sim_kw)
    r2_loss, beta_R2 = r2_model.optimize_sim(target_matrix=B_R, **sim_kw)
    r1_loss, beta_R1 = r1_model.optimize_sim(target_matrix=I_RX, **sim_kw)

    beta_bar = complex(beta_T1) * complex(beta_T2) * complex(beta_R2) * complex(beta_R1)
    beta_g = torch.tensor(beta_bar, dtype=torch.complex64, device=device)

    with torch.no_grad():
        G_T_eff = t2_model.get_cascade() @ t1_model.get_cascade()
        G_R_eff = r1_model.get_cascade() @ r2_model.get_cascade()

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
        f" Disjoint4SIM snr={snr} M={M_int} seed={seed}"
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
            "s_tx": None,
            "s_rx": None,
            "tx_enc": tx_enc,
            "rx_enc": rx_enc,
            "no_mismatch_probe": no_mismatch_probe,
            "no_mismatch_neural": no_mismatch_neural,
            "original_probe": original_probe,
            "original_neural": original_neural,
            "sim_accuracy_probe": sim_probe * 100,
            "sim_accuracy_neural": _pct(sim_neural),
            "lr": lr,
            "beta_T": abs(beta_T1) * abs(beta_T2),
            "beta_R": abs(beta_R2) * abs(beta_R1),
            "loss": t1_loss[-1] if t1_loss else 0.0,
            "loss_history": t1_loss,
            "rx_loss": r1_loss[-1] if r1_loss else 0.0,
            "rx_loss_history": r1_loss,
            # per-SIM detail
            "beta_T1": abs(beta_T1),
            "beta_T2": abs(beta_T2),
            "beta_R2": abs(beta_R2),
            "beta_R1": abs(beta_R1),
            "t2_loss": t2_loss[-1] if t2_loss else 0.0,
            "t2_loss_history": t2_loss,
            "r2_loss": r2_loss[-1] if r2_loss else 0.0,
            "r2_loss_history": r2_loss,
        },
    )
    del t1_model, t2_model, r2_model, r1_model
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

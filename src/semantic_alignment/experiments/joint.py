"""Joint Dual-SIM runner — covers layers, SNR, and s-layer sweeps in one function."""

import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__" and __package__ is None:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    __package__ = "semantic_alignment.experiments"

import gc
import random

import numpy as np
import torch

from semantic_alignment.evaluation.inference import run_evaluation
from semantic_alignment.sim.dualsim import DualSIMoptimizer, DualSIMoptimizerTorch

from ._utils import RESULTS_DIR, ch_dims, make_fname, neural_str, pct, save_result, snr_tag


def build_joint_sim(
    L: int,
    M_int: int,
    wavelength: float,
    dx: float,
    thickness_tx: float,
    thickness_rx: float,
    device,
    tx_in_x: int,
    tx_in_y: int,
    ch_x: int,
    ch_y: int,
    rx_out_x: int,
    rx_out_y: int,
) -> DualSIMoptimizerTorch:
    """Instantiate and return a joint Dual-SIM model on the target device.

    Parameters
    ----------
    L : int
        Number of metasurface layers in each SIM (TX and RX share the same depth).
    M_int : int
        Side length of the square intermediate aperture (``M_int × M_int`` atoms).
    wavelength : float
        Operating wavelength λ (metres).
    dx : float
        Meta-atom spacing used for all apertures (metres).
    thickness_tx : float
        Total physical thickness of the TX-SIM stack (metres).
    thickness_rx : float
        Total physical thickness of the RX-SIM stack (metres).
    device :
        PyTorch device on which to place the model.
    tx_in_x, tx_in_y : int
        TX input aperture dimensions in atoms.
    ch_x, ch_y : int
        Channel-interface aperture dimensions in atoms.
    rx_out_x, rx_out_y : int
        RX output aperture dimensions in atoms.

    Returns
    -------
    DualSIMoptimizerTorch
        Freshly built joint Dual-SIM model with random initial phase shifts.
    """
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


def run_joint_sim(
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
    strategy_name: str = "Linear",
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
    """Run one joint Dual-SIM job.

    Sweep mode is selected by the parameters provided:

    - **layers** (``s_tx/s_rx=None``, ``snr=None``): vary depth L.
    - **SNR**    (``s_tx/s_rx=None``, ``snr`` finite): vary noise level.
    - **slayer** (``s_tx`` and ``s_rx`` provided): vary inter-layer spacing.

    The model is built, optimised via :meth:`DualSIMoptimizerTorch.optimize_alternating`,
    and evaluated with :func:`run_evaluation`.  The result row is written to
    ``results/{result_subdir}/`` as a single-row Parquet file.  The job is
    skipped if the output file already exists.

    Parameters
    ----------
    L : int
        Number of metasurface layers per SIM.
    M_int : int
        Side length of the square intermediate aperture.
    no_mismatch_probe : float
        No-mismatch (oracle) accuracy from the linear probe, in percent.
    original_probe : float
        Oracle linear-alignment accuracy (no SIM, no channel), in percent.
    A_target : array-like
        Target alignment matrix.
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
        Alignment strategy label (``"Linear"`` or ``"PPFE"``).
    seed : int, optional
        Global random seed.
    max_iters : int, optional
        Total gradient-step budget.  The number of TX/RX cycles run is
        ``max_iters // (k_T + k_R)``.
    lr_tx : float, optional
        TX (G_T) learning rate.
    lr_rx : float, optional
        RX (G_R) learning rate.
    clf_neural : optional
        Optional neural classifier (``None`` skips neural accuracy logging).
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
    clip_value : float or None, optional
        Gradient clipping norm; ``None`` disables clipping.
    k_T : int, optional
        Consecutive TX-only gradient steps per cycle.
    k_R : int, optional
        Consecutive RX-only gradient steps per cycle.
    channel_dim : int or None, optional
        Physical channel dimension override; defaults to ``max(A_target.shape)``.
    wavelength : float, optional
        Operating wavelength λ (metres).
    slayer : float, optional
        Default inter-layer spacing in wavelengths (overridden by ``s_tx``/``s_rx``).
    dx : float, optional
        Meta-atom spacing (metres).
    snr : float or None, optional
        AWGN SNR in dB; ``None`` means noiseless.
    s_tx, s_rx : int or None, optional
        TX/RX inter-layer spacings in wavelengths (activates slayer sweep mode).
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

    model = build_joint_sim(
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
            model, test_loader, H_mimo, snr, beta, L_in, mu_in, L_out, mu_out, clf_neural, device
        )
        if clf_neural is not None
        else None
    )

    print(
        f" {log_prefix}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{neural_str('orig_neural', original_neural)}"
        f" | sim={sim_probe * 100:.2f}%{neural_str('sim_neural', pct(sim_neural))}"
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
            "sim_accuracy_probe": sim_probe * 100,
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


if __name__ == "__main__":
    print("run_joint_sim: joint Dual-SIM runner (layers / SNR / s-layer sweeps).")
    print("Invoke via: just run-joint")
    print("         or: uv run python scripts/classification.py --config-name config_joint_layers")

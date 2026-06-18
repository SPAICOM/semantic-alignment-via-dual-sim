"""Mono-SIM (TX-only) runner — covers layers and SNR sweeps in one function."""

import gc
import random

import numpy as np
import torch

from semantic_alignment.evaluation.inference import run_evaluation_mono
from semantic_alignment.sim.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch
from semantic_alignment.utils.utils import complex_gaussian_matrix

from ._utils import RESULTS_DIR, ch_dims, make_fname, neural_str, pct, save_result


def run_mono_sim(
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
    channel_dim: int | None = None,
    wavelength: float = 0.005,
    slayer: float = 0.025,
    dx: float = 0.0025,
    **_,
) -> None:
    """Run one Mono-SIM (TX-only) job.

    Sweep mode is selected by ``snr``:

    - ``snr=None``: ZF equalizer, no noise → ``results/mono/``
    - ``snr`` finite: MMSE equalizer + AWGN → ``results/mono_snr/``

    A new square MIMO channel of dimension ``A_target.shape[0]`` is always
    generated here because the TX and RX embedding spaces may have different
    dimensions.  The job is skipped if the output Parquet file already exists.

    Parameters
    ----------
    L : int
        Number of metasurface layers in the TX-SIM.
    M_int : int
        Side length of the square intermediate aperture.
    no_mismatch_probe : float
        No-mismatch baseline accuracy from the linear probe, in percent.
    original_probe : float
        Oracle linear-alignment accuracy (no SIM, no channel), in percent.
    A_target : array-like
        Target alignment matrix.
    test_loader : DataLoader
        Test set yielding ``(z_tx, labels)`` pairs.
    clf :
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics (Cholesky factor and mean for TX and RX).
    device : torch.device
        Inference and optimization device.
    strategy_name : str, optional
        Alignment strategy label (``"Linear"`` or ``"PPFE"``).
    seed : int, optional
        Global random seed.
    max_iters : int, optional
        Number of optimization iterations.
    lr : float, optional
        Learning rate.
    H_mimo : array-like or None, optional
        Ignored; a fresh channel is always drawn inside this function.
    snr : float or None, optional
        AWGN SNR in dB; ``None`` means noiseless with ZF equalizer.
    clf_neural : optional
        Optional neural classifier.
    no_mismatch_neural, original_neural : float or None, optional
        Neural-classifier baselines.
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
        Default inter-layer spacing in wavelengths.
    dx : float, optional
        Meta-atom spacing (metres).
    """
    subdir = "mono_snr" if snr is not None else "mono"
    ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
    tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
    rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))

    fname = make_fname(strategy_name, seed, L, M_int, ch_dim, snr, tx_enc, rx_enc)
    out_path = RESULTS_DIR / subdir / fname

    if out_path.exists():
        print(f"Skipping {out_path.name}.")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Mono uses a square channel of size = RX complex dim (A_target.shape[0]).
    # _setup passes H_mimo sized to max(A_target.shape), which is wrong when
    # TX and RX have different embedding dimensions — generate it here instead.
    h_dim = int(A_target.shape[0])
    H_mimo = complex_gaussian_matrix(0, 1, size=(h_dim, h_dim)).to(device)

    tx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=tx_in_x,
        num_meta_atoms_in_y=tx_in_y,
        num_meta_atoms_out_x=rx_out_x,
        num_meta_atoms_out_y=rx_out_y,
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

        beta = torch.sum(torch.conj(G_T_f) * A_torch) / (
            torch.sum(torch.conj(G_T_f) * G_T_f) + 1e-12
        )

        mono_probe = run_evaluation_mono(
            G_T_f, test_loader, H_torch, snr, beta, L_in, mu_in, L_out, mu_out, clf, device
        )
        mono_neural = (
            run_evaluation_mono(
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
        f" Mono{'(snr=' + str(snr) + ')' if snr is not None else ''} L={L} M={M_int} seed={seed}"
        f" | no_mismatch={no_mismatch_probe:.2f}%{neural_str('nm_neural', no_mismatch_neural)}"
        f" | orig={original_probe:.2f}%{neural_str('orig_neural', original_neural)}"
        f" | sim={mono_probe * 100:.2f}%{neural_str('sim_neural', pct(mono_neural))}"
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
            "sim_accuracy_probe": mono_probe * 100,
            "sim_accuracy_neural": pct(mono_neural),
            "lr": lr,
            "beta_T": None,
            "beta_R": None,
            "loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "rx_loss": None,
            "rx_loss_history": None,
            "channel_dim": ch_dim,
        },
    )
    del tx_model
    gc.collect()
    torch.cuda.empty_cache()

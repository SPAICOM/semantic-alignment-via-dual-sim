"""Mono-SIM (TX-only) runner — covers layers and SNR sweeps in one function."""

import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__" and __package__ is None:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    __package__ = "semantic_alignment.experiments"

import gc
import random

import numpy as np
import torch

from semantic_alignment.evaluation.inference import run_evaluation_mono
from semantic_alignment.sim.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch
from semantic_alignment.utils import cn_matrix

from ._utils import RESULTS_DIR, ch_dims, make_fname, neural_str, pct, save_result, tipology_tag


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
    n_clusters: int | None = None,
    constructive: bool = False,
    use_mmse: bool = False,
    dataset: str = "cifar10",
    result_subdir: str = "",
    **_,
) -> None:
    """Run one Mono-SIM (TX-only) job.

    ``constructive`` selects the training loss and inference path:

    - ``False`` (standard): loss ``‖β G_T − A‖²``, inference
      ``ẑ = β · G_eq · H · G_T · x`` (G_eq controlled by ``use_mmse``).
    - ``True``: loss ``‖β H G_T − A‖²`` (H in the loss; SIM learns pre-coding),
      inference ``ẑ = β · (H · G_T · x + noise)`` — no G_eq, noise not mitigated.

    ``use_mmse`` (only active when ``constructive=False``):

    - ``False`` (default): ZF equalizer — G_eq = H† (pseudoinverse).
    - ``True``: MMSE equalizer — G_eq regularized with ``snr``.
    """
    subdir = result_subdir if result_subdir else ("mono_snr" if snr is not None else "mono_layers")
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
        left_matrix=H_torch if constructive else None,
    )

    with torch.no_grad():
        G_T_f = tx_model.get_cascade()
        # β aligns the effective TX output (H G_T for constructive, G_T otherwise) to A
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
        },
    )
    del tx_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("run_mono_sim: Mono-SIM (TX-only) runner (layers / SNR sweeps).")
    print("Invoke via: just run-mono")
    print("         or: uv run python scripts/classification.py --config-name config_mono_layers")

"""Warm-start phase computation and disk cache.

Provides _run_warm_start (pure computation) and _run_warm_start_cached
(same, but loads/saves results to results/unrolled/joint/warmstart/).

Cache key logic:
  only_semantic  → keyed on architecture + ws_t only (H=I, no seed/h0_var)
  average        → keyed on seed, h0_var, ch_type
  <int>          → keyed on the int seed and ch_type
"""

import math
from pathlib import Path

import torch

from semantic_alignment.sim.original.joint import JointSIM
from semantic_alignment.utils import channel as _channel_fn

_WS_CACHE_ROOT = Path("results") / "unrolled" / "joint" / "warmstart"


def _run_warm_start(
    sim: JointSIM,
    A_target: torch.Tensor,
    n_R: int,
    n_T: int,
    ws_t: int,
    ws_type: str,
    h0v: float | None,
    ch_type: str,
    lr_tx: float,
    lr_rx: float,
    seed: int,
    device: torch.device,
) -> tuple[list, list]:
    """Run ws_t alternating-Adam steps; return (ws_phT, ws_phR). Resets sim phases after."""
    with torch.no_grad():
        for p in sim.xi_T:
            p.zero_()
        for p in sim.xi_R:
            p.zero_()
    if ws_t == 0:
        return [], []
    match ws_type:
        case "only_semantic":
            H_ws = torch.eye(n_R, n_T, dtype=torch.complex64, device=device)
        case "average":
            _g = torch.Generator().manual_seed(seed + 999)
            if h0v is not None and not math.isinf(h0v):
                H_ws = _channel_fn(n_R, n_T, channel_type=ch_type, generator=_g).to(device)
            else:
                H_ws = torch.eye(n_R, n_T, dtype=torch.complex64, device=device)
        case _:
            gen = torch.Generator().manual_seed(int(ws_type))
            H_ws = _channel_fn(n_R, n_T, channel_type=ch_type, generator=gen).to(device)
    sim.optimize_alternating(A_target, H_ws, max_iters=ws_t, lr_tx=lr_tx, lr_rx=lr_rx)
    ws_phT = [p.data.clone() for p in sim.xi_T]
    ws_phR = [p.data.clone() for p in sim.xi_R]
    with torch.no_grad():
        for p in sim.xi_T:
            p.zero_()
        for p in sim.xi_R:
            p.zero_()
    return ws_phT, ws_phR


def _ws_cache_path(
    ws_type: str,
    L: int,
    M_int: int,
    tx_enc: str,
    rx_enc: str,
    ws_t: int,
    seed: int,
    h0v: float | None,
    ch_type: str,
) -> Path:
    enc_tag = f"{tx_enc.replace('/', '-')}_{rx_enc.replace('/', '-')}"
    arch = f"L{L}_M{M_int}_{enc_tag}_t{ws_t}"
    match ws_type:
        case "only_semantic":
            name = f"ws_only_semantic_{arch}.pt"
        case "average":
            h0_tag = (
                "rayleigh"
                if h0v is None or (isinstance(h0v, float) and math.isinf(h0v))
                else f"h0v{h0v}"
            )
            name = f"ws_average_{arch}_seed{seed}_{ch_type}_{h0_tag}.pt"
        case _:
            name = f"ws_seed{ws_type}_{arch}_{ch_type}.pt"
    return _WS_CACHE_ROOT / name


def _ensure_ws_accuracy(
    sim: JointSIM,
    ws_phT: list,
    ws_phR: list,
    A_target: torch.Tensor,
    ctx: dict,
    n_R: int,
    n_T: int,
    h0v: float | None,
    ch_type: str,
    ws_type: str,
    seed: int,
    ws_t: int,
    L: int,
    M_int: int,
    tx_enc: str,
    rx_enc: str,
    device: torch.device,
    n_total: int = 10000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    rice_factor_db: float | None = None,
    n_eval: int = 500,
) -> None:
    """Evaluate warm-start accuracy on the test set and cache to ws_accuracy.parquet.

    Called once per (ws_type, h0v, arch) config; skips if already present.
    Resets sim phases to zero after evaluation.
    """
    import numpy as np
    import polars as pl

    from semantic_alignment.data import HChannelDataModule
    from semantic_alignment.sim.core import _compute_beta
    from semantic_alignment.sim.unrolled.runner import _eval_accuracies
    from semantic_alignment.utils import standardize_channel

    acc_path = _WS_CACHE_ROOT / "ws_accuracy.parquet"

    _key = dict(
        warm_start_type=ws_type,
        h0_var=h0v,
        ch_type=ch_type,
        L=L,
        M_int=M_int,
        tx_enc=tx_enc,
        rx_enc=rx_enc,
        seed=seed,
        ws_t=ws_t,
    )

    existing: pl.DataFrame | None = None
    if acc_path.exists():
        existing = pl.read_parquet(acc_path)
        required_cols = set(_key.keys())
        if required_cols.issubset(set(existing.columns)):
            h0_expr = pl.col("h0_var").is_null() if h0v is None else (pl.col("h0_var") == h0v)
            already = existing.filter(
                (pl.col("warm_start_type") == ws_type)
                & h0_expr
                & (pl.col("ch_type") == ch_type)
                & (pl.col("L") == L)
                & (pl.col("M_int") == M_int)
                & (pl.col("seed") == seed)
                & (pl.col("ws_t") == ws_t)
            )
            if not already.is_empty():
                return

    dm = HChannelDataModule(
        n_R=n_R,
        n_T=n_T,
        n_total=n_total,
        batch_size=1,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed + 999,
        h0_var=h0v,
        channel_type=ch_type,
        rice_factor_db=rice_factor_db,
    )
    dm.setup()
    test_channels = dm.test_dataloader().dataset.tensors[0]
    N = min(n_eval, len(test_channels))

    with torch.no_grad():
        for j, p in enumerate(sim.xi_T):
            p.data.copy_(ws_phT[j]) if ws_phT else p.zero_()
        for j, p in enumerate(sim.xi_R):
            p.data.copy_(ws_phR[j]) if ws_phR else p.zero_()

    probe_accs = []
    with torch.no_grad():
        for i in range(N):
            H = standardize_channel(test_channels[i].to(device), channel_type=ch_type)
            Z_eff, _ = sim.get_effective_cascade(H)
            beta = _compute_beta(Z_eff, A_target)
            pa, _ = _eval_accuracies(
                Z_eff,
                beta,
                ctx["Z_test"],
                ctx["labels"],
                ctx["clf"],
                ctx["clf_neural"],
                ctx["L_out"],
                ctx["mu_out"],
                device,
            )
            probe_accs.append(pa)

    # Reset phases to zero so the caller's training loop starts clean
    with torch.no_grad():
        for p in sim.xi_T:
            p.zero_()
        for p in sim.xi_R:
            p.zero_()

    ws_acc = float(np.mean(probe_accs))
    new_row = pl.DataFrame([{**_key, "ws_acc_probe": ws_acc, "n_eval": N}])
    updated = (
        pl.concat([existing, new_row], how="diagonal_relaxed") if existing is not None else new_row
    )
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    updated.write_parquet(acc_path)
    print(f" WS accuracy cached: probe={ws_acc:.4f} (N={N})")


def _run_warm_start_cached(
    sim: JointSIM,
    A_target: torch.Tensor,
    n_R: int,
    n_T: int,
    ws_t: int,
    ws_type: str,
    h0v: float | None,
    ch_type: str,
    lr_tx: float,
    lr_rx: float,
    seed: int,
    device: torch.device,
    tx_enc: str,
    rx_enc: str,
    L: int,
    M_int: int,
) -> tuple[list, list]:
    """Like _run_warm_start but loads from / saves to disk cache."""
    if ws_t == 0:
        return [], []
    cache = _ws_cache_path(ws_type, L, M_int, tx_enc, rx_enc, ws_t, seed, h0v, ch_type)
    if cache.exists():
        ckpt = torch.load(cache, map_location=device, weights_only=True)
        ws_phT = [t.to(device) for t in ckpt["ws_phT"]]
        ws_phR = [t.to(device) for t in ckpt["ws_phR"]]
        print(f" Warm-start loaded from cache: {cache.name}")
        return ws_phT, ws_phR
    ws_phT, ws_phR = _run_warm_start(
        sim, A_target, n_R, n_T, ws_t, ws_type, h0v, ch_type, lr_tx, lr_rx, seed, device
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ws_phT": [t.cpu() for t in ws_phT], "ws_phR": [t.cpu() for t in ws_phR]}, cache)
    print(f" Warm-start cached → {cache.name}")
    return ws_phT, ws_phR

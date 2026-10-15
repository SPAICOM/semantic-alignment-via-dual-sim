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

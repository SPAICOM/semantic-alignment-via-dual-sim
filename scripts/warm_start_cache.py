"""Warm-start phase computation and disk cache.

Provides _run_warm_start (pure computation) and _run_warm_start_cached
(same, but loads/saves results to {task results}/unrolled/joint/warmstart/ —
results/classification/... by default). _ensure_ws_accuracy caches the
downstream-task metrics of the warm start, scored by a Task.

Cache key logic (always: dataset + alignment strategy + architecture + encoders + ws_t,
since the warm start fits A_target):
  only_semantic  → nothing else (H=I, no seed/h0_var)
  average        → + seed, h0_var, ch_type
  <int>          → + the int seed and ch_type
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import tqdm

from semantic_alignment.sim._cascade import build_cascade_RX, build_cascade_TX
from semantic_alignment.sim.core import _compute_beta
from semantic_alignment.sim.original._utils import RESULTS_DIR
from semantic_alignment.sim.original.joint import JointSIM
from semantic_alignment.utils import channel as _channel_fn

if TYPE_CHECKING:
    from semantic_alignment.evaluation.tasks import Task

_WS_CACHE_ROOT = RESULTS_DIR / "unrolled" / "joint" / "warmstart"


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
) -> tuple[list, list, dict]:
    """Run ws_t alternating-Adam steps; return (ws_phT, ws_phR, opt_state).

    opt_state = {"T": opt_T.state_dict(), "R": opt_R.state_dict()} — saved so that
    callers can warm-restart Adam instead of cold-starting it.
    """
    if ws_t == 0:
        return [], [], {}
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

    W_T = [getattr(sim, f"W_T_{i}") for i in range(sim.L_T)]
    W_R = [getattr(sim, f"W_R_{i}") for i in range(sim.L_R)]
    sizes_T = [p.shape[0] for p in sim.xi_T]
    sizes_R = [p.shape[0] for p in sim.xi_R]

    phases_T = [nn.Parameter(torch.zeros(s, device=device)) for s in sizes_T]
    phases_R = [nn.Parameter(torch.zeros(s, device=device)) for s in sizes_R]
    opt_T = torch.optim.Adam(phases_T, lr=lr_tx)
    opt_R = torch.optim.Adam(phases_R, lr=lr_rx)

    def _fwd():
        """Effective cascade ``G_R H_ws G_T`` for the current warm-start phases."""
        G_T = build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
        G_R = build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
        return G_R @ H_ws @ G_T

    # Mirror the default k_T=100, k_R=100 and best_tracking=True used by sim.optimize_alternating
    _k_T, _k_R = 100, 100
    best_loss = float("inf")
    best_T = [p.data.clone() for p in phases_T]
    best_R = [p.data.clone() for p in phases_R]
    for _ in tqdm(range(ws_t), desc=f"warm-start ({ws_type})", leave=False):
        for _ in range(_k_T):
            with torch.no_grad():
                beta_k = _compute_beta(_fwd(), A_target)
            opt_T.zero_grad()
            Z = _fwd()
            loss = torch.norm(beta_k * Z - A_target, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
            loss.backward()
            opt_T.step()

        loss_val = float("inf")
        for _ in range(_k_R):
            with torch.no_grad():
                beta_k = _compute_beta(_fwd(), A_target)
            opt_R.zero_grad()
            Z = _fwd()
            loss = torch.norm(beta_k * Z - A_target, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
            loss.backward()
            opt_R.step()
            loss_val = loss.item()

        if loss_val < best_loss:
            best_loss = loss_val
            best_T = [p.data.clone() for p in phases_T]
            best_R = [p.data.clone() for p in phases_R]

    ws_phT = best_T
    ws_phR = best_R
    opt_state = {"T": opt_T.state_dict(), "R": opt_R.state_dict()}

    # Keep sim phases in sync (used by _ensure_ws_accuracy)
    with torch.no_grad():
        for p, ph in zip(sim.xi_T, ws_phT):
            p.data.copy_(ph)
        for p, ph in zip(sim.xi_R, ws_phR):
            p.data.copy_(ph)
    return ws_phT, ws_phR, opt_state


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
    dataset: str,
    strategy: str,
    root: Path = _WS_CACHE_ROOT,
    prefix: str = "",
) -> Path:
    """Path of the warm-start cache file for one configuration.

    Parameters
    ----------
    ws_type : str
        Warm-start type (``only_semantic`` or ``average``).
    L, M_int : int
        SIM layer count and intermediate meta-atoms.
    tx_enc, rx_enc : str
        Encoder names (``/`` is replaced by ``-``).
    ws_t : int
        Warm-start iterations.
    seed : int
        Training seed.
    h0v : float or None
        Channel variance of the ``H_0`` used for ``average`` warm starts.
    ch_type : str
        Channel type.
    dataset, strategy : str
        Dataset name and alignment strategy.
    root : Path, optional
        Cache root directory.
    prefix : str, optional
        Optional filename prefix.

    Returns
    -------
    Path
        Location of the ``.pt`` cache file.
    """
    enc_tag = f"{tx_enc.replace('/', '-')}_{rx_enc.replace('/', '-')}"
    arch = f"{dataset}_{strategy}_L{L}_M{M_int}_{enc_tag}_t{ws_t}"
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
    return root / f"{prefix}{name}"


def _ensure_ws_accuracy(
    sim: JointSIM,
    ws_phT: list,
    ws_phR: list,
    A_target: torch.Tensor,
    task: "Task",
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
    dataset: str,
    strategy: str,
    n_total: int = 10000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    rice_factor_db: float | None = None,
    n_eval: int = 500,
    cache_root: Path = _WS_CACHE_ROOT,
    extra_key: dict | None = None,
) -> None:
    """Evaluate the warm start's task metrics on test channels; cache to ws_accuracy.parquet.

    ``task`` scores each test channel's cascade (``Task.evaluate_cascade``); the
    means of its ``warm_start_columns`` are stored. Called once per
    (ws_type, h0v, dataset, strategy, arch, *extra_key*) config; skips if already present.
    Resets sim phases to zero after evaluation.
    """
    import numpy as np
    import polars as pl

    from semantic_alignment.data import HChannelDataModule
    from semantic_alignment.sim.core import _compute_beta
    from semantic_alignment.utils import standardize_channel

    acc_path = cache_root / "ws_accuracy.parquet"
    extra_key = extra_key or {}

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
        dataset=dataset,
        strategy=strategy,
        **extra_key,
    )

    existing: pl.DataFrame | None = None
    if acc_path.exists():
        existing = pl.read_parquet(acc_path)
        required_cols = set(_key.keys())
        if required_cols.issubset(set(existing.columns)):
            h0_expr = pl.col("h0_var").is_null() if h0v is None else (pl.col("h0_var") == h0v)
            cond = (
                (pl.col("warm_start_type") == ws_type)
                & h0_expr
                & (pl.col("ch_type") == ch_type)
                & (pl.col("L") == L)
                & (pl.col("M_int") == M_int)
                & (pl.col("seed") == seed)
                & (pl.col("ws_t") == ws_t)
                & (pl.col("dataset") == dataset)
                & (pl.col("strategy") == strategy)
            )
            for col, val in extra_key.items():
                cond = cond & (pl.col(col) == val)
            already = existing.filter(cond)
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

    scores: dict[str, list] = {key: [] for key in task.warm_start_columns}
    with torch.no_grad():
        for i in range(N):
            H = standardize_channel(test_channels[i].to(device), channel_type=ch_type)
            Z_eff, _ = sim.get_effective_cascade(H)
            beta = _compute_beta(Z_eff, A_target)
            metrics = task.evaluate_cascade(Z_eff, beta)
            for key in scores:
                scores[key].append(metrics[key])

    # Reset phases to zero so the caller's training loop starts clean
    with torch.no_grad():
        for p in sim.xi_T:
            p.zero_()
        for p in sim.xi_R:
            p.zero_()

    ws_metrics = {col: float(np.mean(scores[key])) for key, col in task.warm_start_columns.items()}
    new_row = pl.DataFrame([{**_key, **ws_metrics, "n_eval": N}])
    updated = (
        pl.concat([existing, new_row], how="diagonal_relaxed") if existing is not None else new_row
    )
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    updated.write_parquet(acc_path)
    print(f" WS metrics cached: {', '.join(f'{k}={v:.4f}' for k, v in ws_metrics.items())} (N={N})")


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
    dataset: str,
    strategy: str,
    cache_root: Path = _WS_CACHE_ROOT,
    cache_prefix: str = "",
) -> tuple[list, list, dict]:
    """Like _run_warm_start but loads from / saves to disk cache.

    ``dataset`` / ``strategy`` are part of the cache key (the warm start fits
    ``A_target``); ``cache_root`` / ``cache_prefix`` place and name the cache file
    (defaults: the classification cache, no prefix).

    Returns (ws_phT, ws_phR, opt_state) where opt_state = {"T": ..., "R": ...}.
    Old caches without optimizer state are invalidated and recomputed.
    """
    if ws_t == 0:
        return [], [], {}
    cache = _ws_cache_path(
        ws_type,
        L,
        M_int,
        tx_enc,
        rx_enc,
        ws_t,
        seed,
        h0v,
        ch_type,
        dataset,
        strategy,
        cache_root,
        cache_prefix,
    )
    if cache.exists():
        ckpt = torch.load(cache, map_location=device, weights_only=True)
        if "ws_opt_T" in ckpt:
            ws_phT = [t.to(device) for t in ckpt["ws_phT"]]
            ws_phR = [t.to(device) for t in ckpt["ws_phR"]]
            opt_state = {"T": ckpt["ws_opt_T"], "R": ckpt["ws_opt_R"]}
            print(f" Warm-start loaded from cache: {cache.name}")
            return ws_phT, ws_phR, opt_state
        else:
            print(f" Cache missing optimizer state — recomputing: {cache.name}")
            cache.unlink()
    ws_phT, ws_phR, opt_state = _run_warm_start(
        sim, A_target, n_R, n_T, ws_t, ws_type, h0v, ch_type, lr_tx, lr_rx, seed, device
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ws_phT": [t.cpu() for t in ws_phT],
            "ws_phR": [t.cpu() for t in ws_phR],
            "ws_opt_T": opt_state["T"],
            "ws_opt_R": opt_state["R"],
        },
        cache,
    )
    print(f" Warm-start cached → {cache.name}")
    return ws_phT, ws_phR, opt_state

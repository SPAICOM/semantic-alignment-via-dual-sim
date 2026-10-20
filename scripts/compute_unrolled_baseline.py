"""Compute classic alternating-Adam baseline for unrolled comparison plots.

For each channel H in the HChannelDataModule test split, runs K_max Adam steps
and records probe/neural accuracy at every step.  Saves one parquet per test
channel so the job is resumable: existing files are skipped.

Output directory:
  results/unrolled/baseline/{ws_type}/{config_tag}/
    0000.parquet  ← test channel 0
    0001.parquet
    ...

Each parquet has one row with columns:
  test_id, probe_curve (list[float]), neural_curve (list[float]),
  warm_start_type, + metadata.

Usage:
  uv run python scripts/compute_unrolled_baseline.py \\
    datasets/channel=default training.gpu=0
    [training.warm_start_type=none|only_semantic|average]
"""

import math
import os
import sys
from pathlib import Path

import hydra
import lightning as lg
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from warm_start_cache import _run_warm_start_cached

from semantic_alignment.data import HChannelDataModule
from semantic_alignment.sim._cascade import build_cascade_RX, build_cascade_TX
from semantic_alignment.sim.core import _compute_beta
from semantic_alignment.sim.original._utils import ch_dims
from semantic_alignment.sim.original.joint import JointSIM
from semantic_alignment.sim.unrolled.runner import _eval_accuracies, _make_eval_fn
from semantic_alignment.utils import standardize_channel
from unrolled import _setup

torch.set_float32_matmul_precision("high")

_RESULTS = Path("results")
_BASELINE_ROOT = _RESULTS / "unrolled" / "baseline"


def _config_tag(cfg: DictConfig, ch_dim: int, K_max: int) -> str:
    dm = cfg.channel_dm
    h0v = dm.get("h0_var", None)
    match h0v:
        case 0:
            ch_tag = "det"
        case h if h is not None and not math.isinf(float(h)):
            ch_tag = f"rayleigh_h0var{h0v}"
        case _:
            ch_tag = "rayleigh"
    enc = f"{cfg.data.tx_enc}__{cfg.data.rx_enc}".replace("/", "-")
    return (
        f"{cfg.datasets.hf_name}_{cfg.alignment.type}"
        f"_L{cfg.sim.get('L', 25)}_M{cfg.sim.get('M_int', 32)}_ch{ch_dim}"
        f"_K{K_max}_seed{cfg.training.seed}_{ch_tag}_{enc}"
    )


@hydra.main(version_base=None, config_path="../config", config_name="config_unrolled_baseline")
def main(cfg: DictConfig) -> None:
    gpu = cfg.training.get("gpu", None)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    lg.seed_everything(cfg.training.seed, workers=True)
    ctx = _setup(cfg)
    device = ctx["device"]
    A_target = ctx["A_target"]
    lr_tx = ctx["lr_tx"]
    lr_rx = ctx["lr_rx"]

    L = cfg.sim.get("L", 25)
    M_int = cfg.sim.get("M_int", 32)
    slayer = cfg.sim.wavelength * cfg.sim.slayer_multiplier
    dx = cfg.sim.wavelength / cfg.sim.spacing_divisor
    ch_dim = cfg.sim.get("channel_dim", int(max(A_target.shape)))
    ch_x, ch_y = ch_dims(ch_dim)
    tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
    rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))
    n_R = n_T = ch_dim

    K_max = cfg.training.get("baseline_iters", 1000)
    ws_t = cfg.training.get("warm_start_t", 150)
    ws_type = str(cfg.training.get("warm_start_type", "none"))

    dm_cfg = cfg.channel_dm
    h0v = dm_cfg.get("h0_var", None)
    ch_type = dm_cfg.get("channel_type", "rayleigh")

    dm = HChannelDataModule(
        n_R=n_R,
        n_T=n_T,
        n_total=dm_cfg.get("n_total", 10000),
        batch_size=1,
        train_ratio=dm_cfg.get("train_ratio", 0.7),
        val_ratio=dm_cfg.get("val_ratio", 0.15),
        seed=cfg.training.seed + 999,
        h0_var=h0v,
        channel_type=ch_type,
        rice_factor_db=dm_cfg.get("rice_factor_db", None),
    )
    dm.setup()
    test_channels = dm.test_dataloader().dataset.tensors[0]  # (N_test, n_R, n_T) complex

    sim = JointSIM.build_model(
        L,
        M_int,
        cfg.sim.wavelength,
        dx,
        slayer * L,
        slayer * L,
        device,
        tx_in_x,
        tx_in_y,
        ch_x,
        ch_y,
        rx_out_x,
        rx_out_y,
    )

    # ── Warm-start phases (cached) ────────────────────────────────────────────
    if ws_type != "none" and ws_t > 0:
        ws_phT, ws_phR = _run_warm_start_cached(
            sim,
            A_target,
            n_R,
            n_T,
            ws_t,
            ws_type,
            h0v,
            ch_type,
            lr_tx,
            lr_rx,
            cfg.training.seed,
            device,
            tx_enc=ctx["tx_enc"],
            rx_enc=ctx["rx_enc"],
            L=L,
            M_int=M_int,
        )
    else:
        ws_phT, ws_phR = [], []
        ws_type = "none"

    full_oracle = cfg.training.get("full_oracle", True)

    # oracle is channel-distribution-agnostic — skip redundant h0_var≠null runs
    if full_oracle and h0v is not None:
        print(f" Skipping oracle run for h0_var={h0v} (oracle is distribution-agnostic).")
        return

    tag = _config_tag(cfg, ch_dim, K_max)
    baseline_type = "oracle" if full_oracle else "h0_transfer"
    out_dir = _BASELINE_ROOT / baseline_type / ws_type / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_limit = cfg.training.get("n_baselines", None)
    if n_limit is not None:
        test_channels = test_channels[: int(n_limit)]
    N_test = len(test_channels)
    print(
        f" Baseline [{baseline_type}]: {N_test} test channels, K_max={K_max}, warm_start={ws_type}"
    )
    print(f" Output → {out_dir}")

    meta = {
        "dataset": cfg.datasets.hf_name,
        "strategy": ctx["strategy_name"],
        "L": L,
        "M_int": f"{M_int}x{M_int}",
        "channel_dim": ch_dim,
        "tx_enc": ctx["tx_enc"],
        "rx_enc": ctx["rx_enc"],
        "seed": cfg.training.seed,
        "h0_var": h0v,
        "channel_type": ch_type,
        "warm_start_type": ws_type,
        "baseline_type": baseline_type,
        "K_max": K_max,
        "n_test": N_test,
    }

    if full_oracle:
        # ── Oracle: Adam on each test channel individually ────────────────────
        n_done = sum(1 for i in range(N_test) if (out_dir / f"{i:04d}.parquet").exists())
        for i, H_raw in enumerate(tqdm(test_channels, desc="oracle", initial=n_done, total=N_test)):
            out = out_dir / f"{i:04d}.parquet"
            if out.exists():
                continue

            H = standardize_channel(H_raw.to(device), channel_type=ch_type)

            with torch.no_grad():
                for j, p in enumerate(sim.xi_T):
                    p.data.copy_(ws_phT[j]) if ws_phT else p.zero_()
                for j, p in enumerate(sim.xi_R):
                    p.data.copy_(ws_phR[j]) if ws_phR else p.zero_()

            store: dict[str, list] = {"probe": [], "neural": []}
            sim.optimize_alternating(
                A_target,
                H,
                max_iters=K_max * 2,
                k_T=1,
                k_R=1,
                lr_tx=lr_tx,
                lr_rx=lr_rx,
                eval_fn=_make_eval_fn(
                    ctx["Z_test"],
                    ctx["labels"],
                    ctx["clf"],
                    ctx["clf_neural"],
                    ctx["L_out"],
                    ctx["mu_out"],
                    device,
                    store,
                ),
            )

            pl.DataFrame(
                [
                    {
                        **meta,
                        "test_id": i,
                        "probe_curve": store["probe"],
                        "neural_curve": store["neural"]
                        if store["neural"] and store["neural"][0] is not None
                        else None,
                    }
                ]
            ).write_parquet(out)

    else:
        # ── H_0-transfer: Adam on H_0, eval on h_var test channels ───────────
        out = out_dir / "curve.parquet"
        if out.exists():
            print(f" Already done: {out.name} — skipping.")
            print(" Done.")
            return

        # Build H_0: generate deterministic channel (h0_var=0, same seed)
        dm_h0 = HChannelDataModule(
            n_R=n_R,
            n_T=n_T,
            n_total=dm_cfg.get("n_total", 10000),
            batch_size=1,
            train_ratio=dm_cfg.get("train_ratio", 0.7),
            val_ratio=dm_cfg.get("val_ratio", 0.15),
            seed=cfg.training.seed + 999,
            h0_var=0,
            channel_type=ch_type,
        )
        dm_h0.setup()
        H0_raw = dm_h0.test_dataloader().dataset.tensors[0][0]
        H0 = standardize_channel(H0_raw.to(device), channel_type=ch_type)

        # Extract W matrices from sim (non-trainable buffers)
        W_T = [getattr(sim, f"W_T_{i}") for i in range(sim.L_T)]
        W_R = [getattr(sim, f"W_R_{i}") for i in range(sim.L_R)]
        sizes_T = [p.shape[0] for p in sim.xi_T]
        sizes_R = [p.shape[0] for p in sim.xi_R]

        # Init phases from warm-start or zeros (same as oracle)
        phases_T = [
            nn.Parameter(ws_phT[j].clone().to(device))
            if ws_phT
            else nn.Parameter(torch.zeros(s, device=device))
            for j, s in enumerate(sizes_T)
        ]
        phases_R = [
            nn.Parameter(ws_phR[j].clone().to(device))
            if ws_phR
            else nn.Parameter(torch.zeros(s, device=device))
            for j, s in enumerate(sizes_R)
        ]

        opt_T = torch.optim.Adam(phases_T, lr=lr_tx)
        opt_R = torch.optim.Adam(phases_R, lr=lr_rx)

        # Pre-standardize and stack test channels for batched eval
        test_channels_dev = [
            standardize_channel(H_raw.to(device), channel_type=ch_type) for H_raw in test_channels
        ]
        H_test_stack = torch.stack(test_channels_dev)  # (N_test, n_R, n_T)

        def _fwd():
            G_T = build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
            G_R = build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
            return G_R @ H0 @ G_T

        store: dict[str, list] = {"probe": [], "probe_std": [], "neural": [], "neural_std": []}

        print(f" Optimizing on H_0, evaluating on {N_test} test channels...")
        # Run K_max*2 alternating cycles (1 TX step + 1 RX step each), same as oracle
        for _ in tqdm(range(K_max * 2), desc="h0_transfer", leave=False):
            # TX step
            with torch.no_grad():
                beta_k = _compute_beta(_fwd(), A_target)
            opt_T.zero_grad()
            Z = _fwd()
            loss = torch.norm(beta_k * Z - A_target, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
            loss.backward()
            opt_T.step()

            # RX step
            with torch.no_grad():
                beta_k = _compute_beta(_fwd(), A_target)
            opt_R.zero_grad()
            Z = _fwd()
            loss = torch.norm(beta_k * Z - A_target, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
            loss.backward()
            opt_R.step()

            # Eval: apply current phases to all test channels (batched matmul)
            with torch.no_grad():
                G_T_cur = build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
                G_R_cur = build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
                # Z_batch: (N_test, d_rx, d_tx) — G_R @ H_i @ G_T for each test channel
                Z_batch = G_R_cur @ (H_test_stack @ G_T_cur)
                probe_accs, neural_accs = [], []
                for Z_eff_i in Z_batch:
                    beta_i = _compute_beta(Z_eff_i, A_target)
                    pa, na = _eval_accuracies(
                        Z_eff_i,
                        beta_i,
                        ctx["Z_test"],
                        ctx["labels"],
                        ctx["clf"],
                        ctx["clf_neural"],
                        ctx["L_out"],
                        ctx["mu_out"],
                        device,
                    )
                    probe_accs.append(pa)
                    if na is not None:
                        neural_accs.append(na)
            pa_arr = np.array(probe_accs)
            store["probe"].append(float(pa_arr.mean()))
            store["probe_std"].append(float(pa_arr.std()))
            if neural_accs:
                na_arr = np.array(neural_accs)
                store["neural"].append(float(na_arr.mean()))
                store["neural_std"].append(float(na_arr.std()))
            else:
                store["neural"].append(None)
                store["neural_std"].append(None)

        has_neural = store["neural"] and store["neural"][0] is not None
        pl.DataFrame(
            [
                {
                    **meta,
                    "test_id": -1,
                    "probe_curve": store["probe"],
                    "probe_std_curve": store["probe_std"],
                    "neural_curve": store["neural"] if has_neural else None,
                    "neural_std_curve": store["neural_std"] if has_neural else None,
                }
            ]
        ).write_parquet(out)

    print(" Done.")


if __name__ == "__main__":
    main()

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
import polars as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from warm_start_cache import _run_warm_start_cached

from semantic_alignment.data import HChannelDataModule
from semantic_alignment.sim.original._utils import ch_dims
from semantic_alignment.sim.original.joint import JointSIM
from semantic_alignment.sim.unrolled.runner import _make_eval_fn
from semantic_alignment.utils import standardize_channel
from unrolled import _setup

torch.set_float32_matmul_precision("high")

_RESULTS = Path("results")


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

    tag = _config_tag(cfg, ch_dim, K_max)
    out_dir = _RESULTS / "unrolled" / "baseline" / ws_type / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_limit = cfg.training.get("n_baselines", None)
    if n_limit is not None:
        test_channels = test_channels[: int(n_limit)]
    N_test = len(test_channels)
    print(f" Baseline: {N_test} test channels, K_max={K_max}, warm_start={ws_type}")
    print(f" Output → {out_dir}")

    n_done = sum(1 for i in range(N_test) if (out_dir / f"{i:04d}.parquet").exists())
    for i, H_raw in enumerate(tqdm(test_channels, desc="baseline", initial=n_done, total=N_test)):
        out = out_dir / f"{i:04d}.parquet"
        if out.exists():
            continue

        H = standardize_channel(H_raw.to(device), channel_type=ch_type)

        # Reset phases then apply warm-start init
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
                    "test_id": i,
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
                    "K_max": K_max,
                    "probe_curve": store["probe"],
                    "neural_curve": store["neural"]
                    if store["neural"] and store["neural"][0] is not None
                    else None,
                }
            ]
        ).write_parquet(out)

    print(" Done.")


if __name__ == "__main__":
    main()

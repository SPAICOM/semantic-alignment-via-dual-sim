"""Unrolled hparam search — independent training only.

For each (seed, warm_start_type):
  - _setup() once  (data loading, probe training)
  - warm start once (ws_t Adam steps on sim phases)
  - classic baseline once
  then for each weight_ratio:
    - train K=10 independent → save parquet

Total runs: n_seeds × n_ws_types × n_weight_ratios.
"""

import math
import os
import sys
from pathlib import Path

import hydra
import lightning as lg
import polars as pl
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, open_dict

sys.path.insert(0, str(Path(__file__).parent))
import wandb
from compare_unrolled import _setup
from semantic_alignment.data import HChannelDataModule
from semantic_alignment.sim.original._utils import ch_dims
from semantic_alignment.sim.original.joint import JointSIM
from semantic_alignment.sim.unrolled import DualSIMUnrolledSCA
from semantic_alignment.sim.unrolled.runner import (
    CompareUnrolledCallback,
    UnrolledLightning,
)
from semantic_alignment.utils import channel as _channel_fn
from semantic_alignment.utils import standardize_channel

torch.set_float32_matmul_precision("high")

_RESULTS = Path("results")

# ── Hparam grid ────────────────────────────────────────────────────────────────
SEEDS = [42]
WARM_START_TYPES = ["only_semantic", "average"]
WEIGHT_RATIOS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
LR_SCHEDULES = ["none", "cosine"]
H0VAR_EXTRA = [0, 0.5, math.inf]  # extra h0_var values: wr=1.0 only
K = 10
LR_W = 5e-4
COUP = "diagonal"
RESULT_SUBDIR = "unrolled_hparam/joint"
# ───────────────────────────────────────────────────────────────────────────────


def _out_path(
    hf_name: str,
    strategy: str,
    seed: int,
    L: int,
    M_int: int,
    ws_t: int,
    ws_type: str,
    h0v: float | None,
    training_type: str,
    weight_ratio: float,
    warmup_new_layers: int,
    lr_inherited_ratio: float,
    lr_schedule: str = "cosine",
) -> Path:
    ws_tag = "none" if ws_t == 0 else ws_type
    match h0v:
        case 0:
            ch_tag = "_det"
        case h if h is not None and not math.isinf(h):
            ch_tag = f"_rayleigh_h0var{h}"
        case _:
            ch_tag = "_rayleigh"
    tt_tag = "_prog" if training_type == "progressive" else ""
    wr_tag = "" if weight_ratio == 1.0 else f"_wr{weight_ratio}"
    wn_tag = "" if warmup_new_layers == 0 else f"_wn{warmup_new_layers}"
    li_tag = "" if lr_inherited_ratio == 1.0 else f"_li{lr_inherited_ratio}"
    sched_tag = f"_lrs{lr_schedule}"
    fname = (
        f"{hf_name}_unrolled_{strategy}_seed{seed}_L{L}_M{M_int}x{M_int}_K{K}"
        f"_ws{ws_t}_wsh{ws_tag}_lrw{LR_W}{ch_tag}{tt_tag}{wr_tag}{wn_tag}{li_tag}"
        f"{sched_tag}_{COUP}.parquet"
    )
    return _RESULTS / RESULT_SUBDIR / fname


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
    """Run ws_t Adam steps and return (ws_phT, ws_phR). Resets sim phases after."""
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
    sim.optimize_alternating(
        A_target,
        H_ws,
        max_iters=ws_t,
        lr_tx=lr_tx,
        lr_rx=lr_rx,
    )
    ws_phT = [p.data.clone() for p in sim.xi_T]
    ws_phR = [p.data.clone() for p in sim.xi_R]
    with torch.no_grad():
        for p in sim.xi_T:
            p.zero_()
        for p in sim.xi_R:
            p.zero_()
    return ws_phT, ws_phR


@hydra.main(version_base=None, config_path="../config", config_name="config_unrolled_hparam_base")
def main(cfg: DictConfig) -> None:
    gpu = cfg.training.get("gpu", None)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    L = cfg.sim.get("L", 5)
    M_int = cfg.sim.get("M_int", 32)
    ws_t = cfg.training.get("warm_start_t", 150)
    h0v = cfg.channel_dm.get("h0_var", None)
    ch_type = cfg.channel_dm.get("channel_type", "rayleigh")
    kdb = cfg.channel_dm.get("rice_factor_db", None)
    epochs = cfg.training.get("epochs", 150)
    n_channels = cfg.training.get("n_channels", 1000)
    batch_size = cfg.training.get("batch_size", 32)

    slayer = cfg.sim.wavelength * cfg.sim.slayer_multiplier
    dx = cfg.sim.wavelength / cfg.sim.spacing_divisor

    h0v_cases = [(h0v, WEIGHT_RATIOS)] + [(v, [1.0]) for v in H0VAR_EXTRA]
    total = (
        len(SEEDS)
        * len(WARM_START_TYPES)
        * len(LR_SCHEDULES)
        * sum(len(wrl) for _, wrl in h0v_cases)
    )
    print(f" Hparam search (independent only): {total} runs")

    for seed in SEEDS:
        with open_dict(cfg):
            cfg.training.seed = seed

        lg.seed_everything(seed, workers=True)
        ctx = _setup(cfg)
        device = ctx["device"]
        A_target = ctx["A_target"]
        lr_tx = ctx["lr_tx"]
        lr_rx = ctx["lr_rx"]
        strategy_name = ctx["strategy_name"]

        ch_dim = cfg.sim.get("channel_dim", int(max(A_target.shape)))
        ch_x, ch_y = ch_dims(ch_dim)
        tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
        rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))
        n_R = n_T = ch_dim

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
        H_std = standardize_channel(ctx["H_mimo"], channel_type=ch_type)
        lg.seed_everything(seed)

        for h0v_active, wr_list in h0v_cases:
            for ws_type in WARM_START_TYPES:
                print(
                    f"\n{'=' * 60}\n seed={seed} | h0v={h0v_active} | warm_start_type={ws_type}\n{'=' * 60}"
                )

                ws_phT, ws_phR = _run_warm_start(
                    sim,
                    A_target,
                    n_R,
                    n_T,
                    ws_t,
                    ws_type,
                    h0v_active,
                    ch_type,
                    lr_tx,
                    lr_rx,
                    seed,
                    device,
                )

                for weight_ratio in wr_list:
                    for lr_schedule in LR_SCHEDULES:
                        out = _out_path(
                            cfg.datasets.hf_name,
                            strategy_name,
                            seed,
                            L,
                            M_int,
                            ws_t,
                            ws_type,
                            h0v_active,
                            "independent",
                            weight_ratio,
                            0,
                            1.0,
                            lr_schedule=lr_schedule,
                        )
                        if out.exists():
                            print(f" Skip: {out.name}")
                            continue

                        print(
                            f" wr={weight_ratio} | sched={lr_schedule} | ws={ws_type} | seed={seed}"
                        )

                        supervision = (
                            "final"
                            if weight_ratio == 0.0
                            else cfg.training.get("supervision", "all")
                        )
                        lg.seed_everything(seed, workers=True)
                        model = DualSIMUnrolledSCA(
                            sim,
                            K,
                            coupling=COUP,
                            init_w=cfg.training.get("init_w", 0.1),
                            innovation_norm=cfg.training.get("innovation_norm", "rms"),
                            innovation_mode=cfg.training.get("innovation_mode", "autograd"),
                            momentum=cfg.training.get("momentum", True),
                            init_mu=cfg.training.get("init_mu", 0.9),
                            learn_S=cfg.training.get("learn_S", False),
                            phases_T_init=ws_phT if ws_phT else None,
                            phases_R_init=ws_phR if ws_phR else None,
                            gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
                        ).to(device)

                        dm = HChannelDataModule(
                            n_R=n_R,
                            n_T=n_T,
                            n_total=n_channels,
                            batch_size=batch_size,
                            train_ratio=cfg.channel_dm.get("train_ratio", 0.7),
                            val_ratio=cfg.channel_dm.get("val_ratio", 0.15),
                            seed=seed + 999,
                            h0_var=h0v_active,
                            channel_type=ch_type,
                            rice_factor_db=kdb,
                        )
                        dm.setup()

                        lightning = UnrolledLightning(
                            mode="joint",
                            A_target=A_target,
                            K=K,
                            unrolled_model=model,
                            lr_w=LR_W,
                            lr_schedule=lr_schedule,
                            grad_clip=cfg.training.get("grad_clip", 1.0),
                            epochs=epochs,
                            supervision=supervision,
                            weight_ratio=weight_ratio,
                            learn_S=cfg.training.get("learn_S", False),
                            reg_S=cfg.training.get("reg_S", 0.01),
                            K_prev=0,
                            lr_inherited_ratio=1.0,
                            warmup_new_layers=0,
                            Z_test=ctx["Z_test"],
                            labels=ctx["labels"],
                            W_probe=ctx["clf"],
                            clf_neural=ctx["clf_neural"],
                            L_out=ctx["L_out"],
                            mu_out=ctx["mu_out"],
                        )

                        if h0v_active is None:
                            h0v_tag = "rayleigh"
                        elif h0v_active == 0:
                            h0v_tag = "det"
                        elif math.isinf(h0v_active):
                            h0v_tag = "inf"
                        else:
                            h0v_tag = f"h0v{h0v_active}"
                        logger = WandbLogger(
                            project="unrolled_hparam",
                            name=f"K{K}_seed{seed}_{ws_type}_wr{weight_ratio}_lrs{lr_schedule}_{h0v_tag}",
                        )
                        trainer = Trainer(
                            max_epochs=epochs,
                            logger=logger,
                            accelerator="auto",
                            devices=1,
                            precision="bf16-mixed"
                            if cfg.training.get("bf16", False)
                            else "32-true",
                            log_every_n_steps=10,
                            gradient_clip_val=cfg.training.get("grad_clip", 1.0),
                            callbacks=[
                                CompareUnrolledCallback(
                                    H_classic=H_std,
                                    Z_test=ctx["Z_test"],
                                    labels=ctx["labels"],
                                    W_probe=ctx["clf"],
                                    clf_neural=ctx["clf_neural"],
                                    L_out=ctx["L_out"],
                                    mu_out=ctx["mu_out"],
                                ),
                                ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
                            ],
                        )

                        trainer.fit(lightning, dm)
                        trainer.test(lightning, dm, ckpt_path="best")
                        wandb.finish()

                        test_probe = lightning._test_probe_curves
                        test_neural = lightning._test_neural_curves

                        rows = [
                            {
                                "seed": seed,
                                "dataset": cfg.datasets.hf_name,
                                "strategy": strategy_name,
                                "M_int": f"{M_int}x{M_int}",
                                "L": L,
                                "K": K,
                                "tx_enc": ctx["tx_enc"],
                                "rx_enc": ctx["rx_enc"],
                                "channel_dim": ch_dim,
                                "warm_start_type": ws_type,
                                "training_type": "independent",
                                "weight_ratio": weight_ratio,
                                "lr_schedule": lr_schedule,
                                "variant": f"Unrolled ({COUP})",
                                "sample_id": i,
                                "probe_acc": float(test_probe[i][-1]),
                                "neural_acc": float(test_neural[i][-1]) if test_neural else None,
                            }
                            for i in range(len(test_probe))
                        ]

                        out.parent.mkdir(parents=True, exist_ok=True)
                        pl.DataFrame(rows).write_parquet(out)
                        print(f" Saved → {out.name}")

                        del model, lightning, trainer, dm
                        torch.cuda.empty_cache()

    print(" Done.")


if __name__ == "__main__":
    main()

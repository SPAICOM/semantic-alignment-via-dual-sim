"""Task-agnostic deep-unrolling pipeline shared by the ``unrolled_*.py`` entry points.

:func:`run_unrolled` builds the SIM(s) and unrolled model(s), trains the
unrolled step sizes on batches of random MIMO channels via
``UnrolledLightning``, and evaluates every unrolled step on the test channels.
The downstream metrics come from a
:class:`~semantic_alignment.evaluation.tasks.Task`:

- ``unrolled_classification.py``  — :class:`ClassificationTask` (probe/neural accuracy)
- ``unrolled_reconstruction.py``  — :class:`ReconstructionTask` (RX-decoder PSNR/MSE)

Each entry point only builds its task and the setup context, then calls
:func:`run_unrolled`.
"""

import math
from pathlib import Path

import polars as pl
import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from warm_start_cache import _ensure_ws_accuracy
from warm_start_cache import _run_warm_start_cached as _ws_cached

from semantic_alignment.data import HChannelDataModule
from semantic_alignment.evaluation.tasks import Task
from semantic_alignment.sim.original._utils import ch_dims, tipology_tag
from semantic_alignment.sim.original.disjoint_sym import DisjointSymSIM
from semantic_alignment.sim.original.joint import JointSIM
from semantic_alignment.sim.original.optimizers.monosim import (
    MonoSIMoptimizer,
    MonoSIMoptimizerTorch,
)
from semantic_alignment.sim.unrolled import (
    DisjointAsymUnrolledSCA,
    DisjointSymUnrolledSCA,
    DualSIMUnrolledSCA,
    MonoSIMUnrolledSCA,
)
from semantic_alignment.sim.unrolled.runner import (
    CompareUnrolledCallback,
    UnrolledLightning,
    _build_mono_sims,
)
from semantic_alignment.utils import channel as _channel_fn
from semantic_alignment.utils import standardize_channel

torch.set_float32_matmul_precision("high")


def _draw_channel(cfg: DictConfig, A_target: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Seed channel ``H_classic`` (``channel_dm`` model), used by the per-epoch eval."""
    ch_dim_cfg = cfg.sim.get("channel_dim", None)
    h_dim = ch_dim_cfg if ch_dim_cfg is not None else int(max(A_target.shape))
    return _channel_fn(
        h_dim,
        h_dim,
        channel_type=cfg.channel_dm.get("channel_type", "rayleigh"),
        rice_factor_db=cfg.channel_dm.get("rice_factor_db", None),
    ).to(device)


def _channel_tag(cfg: DictConfig) -> str:
    """Filename tag of the training-channel distribution (``channel_dm``)."""
    ch_type = cfg.channel_dm.get("channel_type", "rayleigh")
    kdb = cfg.channel_dm.get("rice_factor_db", None)
    h0v = cfg.channel_dm.get("h0_var", None)
    match (ch_type, h0v):
        case ("rician", _):
            return f"_rician{kdb}dB"
        case (_, 0):
            return "_det"
        case (_, h) if h is not None and not math.isinf(h):
            return f"_rayleigh_h0var{h}"
        case _:
            return "_rayleigh"


def _strategy_tag(cfg: DictConfig) -> str:
    """Strategy name for filenames and the warm-start cache key.

    PPFE clustered on the TX training set (``alignment.cluster_on=train_tx``)
    gets its own tag, so it never reuses results or warm starts computed with
    the pilot clustering; every other case keeps the plain strategy name.
    """
    strategy = cfg.alignment.type
    cluster_on = cfg.alignment.get("cluster_on", "pilots")
    return f"{strategy}-{cluster_on}" if strategy == "PPFE" and cluster_on != "pilots" else strategy


def _out_path(
    cfg: DictConfig, ctx: dict, task: Task, coup: str, K: int, lr_w: float | None = None
) -> Path:
    """``{result_root}/unrolled/{mode}[/{tipology}]/{task prefix}{dataset}_unrolled_….parquet``.

    ``result_root`` defaults to ``task.results_dir`` (overridable via ``training.result_root``).
    """
    results_root = Path(cfg.training.get("result_root", None) or task.results_dir)
    mode = cfg.sim.get("sim_type", "joint")
    tip = tipology_tag(mode)
    result_subdir = f"unrolled/{mode}/{tip}" if tip else f"unrolled/{mode}"
    L = cfg.sim.get("L", 25)
    M_int = cfg.sim.get("M_int", 32)
    ws_t = cfg.training.get("warm_start_t", 0)
    ws_type = str(cfg.training.get("warm_start_type", "identity"))
    ws_init_tag = "none" if ws_t == 0 else ws_type
    lr_w_tag = f"_lrw{lr_w}" if lr_w is not None else ""
    ch_tag = _channel_tag(cfg)
    tt = cfg.training.get("training_type", "independent")
    wr = cfg.training.get("weight_ratio", 1.0)
    wn = cfg.training.get("warmup_new_layers", 0)
    li = cfg.training.get("lr_inherited_ratio", 1.0)
    hparam_tag = ""
    if tt != "independent":
        hparam_tag += "_prog"
    if wr != 1.0:
        hparam_tag += f"_wr{wr}"
    if wn != 0:
        hparam_tag += f"_wn{wn}"
    if li != 1.0:
        hparam_tag += f"_li{li}"
    fname = (
        f"{task.fname_prefix}{ctx['dataset']}_unrolled_{_strategy_tag(cfg)}"
        f"_seed{cfg.training.seed}_L{L}_M{M_int}x{M_int}_K{K}"
        f"_ws{ws_t}_wsh{ws_init_tag}{lr_w_tag}{ch_tag}{hparam_tag}"
        f"_{coup}_{ctx['tx_enc']}_{ctx['rx_enc']}.parquet"
    )
    return results_root / result_subdir / fname


def _steps_path(
    cfg: DictConfig, ctx: dict, task: Task, coup: str, K: int, lr_w: float | None = None
) -> Path:
    """Companion .steps.pt file for progressive training (step state dict)."""
    return _out_path(cfg, ctx, task, coup, K, lr_w).with_suffix(".steps.pt")


def _final_metrics(task: Task, curves: dict, i: int) -> dict:
    """Final-step value of every task metric on test channel ``i`` (``None`` if unavailable)."""
    return {
        col: float(curves[key][i][-1]) if curves[key] else None
        for key, col in task.cascade_columns.items()
    }


def _metric_curves(task: Task, curves: dict) -> dict:
    """Per-channel, per-step curve of every task metric (``None`` if unavailable)."""
    return {col: curves[key] for key, col in task.cascade_columns.items()}


def _copy_progressive_init(
    model,
    state_T: dict,
    state_R: dict,
    K_prev: int,
    last_layer_init: bool = False,
) -> None:
    """Copy first K_prev layers from saved step states into model.step_T / step_R.

    All learnable parameters in UnrolledStep subclasses are indexed along dim 0
    as the K dimension, so slicing [:K_prev] copies exactly the trained layers.

    If last_layer_init=True, new layers [K_prev:] are initialised from the last
    trained layer (K_prev-1) instead of the fixed default (init_w, init_mu, s=1).
    """
    with torch.no_grad():
        for name, dst_p in model.step_T.named_parameters():
            if name in state_T:
                dst_p.data[:K_prev].copy_(state_T[name])
                if last_layer_init and dst_p.data.shape[0] > K_prev:
                    last = state_T[name][-1:]  # (1, ...)
                    dst_p.data[K_prev:].copy_(
                        last.expand(dst_p.data.shape[0] - K_prev, *last.shape[1:])
                    )
        for name, dst_p in model.step_R.named_parameters():
            if name in state_R:
                dst_p.data[:K_prev].copy_(state_R[name])
                if last_layer_init and dst_p.data.shape[0] > K_prev:
                    last = state_R[name][-1:]
                    dst_p.data[K_prev:].copy_(
                        last.expand(dst_p.data.shape[0] - K_prev, *last.shape[1:])
                    )


def run_unrolled(cfg: DictConfig, ctx: dict, task: Task) -> None:
    """Train and evaluate the unrolled SCA models for every coupling × K — any task.

    Parameters
    ----------
    cfg : DictConfig
        Composed config (``sim``, ``alignment``, ``training``, ``channel_dm``).
    ctx : dict
        Setup context: ``device``, ``A_target``, ``H_mimo`` (seed channel),
        ``strategy_name``, ``lr_tx``/``lr_rx``, ``tx_enc``/``rx_enc``,
        ``dataset``, ``F_T``/``F_R_H``; optionally ``ws_extra_key`` (extra
        warm-start metric-cache key columns). The warm-start caches are keyed on
        dataset + strategy and prefixed with ``task.fname_prefix``.
    task : Task
        Scores the per-step cascades (``evaluate_cascades``) and names the
        metric columns, results root and filename prefix.
    """
    mode = cfg.sim.get("sim_type", "joint")
    K_default = int(cfg.training.get("K", 30))
    K_list_cfg = list(cfg.training.get("K_list", []))
    _k_display = K_list_cfg if K_list_cfg else [K_default]
    print(
        f" seed={cfg.training.seed} | mode={mode} | "
        f"dataset={ctx['dataset']} | align={cfg.alignment.type}"
        f" | K={_k_display}"
    )

    ch_type = cfg.channel_dm.get("channel_type", "rayleigh")
    kdb = cfg.channel_dm.get("rice_factor_db", None)
    h0v = cfg.channel_dm.get("h0_var", None)

    device = ctx["device"]
    A_target = ctx["A_target"]
    H_mimo = ctx["H_mimo"]
    strategy_name = ctx["strategy_name"]
    lr_tx = ctx["lr_tx"]
    lr_rx = ctx["lr_rx"]
    K_default = int(cfg.training.get("K", 30))
    K_list_cfg = list(cfg.training.get("K_list", []))
    K_list = [int(k) for k in K_list_cfg] if K_list_cfg else [K_default]
    lr_w_cfg = cfg.training.get("lr_w", 1e-3)
    try:
        lr_w_list = [float(v) for v in lr_w_cfg]
        if len(lr_w_list) != len(K_list):
            raise ValueError(
                f"lr_w list length ({len(lr_w_list)}) must match K_list length ({len(K_list)})"
            )
    except TypeError:
        lr_w_list = [float(lr_w_cfg)] * len(K_list)

    lr_w_prog_cfg = cfg.training.get("lr_w_progressive", None)
    if lr_w_prog_cfg is None:
        lr_w_progressive_list = lr_w_list
    else:
        try:
            lr_w_progressive_list = [float(v) for v in lr_w_prog_cfg]
            if len(lr_w_progressive_list) != len(K_list):
                raise ValueError(
                    f"lr_w_progressive list length ({len(lr_w_progressive_list)}) must match K_list length ({len(K_list)})"
                )
        except TypeError:
            lr_w_progressive_list = [float(lr_w_prog_cfg)] * len(K_list)
    constructive = cfg.sim.get("constructive", False)
    ch_tag = _channel_tag(cfg)

    print(f" K_list={K_list}, lr_w_list={lr_w_list}, lr_w_progressive_list={lr_w_progressive_list}")
    slayer = cfg.sim.wavelength * cfg.sim.slayer_multiplier
    dx = cfg.sim.wavelength / cfg.sim.spacing_divisor
    L = cfg.sim.get("L", 25)
    M_int = cfg.sim.get("M_int", 32)
    channel_dim = cfg.sim.get("channel_dim", None)
    ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
    ch_x, ch_y = ch_dims(ch_dim)
    tx_in_x, tx_in_y = ch_dims(int(A_target.shape[1]))
    rx_out_x, rx_out_y = ch_dims(int(A_target.shape[0]))

    epochs = cfg.training.get("epochs", 30)
    coupling_cfg = cfg.training.get("coupling", ["diagonal"])
    coupling_list = [coupling_cfg] if isinstance(coupling_cfg, str) else list(coupling_cfg)

    # ---- Mode-specific build + classic baseline + training ----

    if mode == "joint":
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

        n_R, n_T = H_mimo.shape[0], H_mimo.shape[1]
        H_mimo = standardize_channel(H_mimo, channel_type=ch_type)
        # Pre-compute warm-start phases once for all coupling variants
        ws_t = cfg.training.get("warm_start_t", 0)
        ws_type = str(cfg.training.get("warm_start_type", "identity"))
        ws_init_tag = "none" if ws_t == 0 else ws_type
        ws_phT: list | None = None
        ws_phR: list | None = None
        ws_root = task.results_dir / "unrolled" / "joint" / "warmstart"
        if ws_t > 0:
            ws_phT, ws_phR, *_ = _ws_cached(
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
                dataset=ctx["dataset"],
                strategy=_strategy_tag(cfg),
                cache_root=ws_root,
                cache_prefix=task.fname_prefix,
            )
            _ensure_ws_accuracy(
                sim,
                ws_phT,
                ws_phR,
                A_target,
                task,
                n_R,
                n_T,
                h0v,
                ch_type,
                ws_type,
                cfg.training.seed,
                ws_t,
                L,
                M_int,
                ctx["tx_enc"],
                ctx["rx_enc"],
                device,
                n_total=cfg.channel_dm.get("n_total", 10000),
                train_ratio=cfg.channel_dm.get("train_ratio", 0.7),
                val_ratio=cfg.channel_dm.get("val_ratio", 0.15),
                rice_factor_db=cfg.channel_dm.get("rice_factor_db", None),
                dataset=ctx["dataset"],
                strategy=_strategy_tag(cfg),
                cache_root=ws_root,
                extra_key=ctx.get("ws_extra_key", None),
            )

        for coup in coupling_list:
            print(f"\n{'=' * 60}\n Joint coupling='{coup}'\n{'=' * 60}")

            conv_kwargs = {}
            if coup in ("conv", "conv2d"):
                conv_kwargs = dict(
                    shapes_T=[(M_int, M_int)] * (L - 1) + [(ch_x, ch_y)],
                    shapes_R=[(ch_x, ch_y)] + [(M_int, M_int)] * (L - 1),
                    k_spatial=cfg.training.get("k_spatial", 3),
                    k_layer=cfg.training.get("k_layer", 1),
                    mu_mode=cfg.training.get("mu_mode", "scalar"),
                    s_mode=cfg.training.get("s_mode", "scalar"),
                    k_spatial_s=cfg.training.get("k_spatial_s", None),
                    k_layer_s=cfg.training.get("k_layer_s", None),
                    reg_s_mode=cfg.training.get("reg_s_mode", "none"),
                )

            training_type = cfg.training.get("training_type", "independent")
            prev_step_state: dict | None = None
            prev_K: int | None = None

            for k_idx, K in enumerate(K_list):
                # Progressive steps (k_idx > 0) use lr_w_progressive_list if set.
                is_progressive_k = training_type == "progressive" and k_idx > 0
                lr_w = lr_w_progressive_list[k_idx] if is_progressive_k else lr_w_list[k_idx]
                out = _out_path(cfg, ctx, task, coup, K, lr_w)
                if out.exists():
                    print(f" Already done: {out.name} — skipping.")
                    if training_type == "progressive":
                        sp = _steps_path(cfg, ctx, task, coup, K, lr_w)
                        if sp.exists():
                            ckpt = torch.load(sp, map_location=device, weights_only=True)
                            prev_step_state = {"step_T": ckpt["step_T"], "step_R": ckpt["step_R"]}
                            prev_K = K
                    continue

                model = DualSIMUnrolledSCA(
                    sim,
                    K,
                    coupling=coup,
                    init_w=cfg.training.get("init_w", 0.1),
                    innovation_norm=cfg.training.get("innovation_norm", "rms"),
                    innovation_mode=cfg.training.get("innovation_mode", "autograd"),
                    momentum=cfg.training.get("momentum", True),
                    init_mu=cfg.training.get("init_mu", 0.9),
                    learn_S=cfg.training.get("learn_S", False),
                    phases_T_init=ws_phT,
                    phases_R_init=ws_phR,
                    gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
                    phi_init=cfg.training.get("phi_init", "zero"),
                    **conv_kwargs,
                ).to(device)

                if training_type == "progressive" and prev_step_state is not None:
                    lli = cfg.training.get("init_new_from_last", False)
                    _copy_progressive_init(
                        model,
                        prev_step_state["step_T"],
                        prev_step_state["step_R"],
                        prev_K,
                        last_layer_init=lli,
                    )
                    lli_tag = " (last-layer init)" if lli else ""
                    print(
                        f"  Progressive init: copied K={prev_K} → K={K} step parameters{lli_tag}."
                    )

                dm = HChannelDataModule(
                    n_R=n_R,
                    n_T=n_T,
                    n_total=cfg.channel_dm.get("n_total", 10000),
                    batch_size=cfg.training.get("batch_size", 32),
                    train_ratio=cfg.channel_dm.get("train_ratio", 0.7),
                    val_ratio=cfg.channel_dm.get("val_ratio", 0.15),
                    seed=cfg.training.seed + 999,
                    h0_var=h0v,
                    channel_type=ch_type,
                    rice_factor_db=kdb,
                )
                dm.setup()

                lightning = UnrolledLightning(
                    mode="joint",
                    A_target=A_target,
                    K=K,
                    unrolled_model=model,
                    lr_w=lr_w,
                    lr_schedule=cfg.training.get("lr_schedule", "cosine"),
                    grad_clip=cfg.training.get("grad_clip", 1.0),
                    epochs=epochs,
                    supervision=cfg.training.get("supervision", "final"),
                    weight_ratio=cfg.training.get("weight_ratio", 1.0),
                    learn_S=cfg.training.get("learn_S", False),
                    reg_S=cfg.training.get("reg_S", 0.01),
                    K_prev=prev_K if (training_type == "progressive" and prev_K is not None) else 0,
                    lr_inherited_ratio=cfg.training.get("lr_inherited_ratio", 1.0),
                    warmup_new_layers=cfg.training.get("warmup_new_layers", 0),
                    task=task,
                )

                eval_callback = CompareUnrolledCallback(H_classic=H_mimo, task=task)

                logger = WandbLogger(
                    project=cfg.training.get("wandb_project", "unrolled"),
                    name=f"joint_K{K}_L{L}_M{M_int}_{strategy_name}_{coup}{'_S' if cfg.training.get('learn_S', False) else ''}_ws{ws_t}_wsh{ws_init_tag}{ch_tag}",
                )

                trainer = Trainer(
                    max_epochs=epochs,
                    logger=logger,
                    accelerator="auto",
                    devices=1,
                    precision="bf16-mixed" if cfg.training.get("bf16", False) else "32-true",
                    log_every_n_steps=10,
                    gradient_clip_val=cfg.training.get("grad_clip", 1.0),
                    callbacks=[
                        eval_callback,
                        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
                    ],
                )

                trainer.fit(lightning, dm)
                trainer.test(lightning, dm, ckpt_path="best")
                wandb.finish()

                curves = lightning._test_curves

                init_tag = f"warmstart_{ws_t}_{ws_init_tag}" if ws_t > 0 else "none"
                meta = {
                    "seed": cfg.training.seed,
                    "dataset": ctx["dataset"],
                    "strategy": strategy_name,
                    "M_int": f"{M_int}x{M_int}",
                    "L": L,
                    "tx_enc": ctx["tx_enc"],
                    "rx_enc": ctx["rx_enc"],
                    "channel_dim": ch_dim,
                    "constructive": True,
                    "K": K,
                    "lr_w": lr_w,
                    "init": init_tag,
                    "warm_start_type": ws_type if ws_t > 0 else "none",
                    "channel_type": ch_type,
                    "rice_factor_db": kdb,
                    "h0_var": h0v,
                    **task.cascade_meta(),
                }
                n_channels = len(next(c for c in curves.values() if c))
                rows = [
                    {
                        **meta,
                        "variant": f"Unrolled ({coup})",
                        "sample_id": i,
                        **_final_metrics(task, curves, i),
                    }
                    for i in range(n_channels)
                ]

                out.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(rows).write_parquet(out)
                print(f"Parquet saved → {out}")

                if training_type == "progressive":
                    sp = _steps_path(cfg, ctx, task, coup, K, lr_w)
                    torch.save(
                        {
                            "step_T": {k: v.cpu() for k, v in model.step_T.state_dict().items()},
                            "step_R": {k: v.cpu() for k, v in model.step_R.state_dict().items()},
                        },
                        sp,
                    )
                    prev_step_state = {
                        "step_T": model.step_T.state_dict(),
                        "step_R": model.step_R.state_dict(),
                    }
                    prev_K = K

                del model
                torch.cuda.empty_cache()

    elif mode == "mono":
        mono_sim = MonoSIMoptimizerTorch(
            MonoSIMoptimizer(
                num_layers=L,
                num_meta_atoms_in_x=tx_in_x,
                num_meta_atoms_in_y=tx_in_y,
                num_meta_atoms_out_x=rx_out_x,
                num_meta_atoms_out_y=rx_out_y,
                num_meta_atoms_int_x=M_int,
                num_meta_atoms_int_y=M_int,
                thickness=slayer * L,
                wavelength=cfg.sim.wavelength,
                spacings={"in": dx, "out": dx, "int": dx},
            )
        ).to(device)

        n_R, n_T = H_mimo.shape[0], H_mimo.shape[1]

        for coup in coupling_list:
            print(f"\n{'=' * 60}\n Mono coupling='{coup}'\n{'=' * 60}")

            for K in K_list:
                out = _out_path(cfg, ctx, task, coup, K)
                if out.exists():
                    print(f" Already done: {out.name} — skipping.")
                    continue

                model = MonoSIMUnrolledSCA(
                    mono_sim,
                    K,
                    coupling=coup,
                    init_w=cfg.training.get("init_w", 0.1),
                    innovation_norm=cfg.training.get("innovation_norm", "rms"),
                    innovation_mode=cfg.training.get("innovation_mode", "autograd"),
                    momentum=cfg.training.get("momentum", True),
                    init_mu=cfg.training.get("init_mu", 0.9),
                    learn_S=cfg.training.get("learn_S", False),
                ).to(device)

                dm = HChannelDataModule(
                    n_R=n_R,
                    n_T=n_T,
                    n_total=cfg.channel_dm.get("n_total", 10000),
                    batch_size=cfg.training.get("batch_size", 32),
                    train_ratio=cfg.channel_dm.get("train_ratio", 0.7),
                    val_ratio=cfg.channel_dm.get("val_ratio", 0.15),
                    seed=cfg.training.seed + 999,
                    h0_var=h0v,
                    channel_type=ch_type,
                    rice_factor_db=kdb,
                )
                dm.setup()

                lightning = UnrolledLightning(
                    mode="mono",
                    A_target=A_target,
                    K=K,
                    unrolled_model=model,
                    constructive=constructive,
                    lr_w=cfg.training.get("lr_w", 1e-2),
                    lr_schedule=cfg.training.get("lr_schedule", "cosine"),
                    grad_clip=cfg.training.get("grad_clip", 1.0),
                    epochs=epochs,
                    supervision=cfg.training.get("supervision", "final"),
                    weight_ratio=cfg.training.get("weight_ratio", 1.0),
                    learn_S=cfg.training.get("learn_S", False),
                    reg_S=cfg.training.get("reg_S", 0.01),
                    task=task,
                )

                eval_callback = CompareUnrolledCallback(H_classic=H_mimo, task=task)

                logger = WandbLogger(
                    project=cfg.training.get("wandb_project", "unrolled"),
                    name=f"mono_K{K}_L{L}_M{M_int}_{strategy_name}_{coup}{'_S' if cfg.training.get('learn_S', False) else ''}{ch_tag}",
                )

                trainer = Trainer(
                    max_epochs=epochs,
                    logger=logger,
                    accelerator="auto",
                    devices=1,
                    precision="bf16-mixed" if cfg.training.get("bf16", False) else "32-true",
                    log_every_n_steps=10,
                    gradient_clip_val=cfg.training.get("grad_clip", 1.0),
                    callbacks=[
                        eval_callback,
                        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
                    ],
                )

                trainer.fit(lightning, dm)
                trainer.test(lightning, dm, ckpt_path="best")
                wandb.finish()

                curves = lightning._test_curves

                out.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(
                    [
                        {
                            "seed": cfg.training.seed,
                            "dataset": ctx["dataset"],
                            "strategy": strategy_name,
                            "M_int": f"{M_int}x{M_int}",
                            "L": L,
                            "tx_enc": ctx["tx_enc"],
                            "rx_enc": ctx["rx_enc"],
                            "channel_dim": ch_dim,
                            "constructive": constructive,
                            "variant": f"Unrolled ({coup})",
                            "K": K,
                            **_metric_curves(task, curves),
                            "channel_type": ch_type,
                            "rice_factor_db": kdb,
                            "h0_var": h0v,
                            **task.cascade_meta(),
                        }
                    ]
                ).write_parquet(out)
                print(f"Parquet saved → {out}")
                del model
                torch.cuda.empty_cache()

    elif mode in ("disjoint_sym", "disjoint_sym_c"):
        constructive = mode == "disjoint_sym_c"
        tx_sim, rx_sim = _build_mono_sims(
            L,
            M_int,
            tx_in_x,
            tx_in_y,
            ch_x,
            ch_y,
            rx_out_x,
            rx_out_y,
            cfg.sim.wavelength,
            slayer,
            dx,
            device,
        )
        n_R = n_T = ch_dim

        tx_target, rx_target = DisjointSymSIM.build_alignment_targets(
            A_target,
            ch_dim,
            strategy_name,
            device,
            F_T=ctx["F_T"],
            F_R_H=ctx["F_R_H"],
        )

        for coup in coupling_list:
            print(f"\n{'=' * 60}\n DisjointSym coupling='{coup}'\n{'=' * 60}")

            for K in K_list:
                out = _out_path(cfg, ctx, task, coup, K)
                if out.exists():
                    print(f" Already done: {out.name} — skipping.")
                    continue

                model = DisjointSymUnrolledSCA(
                    tx_sim,
                    rx_sim,
                    K,
                    constructive=constructive,
                    coupling=coup,
                    init_w=cfg.training.get("init_w", 0.1),
                    innovation_norm=cfg.training.get("innovation_norm", "rms"),
                    momentum=cfg.training.get("momentum", True),
                    init_mu=cfg.training.get("init_mu", 0.9),
                    learn_S=cfg.training.get("learn_S", False),
                ).to(device)

                dm = HChannelDataModule(
                    n_R=n_R,
                    n_T=n_T,
                    n_total=cfg.channel_dm.get("n_total", 10000),
                    batch_size=cfg.training.get("batch_size", 32),
                    train_ratio=cfg.channel_dm.get("train_ratio", 0.7),
                    val_ratio=cfg.channel_dm.get("val_ratio", 0.15),
                    seed=cfg.training.seed + 999,
                    precompute_svd=True,
                    channel_type=ch_type,
                    rice_factor_db=kdb,
                )
                dm.setup()

                lightning = UnrolledLightning(
                    mode="disjoint_sym",
                    A_target=A_target,
                    tx_target=tx_target,
                    rx_target=rx_target,
                    K=K,
                    unrolled_model=model,
                    constructive=constructive,
                    lr_w=cfg.training.get("lr_w", 1e-2),
                    lr_schedule=cfg.training.get("lr_schedule", "cosine"),
                    grad_clip=cfg.training.get("grad_clip", 1.0),
                    epochs=epochs,
                    supervision=cfg.training.get("supervision", "final"),
                    weight_ratio=cfg.training.get("weight_ratio", 1.0),
                    learn_S=cfg.training.get("learn_S", False),
                    reg_S=cfg.training.get("reg_S", 0.01),
                    task=task,
                )

                eval_callback = CompareUnrolledCallback(H_classic=H_mimo, task=task)

                logger = WandbLogger(
                    project=cfg.training.get("wandb_project", "unrolled"),
                    name=f"disjoint_sym_K{K}_L{L}_M{M_int}_{strategy_name}_{coup}{ch_tag}",
                )

                trainer = Trainer(
                    max_epochs=epochs,
                    logger=logger,
                    accelerator="auto",
                    devices=1,
                    precision="bf16-mixed" if cfg.training.get("bf16", False) else "32-true",
                    log_every_n_steps=10,
                    gradient_clip_val=cfg.training.get("grad_clip", 1.0),
                    callbacks=[
                        eval_callback,
                        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
                    ],
                )

                trainer.fit(lightning, dm)
                trainer.test(lightning, dm, ckpt_path="best")
                wandb.finish()

                curves = lightning._test_curves

                out.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(
                    [
                        {
                            "seed": cfg.training.seed,
                            "dataset": ctx["dataset"],
                            "strategy": strategy_name,
                            "M_int": f"{M_int}x{M_int}",
                            "L": L,
                            "tx_enc": ctx["tx_enc"],
                            "rx_enc": ctx["rx_enc"],
                            "channel_dim": ch_dim,
                            "constructive": constructive,
                            "variant": f"Unrolled ({coup})",
                            "K": K,
                            **_metric_curves(task, curves),
                            "channel_type": ch_type,
                            "rice_factor_db": kdb,
                            "h0_var": h0v,
                            **task.cascade_meta(),
                        }
                    ]
                ).write_parquet(out)
                print(f"Parquet saved → {out}")
                del model
                torch.cuda.empty_cache()

    elif mode == "disjoint_asym":
        alignment_side = cfg.sim.get("alignment_side", "tx")
        tx_sim, rx_sim = _build_mono_sims(
            L,
            M_int,
            tx_in_x,
            tx_in_y,
            ch_x,
            ch_y,
            rx_out_x,
            rx_out_y,
            cfg.sim.wavelength,
            slayer,
            dx,
            device,
        )
        n_R = n_T = ch_dim

        for coup in coupling_list:
            print(f"\n{'=' * 60}\n DisjointAsym coupling='{coup}'\n{'=' * 60}")

            for K in K_list:
                out = _out_path(cfg, ctx, task, coup, K)
                if out.exists():
                    print(f" Already done: {out.name} — skipping.")
                    continue

                model = DisjointAsymUnrolledSCA(
                    tx_sim,
                    rx_sim,
                    K,
                    constructive=constructive,
                    alignment_side=alignment_side,
                    coupling=coup,
                    init_w=cfg.training.get("init_w", 0.1),
                    innovation_norm=cfg.training.get("innovation_norm", "rms"),
                    momentum=cfg.training.get("momentum", True),
                    init_mu=cfg.training.get("init_mu", 0.9),
                    learn_S=cfg.training.get("learn_S", False),
                ).to(device)

                dm = HChannelDataModule(
                    n_R=n_R,
                    n_T=n_T,
                    n_total=cfg.channel_dm.get("n_total", 10000),
                    batch_size=cfg.training.get("batch_size", 32),
                    train_ratio=cfg.channel_dm.get("train_ratio", 0.7),
                    val_ratio=cfg.channel_dm.get("val_ratio", 0.15),
                    seed=cfg.training.seed + 999,
                    h0_var=h0v,
                    channel_type=ch_type,
                    rice_factor_db=kdb,
                )
                dm.setup()

                lightning = UnrolledLightning(
                    mode="disjoint_asym",
                    A_target=A_target,
                    K=K,
                    unrolled_model=model,
                    constructive=constructive,
                    lr_w=cfg.training.get("lr_w", 1e-2),
                    lr_schedule=cfg.training.get("lr_schedule", "cosine"),
                    grad_clip=cfg.training.get("grad_clip", 1.0),
                    epochs=epochs,
                    supervision=cfg.training.get("supervision", "final"),
                    weight_ratio=cfg.training.get("weight_ratio", 1.0),
                    learn_S=cfg.training.get("learn_S", False),
                    reg_S=cfg.training.get("reg_S", 0.01),
                    task=task,
                )

                eval_callback = CompareUnrolledCallback(H_classic=H_mimo, task=task)

                logger = WandbLogger(
                    project=cfg.training.get("wandb_project", "unrolled"),
                    name=f"disjoint_asym_K{K}_L{L}_M{M_int}_{strategy_name}_{coup}{ch_tag}",
                )

                trainer = Trainer(
                    max_epochs=epochs,
                    logger=logger,
                    accelerator="auto",
                    devices=1,
                    precision="bf16-mixed" if cfg.training.get("bf16", False) else "32-true",
                    log_every_n_steps=10,
                    gradient_clip_val=cfg.training.get("grad_clip", 1.0),
                    callbacks=[
                        eval_callback,
                        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
                    ],
                )

                trainer.fit(lightning, dm)
                trainer.test(lightning, dm, ckpt_path="best")
                wandb.finish()

                curves = lightning._test_curves

                out.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(
                    [
                        {
                            "seed": cfg.training.seed,
                            "dataset": ctx["dataset"],
                            "strategy": strategy_name,
                            "M_int": f"{M_int}x{M_int}",
                            "L": L,
                            "tx_enc": ctx["tx_enc"],
                            "rx_enc": ctx["rx_enc"],
                            "channel_dim": ch_dim,
                            "constructive": constructive,
                            "variant": f"Unrolled ({coup})",
                            "K": K,
                            **_metric_curves(task, curves),
                            "channel_type": ch_type,
                            "rice_factor_db": kdb,
                            "h0_var": h0v,
                            **task.cascade_meta(),
                        }
                    ]
                ).write_parquet(out)
                print(f"Parquet saved → {out}")
                del model
                torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unknown sim_type: {mode!r}")

    torch.cuda.empty_cache()
    print(" Done.")

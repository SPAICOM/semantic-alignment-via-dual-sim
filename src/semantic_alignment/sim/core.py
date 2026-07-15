"""Core SCA optimisation — single-channel, no batch dimension.

Canonical implementations shared by monosim, dualsim, and unrolled warm-start.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from ._cascade import build_cascade_RX, build_cascade_TX


def _compute_beta(Z, A):
    ip = torch.sum(torch.conj(Z) * A)
    return ip / (torch.sum(torch.conj(Z) * Z).real + 1e-12)


def _build_optimizer(params, lr, optimizer_type):
    if optimizer_type == "sgd":
        return optim.SGD(params, lr=lr)
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def _build_scheduler(opt, scheduler, n_iters, warmup_frac, lr):
    if scheduler != "cosine":
        return None
    warmup_iters = max(1, int(n_iters * warmup_frac))
    cosine_iters = max(1, n_iters - warmup_iters)
    constant = optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=warmup_iters)
    cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_iters, eta_min=lr * 1e-3)
    return optim.lr_scheduler.SequentialLR(
        opt, schedulers=[constant, cosine], milestones=[warmup_iters]
    )


def optimize_mono(
    W_list,
    sizes,
    A,
    K,
    lr,
    device,
    side="tx",
    left_matrix=None,
    right_matrix=None,
    phases_init=None,
    optimizer="adam",
    scheduler=None,
    warmup_frac=0.05,
    best_tracking=True,
    clip_value=None,
    eval_fn=None,
    show_progress=None,
    desc="mono",
):
    """K Adam steps on a single SIM cascade.

    Returns ``(phases, loss_history, beta_f)`` where phases are detached (m,) tensors.
    """
    build = build_cascade_TX if side == "tx" else build_cascade_RX

    if phases_init is not None:
        phases = [nn.Parameter(p.clone().to(device).detach()) for p in phases_init]
    else:
        phases = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes]

    L = left_matrix.to(device) if left_matrix is not None else None
    R = right_matrix.to(device) if right_matrix is not None else None

    def _fwd():
        G = build(W_list, [p.unsqueeze(0) for p in phases]).squeeze(0)
        if L is not None:
            G = L @ G
        if R is not None:
            G = G @ R
        return G

    opt = _build_optimizer(phases, lr, optimizer)
    sched = _build_scheduler(opt, scheduler, K, warmup_frac, lr)

    loss_history = []
    best_loss = float("inf")
    best_phases = [p.data.clone() for p in phases]

    _show = (eval_fn is not None) if show_progress is None else show_progress
    bar = tqdm(range(K), desc=desc, leave=False, disable=not _show, dynamic_ncols=True)
    for _ in bar:
        with torch.no_grad():
            beta_k = _compute_beta(_fwd(), A)
        opt.zero_grad()
        Z = _fwd()
        loss = torch.norm(beta_k * Z - A, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
        loss.backward()
        if clip_value is not None:
            nn.utils.clip_grad_norm_(phases, max_norm=clip_value)
        opt.step()
        if sched is not None:
            sched.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        if best_tracking and loss_val < best_loss:
            best_loss = loss_val
            best_phases = [p.data.clone() for p in phases]

        if _show:
            bar.set_postfix(loss=f"{loss_val:.4e}", beta=f"{beta_k.abs().item():.2e}")

        if eval_fn is not None:
            with torch.no_grad():
                Z_eff = _fwd()
                eval_fn(Z_eff, _compute_beta(Z_eff, A))
    bar.close()

    if best_tracking:
        with torch.no_grad():
            for p, best in zip(phases, best_phases):
                p.copy_(best)

    with torch.no_grad():
        beta_f = _compute_beta(_fwd(), A)

    return [p.detach() for p in phases], loss_history, beta_f.item()


def optimize_joint(
    W_T,
    W_R,
    sizes_T,
    sizes_R,
    H,
    A,
    n_cycles,
    lr_tx,
    lr_rx,
    device,
    phases_T_init=None,
    phases_R_init=None,
    k_T=1,
    k_R=1,
    optimizer="adam",
    scheduler=None,
    warmup_frac=0.05,
    best_tracking=True,
    clip_value=None,
    eval_fn=None,
    show_progress=None,
    desc="joint",
):
    """n_cycles alternating Adam cycles, each with k_T TX steps then k_R RX steps.

    Returns ``(phases_T, phases_R, loss_history, beta_f)``.
    """

    def _init(sizes, init):
        if init is not None:
            return [nn.Parameter(p.clone().to(device).detach()) for p in init]
        return [nn.Parameter(torch.zeros(m, device=device)) for m in sizes]

    phases_T = _init(sizes_T, phases_T_init)
    phases_R = _init(sizes_R, phases_R_init)

    def _fwd():
        G_T = build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
        G_R = build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
        return G_R @ H @ G_T

    opt_T = _build_optimizer(phases_T, lr_tx, optimizer)
    opt_R = _build_optimizer(phases_R, lr_rx, optimizer)
    sched_T = _build_scheduler(opt_T, scheduler, n_cycles, warmup_frac, lr_tx)
    sched_R = _build_scheduler(opt_R, scheduler, n_cycles, warmup_frac, lr_rx)

    loss_history = []
    best_loss = float("inf")
    best_T = [p.data.clone() for p in phases_T]
    best_R = [p.data.clone() for p in phases_R]
    loss_val = 0.0

    _show = (eval_fn is not None) if show_progress is None else show_progress
    bar = tqdm(range(n_cycles), desc=desc, leave=False, disable=not _show, dynamic_ncols=True)
    for _ in bar:
        for _ in range(k_T):
            with torch.no_grad():
                beta_k = _compute_beta(_fwd(), A)
            opt_T.zero_grad()
            Z = _fwd()
            loss = torch.norm(beta_k * Z - A, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
            loss.backward()
            if clip_value is not None:
                nn.utils.clip_grad_norm_(phases_T, max_norm=clip_value)
            opt_T.step()
            loss_val = loss.item()

        for _ in range(k_R):
            with torch.no_grad():
                beta_k = _compute_beta(_fwd(), A)
            opt_R.zero_grad()
            Z = _fwd()
            loss = torch.norm(beta_k * Z - A, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
            loss.backward()
            if clip_value is not None:
                nn.utils.clip_grad_norm_(phases_R, max_norm=clip_value)
            opt_R.step()
            loss_val = loss.item()

        if sched_T is not None:
            sched_T.step()
        if sched_R is not None:
            sched_R.step()

        loss_history.append(loss_val)
        if best_tracking and loss_val < best_loss:
            best_loss = loss_val
            best_T = [p.data.clone() for p in phases_T]
            best_R = [p.data.clone() for p in phases_R]

        if _show:
            bar.set_postfix(loss=f"{loss_val:.4e}")

        if eval_fn is not None:
            with torch.no_grad():
                Z_eff = _fwd()
                eval_fn(Z_eff, _compute_beta(Z_eff, A))
    bar.close()

    if best_tracking:
        with torch.no_grad():
            for p, best in zip(phases_T, best_T):
                p.copy_(best)
            for p, best in zip(phases_R, best_R):
                p.copy_(best)

    with torch.no_grad():
        beta_f = _compute_beta(_fwd(), A)

    return (
        [p.detach() for p in phases_T],
        [p.detach() for p in phases_R],
        loss_history,
        beta_f.item(),
    )


# ---------------------------------------------------------------------------
# Backward-compat thin wrappers (same API as the old sim/classic.py)
# ---------------------------------------------------------------------------


def _classic_joint(
    W_T, W_R, sizes_T, sizes_R, H, A, K, lr_tx, lr_rx, device, eval_fn=None, desc="classic joint"
):
    """K alternating Adam cycles (1 TX step + 1 RX step each).

    Returns ``(phases_T, phases_R)`` as lists of detached (m,) tensors.
    """
    phases_T, phases_R, _, _ = optimize_joint(
        W_T,
        W_R,
        sizes_T,
        sizes_R,
        H,
        A,
        n_cycles=K,
        lr_tx=lr_tx,
        lr_rx=lr_rx,
        device=device,
        k_T=1,
        k_R=1,
        best_tracking=False,
        eval_fn=eval_fn,
        desc=desc,
    )
    return phases_T, phases_R


def _classic_mono(W_list, sizes, H_left, A, K, lr, device, eval_fn=None, desc="classic mono"):
    """K Adam steps on a single SIM. H_left is the left-multiplying channel (None if non-constructive).

    Returns phases as list of detached (m,) tensors.
    """
    phases, _, _ = optimize_mono(
        W_list,
        sizes,
        A,
        K,
        lr,
        device,
        left_matrix=H_left,
        best_tracking=False,
        eval_fn=eval_fn,
        desc=desc,
    )
    return phases

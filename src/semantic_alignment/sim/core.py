"""Core SCA optimisation — single-channel, no batch dimension.

Canonical implementations shared by monosim, dualsim, and unrolled warm-start.
Provides closed-form MMSE scaling (``beta``) and alternating-gradient Adam
loops for both mono (single SIM) and joint (dual TX+RX SIM) configurations.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from ._cascade import build_cascade_RX, build_cascade_TX


def _compute_beta(Z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Compute the MMSE-optimal scalar scaling β = ⟨Z, A⟩ / ‖Z‖².

    β aligns the received signal Z to the target A in the least-squares sense:
    min_β ‖βZ - A‖²  ⟹  β* = ⟨Z, A⟩ / ‖Z‖².

    Parameters
    ----------
    Z : torch.Tensor
        Received signal matrix, shape (d_out, d_in).
    A : torch.Tensor
        Target signal matrix, same shape as Z.

    Returns
    -------
    torch.Tensor
        Scalar complex β (shape (1,) or ()).
    """
    # Inner product ⟨Z, A⟩ = Σ_ij Z*_ij A_ij
    ip = torch.sum(torch.conj(Z) * A)
    # ‖Z‖² = Σ_ij |Z_ij|² (real by construction)
    return ip / (torch.sum(torch.conj(Z) * Z).real + 1e-12)


def _build_optimizer(params: list[nn.Parameter], lr: float, optimizer_type: str) -> optim.Optimizer:
    """Instantiate an optimizer for the given phase parameters.

    Parameters
    ----------
    params : list[nn.Parameter]
        Phase logits to optimize.
    lr : float
        Learning rate.
    optimizer_type : str
        ``"sgd"`` for vanilla SGD, anything else defaults to Adam.

    Returns
    -------
    torch.optim.Optimizer
    """
    if optimizer_type == "sgd":
        return optim.SGD(params, lr=lr)
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def _build_scheduler(
    opt: optim.Optimizer,
    scheduler: str | None,
    n_iters: int,
    warmup_frac: float,
    lr: float,
) -> optim.lr_scheduler.LRScheduler | None:
    """Build a learning-rate scheduler with optional linear warmup + cosine decay.

    Parameters
    ----------
    opt : torch.optim.Optimizer
        The optimizer to attach the scheduler to.
    scheduler : str or None
        ``"cosine"`` for warmup + cosine annealing; ``None`` for constant LR.
    n_iters : int
        Total number of optimization iterations.
    warmup_frac : float
        Fraction of ``n_iters`` used for linear warmup (constant LR).
    lr : float
        Base learning rate; cosine decays to ``lr * 1e-3``.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler or None
    """
    if scheduler != "cosine":
        return None
    warmup_iters = max(1, int(n_iters * warmup_frac))
    cosine_iters = max(1, n_iters - warmup_iters)
    # Phase 1: constant LR for warmup_iters
    constant = optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=warmup_iters)
    # Phase 2: cosine annealing from lr down to lr * 1e-3
    cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_iters, eta_min=lr * 1e-3)
    return optim.lr_scheduler.SequentialLR(
        opt, schedulers=[constant, cosine], milestones=[warmup_iters]
    )


def optimize_mono(
    W_list: list[torch.Tensor],
    sizes: list[int],
    A: torch.Tensor,
    K: int,
    lr: float,
    device: torch.device,
    side: str = "tx",
    left_matrix: torch.Tensor | None = None,
    right_matrix: torch.Tensor | None = None,
    phases_init: list[torch.Tensor] | None = None,
    optimizer: str = "adam",
    scheduler: str | None = None,
    warmup_frac: float = 0.05,
    best_tracking: bool = True,
    clip_value: float | None = None,
    eval_fn: Callable | None = None,
    show_progress: bool | None = None,
    desc: str = "mono",
) -> tuple[list[torch.Tensor], list[float], float]:
    """K Adam steps on a single SIM cascade (mono-SIM optimization).

    Optimizes phase logits to minimize ‖β·G·A - A_target‖² where G is the
    SIM cascade (TX or RX) and β is the MMSE-optimal scaling.

    Parameters
    ----------
    W_list : list[Tensor]
        Linear transforms for each SIM layer.
    sizes : list[int]
        Number of phase parameters per layer.
    A : Tensor
        Target signal matrix, shape (d_out, d_in).
    K : int
        Number of Adam steps.
    lr : float
        Learning rate.
    device : torch.device
        Computation device.
    side : str
        ``"tx"`` for TX cascade, ``"rx"`` for RX cascade.
    left_matrix, right_matrix : Tensor or None
        Optional pre/post-multiplication matrices (e.g. channel matrix).
    phases_init : list[Tensor] or None
        Initial phase values; zeros if None.
    optimizer : str
        ``"adam"`` or ``"sgd"``.
    scheduler : str or None
        ``"cosine"`` for warmup + cosine decay.
    warmup_frac : float
        Fraction of K used for warmup.
    best_tracking : bool
        If True, return the best (lowest-loss) phases seen during optimization.
    clip_value : float or None
        Max gradient norm for clipping.
    eval_fn : callable or None
        Optional callback ``eval_fn(Z, beta)`` called after each step.
    show_progress : bool or None
        Show tqdm progress bar; defaults to True if eval_fn is provided.
    desc : str
        Progress bar description.

    Returns
    -------
    tuple[list[Tensor], list[float], float]
        ``(phases, loss_history, beta_f)`` where phases are detached (m,) tensors.
    """
    # Select cascade builder based on SIM side
    build = build_cascade_TX if side == "tx" else build_cascade_RX

    # Initialize phase logits (unconstrained real parameters)
    if phases_init is not None:
        phases = [nn.Parameter(p.clone().to(device).detach()) for p in phases_init]
    else:
        phases = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes]

    # Move optional pre/post matrices to device
    L = left_matrix.to(device) if left_matrix is not None else None
    R = right_matrix.to(device) if right_matrix is not None else None

    def _fwd():
        """Forward pass: compute cascade matrix G, optionally left/right multiply."""
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
        # Compute current β (detached — not part of the computation graph)
        with torch.no_grad():
            beta_k = _compute_beta(_fwd(), A)
        opt.zero_grad()
        Z = _fwd()
        # Loss: normalized Frobenius norm of βZ - A
        loss = torch.norm(beta_k * Z - A, p="fro") ** 2 / (Z.shape[0] * Z.shape[1])
        loss.backward()
        if clip_value is not None:
            nn.utils.clip_grad_norm_(phases, max_norm=clip_value)
        opt.step()
        if sched is not None:
            sched.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        # Track best phases across all iterations
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

    # Restore best phases if tracking was enabled
    if best_tracking:
        with torch.no_grad():
            for p, best in zip(phases, best_phases):
                p.copy_(best)

    with torch.no_grad():
        beta_f = _compute_beta(_fwd(), A)

    return [p.detach() for p in phases], loss_history, beta_f.item()


def optimize_joint(
    W_T: list[torch.Tensor],
    W_R: list[torch.Tensor],
    sizes_T: list[int],
    sizes_R: list[int],
    H: torch.Tensor,
    A: torch.Tensor,
    n_cycles: int,
    lr_tx: float,
    lr_rx: float,
    device: torch.device,
    phases_T_init: list[torch.Tensor] | None = None,
    phases_R_init: list[torch.Tensor] | None = None,
    k_T: int = 1,
    k_R: int = 1,
    optimizer: str = "adam",
    scheduler: str | None = None,
    warmup_frac: float = 0.05,
    best_tracking: bool = True,
    clip_value: float | None = None,
    eval_fn: Callable | None = None,
    show_progress: bool | None = None,
    desc: str = "joint",
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], float]:
    """Alternating Adam optimization for dual-SIM (TX + RX) joint phases.

    Performs ``n_cycles`` alternating updates: each cycle takes ``k_T`` TX steps
    followed by ``k_R`` RX steps. The forward model is:

        Z = G_R · H · G_T · A

    where G_T and G_R are the TX and RX cascades, and H is the channel matrix.

    Parameters
    ----------
    W_T, W_R : list[Tensor]
        Linear transforms for TX and RX SIM layers.
    sizes_T, sizes_R : list[int]
        Phase parameter counts per layer for TX and RX.
    H : Tensor
        Channel matrix, shape (d_rx, d_tx).
    A : Tensor
        Target signal matrix, shape (d_tx, d_in).
    n_cycles : int
        Number of alternating TX/RX cycles.
    lr_tx, lr_rx : float
        Learning rates for TX and RX phases.
    device : torch.device
        Computation device.
    phases_T_init, phases_R_init : list[Tensor] or None
        Initial phase values; zeros if None.
    k_T, k_R : int
        Number of TX/RX steps per cycle.
    optimizer : str
        ``"adam"`` or ``"sgd"``.
    scheduler : str or None
        ``"cosine"`` for warmup + cosine decay.
    warmup_frac : float
        Fraction of n_cycles used for warmup.
    best_tracking : bool
        If True, return the best phases seen during optimization.
    clip_value : float or None
        Max gradient norm for clipping.
    eval_fn : callable or None
        Optional callback ``eval_fn(Z, beta)`` called after each cycle.
    show_progress : bool or None
        Show tqdm progress bar.
    desc : str
        Progress bar description.

    Returns
    -------
    tuple[list[Tensor], list[Tensor], list[float], float]
        ``(phases_T, phases_R, loss_history, beta_f)``.
    """

    def _init(sizes, init):
        """Initialize phase logits from scratch or from a provided initial guess."""
        if init is not None:
            return [nn.Parameter(p.clone().to(device).detach()) for p in init]
        return [nn.Parameter(torch.zeros(m, device=device)) for m in sizes]

    phases_T = _init(sizes_T, phases_T_init)
    phases_R = _init(sizes_R, phases_R_init)

    def _fwd():
        """Forward pass: Z = G_R · H · G_T."""
        G_T = build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
        G_R = build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
        return G_R @ H @ G_T

    # Separate optimizers and schedulers for TX and RX
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
        # --- TX sub-cycle: k_T gradient steps on phases_T ---
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

        # --- RX sub-cycle: k_R gradient steps on phases_R ---
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

        # Step schedulers once per cycle
        if sched_T is not None:
            sched_T.step()
        if sched_R is not None:
            sched_R.step()

        loss_history.append(loss_val)
        # Track best phases across all cycles
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

    # Restore best phases if tracking was enabled
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
    W_T: list[torch.Tensor],
    W_R: list[torch.Tensor],
    sizes_T: list[int],
    sizes_R: list[int],
    H: torch.Tensor,
    A: torch.Tensor,
    K: int,
    lr_tx: float,
    lr_rx: float,
    device: torch.device,
    eval_fn: Callable | None = None,
    desc: str = "classic joint",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """K alternating Adam cycles (1 TX step + 1 RX step each).

    Convenience wrapper around ``optimize_joint`` with ``k_T=k_R=1``.

    Parameters
    ----------
    W_T, W_R : list[Tensor]
        Linear transforms for TX and RX SIM layers.
    sizes_T, sizes_R : list[int]
        Phase parameter counts per layer.
    H : Tensor
        Channel matrix.
    A : Tensor
        Target signal matrix.
    K : int
        Number of alternating cycles.
    lr_tx, lr_rx : float
        Learning rates for TX and RX.
    device : torch.device
        Computation device.
    eval_fn : callable or None
        Optional evaluation callback.
    desc : str
        Progress bar description.

    Returns
    -------
    tuple[list[Tensor], list[Tensor]]
        ``(phases_T, phases_R)`` as lists of detached (m,) tensors.
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


def _classic_mono(
    W_list: list[torch.Tensor],
    sizes: list[int],
    H_left: torch.Tensor | None,
    A: torch.Tensor,
    K: int,
    lr: float,
    device: torch.device,
    eval_fn: Callable | None = None,
    desc: str = "classic mono",
) -> list[torch.Tensor]:
    """K Adam steps on a single SIM.

    Convenience wrapper around ``optimize_mono`` with default settings.

    Parameters
    ----------
    W_list : list[Tensor]
        Linear transforms for each SIM layer.
    sizes : list[int]
        Phase parameter counts per layer.
    H_left : Tensor or None
        Left-multiplying channel matrix (None for non-constructive).
    A : Tensor
        Target signal matrix.
    K : int
        Number of Adam steps.
    lr : float
        Learning rate.
    device : torch.device
        Computation device.
    eval_fn : callable or None
        Optional evaluation callback.
    desc : str
        Progress bar description.

    Returns
    -------
    list[Tensor]
        Phases as list of detached (m,) tensors.
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

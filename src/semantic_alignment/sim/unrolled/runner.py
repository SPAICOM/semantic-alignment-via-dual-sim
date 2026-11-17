"""Comparison experiment — classic vs unrolled SCA, LightningModule version.

Provides ``UnrolledLightning`` that trains an unrolled SCA model on
batches of random MIMO channel matrices and compares its per-layer task
metrics against a static classic (Adam-on-phases) baseline computed on a
single seed-generated channel matrix.

Training only sees the alignment NMSE; the downstream metrics (accuracy for
classification, PSNR for reconstruction, …) come from a
:class:`~semantic_alignment.evaluation.tasks.Task`, which scores whole batches
of cascades through ``evaluate_cascades`` — so this module is task-agnostic.
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import lightning as lg
import matplotlib
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .._cascade import build_cascade_RX as _build_cascade_RX
from .._cascade import build_cascade_TX as _build_cascade_TX
from ..original.optimizers.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch

if TYPE_CHECKING:
    from semantic_alignment.evaluation.tasks import Task

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _precompute_test_data(
    test_loader: Iterable, A_target: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate all test z_tx and labels into single tensors.

    Returns
    -------
    Z_test : Tensor, shape (n_dim, N)
    labels : Tensor, shape (N,)
    """
    z_list, label_list = [], []
    for z, lbl in test_loader:
        z_list.append(z.to(device))
        label_list.append(lbl.to(device))
    Z_test = torch.cat(z_list, dim=0).T.contiguous()
    labels = torch.cat(label_list, dim=0)
    return Z_test, labels


def _compute_beta(Z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Compute the MMSE-optimal scalar scaling β = ⟨Z, A⟩ / ‖Z‖².

    Parameters
    ----------
    Z : Tensor
        Received signal matrix.
    A : Tensor
        Target alignment matrix.

    Returns
    -------
    torch.Tensor
        Scalar complex β.
    """
    ip = torch.sum(torch.conj(Z) * A)
    return ip / (torch.sum(torch.conj(Z) * Z).real + 1e-12)


def _make_eval_fn(task: "Task", store: dict[str, list]) -> Callable:
    """Return a callable ``fn(Z_eff, beta)`` that appends ``task``'s metrics to *store*."""

    def fn(Z_eff: torch.Tensor, beta: torch.Tensor) -> None:
        """Score ``β Z_eff`` and append each metric to its list in ``store``."""
        for key, val in task.evaluate_cascade(Z_eff, beta).items():
            store.setdefault(key, []).append(val)

    return fn


def _batch_beta(Z_eff: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Per-cascade MMSE scalar ``β_b = ⟨Z_b, A⟩ / ‖Z_b‖²`` for a batch ``(B, m, n)``."""
    ip = (Z_eff.conj() * A).sum((-2, -1))
    s = (Z_eff.conj() * Z_eff).sum((-2, -1)).real
    return ip / (s + 1e-12)


def _build_mono_sims(
    L: int,
    M_int: int,
    tx_in_x: int,
    tx_in_y: int,
    ch_x: int,
    ch_y: int,
    rx_out_x: int,
    rx_out_y: int,
    wavelength: float,
    slayer: int,
    dx: float,
    device: torch.device,
) -> tuple[MonoSIMoptimizerTorch, MonoSIMoptimizerTorch]:
    """Build TX and RX MonoSIMoptimizerTorch instances for disjoint mode."""
    tx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=tx_in_x,
        num_meta_atoms_in_y=tx_in_y,
        num_meta_atoms_out_x=ch_x,
        num_meta_atoms_out_y=ch_y,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
        spacings={"in": dx, "out": dx, "int": dx},
        side="tx",
    )
    rx_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=ch_x,
        num_meta_atoms_in_y=ch_y,
        num_meta_atoms_out_x=rx_out_x,
        num_meta_atoms_out_y=rx_out_y,
        num_meta_atoms_int_x=M_int,
        num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength,
        spacings={"in": dx, "out": dx, "int": dx},
        side="rx",
    )
    tx_sim = MonoSIMoptimizerTorch(tx_cpu).to(device)
    rx_sim = MonoSIMoptimizerTorch(rx_cpu).to(device)
    return tx_sim, rx_sim


# ---------------------------------------------------------------------------
# Classic baseline: Adam over SIM phases
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
    eval_fn: Callable,
    device: torch.device,
    steps_per_cycle: int = 1,
    desc: str = "classic joint",
) -> list[torch.Tensor]:
    """K outer cycles; each cycle runs TX steps then RX steps."""
    phases_T = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes_T]
    phases_R = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes_R]
    opt_T = torch.optim.Adam(phases_T, lr=lr_tx)
    opt_R = torch.optim.Adam(phases_R, lr=lr_rx)

    for _ in tqdm(range(K), desc=desc, leave=False):
        for _ in tqdm(range(steps_per_cycle), desc="TX", leave=False, disable=steps_per_cycle == 1):
            with torch.no_grad():
                G_T = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
                G_R = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
                beta_k = _compute_beta(G_R @ H @ G_T, A)
            opt_T.zero_grad()
            G_T = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
            G_R = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
            loss = torch.norm(beta_k * (G_R @ H @ G_T) - A, p="fro") ** 2 / (
                A.shape[0] * A.shape[1]
            )
            loss.backward()
            opt_T.step()

        for _ in tqdm(range(steps_per_cycle), desc="RX", leave=False, disable=steps_per_cycle == 1):
            with torch.no_grad():
                G_T = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
                G_R = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
                beta_k = _compute_beta(G_R @ H @ G_T, A)
            opt_R.zero_grad()
            G_T = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
            G_R = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
            loss = torch.norm(beta_k * (G_R @ H @ G_T) - A, p="fro") ** 2 / (
                A.shape[0] * A.shape[1]
            )
            loss.backward()
            opt_R.step()

        with torch.no_grad():
            G_T = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
            G_R = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
            Z_eff = G_R @ H @ G_T
            beta_opt = _compute_beta(Z_eff, A)
        eval_fn(Z_eff, beta_opt)

    return [p.detach() for p in phases_T + phases_R]


def _classic_mono(
    W_list: list[torch.Tensor],
    sizes: list[int],
    H: torch.Tensor,
    A: torch.Tensor,
    K: int,
    lr: float,
    constructive: bool,
    eval_fn: Callable,
    device: torch.device,
    desc: str = "classic mono",
) -> list[torch.Tensor]:
    """Run classic (non-unrolled) gradient-descent TX-only optimization on a fixed H.

    Parameters
    ----------
    W_list : list[Tuple[Tensor, Tensor]]
        Propagation matrices ``(W_in, W_out)`` for each SIM layer.
    sizes : list[int]
        Number of meta-atoms per layer.
    H : Tensor
        Channel matrix (used if constructive).
    A : Tensor
        Target alignment matrix.
    K : int
        Number of GD iterations.
    lr : float
        Adam learning rate.
    constructive : bool
        If True, pre-multiply G by H (constructive mode).
    eval_fn : callable
        Callback ``fn(Z_eff, beta)`` to record per-step accuracies.
    device : torch.device
        Target device.
    desc : str, optional
        Progress bar description.

    Returns
    -------
    list[Tensor]
        Optimized phase vectors, detached.
    """
    phases = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes]
    opt = torch.optim.Adam(phases, lr=lr)

    for _ in tqdm(range(K), desc=desc, leave=False):
        G = _build_cascade_TX(W_list, [p.unsqueeze(0) for p in phases]).squeeze(0)
        left_mat = H if constructive else None
        if left_mat is not None:
            G = left_mat @ G
        with torch.no_grad():
            beta_k = _compute_beta(G, A)
        opt.zero_grad()
        G = _build_cascade_TX(W_list, [p.unsqueeze(0) for p in phases]).squeeze(0)
        if left_mat is not None:
            G = left_mat @ G
        loss = torch.norm(beta_k * G - A, p="fro") ** 2 / (G.shape[0] * G.shape[1])
        loss.backward()
        opt.step()

        with torch.no_grad():
            G = _build_cascade_TX(W_list, [p.unsqueeze(0) for p in phases]).squeeze(0)
            if left_mat is not None:
                G = left_mat @ G
            beta_opt = _compute_beta(G, A)
        eval_fn(G, beta_opt)

    return [p.detach() for p in phases]


def _classic_disjoint(
    W_T: list[torch.Tensor],
    W_R: list[torch.Tensor],
    sizes_T: list[int],
    sizes_R: list[int],
    tx_target: torch.Tensor | None,
    rx_target: torch.Tensor | None,
    tx_left: torch.Tensor | None,
    tx_right: torch.Tensor | None,
    rx_left: torch.Tensor | None,
    rx_right: torch.Tensor | None,
    H: torch.Tensor,
    A: torch.Tensor,
    K: int,
    lr: float,
    eval_fn: Callable,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run classic (non-unrolled) disjoint gradient-descent optimization for TX and RX SIMs.

    Parameters
    ----------
    W_T : list[Tuple[Tensor, Tensor]]
        TX propagation matrices ``(W_in, W_out)`` for each layer.
    W_R : list[Tuple[Tensor, Tensor]]
        RX propagation matrices ``(W_in, W_out)`` for each layer.
    sizes_T : list[int]
        Number of TX meta-atoms per layer.
    sizes_R : list[int]
        Number of RX meta-atoms per layer.
    tx_target, rx_target : Tensor or None
        Per-SIM alignment targets (None for non-existent side).
    tx_left, tx_right, rx_left, rx_right : Tensor
        Boundary SVD factors for each side of the channel.
    H : Tensor
        Channel matrix.
    A : Tensor
        Joint target alignment matrix.
    K : int
        Number of GD iterations.
    lr : float
        Adam learning rate.
    eval_fn : callable
        Callback ``fn(Z_eff, beta)`` to record per-step accuracies.
    device : torch.device
        Target device.

    Returns
    -------
    list[Tensor]
        Optimized phase vectors (TX + RX concatenated), detached.
    """
    phases_T = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes_T]
    phases_R = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes_R]
    I_T = tx_target is not None
    I_R = rx_target is not None
    opt_T = torch.optim.Adam(phases_T, lr=lr) if I_T else None
    opt_R = torch.optim.Adam(phases_R, lr=lr) if I_R else None

    for _ in tqdm(range(K), desc="classic disjoint", leave=False):
        if I_T:
            G_T = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
            if tx_left is not None:
                G_T = tx_left @ G_T
            if tx_right is not None:
                G_T = G_T @ tx_right
            with torch.no_grad():
                beta_tx = _compute_beta(G_T, tx_target)
            opt_T.zero_grad()
            G_T = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
            if tx_left is not None:
                G_T = tx_left @ G_T
            if tx_right is not None:
                G_T = G_T @ tx_right
            loss = torch.norm(beta_tx * G_T - tx_target, p="fro") ** 2 / (
                tx_target.shape[0] * tx_target.shape[1]
            )
            loss.backward()
            opt_T.step()

        if I_R:
            G_R = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
            if rx_left is not None:
                G_R = rx_left @ G_R
            if rx_right is not None:
                G_R = G_R @ rx_right
            with torch.no_grad():
                beta_rx = _compute_beta(G_R, rx_target)
            opt_R.zero_grad()
            G_R = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
            if rx_left is not None:
                G_R = rx_left @ G_R
            if rx_right is not None:
                G_R = G_R @ rx_right
            loss = torch.norm(beta_rx * G_R - rx_target, p="fro") ** 2 / (
                rx_target.shape[0] * rx_target.shape[1]
            )
            loss.backward()
            opt_R.step()

        with torch.no_grad():
            G_T_f = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
            G_R_f = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
            Z_eff = G_R_f @ H @ G_T_f
            beta_opt = _compute_beta(Z_eff, A)
        eval_fn(Z_eff, beta_opt)

    return (
        [p.detach() for p in phases_T],
        [p.detach() for p in phases_R],
    )


# ---------------------------------------------------------------------------
# Unrolled evaluation on H_classic
# ---------------------------------------------------------------------------


def _unrolled_eval_curve_joint(
    unrolled: nn.Module,
    H: torch.Tensor,
    A: torch.Tensor,
    eval_fn: Callable,
    device: torch.device,
    k_max: int | None = None,
) -> list[torch.Tensor]:
    """Evaluate per-step probe accuracy for a trained DualSIMUnrolledSCA on a fixed H."""
    H = H.to(device)
    A = A.to(device)
    W_T = unrolled._get_W_T()
    W_R = unrolled._get_W_R()
    n_T = len(unrolled.sizes_T)

    def step_cb(k: int, phases: list[torch.Tensor]) -> None:
        """Rebuild ``G_R H G_T`` from the step-``k`` phases and score it with the optimal β."""
        with torch.no_grad():
            dev = phases[0].device
            phases_T = phases[:n_T]
            phases_R = phases[n_T:]
            G_T = _build_cascade_TX(W_T, phases_T).squeeze(0)
            G_R = _build_cascade_RX(W_R, phases_R).squeeze(0)
            Z_eff = G_R @ H.to(dev) @ G_T
            beta_opt = _compute_beta(Z_eff, A.to(dev))
        eval_fn(Z_eff, beta_opt)

    unrolled.eval()
    with torch.enable_grad():
        _, curve = unrolled.forward(H, A, B=1, step_callback=step_cb, k_max=k_max)
    return curve


def _unrolled_eval_curve_mono(
    unrolled: nn.Module,
    H: torch.Tensor,
    A: torch.Tensor,
    constructive: bool,
    eval_fn: Callable,
    device: torch.device,
    k_max: int | None = None,
) -> list[torch.Tensor]:
    """Evaluate per-step probe accuracy for a trained MonoSIMUnrolledSCA on a fixed H.

    Parameters
    ----------
    unrolled : MonoSIMUnrolledSCA
        Trained unrolled mono-SIM model.
    H : Tensor
        Fixed channel matrix.
    A : Tensor
        Target alignment matrix.
    constructive : bool
        If True, apply H before the cascade (constructive mode).
    eval_fn : callable
        Callback ``fn(Z_eff, beta)`` to record accuracies.
    device : torch.device
        Target device.
    k_max : int or None
        Override number of steps.

    Returns
    -------
    list[Tensor]
        Per-step NMSE curve.
    """
    H = H.to(device)
    A = A.to(device)
    W_list = unrolled._get_W()

    def step_cb(k: int, phases: list[torch.Tensor]) -> None:
        """Rebuild the mono-SIM cascade (with ``H`` if constructive) at step ``k`` and score it."""
        with torch.no_grad():
            dev = phases[0].device
            G = _build_cascade_TX(W_list, phases).squeeze(0)
            if constructive:
                G = H.to(dev) @ G
            beta_opt = _compute_beta(G, A.to(dev))
        eval_fn(G, beta_opt)

    unrolled.eval()
    left_mat = H if constructive else None
    with torch.enable_grad():
        _, curve = unrolled.forward(A, left_mat=left_mat, B=1, step_callback=step_cb, k_max=k_max)
    return curve


def _unrolled_eval_curve_disjoint(
    tx_model: nn.Module,
    rx_model: nn.Module,
    H: torch.Tensor,
    A: torch.Tensor,
    K: int,
    tx_target: torch.Tensor | None,
    rx_target: torch.Tensor | None,
    tx_left: torch.Tensor | None,
    tx_right: torch.Tensor | None,
    rx_left: torch.Tensor | None,
    rx_right: torch.Tensor | None,
    init_T: list[torch.Tensor],
    init_R: list[torch.Tensor],
    eval_fn: Callable,
    device: torch.device,
    k_max: int | None = None,
) -> list[float]:
    """Manual per-step loop for disjoint (two independent models)."""
    tx_phases = [p.clone() for p in init_T]
    rx_phases = [p.clone() for p in init_R]
    A2_T = torch.norm(tx_target) ** 2 if tx_target is not None else None
    A2_R = torch.norm(rx_target) ** 2 if rx_target is not None else None
    W_T = tx_model._get_W()
    W_R = rx_model._get_W()

    k_run = k_max if k_max is not None else K
    tx_model.eval()
    rx_model.eval()
    with torch.enable_grad():
        for k in range(k_run):
            if tx_target is not None:
                _, g_list = tx_model._innovation(
                    tx_phases, tx_target, A2_T, None, tx_left, tx_right, False
                )
                g_list = tx_model._normalize(g_list)
                if tx_model.momentum:
                    if k == 0:
                        tx_model._v_mom = [torch.zeros_like(g) for g in g_list]
                    mu_t = tx_model._mu_tensors(k)
                    tx_model._v_mom = [
                        mu * v + (1 - mu) * g for mu, v, g in zip(mu_t, tx_model._v_mom, g_list)
                    ]
                    g_list = [v / (1 - mu ** (k + 1)) for v, mu in zip(tx_model._v_mom, mu_t)]
                step = tx_model._step_tensors(k)
                tx_phases = [x + s * g for x, s, g in zip(tx_phases, step, g_list)]
                tx_phases = [x.detach() for x in tx_phases]
                if tx_model.momentum:
                    tx_model._v_mom = [v.detach() for v in tx_model._v_mom]

            if rx_target is not None:
                _, g_list = rx_model._innovation(
                    rx_phases, rx_target, A2_R, None, rx_left, rx_right, False
                )
                g_list = rx_model._normalize(g_list)
                if rx_model.momentum:
                    if k == 0:
                        rx_model._v_mom = [torch.zeros_like(g) for g in g_list]
                    mu_r = rx_model._mu_tensors(k)
                    rx_model._v_mom = [
                        mu * v + (1 - mu) * g for mu, v, g in zip(mu_r, rx_model._v_mom, g_list)
                    ]
                    g_list = [v / (1 - mu ** (k + 1)) for v, mu in zip(rx_model._v_mom, mu_r)]
                step = rx_model._step_tensors(k)
                rx_phases = [x + s * g for x, s, g in zip(rx_phases, step, g_list)]
                rx_phases = [x.detach() for x in rx_phases]
                if rx_model.momentum:
                    rx_model._v_mom = [v.detach() for v in rx_model._v_mom]

            with torch.no_grad():
                G_T = _build_cascade_TX(W_T, tx_phases).squeeze(0)
                G_R = _build_cascade_RX(W_R, rx_phases).squeeze(0)
                Z_eff = G_R @ H @ G_T
                beta_opt = _compute_beta(Z_eff, A)
            eval_fn(Z_eff, beta_opt)

    curve = []
    with torch.no_grad():
        G_T = _build_cascade_TX(W_T, tx_phases).squeeze(0)
        G_R = _build_cascade_RX(W_R, rx_phases).squeeze(0)
        Z_eff = G_R @ H @ G_T
        ns = torch.norm(beta_opt * Z_eff - A, p="fro") ** 2 / (A.shape[0] * A.shape[1])
        curve.append(ns.item())
    return curve


def _unrolled_eval_curve_disjoint_sym(
    model: nn.Module,
    H: torch.Tensor,
    A: torch.Tensor,
    tx_target: torch.Tensor,
    rx_target: torch.Tensor,
    eval_fn: Callable,
    device: torch.device,
    k_max: int | None = None,
) -> None:
    """Per-step probe accuracy for DisjointSymUnrolledSCA on a fixed H."""
    H = H.to(device)
    A = A.to(device)
    U, s, Vh = torch.linalg.svd(H.unsqueeze(0), full_matrices=False)
    sim_tx = model.sim_tx
    sim_rx = model.sim_rx

    def _step_cb(k: int, tx_phases: list[torch.Tensor], rx_phases: list[torch.Tensor]) -> None:
        """Rebuild ``G_R H G_T`` from the step-``k`` TX/RX phases and score it with the optimal β."""
        with torch.no_grad():
            G_T = sim_tx._cascade(tx_phases).squeeze(0)
            G_R = sim_rx._cascade(rx_phases).squeeze(0)
            Z_eff = G_R @ H @ G_T
            beta = _compute_beta(Z_eff, A)
        eval_fn(Z_eff, beta)

    model.eval()
    with torch.enable_grad():
        model.forward(
            U, s, Vh, tx_target, rx_target, create_graph=False, k_max=k_max, step_callback=_step_cb
        )


def _unrolled_eval_curve_disjoint_asym(
    model: nn.Module,
    H: torch.Tensor,
    A: torch.Tensor,
    eval_fn: Callable,
    device: torch.device,
    k_max: int | None = None,
) -> None:
    """Per-step probe accuracy for DisjointAsymUnrolledSCA on a fixed H."""
    H = H.to(device)
    A = A.to(device)
    H_batch = H.unsqueeze(0)
    sim_tx = model.sim_tx
    sim_rx = model.sim_rx

    def _step_cb(k: int, tx_phases: list[torch.Tensor], rx_phases: list[torch.Tensor]) -> None:
        """Rebuild ``G_R H G_T`` from the step-``k`` TX/RX phases and score it with the optimal β."""
        with torch.no_grad():
            G_T = sim_tx._cascade(tx_phases).squeeze(0)
            G_R = sim_rx._cascade(rx_phases).squeeze(0)
            Z_eff = G_R @ H @ G_T
            beta = _compute_beta(Z_eff, A)
        eval_fn(Z_eff, beta)

    model.eval()
    with torch.enable_grad():
        model.forward(H_batch, A, create_graph=False, k_max=k_max, step_callback=_step_cb)


# ---------------------------------------------------------------------------
# LightningModule  (training only)
# ---------------------------------------------------------------------------


class UnrolledLightning(lg.LightningModule):
    """Train unrolled SCA step sizes on batches of H.

    Parameters
    ----------
    mode : str
        ``'joint'`` or ``'mono'``.
    A_target : Tensor
        Alignment target matrix.
    K : int
        Number of unrolled layers.
    unrolled_model : DualSIMUnrolledSCA | MonoSIMUnrolledSCA
    constructive : bool
        Whether to include H in the loss (mono mode).
    supervision : str
        ``'all'`` (deep supervision) or ``'final'``.
    weight_ratio : float
        w[0]/w[K-1] ratio for deep supervision (ignored if supervision='final').
        1.0=uniform; 0.0=last-layer only. Converted to per-step gamma internally.
    lr_w : float
        Learning rate for unrolled step-size parameters.
    lr_schedule : str
        ``'none'`` or ``'cosine'``.
    grad_clip : float
        Gradient clipping max norm.
    epochs : int
        Number of epochs (used for scheduler fallback if trainer not set).
    learn_S : bool
        Whether to learn S values (TX/RX).
    reg_S : float
        Regularisation coefficient for S values.
    task : Task
        Downstream task scoring the per-step cascades in ``test_step``.
    """

    def __init__(
        self,
        mode: str,
        A_target: torch.Tensor,
        K: int,
        unrolled_model: nn.Module | None = None,
        constructive: bool = False,
        supervision: str = "final",
        weight_ratio: float = 1.0,
        lr_w: float = 1e-2,
        lr_schedule: str = "none",
        grad_clip: float = 1.0,
        epochs: int = 30,
        learn_S: bool = False,
        reg_S: float = 0.01,
        K_prev: int = 0,
        lr_inherited_ratio: float = 1.0,
        warmup_new_layers: int = 0,
        tx_target: torch.Tensor | None = None,
        rx_target: torch.Tensor | None = None,
        task: "Task | None" = None,
    ):
        """Initialize the Lightning training module.

        Parameters
        ----------
        mode : str
            Optimization mode: ``"joint"``, ``"mono"``, ``"disjoint_sym"``, or ``"disjoint_asym"``.
        A_target : Tensor
            Target alignment matrix.
        K : int
            Number of unrolled steps.
        unrolled_model : nn.Module or None
            The unrolled SCA model to train.
        constructive : bool
            If True, use constructive (full-CSI) mode.
        supervision : str
            ``"final"`` for last-step loss, ``"all"`` for deep supervision.
        weight_ratio : float
            Deep supervision weight ratio.
        lr_w : float
            Learning rate for step module parameters.
        lr_schedule : str
            ``"none"`` or ``"cosine"`` annealing.
        grad_clip : float
            Max gradient norm for clipping.
        epochs : int
            Number of training epochs.
        learn_S : bool
            Whether the model learns S scaling parameters.
        reg_S : float
            Regularization coefficient for S.
        tx_target, rx_target : Tensor or None
            Alignment targets for disjoint modes.
        task : Task or None
            Downstream task scoring the per-step cascades at test time.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "unrolled_model",
                "A_target",
                "tx_target",
                "rx_target",
                "task",
            ]
        )

        self.mode = mode
        self.A_target = A_target
        self.K = K
        self.lr_w = lr_w
        self.lr_schedule = lr_schedule
        self.grad_clip = grad_clip
        self.constructive = constructive
        self.supervision = supervision
        self.weight_ratio = weight_ratio
        self.epochs = epochs
        self.reg_S_coef = reg_S
        self.K_prev = K_prev
        self.lr_inherited_ratio = lr_inherited_ratio
        self.warmup_new_layers = warmup_new_layers

        self.unrolled_model = unrolled_model
        if K_prev > 0 and lr_inherited_ratio == 0.0 and hasattr(unrolled_model, "detach_before"):
            unrolled_model.detach_before = K_prev
        self.tx_target = tx_target
        self.rx_target = rx_target

        self.task = task

    # ---- optimiser ----

    def configure_optimizers(self) -> dict | torch.optim.Optimizer:
        """Configure Adam optimizer with optional cosine annealing schedule.

        Returns
        -------
        dict or torch.optim.Optimizer
            Optimizer dict with scheduler (if cosine), or bare optimizer.
        """
        opt = torch.optim.Adam(self.unrolled_model.parameters(), lr=self.lr_w)
        if self.lr_schedule == "cosine":
            T = self.trainer.estimated_stepping_batches if self.trainer else self.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=T,
                eta_min=self.lr_w * 1e-2,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return opt

    # ---- training ----

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Run one training step across all optimization modes.

        Dispatches to the appropriate unrolled model forward pass based on
        ``self.mode``, adds S-regularization, and logs metrics.

        Parameters
        ----------
        batch : tuple
            Batch data (format depends on mode).
        batch_idx : int
            Batch index (unused).

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        H_batch = batch[0]

        if self.mode == "joint":
            loss, _ = self.unrolled_model(
                H_batch,
                self.A_target,
                B=H_batch.shape[0],
                supervision=self.supervision,
                weight_ratio=self.weight_ratio,
            )
        elif self.mode == "mono":
            left_mat = H_batch if self.constructive else None
            loss, _ = self.unrolled_model(
                self.A_target,
                left_mat=left_mat,
                B=H_batch.shape[0],
                supervision=self.supervision,
                weight_ratio=self.weight_ratio,
            )
        elif self.mode == "disjoint_sym":
            H_batch, U_batch, s_batch, Vh_batch = batch
            loss, _, _ = self.unrolled_model(
                U_batch,
                s_batch,
                Vh_batch,
                self.tx_target,
                self.rx_target,
                supervision=self.supervision,
                weight_ratio=self.weight_ratio,
            )
        elif self.mode == "disjoint_asym":
            loss, _, _ = self.unrolled_model(
                H_batch,
                self.A_target,
                supervision=self.supervision,
                weight_ratio=self.weight_ratio,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        reg_val = self.unrolled_model.reg_S()
        if self.reg_S_coef > 0:
            loss = loss + self.reg_S_coef * reg_val

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/reg_S", reg_val, prog_bar=False)
        return loss

    def on_after_backward(self) -> None:
        """Clip gradients and apply progressive layer-wise gradient scaling."""
        if self.grad_clip > 0:
            for p in self.unrolled_model.parameters():
                if p.grad is not None:
                    nn.utils.clip_grad_norm_([p], max_norm=self.grad_clip)

        if self.K_prev > 0 and (self.lr_inherited_ratio != 1.0 or self.warmup_new_layers > 0):
            warmup_factor = (
                min(1.0, (self.current_epoch + 1) / self.warmup_new_layers)
                if self.warmup_new_layers > 0
                else 1.0
            )
            for param in self.unrolled_model.parameters():
                if param.grad is None or param.grad.shape[0] != self.K:
                    continue
                param.grad[: self.K_prev].mul_(self.lr_inherited_ratio)
                if warmup_factor < 1.0:
                    param.grad[self.K_prev :].mul_(warmup_factor)

    # ---- validation ----

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Run one validation step (same logic as training step, no gradient update).

        Parameters
        ----------
        batch : tuple
            Batch data (format depends on mode).
        batch_idx : int
            Batch index (unused).

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        H_batch = batch[0]

        if self.mode == "joint":
            loss, _ = self.unrolled_model(
                H_batch,
                self.A_target,
                B=H_batch.shape[0],
                supervision=self.supervision,
                weight_ratio=self.weight_ratio,
            )
        elif self.mode == "mono":
            left_mat = H_batch if self.constructive else None
            loss, _ = self.unrolled_model(
                self.A_target,
                left_mat=left_mat,
                B=H_batch.shape[0],
                supervision=self.supervision,
                weight_ratio=self.weight_ratio,
            )
        elif self.mode == "disjoint_sym":
            H_batch, U_batch, s_batch, Vh_batch = batch
            loss, _, _ = self.unrolled_model(
                U_batch,
                s_batch,
                Vh_batch,
                self.tx_target,
                self.rx_target,
                supervision=self.supervision,
                weight_ratio=self.weight_ratio,
            )
        elif self.mode == "disjoint_asym":
            loss, _, _ = self.unrolled_model(
                H_batch,
                self.A_target,
                supervision=self.supervision,
                weight_ratio=self.weight_ratio,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ---- test ----

    def on_test_start(self) -> None:
        """Initialize per-step metric accumulators for test evaluation."""
        self._test_curve = {key: [[] for _ in range(self.K)] for key in self.task.cascade_columns}

    def _record_step(self, k: int, Z_eff: torch.Tensor) -> None:
        """Score the batch of step-``k`` cascades and append to the test accumulators."""
        beta = _batch_beta(Z_eff, self.A_target)
        for key, vals in self.task.evaluate_cascades(Z_eff, beta).items():
            if vals is not None:
                self._test_curve[key][k].extend(vals)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Run one test step, recording the task metrics after every unrolled step.

        Uses ``torch.inference_mode(False)`` to allow autograd tracking
        for the step callback. Dispatches to mode-specific helpers for
        disjoint variants.
        """
        H_batch = batch[0].clone()  # clone outside inference_mode so autograd can track it
        B = H_batch.shape[0]

        def _eval_batch(k: int, phases: list[torch.Tensor]) -> None:
            """Step callback: rebuild the effective cascades from ``phases`` and record metrics."""
            if self.mode == "joint":
                n_T = len(self.unrolled_model.sizes_T)
                Z_eff = self.unrolled_model._cascade(H_batch, phases[:n_T], phases[n_T:])
            else:
                left_mat = H_batch if self.constructive else None
                Z_eff = self.unrolled_model._cascade(phases, left_mat)
            self._record_step(k, Z_eff)

        if self.mode == "joint":
            self.unrolled_model.forward(
                H_batch,
                self.A_target,
                B=B,
                step_callback=_eval_batch,
                k_max=self.K,
            )
        elif self.mode == "mono":
            left_mat = H_batch if self.constructive else None
            self.unrolled_model.forward(
                self.A_target,
                left_mat=left_mat,
                B=B,
                step_callback=_eval_batch,
                k_max=self.K,
            )
        elif self.mode == "disjoint_sym":
            self._test_step_disjoint_sym(batch)
        elif self.mode == "disjoint_asym":
            self._test_step_disjoint_asym(batch)

    def _test_step_disjoint_sym(self, batch: tuple) -> None:
        """Test step for symmetric disjoint mode — per-step metrics via SVD components.

        Parameters
        ----------
        batch : tuple
            ``(H_batch, U_batch, s_batch, Vh_batch)`` — batched SVD of H.
        """
        H_batch, U_batch, s_batch, Vh_batch = [b.clone() for b in batch]
        sim_tx = self.unrolled_model.sim_tx
        sim_rx = self.unrolled_model.sim_rx

        def _step_cb(k: int, tx_phases: list[torch.Tensor], rx_phases: list[torch.Tensor]) -> None:
            """Step callback: record metrics of ``G_R H G_T`` for the whole batch."""
            with torch.no_grad():
                G_T = sim_tx._cascade(tx_phases)
                G_R = sim_rx._cascade(rx_phases)
                self._record_step(k, G_R @ H_batch @ G_T)

        self.unrolled_model.forward(
            U_batch,
            s_batch,
            Vh_batch,
            self.tx_target,
            self.rx_target,
            create_graph=False,
            k_max=self.K,
            step_callback=_step_cb,
        )

    def _test_step_disjoint_asym(self, batch: tuple) -> None:
        """Test step for asymmetric disjoint mode — per-step metrics via raw H.

        Parameters
        ----------
        batch : tuple
            ``(H_batch,)`` — batch of channel matrices.
        """
        H_batch = batch[0].clone()
        sim_tx = self.unrolled_model.sim_tx
        sim_rx = self.unrolled_model.sim_rx

        def _step_cb(k: int, tx_phases: list[torch.Tensor], rx_phases: list[torch.Tensor]) -> None:
            """Step callback: record metrics of ``G_R H G_T`` for the whole batch."""
            with torch.no_grad():
                G_T = sim_tx._cascade(tx_phases)
                G_R = sim_rx._cascade(rx_phases)
                self._record_step(k, G_R @ H_batch @ G_T)

        self.unrolled_model.forward(
            H_batch,
            self.A_target,
            create_graph=False,
            k_max=self.K,
            step_callback=_step_cb,
        )


# ---------------------------------------------------------------------------
# Callback  (eval curves + wandb logging)
# ---------------------------------------------------------------------------


class CompareUnrolledCallback(lg.Callback):
    """Per-epoch evaluation of the task metrics and logging to wandb.

    Parameters
    ----------
    H_classic : Tensor
        Single seed channel matrix used for the per-epoch eval of non-joint modes.
    task : Task
        Downstream task scoring the cascades; metric ``key`` is logged as
        ``val/unrolled_final_{key}`` and ``test/unrolled_final_{key}``.
    """

    def __init__(self, H_classic: torch.Tensor, task: "Task"):
        """Initialize the comparison callback.

        Parameters
        ----------
        H_classic : Tensor
            Single seed channel matrix for the per-epoch eval of non-joint modes.
        task : Task
            Downstream task scoring the cascades.
        """
        super().__init__()
        self.H_classic = H_classic
        self.task = task

    # ---- eval curve ----

    def on_validation_epoch_end(self, trainer: lg.Trainer, pl_module: lg.LightningModule) -> None:
        """Evaluate the final-step task metrics and log to wandb.

        Joint mode: iterates over the val dataloader and averages the per-channel
        metrics at step K (same logic as test_step).
        Other modes: evaluates on H_classic as a single-channel proxy.
        """
        if trainer.sanity_checking:
            return

        device = pl_module.device

        if pl_module.mode == "joint":
            val_dl = trainer.val_dataloaders
            if isinstance(val_dl, (list, tuple)):
                val_dl = val_dl[0]
            if val_dl is None:
                return

            finals: dict[str, list[float]] = {key: [] for key in self.task.cascade_columns}
            n_T = len(pl_module.unrolled_model.sizes_T)
            K_last = pl_module.K - 1

            for batch in val_dl:
                H_batch = batch[0].to(device)
                B = H_batch.shape[0]

                def _step_cb(k: int, phases: list[torch.Tensor], _H=H_batch) -> None:
                    """Collect final-step metrics only; ``_H`` pins this batch's channels."""
                    if k != K_last:
                        return
                    Z_eff = pl_module.unrolled_model._cascade(_H, phases[:n_T], phases[n_T:])
                    beta = _batch_beta(Z_eff, pl_module.A_target)
                    for key, vals in self.task.evaluate_cascades(Z_eff, beta).items():
                        if vals is not None:
                            finals[key].extend(vals)

                with torch.no_grad():
                    pl_module.unrolled_model.forward(
                        H_batch,
                        pl_module.A_target,
                        B=B,
                        step_callback=_step_cb,
                        k_max=pl_module.K,
                    )

            for i, (key, vals) in enumerate(finals.items()):
                if vals:
                    pl_module.log(
                        f"val/unrolled_final_{key}", sum(vals) / len(vals), prog_bar=i == 0
                    )

        else:
            # Non-joint modes: single H_classic proxy (fast, avoids iterating the full val set)
            store: dict[str, list] = {}
            eval_fn = _make_eval_fn(self.task, store)
            if pl_module.mode == "mono":
                _unrolled_eval_curve_mono(
                    pl_module.unrolled_model,
                    self.H_classic,
                    pl_module.A_target,
                    pl_module.constructive,
                    eval_fn,
                    device,
                    k_max=pl_module.K,
                )
            elif pl_module.mode == "disjoint_sym":
                _unrolled_eval_curve_disjoint_sym(
                    pl_module.unrolled_model,
                    self.H_classic,
                    pl_module.A_target,
                    pl_module.tx_target,
                    pl_module.rx_target,
                    eval_fn,
                    device,
                    k_max=pl_module.K,
                )
            elif pl_module.mode == "disjoint_asym":
                _unrolled_eval_curve_disjoint_asym(
                    pl_module.unrolled_model,
                    self.H_classic,
                    pl_module.A_target,
                    eval_fn,
                    device,
                    k_max=pl_module.K,
                )
            for i, (key, vals) in enumerate(store.items()):
                if vals and vals[-1] is not None:
                    pl_module.log(f"val/unrolled_final_{key}", float(vals[-1]), prog_bar=i == 0)

        # Log learnable S values
        unrolled = pl_module.unrolled_model
        if hasattr(unrolled, "learn_S") and unrolled.learn_S:
            if hasattr(unrolled, "s_T"):
                pl_module.log("val/s_T_mean", unrolled.s_T.mean().item())
                pl_module.log("val/s_R_mean", unrolled.s_R.mean().item())
            if hasattr(unrolled, "s"):
                pl_module.log("val/s_mean", unrolled.s.mean().item())

    def on_test_epoch_end(self, trainer: lg.Trainer, pl_module: lg.LightningModule) -> None:
        """Aggregate per-channel test curves and log the mean final metrics.

        Transposes each per-step accumulator from (K, n_channels) to
        (n_channels, K) into ``pl_module._test_curves[key]`` (``None`` for a
        metric the task could not compute, e.g. no neural classifier).
        """
        curves: dict[str, list | None] = {}
        logs = {}
        for key, per_step in pl_module._test_curve.items():
            if not per_step or not per_step[0]:
                curves[key] = None
                continue
            # transpose (K, n_channels) → (n_channels, K)
            curves[key] = list(map(list, zip(*per_step)))
            mean = sum(c[-1] for c in curves[key]) / len(curves[key])
            pl_module.log(f"test/unrolled_final_{key}", mean)
            logs[f"test/unrolled_final_{key}"] = mean
        pl_module._test_curves = curves

        if trainer.logger is not None:
            trainer.logger.experiment.log(logs)

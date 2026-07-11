"""Comparison experiment — classic vs unrolled SCA, LightningModule version.

Provides ``UnrolledLightning`` that trains an unrolled SCA model on
batches of random MIMO channel matrices and compares its per-layer accuracy
against a static classic (Adam-on-phases) baseline computed on a single
seed-generated channel matrix.
"""

import lightning as lg
import matplotlib
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from semantic_alignment.evaluation.linear_probe import clf_accuracy, linear_probe_accuracy
from semantic_alignment.sim.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch
from semantic_alignment.sim.unrolled import _build_cascade_RX, _build_cascade_TX
from semantic_alignment.utils.utils import decompress_complex_tensor

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _precompute_test_data(test_loader, A_target, device):
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


def _compute_beta(Z, A):
    ip = torch.sum(torch.conj(Z) * A)
    return ip / (torch.sum(torch.conj(Z) * Z).real + 1e-12)


def _eval_accuracies(Z_eff, beta, Z_test, labels, W_probe, clf_neural, L_out, mu_out, device):
    dev = Z_eff.device
    cdtype = Z_eff.dtype
    Z_test = Z_test.to(device=dev, dtype=cdtype)
    labels = labels.to(dev)
    W_probe = W_probe.to(device=dev, dtype=torch.float32)
    L_out = L_out.to(device=dev, dtype=cdtype)
    mu_out = mu_out.to(device=dev, dtype=cdtype)
    if clf_neural is not None:
        clf_neural = clf_neural.to(dev)
    z_aligned = beta * (Z_eff @ Z_test)
    z_aligned = (L_out @ z_aligned) + mu_out
    z_aligned = decompress_complex_tensor(z_aligned, device).T
    probe_acc = linear_probe_accuracy(W_probe, z_aligned.float(), labels)
    neural_acc = None
    if clf_neural is not None:
        neural_acc = clf_accuracy(clf_neural, z_aligned.float(), labels)
    return probe_acc, neural_acc


def _make_eval_fn(Z_test, labels, W_probe, clf_neural, L_out, mu_out, device, store):
    """Return a callable ``fn(Z_eff, beta)`` that appends accuracies to *store*."""

    def fn(Z_eff, beta):
        pa, na = _eval_accuracies(
            Z_eff, beta, Z_test, labels, W_probe, clf_neural, L_out, mu_out, device
        )
        store["probe"].append(pa)
        store["neural"].append(na)

    return fn


def _build_mono_sims(
    L, M_int, tx_in_x, tx_in_y, ch_x, ch_y, rx_out_x, rx_out_y, wavelength, slayer, dx, device
):
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
    W_T,
    W_R,
    sizes_T,
    sizes_R,
    H,
    A,
    K,
    lr_tx,
    lr_rx,
    eval_fn,
    device,
    steps_per_cycle: int = 1,
    desc: str = "classic joint",
):
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
    W_list, sizes, H, A, K, lr, constructive, eval_fn, device, desc: str = "classic mono"
):
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
    W_T,
    W_R,
    sizes_T,
    sizes_R,
    target_tx,
    target_rx,
    tx_left,
    tx_right,
    rx_left,
    rx_right,
    H,
    A,
    K,
    lr,
    eval_fn,
    device,
):
    phases_T = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes_T]
    phases_R = [nn.Parameter(torch.zeros(m, device=device)) for m in sizes_R]
    I_T = target_tx is not None
    I_R = target_rx is not None
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
                beta_tx = _compute_beta(G_T, target_tx)
            opt_T.zero_grad()
            G_T = _build_cascade_TX(W_T, [p.unsqueeze(0) for p in phases_T]).squeeze(0)
            if tx_left is not None:
                G_T = tx_left @ G_T
            if tx_right is not None:
                G_T = G_T @ tx_right
            loss = torch.norm(beta_tx * G_T - target_tx, p="fro") ** 2 / (
                target_tx.shape[0] * target_tx.shape[1]
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
                beta_rx = _compute_beta(G_R, target_rx)
            opt_R.zero_grad()
            G_R = _build_cascade_RX(W_R, [p.unsqueeze(0) for p in phases_R]).squeeze(0)
            if rx_left is not None:
                G_R = rx_left @ G_R
            if rx_right is not None:
                G_R = G_R @ rx_right
            loss = torch.norm(beta_rx * G_R - target_rx, p="fro") ** 2 / (
                target_rx.shape[0] * target_rx.shape[1]
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


def _unrolled_eval_curve_joint(unrolled, H, A, eval_fn, device, k_max=None):
    """Evaluate per-step probe accuracy for a trained DualSIMUnrolledSCA on a fixed H."""
    H = H.to(device)
    A = A.to(device)
    W_T = unrolled._get_W_T()
    W_R = unrolled._get_W_R()
    n_T = len(unrolled.sizes_T)

    def step_cb(k, phases):
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


def _unrolled_eval_curve_mono(unrolled, H, A, constructive, eval_fn, device, k_max=None):
    H = H.to(device)
    A = A.to(device)
    W_list = unrolled._get_W()

    def step_cb(k, phases):
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
    tx_model,
    rx_model,
    H,
    A,
    K,
    target_tx,
    target_rx,
    tx_left,
    tx_right,
    rx_left,
    rx_right,
    init_T,
    init_R,
    eval_fn,
    device,
    k_max=None,
):
    """Manual per-step loop for disjoint (two independent models)."""
    tx_phases = [p.clone() for p in init_T]
    rx_phases = [p.clone() for p in init_R]
    A2_T = torch.norm(target_tx) ** 2 if target_tx is not None else None
    A2_R = torch.norm(target_rx) ** 2 if target_rx is not None else None
    W_T = tx_model._get_W()
    W_R = rx_model._get_W()

    k_run = k_max if k_max is not None else K
    tx_model.eval()
    rx_model.eval()
    with torch.enable_grad():
        for k in range(k_run):
            if target_tx is not None:
                _, g_list = tx_model._innovation(
                    tx_phases, target_tx, A2_T, None, tx_left, tx_right, False
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

            if target_rx is not None:
                _, g_list = rx_model._innovation(
                    rx_phases, target_rx, A2_R, None, rx_left, rx_right, False
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
    Z_test, labels, W_probe, clf_neural, L_out, mu_out : eval artifacts
        Passed to enable batched accuracy computation in test_step.
    """

    def __init__(
        self,
        mode: str,
        A_target: torch.Tensor,
        K: int,
        unrolled_model=None,
        constructive: bool = False,
        supervision: str = "final",
        weight_ratio: float = 1.0,
        lr_w: float = 1e-2,
        lr_schedule: str = "none",
        grad_clip: float = 1.0,
        epochs: int = 30,
        learn_S: bool = False,
        reg_S: float = 0.01,
        Z_test=None,
        labels=None,
        W_probe=None,
        clf_neural=None,
        L_out=None,
        mu_out=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "unrolled_model",
                "A_target",
                "Z_test",
                "labels",
                "W_probe",
                "clf_neural",
                "L_out",
                "mu_out",
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

        self.unrolled_model = unrolled_model

        self.Z_test = Z_test
        self.labels = labels
        self.W_probe = W_probe
        self.clf_neural = clf_neural
        self.L_out = L_out
        self.mu_out = mu_out

    # ---- optimiser ----

    def configure_optimizers(self):
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

    def training_step(self, batch, batch_idx):
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
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        reg_val = self.unrolled_model.reg_S()
        if self.reg_S_coef > 0:
            loss = loss + self.reg_S_coef * reg_val

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/reg_S", reg_val, prog_bar=False)
        return loss

    def on_after_backward(self):
        if self.grad_clip > 0:
            for p in self.unrolled_model.parameters():
                if p.grad is not None:
                    nn.utils.clip_grad_norm_([p], max_norm=self.grad_clip)

    # ---- validation ----

    def validation_step(self, batch, batch_idx):
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
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ---- test ----

    def on_test_start(self):
        self._test_probe_curve = [[] for _ in range(self.K)]
        self._test_neural_curve = [[] for _ in range(self.K)]

    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx):
        H_batch = batch[0].clone()  # clone outside inference_mode so autograd can track it
        B = H_batch.shape[0]

        def _eval_batch(k, phases):
            if self.mode == "joint":
                n_T = len(self.unrolled_model.sizes_T)
                Z_eff = self.unrolled_model._cascade(H_batch, phases[:n_T], phases[n_T:])
            else:
                left_mat = H_batch if self.constructive else None
                Z_eff = self.unrolled_model._cascade(phases, left_mat)

            ip = (Z_eff.conj() * self.A_target).sum((-2, -1))
            s = (Z_eff.conj() * Z_eff).sum((-2, -1)).real
            beta = ip / (s + 1e-12)

            z_aw = beta[:, None, None] * (Z_eff @ self.Z_test)
            z_a = self.L_out @ z_aw + self.mu_out
            z_real = decompress_complex_tensor(z_a, self.device).transpose(-1, -2).float()

            W = self.W_probe.to(z_real.device)
            labels = self.labels.to(z_real.device)
            preds = (z_real @ W).argmax(dim=-1)
            self._test_probe_curve[k].extend(
                (preds == labels[None, :]).float().mean(dim=-1).tolist()
            )

            if self.clf_neural is not None:
                dev = next(self.clf_neural.parameters()).device
                N_test = z_real.shape[1]
                z_flat = z_real.reshape(B * N_test, -1).to(dev)
                preds_n = self.clf_neural(z_flat).argmax(dim=-1).reshape(B, N_test)
                self._test_neural_curve[k].extend(
                    (preds_n == labels[None, :].to(dev)).float().mean(dim=-1).tolist()
                )

        if self.mode == "joint":
            self.unrolled_model.forward(
                H_batch,
                self.A_target,
                B=B,
                step_callback=_eval_batch,
                k_max=self.K,
            )
        else:
            left_mat = H_batch if self.constructive else None
            self.unrolled_model.forward(
                self.A_target,
                left_mat=left_mat,
                B=B,
                step_callback=_eval_batch,
                k_max=self.K,
            )


# ---------------------------------------------------------------------------
# Callback  (eval curves + wandb logging)
# ---------------------------------------------------------------------------


class CompareUnrolledCallback(lg.Callback):
    """Per-epoch evaluation on a fixed H and logging to wandb.

    Parameters
    ----------
    H_classic : Tensor
        Single seed channel matrix used for the classic baseline and per-epoch eval.
    Z_test : Tensor
        Precomputed test input matrix.
    labels : Tensor
        Test labels.
    W_probe : Tensor or None
        Linear probe weight matrix.
    clf_neural : nn.Module or None
        Neural classifier (optional).
    L_out : Tensor or None
        Output whitening lower-triangular factor.
    mu_out : Tensor or None
        Output whitening mean.
    classic_probe_curve : list or Tensor or None
        Pre-computed classic baseline probe accuracy (len K+1).
    classic_neural_curve : list or Tensor or None
        Pre-computed classic baseline neural accuracy (len K+1 or None).
    """

    def __init__(
        self,
        H_classic: torch.Tensor,
        Z_test: torch.Tensor,
        labels: torch.Tensor,
        W_probe=None,
        clf_neural=None,
        L_out=None,
        mu_out=None,
        classic_probe_curve=None,
        classic_neural_curve=None,
    ):
        super().__init__()
        self.H_classic = H_classic
        self.Z_test = Z_test
        self.labels = labels
        self.W_probe = W_probe
        self.clf_neural = clf_neural
        self.L_out = L_out
        self.mu_out = mu_out
        self.classic_probe_curve = classic_probe_curve
        self.classic_neural_curve = classic_neural_curve

    # ---- eval curve ----

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        store: dict[str, list] = {"probe": [], "neural": []}
        eval_fn = _make_eval_fn(
            self.Z_test,
            self.labels,
            self.W_probe,
            self.clf_neural,
            self.L_out,
            self.mu_out,
            pl_module.device,
            store,
        )

        if pl_module.mode == "joint":
            _unrolled_eval_curve_joint(
                pl_module.unrolled_model,
                self.H_classic,
                pl_module.A_target,
                eval_fn,
                pl_module.device,
                k_max=pl_module.K,
            )
        elif pl_module.mode == "mono":
            _unrolled_eval_curve_mono(
                pl_module.unrolled_model,
                self.H_classic,
                pl_module.A_target,
                pl_module.constructive,
                eval_fn,
                pl_module.device,
                k_max=pl_module.K,
            )

        probe_arr = torch.tensor(store["probe"], dtype=torch.float32, device=pl_module.device)
        neural_arr = (
            torch.tensor(store["neural"], dtype=torch.float32, device=pl_module.device)
            if store["neural"][0] is not None
            else None
        )

        pl_module.log("val/unrolled_final_probe", probe_arr[-1].item(), prog_bar=True)
        pl_module.log("val/classic_final_probe", self.classic_probe_curve[-1], prog_bar=False)
        if neural_arr is not None:
            pl_module.log("val/unrolled_final_neural", neural_arr[-1].item())

        # Log learnable S values
        unrolled = pl_module.unrolled_model
        if hasattr(unrolled, "learn_S") and unrolled.learn_S:
            if hasattr(unrolled, "s_T"):
                pl_module.log("val/s_T_mean", unrolled.s_T.mean().item())
                pl_module.log("val/s_R_mean", unrolled.s_R.mean().item())
            if hasattr(unrolled, "s"):
                pl_module.log("val/s_mean", unrolled.s.mean().item())

    def on_test_epoch_end(self, trainer, pl_module):
        # transpose (K, n_channels) → (n_channels, K)
        probe_curves = list(map(list, zip(*pl_module._test_probe_curve)))
        pl_module._test_probe_curves = probe_curves

        mean_probe = sum(c[-1] for c in probe_curves) / len(probe_curves)
        pl_module.log("test/unrolled_final_probe", mean_probe)
        logs = {"test/unrolled_final_probe": mean_probe}

        if pl_module._test_neural_curve and pl_module._test_neural_curve[0]:
            neural_curves = list(map(list, zip(*pl_module._test_neural_curve)))
            pl_module._test_neural_curves = neural_curves
            mean_neural = sum(c[-1] for c in neural_curves) / len(neural_curves)
            pl_module.log("test/unrolled_final_neural", mean_neural)
            logs["test/unrolled_final_neural"] = mean_neural
        else:
            pl_module._test_neural_curves = None

        if trainer.logger is not None:
            trainer.logger.experiment.log(logs)

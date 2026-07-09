"""Single Stacked Intelligent Metasurface (Mono-SIM) optimizer — NumPy and PyTorch versions.

A Mono-SIM is a single metasurface stack placed at either the TX or the RX side.
The signal flow and phase-shift layout differ between the two sides:

TX (``side="tx"``):
    G_T = Υ_L W_L … Υ_1 W_1
    Phase layout: [M_int] × (L-1) intermediate layers, then [N_out] at the output.

RX (``side="rx"``):
    G_R = W_{L-1} Υ_{L-1} … W_0 Υ_0
    Phase layout: [N_in] at the input aperture, then [M_int] × (L-1) intermediate.

The optimizer minimises ‖β G − A‖²_F (``optimize_sim``) or
‖β G_R H_eff − A‖²_F for RX-side training (``optimize_rx_sequential``).

Classes
-------
MonoSIMoptimizer      — NumPy reference: builds propagation matrices W.
MonoSIMoptimizerTorch — PyTorch wrapper with gradient-based phase optimisation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


class MonoSIMoptimizer:
    """Single-SIM optimizer — builds propagation matrices and holds phase shifts (NumPy).

    Parameters
    ----------
    num_layers : int
        Number of metasurface layers.
    num_meta_atoms_in_x : int
        Meta-atoms in the x-direction of the input aperture.
    num_meta_atoms_in_y : int
        Meta-atoms in the y-direction of the input aperture.
    num_meta_atoms_out_x : int
        Meta-atoms in the x-direction of the output aperture.
    num_meta_atoms_out_y : int
        Meta-atoms in the y-direction of the output aperture.
    num_meta_atoms_int_x : int
        Meta-atoms in the x-direction of intermediate layers.
    num_meta_atoms_int_y : int
        Meta-atoms in the y-direction of intermediate layers.
    thickness : float
        Total physical thickness of the SIM stack (metres).
    wavelength : float
        Operating wavelength λ (metres).
    verbose : bool, optional
        If ``True``, print construction details.
    spacings : dict, optional
        Meta-atom spacings for each aperture type.  Defaults to ``λ/2`` for all.
    side : str, optional
        ``"tx"`` (default) or ``"rx"``.  Controls phase-shift layout and cascade order.
    """

    def __init__(
        self,
        num_layers: int,
        num_meta_atoms_in_x: int,
        num_meta_atoms_in_y: int,
        num_meta_atoms_out_x: int,
        num_meta_atoms_out_y: int,
        num_meta_atoms_int_x: int,
        num_meta_atoms_int_y: int,
        thickness: float,
        wavelength: float,
        verbose: bool = False,
        spacings: dict = None,
        side: str = "tx",
    ):
        if num_layers <= 0:
            raise ValueError("Number of layers must be positive.")
        if side not in ("tx", "rx"):
            raise ValueError(f"side must be 'tx' or 'rx', got {side!r}.")

        self.L = num_layers
        self.N_in = num_meta_atoms_in_x * num_meta_atoms_in_y
        self.N_out = num_meta_atoms_out_x * num_meta_atoms_out_y
        self.M_int = num_meta_atoms_int_x * num_meta_atoms_int_y
        self.slayer = thickness / num_layers
        self.side = side

        self.wavelength = wavelength
        self.kappa = 2 * np.pi / wavelength
        self.verbose = verbose

        self.spacings = spacings or {
            "in": 0.5 * wavelength,
            "out": 0.5 * wavelength,
            "int": 0.5 * wavelength,
        }

        std_dev = 0.01
        if side == "rx":
            # RX: first phase at input aperture (N_in), rest at intermediate (M_int)
            self._phase_shifts = [np.random.normal(0, std_dev, self.N_in)] + [
                np.random.normal(0, std_dev, self.M_int) for _ in range(self.L - 1)
            ]
        else:
            # TX: intermediate phases (M_int), last at output aperture (N_out)
            self._phase_shifts = [
                np.random.normal(0, std_dev, self.M_int) for _ in range(self.L - 1)
            ] + [np.random.normal(0, std_dev, self.N_out)]

        self.W = self._build_W_list(self.L)

    def _build_W_list(self, L: int) -> list:
        """Build the list of L propagation matrices for the SIM stack.

        Parameters
        ----------
        L : int
            Number of layers.

        Returns
        -------
        list of np.ndarray
            Propagation matrices ``[W_1, …, W_L]``.
        """
        if L == 1:
            return [self._calculate_W("input", "output")]
        W_list = [self._calculate_W("input", "intermediate")]
        for _ in range(L - 2):
            W_list.append(self._calculate_W("intermediate", "intermediate"))
        W_list.append(self._calculate_W("intermediate", "output"))
        return W_list

    def _calculate_W(self, from_type: str, to_type: str) -> np.ndarray:
        """Compute the free-space propagation matrix between two adjacent meta-atom layers.

        Entries follow the scalar Green's function approximation:
            W[to, from] = (s_layer · A_cell / d²) · (1/(2πd) − j/λ) · exp(jκd)

        Vectorised over both aperture axes at once (NumPy broadcasting).

        Parameters
        ----------
        from_type : str
            ``"input"``, ``"intermediate"``, or ``"output"`` — source layer.
        to_type : str
            ``"input"``, ``"intermediate"``, or ``"output"`` — destination layer.

        Returns
        -------
        np.ndarray
            Complex propagation matrix of shape ``(to_total, from_total)``.
        """
        params = {
            "input": (self.N_in, int(np.sqrt(self.N_in)), self.spacings["in"]),
            "intermediate": (self.M_int, int(np.sqrt(self.M_int)), self.spacings["int"]),
            "output": (self.N_out, int(np.sqrt(self.N_out)), self.spacings["out"]),
        }
        from_total, from_nx, from_sx = params[from_type]
        to_total, to_nx, _ = params[to_type]
        A_cell = from_sx * from_sx  # unit-cell area

        from_idx = np.arange(1, from_total + 1)
        from_my = np.ceil(from_idx / from_nx).astype(np.int64)
        from_mx = from_idx - (from_my - 1) * from_nx

        to_idx = np.arange(1, to_total + 1)
        to_my = np.ceil(to_idx / to_nx).astype(np.int64)
        to_mx = to_idx - (to_my - 1) * to_nx

        # (to_total, from_total) pairwise distance.
        dx = (to_mx[:, None] - from_mx[None, :]) * from_sx
        dy = (to_my[:, None] - from_my[None, :]) * from_sx
        d = np.sqrt(dx**2 + dy**2 + self.slayer**2)

        W = np.zeros((to_total, from_total), dtype=complex)
        nonzero = d > 0
        d_nz = d[nonzero]
        W[nonzero] = (
            (self.slayer * A_cell / d_nz**2)
            * (1 / (2 * np.pi * d_nz) - 1j / self.wavelength)
            * np.exp(1j * self.kappa * d_nz)
        )
        return W


class MonoSIMoptimizerTorch(nn.Module):
    """PyTorch Mono-SIM with trainable phase shifts and gradient-based optimization.

    Parameters
    ----------
    sim_cpu : MonoSIMoptimizer
        Pre-built NumPy optimizer whose propagation matrices and initial phase
        shifts are copied into this module as buffers and parameters.
    """

    def __init__(self, sim_cpu: MonoSIMoptimizer):
        super().__init__()
        self.verbose = sim_cpu.verbose
        self.L = sim_cpu.L
        self.input_dim = sim_cpu.N_in
        self.side = sim_cpu.side

        for i, W in enumerate(sim_cpu.W):
            self.register_buffer(f"W_{i}", torch.tensor(W, dtype=torch.complex64))

        self.xi = nn.ParameterList(
            [nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts]
        )

    def get_cascade(self):
        """Compute the cascade matrix from the current phase parameters.

        TX (``side='tx'``): G_T = Υ_L W_L … Υ_1 W_1
            Phase applied at intermediate/output apertures.
        RX (``side='rx'``): G_R = W_{L-1} Υ_{L-1} … W_0 Υ_0
            Phase applied at input/intermediate apertures; signal hits Y before W.

        Returns
        -------
        torch.Tensor
            Complex64 cascade matrix ``G`` of shape ``(N_in, N_out)`` for TX
            or ``(N_out, N_in)`` for RX.
        """
        device = self.xi[0].device
        if self.side == "rx":
            G = None
            for i in range(self.L):
                W_i = getattr(self, f"W_{i}")
                phase = 2 * torch.pi * torch.sigmoid(self.xi[i])
                Y_i = torch.diag(torch.exp(1j * phase))
                G = Y_i if G is None else Y_i @ G
                G = W_i @ G
            return G
        else:
            G = torch.eye(self.input_dim, dtype=torch.complex64, device=device)
            for i in range(self.L):
                W_i = getattr(self, f"W_{i}")
                # Map unconstrained parameter to [0, 2π) via sigmoid — smooth, bounded gradient
                phase = 2 * torch.pi * torch.sigmoid(self.xi[i])
                Y_i = torch.diag(torch.exp(1j * phase))
                G = Y_i @ W_i @ G
            return G

    @staticmethod
    def _build_optimizer(params, lr: float, optimizer: str):
        """Construct an SGD or Adam optimizer for ``params``.

        Parameters
        ----------
        params : iterable of torch.Tensor
            Parameters to optimize.
        lr : float
            Learning rate.
        optimizer : str
            ``"sgd"`` or ``"adam"``.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        if optimizer == "sgd":
            return optim.SGD(params, lr=lr)
        return optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    @staticmethod
    def _build_scheduler(opt, scheduler: str, max_iters: int, warmup_frac: float, lr: float):
        """Return a cosine-annealing LR scheduler with warm-up, or ``None`` for fixed LR.

        Parameters
        ----------
        opt : torch.optim.Optimizer
            Optimizer to schedule.
        scheduler : str or None
            ``"cosine"`` to enable scheduling, any other value returns ``None``.
        max_iters : int
            Total number of training iterations.
        warmup_frac : float
            Fraction of ``max_iters`` held at constant LR before decay.
        lr : float
            Base learning rate (used for ``eta_min = lr * 1e-3``).

        Returns
        -------
        torch.optim.lr_scheduler.SequentialLR or None
        """
        if scheduler != "cosine":
            return None
        warmup_iters = max(1, int(max_iters * warmup_frac))
        cosine_iters = max(1, max_iters - warmup_iters)
        constant = optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=warmup_iters)
        cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_iters, eta_min=lr * 1e-3)
        return optim.lr_scheduler.SequentialLR(
            opt, schedulers=[constant, cosine], milestones=[warmup_iters]
        )

    def optimize_sim(
        self,
        target_matrix,
        max_iters: int = 5000,
        lr: float = 0.01,
        optimizer: str = "adam",
        scheduler: str | None = None,
        warmup_frac: float = 0.05,
        left_matrix=None,
        right_matrix=None,
    ):
        """Optimize phase shifts to minimise ``‖β (L G R) − T‖²_F / (n·m)``.

        The effective cascade used in the loss is ``G_eff = L @ G @ R`` where
        ``L`` (``left_matrix``) and ``R`` (``right_matrix``) are optional fixed
        complex matrices that incorporate the channel into the objective.
        When both are ``None`` the loss reduces to the standard ``‖β G − T‖²``.

        Parameters
        ----------
        target_matrix : array-like or torch.Tensor
            Target matrix T, complex.
        max_iters : int, optional
            Number of optimization steps.
        lr : float, optional
            Learning rate.
        optimizer : str, optional
            ``"sgd"`` or ``"adam"`` (default).
        scheduler : str or None, optional
            ``None`` — fixed LR; ``"cosine"`` — cosine decay after warm-up.
        warmup_frac : float, optional
            Fraction of ``max_iters`` kept at constant LR (cosine only).
        left_matrix : torch.Tensor or None, optional
            Fixed matrix L applied on the left: loss uses ``L @ G``.
        right_matrix : torch.Tensor or None, optional
            Fixed matrix R applied on the right: loss uses ``G @ R``.

        Returns
        -------
        loss_history : list of float
        beta_f : float
        """
        dev = self.xi[0].device
        if isinstance(target_matrix, torch.Tensor):
            Target = target_matrix.clone().detach().to(dev).to(torch.complex64)
        else:
            Target = torch.tensor(target_matrix, dtype=torch.complex64, device=dev)

        L = left_matrix.to(dev).to(torch.complex64).detach() if left_matrix is not None else None
        R = right_matrix.to(dev).to(torch.complex64).detach() if right_matrix is not None else None

        def _eff(G):
            G_e = G
            if L is not None:
                G_e = L @ G_e
            if R is not None:
                G_e = G_e @ R
            return G_e

        opt = self._build_optimizer(self.xi.parameters(), lr, optimizer)
        sched = self._build_scheduler(opt, scheduler, max_iters, warmup_frac, lr)
        loss_history = []
        best_loss = float("inf")
        best_xi = [p.data.clone() for p in self.xi]

        bar = tqdm(
            range(max_iters), desc=f"MonoSIM {self.side.upper()}", leave=False, dynamic_ncols=True
        )
        for k in bar:
            with torch.no_grad():
                G_eff_k = _eff(self.get_cascade())
                beta_k = torch.sum(torch.conj(G_eff_k) * Target) / (
                    torch.sum(torch.conj(G_eff_k) * G_eff_k) + 1e-12
                )

            opt.zero_grad()
            G_eff = _eff(self.get_cascade())
            loss = torch.norm(beta_k * G_eff - Target, p="fro") ** 2 / (
                G_eff.shape[0] * G_eff.shape[1]
            )
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()

            loss_history.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_xi = [p.data.clone() for p in self.xi]
            bar.set_postfix(loss=f"{loss.item():.4e}", beta=f"{torch.abs(beta_k).item():.2e}")

        bar.close()
        with torch.no_grad():
            for p, best in zip(self.xi, best_xi):
                p.copy_(best)
            G_eff_f = _eff(self.get_cascade())
            beta_f = torch.sum(torch.conj(G_eff_f) * Target) / (
                torch.sum(torch.conj(G_eff_f) * G_eff_f) + 1e-12
            )

        return loss_history, beta_f.item()

    def optimize_rx_sequential(
        self,
        H_eff,
        A_target,
        max_iters: int = 5000,
        lr: float = 0.1,
        optimizer: str = "adam",
        scheduler: str | None = None,
        warmup_frac: float = 0.05,
    ):
        """Optimize RX phase shifts given the effective channel ``H_eff = H · G_T``.

        Minimises ``‖β G_R H_eff − A‖²_F / (n·m)`` with respect to the RX
        phase shifts.  Used in the disjoint Dual-SIM training pipeline after
        the TX-SIM has been frozen.

        Parameters
        ----------
        H_eff : array-like or torch.Tensor
            Effective channel ``H · G_T``, complex.
        A_target : array-like or torch.Tensor
            Target alignment matrix A, complex.
        max_iters : int, optional
            Number of optimization steps.
        lr : float, optional
            Learning rate.
        optimizer : str, optional
            ``"sgd"`` — SGD; ``"adam"`` — Adam (default).
        scheduler : str or None, optional
            ``None`` — fixed LR (default);
            ``"cosine"`` — constant LR for ``warmup_frac`` iters, then cosine → ``lr*1e-3``.
        warmup_frac : float, optional
            Fraction of ``max_iters`` kept at constant LR (only for ``scheduler="cosine"``).

        Returns
        -------
        loss_history : list of float
            Per-step loss values.
        beta_f : float
            Optimal scalar β at the final iteration.
        """
        if isinstance(H_eff, torch.Tensor):
            H_eff_t = H_eff.clone().detach().to(self.xi[0].device).to(torch.complex64)
        else:
            H_eff_t = torch.tensor(H_eff, dtype=torch.complex64, device=self.xi[0].device)

        if isinstance(A_target, torch.Tensor):
            A_t = A_target.clone().detach().to(self.xi[0].device).to(torch.complex64)
        else:
            A_t = torch.tensor(A_target, dtype=torch.complex64, device=self.xi[0].device)

        opt = self._build_optimizer(self.xi.parameters(), lr, optimizer)
        sched = self._build_scheduler(opt, scheduler, max_iters, warmup_frac, lr)
        loss_history = []
        best_loss = float("inf")
        best_xi = [p.data.clone() for p in self.xi]

        bar = tqdm(range(max_iters), desc="MonoSIM RX seq", leave=False, dynamic_ncols=True)
        for k in bar:
            with torch.no_grad():
                G_R = self.get_cascade()
                Z = G_R @ H_eff_t
                beta_k = torch.sum(torch.conj(Z) * A_t) / (torch.sum(torch.conj(Z) * Z) + 1e-12)

            opt.zero_grad()
            Z_cur = self.get_cascade() @ H_eff_t
            loss = torch.norm(beta_k * Z_cur - A_t, p="fro") ** 2 / (
                Z_cur.shape[0] * Z_cur.shape[1]
            )
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()

            loss_history.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_xi = [p.data.clone() for p in self.xi]
            bar.set_postfix(loss=f"{loss.item():.4e}", beta=f"{torch.abs(beta_k).item():.2e}")
        bar.close()

        with torch.no_grad():
            for p, best in zip(self.xi, best_xi):
                p.copy_(best)
            G_R_f = self.get_cascade()
            Z_f = G_R_f @ H_eff_t
            beta_f = torch.sum(torch.conj(Z_f) * A_t) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

        return loss_history, beta_f.item()


if __name__ == "__main__":
    import torch

    cpu = MonoSIMoptimizer(
        num_layers=2,
        num_meta_atoms_in_x=2,
        num_meta_atoms_in_y=2,
        num_meta_atoms_out_x=2,
        num_meta_atoms_out_y=2,
        num_meta_atoms_int_x=2,
        num_meta_atoms_int_y=2,
        thickness=0.05,
        wavelength=0.005,
        side="tx",
    )
    model = MonoSIMoptimizerTorch(cpu)
    G = model.get_cascade()
    print(f"MonoSIMoptimizer      : W[0]={cpu.W[0].shape}")
    print(f"MonoSIMoptimizerTorch : cascade={G.shape}, dtype={G.dtype}")

"""Dual Stacked Intelligent Metasurface (Dual-SIM) optimizer — NumPy and PyTorch versions.

A Dual-SIM consists of two cascaded metasurface stacks:
  - TX-SIM  (L_T layers) placed before the MIMO channel to pre-process the transmitted signal.
  - RX-SIM  (L_R layers) placed after  the MIMO channel to post-process the received signal.

Each layer applies a diagonal phase matrix Υ_l followed by a fixed free-space
propagation matrix W_l computed from scalar-wave diffraction (Green's function).
The trainable parameters are the phase shifts ξ at every layer.

The effective end-to-end cascade is:
    Z = G_R · H · G_T
where G_T and G_R are the cascaded transfer matrices of the TX and RX metasurfaces
and H is the MIMO channel matrix.  The optimizer minimises ‖β Z − A‖²_F, i.e. it
finds phase profiles such that β Z best approximates the target alignment matrix A.

Classes
-------
DualSIMoptimizer      — NumPy reference implementation (builds propagation matrices).
DualSIMoptimizerTorch — PyTorch wrapper with gradient-based alternating optimisation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


class DualSIMoptimizer:
    """Dual-SIM optimizer — builds propagation matrices and holds phase shifts (NumPy).

    Constructs the fixed free-space propagation matrices W for both the TX-SIM
    (``L_T`` layers) and the RX-SIM (``L_R`` layers) using scalar Green's function
    diffraction.  Phase-shift vectors are randomly initialised and can be
    optimised externally.

    Parameters
    ----------
    num_layers_TX : int
        Number of metasurface layers in the TX-SIM.
    num_meta_atoms_TX_in_x : int
        Number of meta-atoms in the x-direction of the TX input aperture.
    num_meta_atoms_TX_in_y : int
        Number of meta-atoms in the y-direction of the TX input aperture.
    num_meta_atoms_TX_out_x : int
        Number of meta-atoms in the x-direction of the TX output aperture.
    num_meta_atoms_TX_out_y : int
        Number of meta-atoms in the y-direction of the TX output aperture.
    num_meta_atoms_TX_int_x : int
        Number of meta-atoms in the x-direction of TX intermediate layers.
    num_meta_atoms_TX_int_y : int
        Number of meta-atoms in the y-direction of TX intermediate layers.
    thickness_TX : float
        Total physical thickness of the TX-SIM (metres).
    num_layers_RX : int
        Number of metasurface layers in the RX-SIM.
    num_meta_atoms_RX_in_x : int
        Number of meta-atoms in the x-direction of the RX input aperture.
    num_meta_atoms_RX_in_y : int
        Number of meta-atoms in the y-direction of the RX input aperture.
    num_meta_atoms_RX_out_x : int
        Number of meta-atoms in the x-direction of the RX output aperture.
    num_meta_atoms_RX_out_y : int
        Number of meta-atoms in the y-direction of the RX output aperture.
    num_meta_atoms_RX_int_x : int
        Number of meta-atoms in the x-direction of RX intermediate layers.
    num_meta_atoms_RX_int_y : int
        Number of meta-atoms in the y-direction of RX intermediate layers.
    thickness_RX : float
        Total physical thickness of the RX-SIM (metres).
    wavelength : float
        Operating wavelength λ (metres).
    verbose : bool, optional
        If ``True``, print construction details.
    spacings : dict, optional
        Meta-atom spacings for each aperture type.  Defaults to ``λ/2`` for all.
    """

    def __init__(
        self,
        num_layers_TX: int,
        num_meta_atoms_TX_in_x: int,
        num_meta_atoms_TX_in_y: int,
        num_meta_atoms_TX_out_x: int,
        num_meta_atoms_TX_out_y: int,
        num_meta_atoms_TX_int_x: int,
        num_meta_atoms_TX_int_y: int,
        thickness_TX: float,
        num_layers_RX: int,
        num_meta_atoms_RX_in_x: int,
        num_meta_atoms_RX_in_y: int,
        num_meta_atoms_RX_out_x: int,
        num_meta_atoms_RX_out_y: int,
        num_meta_atoms_RX_int_x: int,
        num_meta_atoms_RX_int_y: int,
        thickness_RX: float,
        wavelength: float,
        verbose: bool = False,
        spacings: dict = None,
    ):
        if num_layers_TX <= 0 or num_layers_RX <= 0:
            raise ValueError("Number of layers must be positive for both SIMs.")

        self.L_T = num_layers_TX
        self.N_in_T = num_meta_atoms_TX_in_x * num_meta_atoms_TX_in_y
        self.N_out_T = num_meta_atoms_TX_out_x * num_meta_atoms_TX_out_y
        self.M_int_T = num_meta_atoms_TX_int_x * num_meta_atoms_TX_int_y
        self.slayer_TX = thickness_TX / num_layers_TX

        self.L_R = num_layers_RX
        self.N_in_R = num_meta_atoms_RX_in_x * num_meta_atoms_RX_in_y
        self.N_out_R = num_meta_atoms_RX_out_x * num_meta_atoms_RX_out_y
        self.M_int_R = num_meta_atoms_RX_int_x * num_meta_atoms_RX_int_y
        self.slayer_RX = thickness_RX / num_layers_RX

        self.wavelength = wavelength
        self.kappa = 2 * np.pi / wavelength
        self.verbose = verbose

        self.spacings = spacings or {
            "tx_in": 0.5 * wavelength,
            "tx_out": 0.5 * wavelength,
            "tx_int": 0.5 * wavelength,
            "rx_in": 0.5 * wavelength,
            "rx_out": 0.5 * wavelength,
            "rx_int": 0.5 * wavelength,
        }

        std_dev = 0.01
        self._phase_shifts_TX = [
            np.random.normal(0, std_dev, self.M_int_T) for _ in range(self.L_T - 1)
        ] + [np.random.normal(0, std_dev, self.N_out_T)]

        # RX phase shifts: first at input aperture (N_in_R), rest at intermediate (M_int_R)
        self._phase_shifts_RX = [np.random.normal(0, std_dev, self.N_in_R)] + [
            np.random.normal(0, std_dev, self.M_int_R) for _ in range(self.L_R - 1)
        ]

        self.W_TX = self._build_W_list(self.L_T, "TX_input", "TX_intermediate", "TX_output")
        # RX uses L_R propagation matrices (W·Y scheme, signal hits Y before W)
        self.W_RX = self._build_W_list(self.L_R, "RX_input", "RX_intermediate", "RX_output")

    def _build_W_list(self, L: int, type_in: str, type_int: str, type_out: str) -> list:
        """Build the list of L propagation matrices for the TX-SIM (Y·W scheme).

        Layer layout: input → [intermediate × (L-2)] → output.
        For L=1 the single layer maps input directly to output.
        """
        W_list = []
        if L == 1:
            W_list.append(self._calculate_W(type_in, type_out))
        else:
            W_list.append(self._calculate_W(type_in, type_int))
            for _ in range(L - 2):
                W_list.append(self._calculate_W(type_int, type_int))
            W_list.append(self._calculate_W(type_int, type_out))
        return W_list

    def _get_2d_coords(self, linear_idx: int, num_atoms_x: int) -> tuple[int, int]:
        """Convert a 1-based linear meta-atom index to 2-D grid coordinates (mx, my).

        Parameters
        ----------
        linear_idx : int
            1-based flat index of the meta-atom.
        num_atoms_x : int
            Number of meta-atoms along the x-axis.

        Returns
        -------
        tuple of int
            ``(mx, my)`` grid coordinates, both 1-based.
        """
        my = int(np.ceil(linear_idx / num_atoms_x))
        mx = linear_idx - (my - 1) * num_atoms_x
        return mx, my

    def _calculate_propagation_distance(
        self, from_nx, from_idx, to_nx, to_idx, sx, sy, slayer
    ) -> float:
        """Compute the 3-D Euclidean distance between two meta-atom centres.

        Parameters
        ----------
        from_nx : int
            Number of atoms along x in the source layer.
        from_idx : int
            1-based flat index of the source atom.
        to_nx : int
            Number of atoms along x in the destination layer.
        to_idx : int
            1-based flat index of the destination atom.
        sx : float
            Inter-atom spacing in x (metres).
        sy : float
            Inter-atom spacing in y (metres).
        slayer : float
            Layer-to-layer thickness (metres).

        Returns
        -------
        float
            Euclidean distance ``d`` (metres).
        """
        fx, fy = self._get_2d_coords(from_idx, from_nx)
        tx, ty = self._get_2d_coords(to_idx, to_nx)
        return np.sqrt(((fx - tx) * sx) ** 2 + ((fy - ty) * sy) ** 2 + slayer**2)

    def _calculate_W(self, from_layer_type: str, to_layer_type: str) -> np.ndarray:
        """Compute the free-space propagation matrix between two adjacent meta-atom layers.

        Uses the scalar Green's function approximation:
            W[to, from] = (s_layer · A_cell / d²) · (1/(2πd) − j/λ) · exp(jκd)
        where d is the 3-D Euclidean distance between meta-atom positions,
        A_cell = s² is the unit-cell area, and κ = 2π/λ is the wavenumber.
        """
        params = {
            "TX_input": (
                self.N_in_T,
                int(np.sqrt(self.N_in_T)),
                self.spacings["tx_in"],
                self.slayer_TX,
            ),
            "TX_intermediate": (
                self.M_int_T,
                int(np.sqrt(self.M_int_T)),
                self.spacings["tx_int"],
                self.slayer_TX,
            ),
            "TX_output": (
                self.N_out_T,
                int(np.sqrt(self.N_out_T)),
                self.spacings["tx_out"],
                self.slayer_TX,
            ),
            "RX_input": (
                self.N_in_R,
                int(np.sqrt(self.N_in_R)),
                self.spacings["rx_in"],
                self.slayer_RX,
            ),
            "RX_intermediate": (
                self.M_int_R,
                int(np.sqrt(self.M_int_R)),
                self.spacings["rx_int"],
                self.slayer_RX,
            ),
            "RX_output": (
                self.N_out_R,
                int(np.sqrt(self.N_out_R)),
                self.spacings["rx_out"],
                self.slayer_RX,
            ),
        }

        from_total, from_nx, from_sx, slayer = params[from_layer_type]
        to_total, to_nx, _, _ = params[to_layer_type]

        A_cell = from_sx * from_sx
        W = np.zeros((to_total, from_total), dtype=complex)

        for to_idx in range(1, to_total + 1):
            for from_idx in range(1, from_total + 1):
                d = self._calculate_propagation_distance(
                    from_nx, from_idx, to_nx, to_idx, from_sx, from_sx, slayer
                )
                if d > 0:
                    W[to_idx - 1, from_idx - 1] = (
                        (slayer * A_cell / d**2)
                        * (1 / (2 * np.pi * d) - 1j / self.wavelength)
                        * np.exp(1j * self.kappa * d)
                    )
        return W

    def _calculate_G_T(self) -> np.ndarray:
        """G_T = ∏_{l=1}^{L_T} Υ_l^(T) W_l^(T)  — TX cascade transfer matrix."""
        G = np.eye(self.N_in_T, dtype=complex)
        for i in range(self.L_T):
            # Υ_l = diag(exp(j ξ_l)) applies the learnable phase shifts at layer l
            Y_l = np.diag(np.exp(1j * self._phase_shifts_TX[i]))
            G = Y_l @ self.W_TX[i] @ G
        return G

    def _calculate_G_R(self) -> np.ndarray:
        """G_R = W_{L-1} · Y_{L-1} · … · W_0 · Y_0  — RX cascade (Y·W scheme).

        Signal flow: x → Y_0 → W_0 → Y_1 → W_1 → … → Y_{L-1} → W_{L-1}.
        First phase Y_0 acts on the input aperture (N_in_R); the rest are
        intermediate (M_int_R).  Uses L_R propagation matrices and L_R phases.
        """
        G = None
        for i in range(self.L_R):
            Y_l = np.diag(np.exp(1j * self._phase_shifts_RX[i]))
            G = Y_l if G is None else Y_l @ G  # phase shift first
            G = self.W_RX[i] @ G  # then propagation
        return G

    def calculate_effective_cascade(self, H_mimo: np.ndarray) -> np.ndarray:
        """Compute the end-to-end cascade Z = G_R · H · G_T.

        Parameters
        ----------
        H_mimo : np.ndarray
            Complex MIMO channel matrix.

        Returns
        -------
        np.ndarray
            Effective cascade matrix Z.
        """
        return self._calculate_G_R() @ H_mimo @ self._calculate_G_T()

    def calculate_optimal_beta(self, Z: np.ndarray, A: np.ndarray) -> complex:
        """Compute the optimal scalar β = ⟨Z, A⟩_F / ⟨Z, Z⟩_F.

        Parameters
        ----------
        Z : np.ndarray
            Cascade matrix, shape ``(D_rx, D_tx)``.
        A : np.ndarray
            Target alignment matrix, same shape as Z.

        Returns
        -------
        complex
            Optimal scaling scalar β.
        """
        z_vec = Z.flatten()
        a_vec = A.flatten()
        return np.vdot(z_vec, a_vec) / np.vdot(z_vec, z_vec)


class DualSIMoptimizerTorch(nn.Module):
    """PyTorch Dual-SIM with trainable phase shifts and gradient-based optimization.

    Wraps a :class:`DualSIMoptimizer` by registering its propagation matrices as
    non-trainable buffers and the phase-shift vectors as ``nn.Parameter`` lists.
    Optimization uses an alternating Adam scheme with cosine-annealing LR.

    Parameters
    ----------
    sim_cpu : DualSIMoptimizer
        Pre-built NumPy optimizer whose propagation matrices and initial phase
        shifts are copied into this module.
    """

    def __init__(self, sim_cpu: DualSIMoptimizer):
        super().__init__()
        self.verbose = sim_cpu.verbose
        self.L_T = sim_cpu.L_T
        self.L_R = sim_cpu.L_R

        for i, W in enumerate(sim_cpu.W_TX):
            self.register_buffer(f"W_T_{i}", torch.tensor(W, dtype=torch.complex64))
        # W_RX has L_R+1 matrices (W·Y·W scheme)
        for i, W in enumerate(sim_cpu.W_RX):
            self.register_buffer(f"W_R_{i}", torch.tensor(W, dtype=torch.complex64))

        self.xi_T = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32)))
                for p in sim_cpu._phase_shifts_TX
            ]
        )
        # xi_R has L_R parameters all of size M_int_R (intermediate layers only)
        self.xi_R = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32)))
                for p in sim_cpu._phase_shifts_RX
            ]
        )

    def _get_W_list(self, prefix, L):
        """Retrieve the registered propagation buffers W_0 … W_{L-1}."""
        return [getattr(self, f"{prefix}_{i}") for i in range(L)]

    def _calculate_G(self, W_list, phases, L, input_dim):
        """Compute the cascade G = Υ_L W_L … Υ_1 W_1 for a single SIM stack."""
        device = phases[0].device
        G = torch.eye(input_dim, dtype=torch.complex64, device=device)
        for i in range(L):
            # Map unconstrained parameter to [0, 2π) via sigmoid — smooth, bounded gradient
            phase = 2 * torch.pi * torch.sigmoid(phases[i])
            Y_l = torch.diag(torch.exp(1j * phase))
            G = Y_l @ W_list[i] @ G
        return G

    def _calculate_G_T(self):
        """Compute the TX cascade G_T from the current phase parameters."""
        W_T = self._get_W_list("W_T", self.L_T)
        return self._calculate_G(W_T, self.xi_T, self.L_T, W_T[0].shape[1])

    def _calculate_G_R(self):
        """Compute G_R = W_{L-1} · Y_{L-1} · … · W_0 · Y_0 (Y·W scheme).

        Signal flow: x → Y_0 → W_0 → … → Y_{L-1} → W_{L-1}.
        First phase Y_0 acts on the input aperture; the rest are intermediate.
        Uses L_R sigmoid-mapped phases.
        """
        W_R = self._get_W_list("W_R", self.L_R)
        G = None
        for i in range(self.L_R):
            phase = 2 * torch.pi * torch.sigmoid(self.xi_R[i])
            Y_l = torch.diag(torch.exp(1j * phase))
            G = Y_l if G is None else Y_l @ G  # phase shift first
            G = W_R[i] @ G  # then propagation
        return G

    def get_effective_cascade(self, H):
        """Compute the end-to-end cascade Z = G_R · H · G_T.

        Parameters
        ----------
        H : torch.Tensor
            Complex MIMO channel matrix.

        Returns
        -------
        Z : torch.Tensor
            Effective cascade matrix.
        G_R : torch.Tensor
            RX cascade matrix (returned for downstream beta computation).
        """
        GT = self._calculate_G_T()
        GR = self._calculate_G_R()
        return GR @ H @ GT, GR

    def get_TX_cascade(self):
        """Return the TX cascade matrix G_T from the current phase parameters.

        Returns
        -------
        torch.Tensor
            Complex64 cascade matrix G_T.
        """
        return self._calculate_G_T()

    def get_RX_cascade(self):
        """Return the RX cascade matrix G_R from the current phase parameters.

        Returns
        -------
        torch.Tensor
            Complex64 cascade matrix G_R.
        """
        return self._calculate_G_R()

    def get_effective_cascade_functional(self, H, phases_T, phases_R):
        """Compute Z = G_R · H · G_T using externally supplied phase lists.

        Useful for evaluation without modifying the module's own parameters.

        Parameters
        ----------
        H : torch.Tensor
            Complex MIMO channel matrix.
        phases_T : list of torch.Tensor
            TX phase-shift tensors, one per layer.
        phases_R : list of torch.Tensor
            RX phase-shift tensors, one per layer.

        Returns
        -------
        torch.Tensor
            Effective cascade matrix Z.
        """
        W_T = self._get_W_list("W_T", self.L_T)
        GT = self._calculate_G(W_T, phases_T, self.L_T, W_T[0].shape[1])

        # G_R uses Y·W scheme: signal hits the phase shift before each propagation
        W_R = self._get_W_list("W_R", self.L_R)
        GR = None
        for i in range(self.L_R):
            phase = 2 * torch.pi * torch.sigmoid(phases_R[i])
            Y_l = torch.diag(torch.exp(1j * phase))
            GR = Y_l if GR is None else Y_l @ GR
            GR = W_R[i] @ GR

        return GR @ H @ GT

    def optimize_alternating(
        self,
        A_target,
        H_mimo,
        max_iters: int = 3000,
        lr_tx: float = 0.1,
        lr_rx: float = 0.1,
        warmup_frac: float = 0.05,
        clip_value: float | None = None,
        k_T: int = 100,
        k_R: int = 100,
        optimizer: str = "adam",
        scheduler: str | None = None,
    ):
        """Alternating gradient optimization of TX and RX phase shifts.

        Minimises ``‖β Z − A‖²_F / (n·m)`` where
        ``β = ⟨Z, A⟩_F / ‖Z‖²_F`` is the closed-form optimal scalar computed
        without gradient (``torch.no_grad``), so the loss gradient flows only
        through ``Z``.  Dividing by ``n·m`` keeps the loss scale consistent
        with :class:`MonoSIMoptimizerTorch`.

        Each outer iteration (cycle) performs ``k_T`` consecutive TX-only gradient
        steps followed by ``k_R`` consecutive RX-only gradient steps.
        Best phase profiles are restored at the end.

        Parameters
        ----------
        A_target : array-like or torch.Tensor
            Target alignment matrix, shape ``(D_rx/2, D_tx/2)``, complex.
        H_mimo : array-like or torch.Tensor
            MIMO channel matrix, same shape as required by the SIM.
        max_iters : int, optional
            Total gradient-step budget.  The number of TX/RX cycles run is
            ``max_iters // (k_T + k_R)``.
        lr_tx : float, optional
            Learning rate for the TX (G_T) optimizer.
        lr_rx : float, optional
            Learning rate for the RX (G_R) optimizer.
        warmup_frac : float, optional
            Fraction of ``max_iters`` kept at constant LR before cosine decay
            (only used when ``scheduler="cosine"``).
        clip_value : float or None, optional
            Gradient clipping norm applied after each backward pass.
            ``None`` disables gradient clipping entirely.
        k_T : int, optional
            Consecutive TX-only gradient steps per cycle.
        k_R : int, optional
            Consecutive RX-only gradient steps per cycle.
        optimizer : str, optional
            ``"sgd"`` — SGD; ``"adam"`` — Adam (default).
        scheduler : str or None, optional
            ``None`` — fixed LR throughout (default);
            ``"cosine"`` — constant LR for the first ``warmup_frac`` cycles,
            then cosine annealing down to ``lr * 1e-3``.

        Returns
        -------
        list of float
            Per-cycle loss values ``‖β Z − A‖²_F / (n·m)``.
        """
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)

        device = next(self.parameters()).device
        A_torch = A_torch.to(device)
        H_torch = H_torch.to(device)

        if optimizer == "sgd":
            opt_T = optim.SGD(self.xi_T.parameters(), lr=lr_tx)
            opt_R = optim.SGD(self.xi_R.parameters(), lr=lr_rx)
        else:
            opt_T = optim.Adam(self.xi_T.parameters(), lr=lr_tx, betas=(0.9, 0.999), eps=1e-8)
            opt_R = optim.Adam(self.xi_R.parameters(), lr=lr_rx, betas=(0.9, 0.999), eps=1e-8)

        n_cycles = max(1, max_iters // (k_T + k_R))

        if scheduler == "cosine":
            warmup_iters = max(1, int(n_cycles * warmup_frac))
            cosine_iters = max(1, n_cycles - warmup_iters)

            def _make_scheduler(opt, base_lr):
                constant = optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=warmup_iters)
                cosine = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cosine_iters, eta_min=base_lr * 1e-3
                )
                return optim.lr_scheduler.SequentialLR(
                    opt, schedulers=[constant, cosine], milestones=[warmup_iters]
                )

            sched_T = _make_scheduler(opt_T, lr_tx)
            sched_R = _make_scheduler(opt_R, lr_rx)

        def _step_side(opt, params):
            with torch.no_grad():
                Z_tmp, _ = self.get_effective_cascade(H_torch)
                Z_norm = torch.norm(Z_tmp, p="fro") + 1e-12
                # Closed-form optimal β — computed without gradient so the loss
                # gradient flows only through Z, not through β
                beta_k = torch.sum(torch.conj(Z_tmp) * A_torch) / (
                    torch.sum(torch.conj(Z_tmp) * Z_tmp) + 1e-12
                )

            opt.zero_grad()
            Z, _ = self.get_effective_cascade(H_torch)
            loss_J = torch.norm(beta_k * Z - A_torch, p="fro") ** 2 / (
                A_torch.shape[0] * A_torch.shape[1]
            )
            loss_J.backward()
            if clip_value is not None:
                nn.utils.clip_grad_norm_(params, max_norm=clip_value)
            opt.step()
            return loss_J.item(), beta_k.detach(), Z_norm.detach()

        loss_history = []
        best_loss = float("inf")
        best_xi_T = [p.data.clone() for p in self.xi_T]
        best_xi_R = [p.data.clone() for p in self.xi_R]

        bar = tqdm(range(n_cycles), desc="DualSIM", leave=False, dynamic_ncols=True)
        for k in bar:
            current_loss = 0.0
            beta_k = torch.tensor(1.0 + 0j, device=device)
            Z_norm_k = torch.tensor(1.0, device=device)

            for _ in range(k_T):
                current_loss, beta_k, Z_norm_k = _step_side(opt_T, list(self.xi_T.parameters()))
            for _ in range(k_R):
                current_loss, beta_k, Z_norm_k = _step_side(opt_R, list(self.xi_R.parameters()))

            if scheduler == "cosine":
                sched_T.step()
                sched_R.step()
            loss_history.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_xi_T = [p.data.clone() for p in self.xi_T]
                best_xi_R = [p.data.clone() for p in self.xi_R]

            bar.set_postfix(
                loss=f"{current_loss:.4e}",
                lr=f"{opt_T.param_groups[0]['lr']:.1e}",
                Z=f"{Z_norm_k.item():.2e}",
            )
        bar.close()

        with torch.no_grad():
            for p, best in zip(self.xi_T, best_xi_T):
                p.copy_(best)
            for p, best in zip(self.xi_R, best_xi_R):
                p.copy_(best)

        with torch.no_grad():
            Z_f, _ = self.get_effective_cascade(H_torch)
            pass  # accuracy printed by calling experiment_runner

        return loss_history


if __name__ == "__main__":
    import torch

    cpu = DualSIMoptimizer(
        num_layers_TX=2,
        num_meta_atoms_TX_in_x=2,
        num_meta_atoms_TX_in_y=2,
        num_meta_atoms_TX_out_x=2,
        num_meta_atoms_TX_out_y=2,
        num_meta_atoms_TX_int_x=2,
        num_meta_atoms_TX_int_y=2,
        thickness_TX=0.05,
        num_layers_RX=2,
        num_meta_atoms_RX_in_x=2,
        num_meta_atoms_RX_in_y=2,
        num_meta_atoms_RX_out_x=2,
        num_meta_atoms_RX_out_y=2,
        num_meta_atoms_RX_int_x=2,
        num_meta_atoms_RX_int_y=2,
        thickness_RX=0.05,
        wavelength=0.005,
        verbose=False,
    )
    model = DualSIMoptimizerTorch(cpu)
    H = torch.eye(4, dtype=torch.complex64)
    Z, G_R = model.get_effective_cascade(H)
    G_T = model.get_TX_cascade()
    print(f"DualSIMoptimizerTorch : Z={Z.shape}, G_T={G_T.shape}, G_R={G_R.shape}, dtype={Z.dtype}")

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

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm  # noqa: F401  (kept for any downstream importers)

from ..._cascade import xi_to_phasor
from ...core import optimize_joint


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
        spacings: dict | None = None,
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

    def _build_W_list(self, L: int, type_in: str, type_int: str, type_out: str) -> list[np.ndarray]:
        """Build the list of L propagation matrices for a SIM stack."""
        W_list = []
        if L == 1:
            W_list.append(self._calculate_W(type_in, type_out))
        else:
            W_list.append(self._calculate_W(type_in, type_int))
            for _ in range(L - 2):
                W_list.append(self._calculate_W(type_int, type_int))
            W_list.append(self._calculate_W(type_int, type_out))
        return W_list

    def _calculate_W(self, from_layer_type: str, to_layer_type: str) -> np.ndarray:
        """Compute the free-space propagation matrix between two adjacent meta-atom layers."""
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

        from_idx = np.arange(1, from_total + 1)
        from_my = np.ceil(from_idx / from_nx).astype(np.int64)
        from_mx = from_idx - (from_my - 1) * from_nx

        to_idx = np.arange(1, to_total + 1)
        to_my = np.ceil(to_idx / to_nx).astype(np.int64)
        to_mx = to_idx - (to_my - 1) * to_nx

        dx = (to_mx[:, None] - from_mx[None, :]) * from_sx
        dy = (to_my[:, None] - from_my[None, :]) * from_sx
        d = np.sqrt(dx**2 + dy**2 + slayer**2)

        W = np.zeros((to_total, from_total), dtype=complex)
        nonzero = d > 0
        d_nz = d[nonzero]
        W[nonzero] = (
            (slayer * A_cell / d_nz**2)
            * (1 / (2 * np.pi * d_nz) - 1j / self.wavelength)
            * np.exp(1j * self.kappa * d_nz)
        )
        return W

    def _calculate_G_T(self) -> np.ndarray:
        """Compute the TX cascade transfer matrix G_T = ∏_{l=1}^{L_T} Υ_l W_l."""
        G = np.eye(self.N_in_T, dtype=complex)
        for i in range(self.L_T):
            Y_l = np.diag(np.exp(1j * self._phase_shifts_TX[i]))
            G = Y_l @ self.W_TX[i] @ G
        return G

    def _calculate_G_R(self) -> np.ndarray:
        """Compute G_R = W_{L-1} · Y_{L-1} · … · W_0 · Y_0 (Y·W scheme)."""
        G = None
        for i in range(self.L_R):
            Y_l = np.diag(np.exp(1j * self._phase_shifts_RX[i]))
            G = Y_l if G is None else Y_l @ G
            G = self.W_RX[i] @ G
        return G

    def calculate_effective_cascade(self, H_mimo: np.ndarray) -> np.ndarray:
        """Compute the end-to-end cascade Z = G_R · H · G_T."""
        return self._calculate_G_R() @ H_mimo @ self._calculate_G_T()

    def calculate_optimal_beta(self, Z: np.ndarray, A: np.ndarray) -> complex:
        """Compute the optimal scalar β = ⟨Z, A⟩_F / ⟨Z, Z⟩_F."""
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
        """Initialize Torch wrapper from a pre-built NumPy DualSIMoptimizer.

        Copies TX and RX propagation matrices as non-trainable buffers and
        phase shifts as trainable parameters.

        Parameters
        ----------
        sim_cpu : DualSIMoptimizer
            Pre-built NumPy optimizer.
        """
        super().__init__()
        self.verbose = sim_cpu.verbose
        self.L_T = sim_cpu.L_T
        self.L_R = sim_cpu.L_R

        # Register TX propagation matrices as buffers
        for i, W in enumerate(sim_cpu.W_TX):
            self.register_buffer(f"W_T_{i}", torch.tensor(W, dtype=torch.complex64))
        # Register RX propagation matrices as buffers
        for i, W in enumerate(sim_cpu.W_RX):
            self.register_buffer(f"W_R_{i}", torch.tensor(W, dtype=torch.complex64))

        # TX phase logits (trainable)
        self.xi_T = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32)))
                for p in sim_cpu._phase_shifts_TX
            ]
        )
        # RX phase logits (trainable)
        self.xi_R = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32)))
                for p in sim_cpu._phase_shifts_RX
            ]
        )

    def _get_W_list(self, prefix: str, L: int) -> list[torch.Tensor]:
        """Retrieve L propagation matrices by name prefix.

        Parameters
        ----------
        prefix : str
            Buffer name prefix (``"W_T"`` or ``"W_R"``).
        L : int
            Number of layers.

        Returns
        -------
        list[Tensor]
            Propagation matrices.
        """
        return [getattr(self, f"{prefix}_{i}") for i in range(L)]

    def _calculate_G(
        self, W_list: list[torch.Tensor], phases: list[torch.Tensor], L: int, input_dim: int
    ):
        """Compute TX cascade: G = Υ_L W_L … Υ_1 W_1.

        Parameters
        ----------
        W_list : list[Tensor]
            Linear transforms for each layer.
        phases : list[Tensor]
            Phase logits for each layer.
        L : int
            Number of layers.
        input_dim : int
            Input dimension (rows of first W).

        Returns
        -------
        torch.Tensor
            Complex cascade matrix.
        """
        device = phases[0].device
        # Start with identity: G_0 = I
        G = torch.eye(input_dim, dtype=torch.complex64, device=device)
        for i in range(L):
            # Y·W scheme: phasor after linear transform
            Y_l = torch.diag(xi_to_phasor(phases[i]))
            G = Y_l @ W_list[i] @ G
        return G

    def _calculate_G_T(self) -> torch.Tensor:
        """Compute the TX cascade G_T from current TX phase parameters."""
        W_T = self._get_W_list("W_T", self.L_T)
        return self._calculate_G(W_T, self.xi_T, self.L_T, W_T[0].shape[1])

    def _calculate_G_R(self) -> torch.Tensor:
        """Compute the RX cascade G_R from current RX phase parameters.

        Uses the W·Y scheme: G_R = W_{L-1} · Y_{L-1} · … · W_0 · Y_0.
        """
        W_R = self._get_W_list("W_R", self.L_R)
        G = None
        for i in range(self.L_R):
            Y_l = torch.diag(xi_to_phasor(self.xi_R[i]))
            # First iteration: G = Y_0; subsequent: G = Y_i @ G
            G = Y_l if G is None else Y_l @ G
            # Apply linear transform after phasor modulation
            G = W_R[i] @ G
        return G

    def get_effective_cascade(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the end-to-end cascade Z = G_R · H · G_T.

        Parameters
        ----------
        H : Tensor
            MIMO channel matrix, shape (d_rx, d_tx).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(Z, G_R)`` where Z is the end-to-end cascade and G_R is the RX cascade.
        """
        GT = self._calculate_G_T()
        GR = self._calculate_G_R()
        return GR @ H @ GT, GR

    def get_TX_cascade(self) -> torch.Tensor:
        """Return the current TX cascade matrix G_T."""
        return self._calculate_G_T()

    def get_RX_cascade(self) -> torch.Tensor:
        """Return the current RX cascade matrix G_R."""
        return self._calculate_G_R()

    def get_effective_cascade_functional(
        self, H: torch.Tensor, phases_T: list[torch.Tensor], phases_R: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute Z = G_R · H · G_T using externally supplied phase lists."""
        W_T = self._get_W_list("W_T", self.L_T)
        GT = self._calculate_G(W_T, phases_T, self.L_T, W_T[0].shape[1])

        W_R = self._get_W_list("W_R", self.L_R)
        GR = None
        for i in range(self.L_R):
            Y_l = torch.diag(xi_to_phasor(phases_R[i]))
            GR = Y_l if GR is None else Y_l @ GR
            GR = W_R[i] @ GR

        return GR @ H @ GT

    def optimize_alternating(
        self,
        A_target: torch.Tensor | np.ndarray,
        H_mimo: torch.Tensor | np.ndarray,
        max_iters: int = 3000,
        lr_tx: float = 0.1,
        lr_rx: float = 0.1,
        warmup_frac: float = 0.05,
        clip_value: float | None = None,
        k_T: int = 100,
        k_R: int = 100,
        optimizer: str = "adam",
        scheduler: str | None = None,
        eval_fn: Callable | None = None,
    ) -> list[float]:
        """Alternating gradient optimization of TX and RX phase shifts.

        Each outer iteration performs ``k_T`` consecutive TX-only gradient steps
        followed by ``k_R`` consecutive RX-only gradient steps.
        Best phase profiles are restored at the end.

        Parameters
        ----------
        A_target : array-like or torch.Tensor
            Target alignment matrix, shape ``(D_rx/2, D_tx/2)``, complex.
        H_mimo : array-like or torch.Tensor
            MIMO channel matrix.
        max_iters : int
            Number of alternating cycles (each cycle = k_T TX steps + k_R RX steps).
        lr_tx, lr_rx : float
            Learning rates for TX and RX optimizers.
        k_T, k_R : int
            Consecutive TX-only / RX-only steps per cycle.

        Returns
        -------
        list of float
            Per-cycle loss values.
        """
        device = next(self.parameters()).device
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

        W_T = [getattr(self, f"W_T_{i}") for i in range(self.L_T)]
        W_R = [getattr(self, f"W_R_{i}") for i in range(self.L_R)]
        sizes_T = [p.shape[0] for p in self.xi_T]
        sizes_R = [p.shape[0] for p in self.xi_R]
        phases_T_init = [p.data.clone() for p in self.xi_T]
        phases_R_init = [p.data.clone() for p in self.xi_R]

        n_cycles = max(1, max_iters)

        phases_T, phases_R, loss_history, _ = optimize_joint(
            W_T,
            W_R,
            sizes_T,
            sizes_R,
            H_torch,
            A_torch,
            n_cycles,
            lr_tx,
            lr_rx,
            device,
            phases_T_init=phases_T_init,
            phases_R_init=phases_R_init,
            k_T=k_T,
            k_R=k_R,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            best_tracking=eval_fn is None,
            clip_value=clip_value,
            eval_fn=eval_fn,
            show_progress=True,
            desc="DualSIM",
        )
        with torch.no_grad():
            for p, ph in zip(self.xi_T, phases_T):
                p.copy_(ph)
            for p, ph in zip(self.xi_R, phases_R):
                p.copy_(ph)
        return loss_history

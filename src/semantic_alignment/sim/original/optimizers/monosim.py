"""Single Stacked Intelligent Metasurface (Mono-SIM) optimizer — NumPy and PyTorch versions.

A Mono-SIM is a single metasurface stack placed at either the TX or the RX side.

TX (``side="tx"``):
    G_T = Υ_L W_L … Υ_1 W_1
    Phase layout: [M_int] × (L-1) intermediate layers, then [N_out] at the output.

RX (``side="rx"``):
    G_R = W_{L-1} Υ_{L-1} … W_0 Υ_0
    Phase layout: [N_in] at the input aperture, then [M_int] × (L-1) intermediate.

Classes
-------
MonoSIMoptimizer      — NumPy reference: builds propagation matrices W.
MonoSIMoptimizerTorch — PyTorch wrapper with gradient-based phase optimisation.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm  # noqa: F401  (kept for any downstream importers)

from ..._cascade import xi_to_phasor
from ...core import optimize_mono
from ._tied import TiedBuffersModule


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
        spacings: dict | None = None,
        side: str = "tx",
    ):
        """Initialize Mono-SIM with given geometry and physical parameters.

        Constructs propagation matrices W and randomly initialises phase shifts.

        Parameters
        ----------
        num_layers : int
            Number of metasurface layers.
        num_meta_atoms_in_x, num_meta_atoms_in_y : int
            Meta-atom grid dimensions of the input aperture.
        num_meta_atoms_out_x, num_meta_atoms_out_y : int
            Meta-atom grid dimensions of the output aperture.
        num_meta_atoms_int_x, num_meta_atoms_int_y : int
            Meta-atom grid dimensions of intermediate layers.
        thickness : float
            Total physical thickness of the SIM stack (metres).
        wavelength : float
            Operating wavelength λ (metres).
        verbose : bool
            If True, print construction details.
        spacings : dict or None
            Meta-atom spacings for each aperture type. Defaults to λ/2.
        side : str
            ``"tx"`` or ``"rx"``; controls phase layout and cascade order.
        """
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
        # RX: phase at input aperture (N_in) first, then intermediates (M_int)
        # TX: intermediates (M_int) first, then at output aperture (N_out)
        if side == "rx":
            self._phase_shifts = [np.random.normal(0, std_dev, self.N_in)] + [
                np.random.normal(0, std_dev, self.M_int) for _ in range(self.L - 1)
            ]
        else:
            self._phase_shifts = [
                np.random.normal(0, std_dev, self.M_int) for _ in range(self.L - 1)
            ] + [np.random.normal(0, std_dev, self.N_out)]

        # Build the list of L propagation matrices (Green's function)
        self.W = self._build_W_list(self.L)

    def _build_W_list(self, L: int) -> list[np.ndarray]:
        """Build the list of L propagation matrices for a single SIM stack.

        For L=1: ``[W(input → output)]``.
        For L>1: ``[W(input → int), W(int → int) × (L-2), W(int → output)]``.

        Parameters
        ----------
        L : int
            Number of layers.

        Returns
        -------
        list[np.ndarray]
            Propagation matrices, each shape ``(to_dim, from_dim)``.
        """
        if L == 1:
            return [self._calculate_W("input", "output")]
        W_list = [self._calculate_W("input", "intermediate")]
        if L > 2:
            # All interior matrices are identical: compute once, share the array.
            W_list.extend([self._calculate_W("intermediate", "intermediate")] * (L - 2))
        W_list.append(self._calculate_W("intermediate", "output"))
        return W_list

    def _calculate_W(self, from_type: str, to_type: str) -> np.ndarray:
        """Compute the free-space propagation matrix between two adjacent meta-atom layers.

        Uses the scalar Green's function of a point source in free space:

            W_ij = (h · A_cell / d²) · (1/(2πd) - j/λ) · exp(jκd)

        where h is the layer thickness, A_cell is the meta-atom area, d is the
        3D distance between source and receiver meta-atoms, and κ = 2π/λ.

        Parameters
        ----------
        from_type : str
            Source layer type: ``"input"``, ``"intermediate"``, or ``"output"``.
        to_type : str
            Destination layer type.

        Returns
        -------
        np.ndarray
            Complex propagation matrix, shape ``(to_total, from_total)``.
        """
        params = {
            "input": (self.N_in, int(np.sqrt(self.N_in)), self.spacings["in"]),
            "intermediate": (self.M_int, int(np.sqrt(self.M_int)), self.spacings["int"]),
            "output": (self.N_out, int(np.sqrt(self.N_out)), self.spacings["out"]),
        }
        from_total, from_nx, from_sx = params[from_type]
        to_total, to_nx, _ = params[to_type]
        A_cell = from_sx * from_sx  # meta-atom area (assuming square lattice)

        # Compute 2D grid indices for source meta-atoms (row-major ordering)
        from_idx = np.arange(1, from_total + 1)
        from_my = np.ceil(from_idx / from_nx).astype(np.int64)
        from_mx = from_idx - (from_my - 1) * from_nx

        # Compute 2D grid indices for destination meta-atoms
        to_idx = np.arange(1, to_total + 1)
        to_my = np.ceil(to_idx / to_nx).astype(np.int64)
        to_mx = to_idx - (to_my - 1) * to_nx

        # 2D displacement between each source-destination pair
        dx = (to_mx[:, None] - from_mx[None, :]) * from_sx
        dy = (to_my[:, None] - from_my[None, :]) * from_sx
        # 3D distance including layer thickness
        d = np.sqrt(dx**2 + dy**2 + self.slayer**2)

        W = np.zeros((to_total, from_total), dtype=complex)
        nonzero = d > 0  # exclude self-coupling (d=0)
        d_nz = d[nonzero]
        # Green's function: W = (h·A/d²) · (1/(2πd) - j/λ) · exp(jκd)
        W[nonzero] = (
            (self.slayer * A_cell / d_nz**2)
            * (1 / (2 * np.pi * d_nz) - 1j / self.wavelength)
            * np.exp(1j * self.kappa * d_nz)
        )
        return W


class MonoSIMoptimizerTorch(TiedBuffersModule):
    """PyTorch Mono-SIM with trainable phase shifts and gradient-based optimization.

    Parameters
    ----------
    sim_cpu : MonoSIMoptimizer
        Pre-built NumPy optimizer whose propagation matrices and initial phase
        shifts are copied into this module as buffers and parameters.
    """

    def __init__(self, sim_cpu: MonoSIMoptimizer):
        """Initialize Torch wrapper from a pre-built NumPy MonoSIMoptimizer.

        Copies propagation matrices as non-trainable buffers and phase shifts
        as trainable parameters.

        Parameters
        ----------
        sim_cpu : MonoSIMoptimizer
            Pre-built NumPy optimizer.
        """
        super().__init__()
        self.verbose = sim_cpu.verbose
        self.L = sim_cpu.L
        self.input_dim = sim_cpu.N_in
        self.side = sim_cpu.side

        # Register propagation matrices as buffers (not trainable)
        self.register_W_buffers("W", sim_cpu.W)

        # Phase logits as trainable parameters
        self.xi = nn.ParameterList(
            [nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts]
        )

    def get_cascade(self) -> torch.Tensor:
        """Compute the cascade matrix from the current phase parameters.

        TX (``side='tx'``): G_T = Υ_L W_L … Υ_1 W_1
        RX (``side='rx'``): G_R = W_{L-1} Υ_{L-1} … W_0 Υ_0

        Returns
        -------
        torch.Tensor
            Complex cascade matrix, shape (d_out, d_in).
        """
        device = self.xi[0].device
        if self.side == "rx":
            # RX: W·Y scheme — phasor before linear transform
            G = None
            for i in range(self.L):
                W_i = getattr(self, f"W_{i}")
                Y_i = torch.diag(xi_to_phasor(self.xi[i]))
                G = Y_i if G is None else Y_i @ G
                G = W_i @ G
            return G
        else:
            # TX: Y·W scheme — phasor after linear transform
            G = torch.eye(self.input_dim, dtype=torch.complex64, device=device)
            for i in range(self.L):
                W_i = getattr(self, f"W_{i}")
                Y_i = torch.diag(xi_to_phasor(self.xi[i]))
                G = Y_i @ W_i @ G
            return G

    def optimize_sim(
        self,
        target_matrix: torch.Tensor | np.ndarray,
        max_iters: int = 5000,
        lr: float = 0.01,
        optimizer: str = "adam",
        scheduler: str | None = None,
        warmup_frac: float = 0.05,
        left_matrix: torch.Tensor | np.ndarray | None = None,
        right_matrix: torch.Tensor | np.ndarray | None = None,
        eval_fn=None,
    ) -> tuple[list[float], float]:
        """Optimize phase shifts to minimise ``‖β (L G R) − T‖²_F / (n·m)``.

        Parameters
        ----------
        target_matrix : array-like or torch.Tensor
            Target matrix T, complex.
        max_iters : int
            Number of optimization steps.
        lr : float
            Learning rate.
        left_matrix : torch.Tensor or None
            Fixed matrix L applied on the left: loss uses ``L @ G``.
        right_matrix : torch.Tensor or None
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

        W_list = [getattr(self, f"W_{i}") for i in range(self.L)]
        sizes = [p.shape[0] for p in self.xi]
        phases_init = [p.data.clone() for p in self.xi]

        phases, loss_history, beta_f = optimize_mono(
            W_list,
            sizes,
            Target,
            max_iters,
            lr,
            dev,
            side=self.side,
            left_matrix=L,
            right_matrix=R,
            phases_init=phases_init,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            best_tracking=eval_fn is None,
            show_progress=True,
            eval_fn=eval_fn,
            desc=f"MonoSIM {self.side.upper()}",
        )
        with torch.no_grad():
            for p, ph in zip(self.xi, phases):
                p.copy_(ph)
        return loss_history, beta_f

    def optimize_rx_sequential(
        self,
        H_eff: torch.Tensor | np.ndarray,
        A_target: torch.Tensor | np.ndarray,
        max_iters: int = 5000,
        lr: float = 0.1,
        optimizer: str = "adam",
        scheduler: str | None = None,
        warmup_frac: float = 0.05,
    ) -> tuple[list[float], float]:
        """Optimize RX phase shifts given the effective channel ``H_eff = H · G_T``.

        Minimises ``‖β G_R H_eff − A‖²_F / (n·m)`` with respect to the RX phases.
        Used in sequential (non-alternating) dual-SIM optimization where TX is
        optimized first and RX is optimized second with a fixed TX cascade.

        Parameters
        ----------
        H_eff : Tensor or np.ndarray
            Effective channel after TX cascade, shape (d_rx, d_tx).
        A_target : Tensor or np.ndarray
            Target alignment matrix, shape (d_rx, d_in).
        max_iters : int
            Number of Adam steps.
        lr : float
            Learning rate.
        optimizer : str
            ``"adam"`` or ``"sgd"``.
        scheduler : str or None
            ``"cosine"`` for warmup + cosine decay.
        warmup_frac : float
            Fraction of max_iters used for warmup.

        Returns
        -------
        tuple[list[float], float]
            ``(loss_history, beta_f)``.
        """
        dev = self.xi[0].device
        # Convert to complex64 tensors on the correct device
        if isinstance(H_eff, torch.Tensor):
            H_eff_t = H_eff.clone().detach().to(dev).to(torch.complex64)
        else:
            H_eff_t = torch.tensor(H_eff, dtype=torch.complex64, device=dev)

        if isinstance(A_target, torch.Tensor):
            A_t = A_target.clone().detach().to(dev).to(torch.complex64)
        else:
            A_t = torch.tensor(A_target, dtype=torch.complex64, device=dev)

        W_list = [getattr(self, f"W_{i}") for i in range(self.L)]
        sizes = [p.shape[0] for p in self.xi]
        phases_init = [p.data.clone() for p in self.xi]

        # Use core.optimize_mono with side="rx" and right_matrix=H_eff
        phases, loss_history, beta_f = optimize_mono(
            W_list,
            sizes,
            A_t,
            max_iters,
            lr,
            dev,
            side="rx",
            right_matrix=H_eff_t,
            phases_init=phases_init,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_frac=warmup_frac,
            best_tracking=True,
            show_progress=True,
            desc="MonoSIM RX seq",
        )
        # Copy optimized phases back into the module
        with torch.no_grad():
            for p, ph in zip(self.xi, phases):
                p.copy_(ph)
        return loss_history, beta_f

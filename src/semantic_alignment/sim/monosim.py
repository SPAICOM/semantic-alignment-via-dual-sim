"""Single Stacked Intelligent Metasurface (Mono-SIM) optimizer — NumPy and PyTorch versions."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


class MonoSIMoptimizer:
    """Mono-SIM Optimizer (NumPy). Can be instantiated as TX or RX."""

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
    ):
        if num_layers <= 0:
            raise ValueError("Number of layers must be positive.")

        self.L = num_layers
        self.N_in = num_meta_atoms_in_x * num_meta_atoms_in_y
        self.N_out = num_meta_atoms_out_x * num_meta_atoms_out_y
        self.M_int = num_meta_atoms_int_x * num_meta_atoms_int_y
        self.slayer = thickness / num_layers

        self.wavelength = wavelength
        self.kappa = 2 * np.pi / wavelength
        self.verbose = verbose

        self.spacings = spacings or {
            "in": 0.5 * wavelength,
            "out": 0.5 * wavelength,
            "int": 0.5 * wavelength,
        }

        std_dev = 0.01
        self._phase_shifts = [
            np.random.normal(0, std_dev, self.M_int) for _ in range(self.L - 1)
        ] + [np.random.normal(0, std_dev, self.N_out)]

        self.W = self._build_W_list(self.L)

    def _build_W_list(self, L: int) -> list:
        if L == 1:
            return [self._calculate_W("input", "output")]
        W_list = [self._calculate_W("input", "intermediate")]
        for _ in range(L - 2):
            W_list.append(self._calculate_W("intermediate", "intermediate"))
        W_list.append(self._calculate_W("intermediate", "output"))
        return W_list

    def _get_2d_coords(self, linear_idx: int, num_atoms_x: int) -> tuple[int, int]:
        my = int(np.ceil(linear_idx / num_atoms_x))
        mx = linear_idx - (my - 1) * num_atoms_x
        return mx, my

    def _calculate_propagation_distance(
        self, nx_from, idx_from, nx_to, idx_to, sx, sy, slayer
    ) -> float:
        x_from, y_from = self._get_2d_coords(idx_from, nx_from)
        x_to, y_to = self._get_2d_coords(idx_to, nx_to)
        return np.sqrt(((x_from - x_to) * sx) ** 2 + ((y_from - y_to) * sy) ** 2 + slayer**2)

    def _calculate_W(self, from_type: str, to_type: str) -> np.ndarray:
        params = {
            "input": (self.N_in, int(np.sqrt(self.N_in)), self.spacings["in"]),
            "intermediate": (self.M_int, int(np.sqrt(self.M_int)), self.spacings["int"]),
            "output": (self.N_out, int(np.sqrt(self.N_out)), self.spacings["out"]),
        }
        from_total, from_nx, from_sx = params[from_type]
        to_total, to_nx, _ = params[to_type]
        A_cell = from_sx * from_sx
        W = np.zeros((to_total, from_total), dtype=complex)
        for to_idx in range(1, to_total + 1):
            for from_idx in range(1, from_total + 1):
                d = self._calculate_propagation_distance(
                    from_nx, from_idx, to_nx, to_idx, from_sx, from_sx, self.slayer
                )
                if d > 0:
                    W[to_idx - 1, from_idx - 1] = (
                        (self.slayer * A_cell / d**2)
                        * (1 / (2 * np.pi * d) - 1j / self.wavelength)
                        * np.exp(1j * self.kappa * d)
                    )
        return W


class MonoSIMoptimizerTorch(nn.Module):
    """PyTorch version of MonoSIMoptimizer with trainable phase shifts."""

    def __init__(self, sim_cpu: MonoSIMoptimizer):
        super().__init__()
        self.verbose = sim_cpu.verbose
        self.L = sim_cpu.L
        self.input_dim = sim_cpu.N_in

        for i, W in enumerate(sim_cpu.W):
            self.register_buffer(f"W_{i}", torch.tensor(W, dtype=torch.complex64))

        self.xi = nn.ParameterList(
            [nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts]
        )

    def get_cascade(self):
        device = self.xi[0].device
        G = torch.eye(self.input_dim, dtype=torch.complex64, device=device)
        for i in range(self.L):
            W_i = getattr(self, f"W_{i}")
            Y_i = torch.diag(torch.exp(1j * self.xi[i]))
            G = Y_i @ W_i @ G
        return G

    def optimize_sim(self, target_matrix, max_iters: int = 5000, lr: float = 0.01):
        """Optimise TX phase shifts towards target_matrix (geometric, no regularisation)."""
        if isinstance(target_matrix, torch.Tensor):
            Target = target_matrix.clone().detach().to(self.xi[0].device).to(torch.complex64)
        else:
            Target = torch.tensor(target_matrix, dtype=torch.complex64, device=self.xi[0].device)

        opt = optim.Adam(self.xi.parameters(), lr=lr)
        loss_history = []

        bar = tqdm(range(max_iters), desc="MonoSIM TX", leave=False, dynamic_ncols=True)
        for k in bar:
            with torch.no_grad():
                G_k = self.get_cascade()
                beta_k = torch.sum(torch.conj(G_k) * Target) / (
                    torch.sum(torch.conj(G_k) * G_k) + 1e-12
                )

            opt.zero_grad()
            G = self.get_cascade()
            loss = torch.norm(beta_k * G - Target, p="fro") ** 2
            loss.backward()
            opt.step()

            with torch.no_grad():
                for p in self.xi:
                    p.copy_(p % (2 * torch.pi))

            loss_history.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4e}", beta=f"{torch.abs(beta_k).item():.2e}")

        bar.close()
        with torch.no_grad():
            G_f = self.get_cascade()
            beta_f = torch.sum(torch.conj(G_f) * Target) / (
                torch.sum(torch.conj(G_f) * G_f) + 1e-12
            )

        return loss_history, beta_f.item()

    def optimize_rx_sequential(self, H_eff, A_target, max_iters: int = 5000, lr: float = 0.1):
        """Optimise RX phase shifts using the effective channel H_eff = H * G_T."""
        if isinstance(H_eff, torch.Tensor):
            H_eff_t = H_eff.clone().detach().to(self.xi[0].device).to(torch.complex64)
        else:
            H_eff_t = torch.tensor(H_eff, dtype=torch.complex64, device=self.xi[0].device)

        if isinstance(A_target, torch.Tensor):
            A_t = A_target.clone().detach().to(self.xi[0].device).to(torch.complex64)
        else:
            A_t = torch.tensor(A_target, dtype=torch.complex64, device=self.xi[0].device)

        opt = optim.Adam(self.xi.parameters(), lr=lr)
        loss_history = []

        bar = tqdm(range(max_iters), desc="MonoSIM RX", leave=False, dynamic_ncols=True)
        for k in bar:
            with torch.no_grad():
                G_R = self.get_cascade()
                Z = G_R @ H_eff_t
                beta_k = torch.sum(torch.conj(Z) * A_t) / (torch.sum(torch.conj(Z) * Z) + 1e-12)

            opt.zero_grad()
            Z_cur = self.get_cascade() @ H_eff_t
            loss = torch.norm(beta_k * Z_cur - A_t, p="fro") ** 2
            loss.backward()
            opt.step()

            with torch.no_grad():
                for p in self.xi:
                    p.copy_(p % (2 * torch.pi))

            loss_history.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4e}", beta=f"{torch.abs(beta_k).item():.2e}")
        bar.close()

        with torch.no_grad():
            G_R_f = self.get_cascade()
            Z_f = G_R_f @ H_eff_t
            beta_f = torch.sum(torch.conj(Z_f) * A_t) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

        return loss_history, beta_f.item()

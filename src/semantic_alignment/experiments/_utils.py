"""Shared helpers for all SIM experiment runners.

Provides:
- :class:`_FixedCascade`   — thin wrapper so a pre-computed matrix satisfies
  the ``get_cascade()`` interface of :func:`run_evaluation_disjoint_fixed`.
- :func:`inter_sim_propagation` — free-space propagation matrix between two
  co-located apertures (used by the 4-SIM runner).
- :func:`save_result`  — write a single-row result to a Parquet file.
- :func:`make_fname`   — build a canonical, self-documenting filename.
- :func:`ch_dims`      — factorize a channel dimension into ``(n_x, n_y)``.
- :func:`snr_tag`, :func:`pct`, :func:`neural_str` — small formatting helpers.
"""

from pathlib import Path

import polars as pl
import torch

from semantic_alignment.sim.monosim import MonoSIMoptimizer

RESULTS_DIR = Path("results")


class _FixedCascade:
    """Minimal wrapper so a pre-computed cascade matrix satisfies the
    ``get_cascade()`` interface expected by ``run_evaluation_disjoint_fixed``."""

    def __init__(self, G: torch.Tensor) -> None:
        self._G = G

    def eval(self):
        """No-op to match the ``nn.Module.eval()`` interface."""
        return self

    def get_cascade(self) -> torch.Tensor:
        """Return the pre-computed cascade matrix."""
        return self._G


def inter_sim_propagation(
    n_x: int, n_y: int, slayer: float, dx: float, wavelength: float, device
) -> torch.Tensor:
    """Compute the free-space propagation matrix between two identical n_x × n_y apertures.

    Used in the 4-SIM runner to model the physical gap between the two TX SIMs
    and between the two RX SIMs.

    Parameters
    ----------
    n_x : int
        Number of meta-atoms along x.
    n_y : int
        Number of meta-atoms along y.
    slayer : float
        Inter-layer spacing (metres).
    dx : float
        Meta-atom spacing (metres).
    wavelength : float
        Operating wavelength λ (metres).
    device :
        PyTorch device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Complex64 propagation matrix of shape ``(n_x*n_y, n_x*n_y)``.
    """
    tmp = MonoSIMoptimizer(
        num_layers=1,
        num_meta_atoms_in_x=n_x,
        num_meta_atoms_in_y=n_y,
        num_meta_atoms_out_x=n_x,
        num_meta_atoms_out_y=n_y,
        num_meta_atoms_int_x=n_x,
        num_meta_atoms_int_y=n_y,
        thickness=slayer,
        wavelength=wavelength,
        spacings={"in": dx, "out": dx, "int": dx},
        side="tx",
    )
    return torch.tensor(tmp.W[0], dtype=torch.complex64).to(device)


def save_result(path: Path, row: dict) -> None:
    """Write a single result row to a Parquet file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_parquet(path)


def snr_tag(snr: float | None) -> str:
    """Return ``'inf'`` for ``None`` SNR (noiseless), otherwise its string value."""
    return "inf" if snr is None else str(snr)


def make_fname(
    strategy: str,
    seed: int,
    L: int,
    M_int: int,
    ch_dim: int,
    snr: float | None,
    tx_enc: str,
    rx_enc: str,
    s_tx: int | None = None,
    s_rx: int | None = None,
) -> str:
    """Build a canonical Parquet filename from all experiment fields.

    All fields are always present; ``None`` values use ``"none"`` for s-layer
    and ``"inf"`` for SNR so filenames are self-documenting without branching.
    """
    s_tx_tag = "none" if s_tx is None else str(s_tx)
    s_rx_tag = "none" if s_rx is None else str(s_rx)
    return (
        f"{strategy}_seed{seed}_L{L}_M{M_int}x{M_int}_ch{ch_dim}"
        f"_snr{snr_tag(snr)}_stx{s_tx_tag}_srx{s_rx_tag}"
        f"_{tx_enc}_{rx_enc}.parquet"
    )


def ch_dims(N: int) -> tuple[int, int]:
    """Factorize N into ``(n_x, n_y)`` with ``n_x * n_y == N``, as square as possible.

    Parameters
    ----------
    N : int
        Total number of meta-atoms or channel elements to factorize.

    Returns
    -------
    tuple of int
        ``(n_x, n_y)`` such that ``n_x * n_y == N`` and the aspect ratio
        ``n_x / n_y`` is as close to 1 as possible.
    """
    for n_y in range(int(N**0.5), 0, -1):
        if N % n_y == 0:
            return N // n_y, n_y
    return N, 1


def pct(v: float | None) -> float | None:
    """Convert a fraction in ``[0, 1]`` to a percentage; pass ``None`` through unchanged."""
    return v * 100 if v is not None else None


def neural_str(label: str, acc_pct: float | None) -> str:
    """Format an optional neural-classifier accuracy for console logging.

    Returns an empty string when the neural classifier was not available so
    the caller does not need to branch.
    """
    return f" {label}={acc_pct:.2f}%" if acc_pct is not None else ""

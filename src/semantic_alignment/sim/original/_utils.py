"""Shared helpers for all SIM runner classes.

Provides:
- :class:`_FixedCascade`      — thin wrapper so a pre-computed matrix satisfies
  the ``get_cascade()`` interface.
- :func:`inter_sim_propagation` — free-space propagation matrix between two
  co-located apertures (used by the 4-SIM runner).
- :func:`save_result`  — write a single-row result to a Parquet file.
- :func:`save_params` / :func:`load_params` — trained SIM state next to that Parquet.
- :func:`make_fname`   — build a canonical, self-documenting filename.
- :func:`ch_dims`      — factorize a channel dimension into ``(n_x, n_y)``.
- :func:`snr_tag`, :func:`pct`, :func:`neural_str` — small formatting helpers.
"""

from pathlib import Path

import polars as pl
import torch

from .optimizers.monosim import MonoSIMoptimizer

# Results are split by downstream task: classification (accuracy) vs. reconstruction (PSNR).
RESULTS_ROOT = Path("results")
RESULTS_DIR = RESULTS_ROOT / "classification"
RECONSTRUCTION_RESULTS_DIR = RESULTS_ROOT / "reconstruction"


class _FixedCascade:
    """Minimal wrapper so a pre-computed cascade matrix satisfies the
    ``get_cascade()`` interface expected by ``run_evaluation_disjoint_fixed``.

    Parameters
    ----------
    G : torch.Tensor
        Pre-computed cascade matrix, shape (d_out, d_in).
    """

    def __init__(self, G: torch.Tensor) -> None:
        self._G = G

    def eval(self) -> "_FixedCascade":
        """Return self (no-op; satisfies ``nn.Module.eval()`` interface)."""
        return self

    def get_cascade(self) -> torch.Tensor:
        """Return the stored cascade matrix."""
        return self._G


def inter_sim_propagation(
    n_x: int, n_y: int, slayer: float, dx: float, wavelength: float, device: torch.device
) -> torch.Tensor:
    """Compute the free-space propagation matrix between two identical n_x × n_y apertures.

    Used in the 4-SIM runner to model the physical gap between the two TX SIMs
    and between the two RX SIMs. The propagation is computed via the Green's
    function of a single-layer metasurface with the given thickness and spacing.

    Parameters
    ----------
    n_x, n_y : int
        Aperture dimensions in the x and y directions.
    slayer : float
        Metasurface layer thickness (m).
    dx : float
        Inter-element spacing (m).
    wavelength : float
        Carrier wavelength (m).
    device : torch.device
        Target device for the output tensor.

    Returns
    -------
    torch.Tensor
        Propagation matrix, shape (n_x*n_y, n_x*n_y), complex64.
    """
    # Build a temporary MonoSIMoptimizer to compute the W matrix via its
    # closed-form Green's function propagation
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


def tipology_tag(
    sim_type: str,
    constructive: bool = False,
    alignment_side: str | None = None,
) -> str:
    """Return the tipology subfolder name for an experiment variant.

    - ``disjoint_asym``: ``"{equalized|constructive}/{tx|rx}"``
    - ``disjoint_sym`` / ``disjoint_4sim`` / ``mono``:
      ``"equalized"`` or ``"constructive"``.
    - ``joint``: ``""`` — no tipology subfolder.
    """
    if sim_type == "disjoint_asym":
        side = alignment_side or "tx"
        prefix = "constructive" if constructive else "equalized"
        return f"{prefix}/{side}"
    if sim_type in ("disjoint_sym", "disjoint_4sim"):
        return "constructive" if constructive else "equalized"
    if sim_type == "mono":
        side = alignment_side or "tx"
        prefix = "constructive" if constructive else "equalized"
        return f"{prefix}/{side}"
    return ""


def save_result(path: Path, row: dict) -> None:
    """Write a single result row to a Parquet file, creating parent directories as needed.

    Parameters
    ----------
    path : Path
        Destination file path (will be created/overwritten).
    row : dict
        Key-value pairs for the result row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_parquet(path)


def phase_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Trainable phases of a SIM module on CPU, keyed as in its ``state_dict``.

    Restore with ``model.load_state_dict(state, strict=False)`` on a model built
    with the same geometry (the propagation buffers are rebuilt, not stored).
    """
    return {k: v.detach().cpu() for k, v in model.named_parameters()}


def params_path(path: Path) -> Path:
    """The ``.pt`` file holding the trained state of the run whose result is ``path``."""
    return path.with_suffix(".pt")


def save_params(path: Path, params: dict) -> None:
    """Save a run's trained state next to its result Parquet ``path``.

    Tensors are moved to CPU; ``params`` typically holds ``phases`` (one
    :func:`phase_state` per SIM), the final cascades, ``H`` and ``beta``, so the
    link can be re-evaluated (other SNR, previews, ...) without retraining.
    """

    def _cpu(v):
        """Recursively detach tensors and move them to CPU (dicts are traversed)."""
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        if isinstance(v, dict):
            return {k: _cpu(x) for k, x in v.items()}
        return v

    path = params_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu(params), path)


def load_params(path: Path, map_location: str | torch.device = "cpu") -> dict:
    """Load the trained state saved by :func:`save_params` (``path``: Parquet or ``.pt``)."""
    return torch.load(params_path(Path(path)), map_location=map_location, weights_only=True)


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
    n_clusters: int | None = None,
    dataset: str = "cifar10",
    preconditioner: str = "none",
    extra_tag: str = "",
    channel_type: str = "rayleigh",
    rice_factor_db: float | None = None,
    cluster_on: str = "pilots",
    **_,
) -> str:
    """Build a canonical Parquet filename from all experiment fields.

    The filename encodes every independent variable so that results are
    self-documenting and unambiguous when stored on disk.

    Parameters
    ----------
    strategy : str
        Optimization strategy name (e.g. ``"joint"``, ``"mono"``).
    seed : int
        Random seed for reproducibility.
    L : int
        Number of SIM layers.
    M_int : int
        Intermediate dimension (number of meta-atoms per side).
    ch_dim : int
        Total channel dimension (n_x × n_y).
    snr : float or None
        Signal-to-noise ratio; ``None`` means noiseless.
    tx_enc, rx_enc : str
        TX and RX encoder types.
    s_tx, s_rx : int or None
        Sparsity parameters for TX and RX.
    n_clusters : int or None
        Number of clusters (for clustered experiments).
    dataset : str
        Dataset name (default ``"cifar10"``).
    preconditioner : str
        Preconditioner type (default ``"none"``).
    channel_type, rice_factor_db
        MIMO channel model; ``"rician"`` adds ``_rician{K}dB`` (Rayleigh adds nothing).
    cluster_on : str
        PPFE clustering (``alignment.cluster_on``); ``"train_tx"`` turns the
        cluster tag into ``_k{n}tx`` (pilot clustering keeps ``_k{n}``).

    Returns
    -------
    str
        Canonical filename ending in ``.parquet``.
    """
    s_tx_tag = "none" if s_tx is None else str(s_tx)
    s_rx_tag = "none" if s_rx is None else str(s_rx)
    k_tag = f"_k{n_clusters}" if n_clusters is not None else ""
    if k_tag and cluster_on == "train_tx":
        k_tag += "tx"
    prec_tag = f"_prec{preconditioner}" if preconditioner != "none" else ""
    ch_tag = f"_rician{rice_factor_db}dB" if channel_type == "rician" else ""
    return (
        f"{dataset}_{strategy}_seed{seed}_L{L}_M{M_int}x{M_int}_ch{ch_dim}"
        f"_snr{snr_tag(snr)}_stx{s_tx_tag}_srx{s_rx_tag}{k_tag}{prec_tag}"
        f"{extra_tag}{ch_tag}_{tx_enc}_{rx_enc}.parquet"
    )


def ch_dims(N: int) -> tuple[int, int]:
    """Factorize N into ``(n_x, n_y)`` with ``n_x * n_y == N``, as square as possible.

    Parameters
    ----------
    N : int
        Total channel dimension to factorize.

    Returns
    -------
    tuple[int, int]
        ``(n_x, n_y)`` with ``n_x >= n_y`` and ``n_x * n_y == N``.
    """
    # Search downward from sqrt(N) to find the largest divisor
    for n_y in range(int(N**0.5), 0, -1):
        if N % n_y == 0:
            return N // n_y, n_y
    # Prime N: return N×1
    return N, 1


def pct(v: float | None) -> float | None:
    """Pass through a value in ``[0, 1]``; pass ``None`` through unchanged."""
    return v


def neural_str(label: str, acc_pct: float | None) -> str:
    """Format an optional neural-classifier accuracy for console logging."""
    return f" {label}={acc_pct * 100:.2f}%" if acc_pct is not None else ""

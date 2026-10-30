"""Original (non-unrolled) SIM runners and optimizer classes."""

from ._base import SIMRunner
from ._utils import (
    RECONSTRUCTION_RESULTS_DIR,
    RESULTS_DIR,
    _FixedCascade,
    ch_dims,
    inter_sim_propagation,
    load_params,
    make_fname,
    neural_str,
    pct,
    phase_state,
    save_params,
    save_result,
    snr_tag,
    tipology_tag,
)
from .disjoint_2sim import Disjoint2SIM, DisjointTargets
from .disjoint_4sim import Disjoint4SIM
from .disjoint_asym import DisjointAsymSIM
from .disjoint_sym import DisjointSymSIM
from .joint import JointSIM
from .mono import MonoSIM

# Optimizer classes — both NumPy and Torch variants for mono/dual SIM.
from .optimizers import (
    DualSIMoptimizer,
    DualSIMoptimizerTorch,
    MonoSIMoptimizer,
    MonoSIMoptimizerTorch,
)

# All public names: runner ABC, utility helpers, runner classes, optimizers.
__all__ = [
    "SIMRunner",
    "RESULTS_DIR",
    "RECONSTRUCTION_RESULTS_DIR",
    "_FixedCascade",
    "ch_dims",
    "inter_sim_propagation",
    "load_params",
    "make_fname",
    "neural_str",
    "pct",
    "phase_state",
    "save_params",
    "save_result",
    "snr_tag",
    "tipology_tag",
    "DisjointTargets",
    "Disjoint2SIM",
    "Disjoint4SIM",
    "DisjointAsymSIM",
    "DisjointSymSIM",
    "JointSIM",
    "MonoSIM",
    "DualSIMoptimizer",
    "DualSIMoptimizerTorch",
    "MonoSIMoptimizer",
    "MonoSIMoptimizerTorch",
]

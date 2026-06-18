"""Experiment runners — one entry point per SIM architecture.

Each runner's YAML config carries a ``_target_`` field; Hydra's
``instantiate`` calls the right function automatically with no match
statement in the calling script.

Runners
-------
- :func:`run_joint_sim`      — joint Dual-SIM (layers / SNR / s-layer sweeps)
- :func:`run_mono_sim`       — Mono-SIM TX-only (layers / SNR sweeps)
- :func:`run_disjoint_asym`  — disjoint asym (TX→A, RX→H†)
- :func:`run_disjoint_sym`   — disjoint sym with full CSI
- :func:`run_disjoint_4sim`  — disjoint 4-SIM with full CSI
- :func:`run_disjoint_2sim`  — core 2-SIM runner (accepts pre-computed targets)

Target helpers
--------------
- :func:`compute_asym_targets` — asymmetric TX/RX targets
- :func:`compute_sym_targets`  — symmetric (CSI) TX/RX targets
"""

from semantic_alignment.experiments.disjoint_2sim import run_disjoint_2sim
from semantic_alignment.experiments.disjoint_4sim import run_disjoint_4sim
from semantic_alignment.experiments.disjoint_asym import compute_asym_targets, run_disjoint_asym
from semantic_alignment.experiments.disjoint_sym import compute_sym_targets, run_disjoint_sym
from semantic_alignment.experiments.joint import run_joint_sim
from semantic_alignment.experiments.mono import run_mono_sim

__all__ = [
    "run_joint_sim",
    "run_mono_sim",
    "run_disjoint_asym",
    "run_disjoint_sym",
    "run_disjoint_4sim",
    "run_disjoint_2sim",
    "compute_asym_targets",
    "compute_sym_targets",
]

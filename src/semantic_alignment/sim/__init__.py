"""Sim sub-package: SIM phase-profile optimizers.

:class:`DualSIMoptimizer` / :class:`DualSIMoptimizerTorch`
    Optimize the phase profiles of a *dual* SIM stack (one SIM at TX, one at RX)
    to jointly maximize semantic alignment through the MIMO channel.

:class:`MonoSIMoptimizer` / :class:`MonoSIMoptimizerTorch`
    Single-SIM variant where only the TX-side metasurface is optimized; the RX
    applies a conventional MMSE equalizer.

NumPy variants use closed-form or iterative solvers; Torch variants enable
gradient-based optimization on GPU.
"""

from semantic_alignment.sim.dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from semantic_alignment.sim.monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch

__all__ = [
    "DualSIMoptimizer",
    "DualSIMoptimizerTorch",
    "MonoSIMoptimizer",
    "MonoSIMoptimizerTorch",
]

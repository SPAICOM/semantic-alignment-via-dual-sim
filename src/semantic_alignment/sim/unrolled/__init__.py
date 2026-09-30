"""Unrolled SIM optimizers — end-to-end differentiable cascade simulation.

This sub-package provides:

- **Backbone classes** (``MonoUnrolledSCA``, ``DualUnrolledSCA``) that implement
  the forward cascade, analytic backward pass, and loss evaluation.
- **Concrete subclasses** for mono, joint, disjoint-symmetric, and disjoint-asymmetric
  SIM configurations.
- **Step modules** (``ScalarStep``, ``DiagonalStep``, ``ConvStep``) that parameterize
  the per-layer linear transforms.
- **Cascade helpers** re-exported from ``_cascade.py`` for building TX/RX cascades
  and their backward passes.
"""

# Re-export cascade helpers with leading-underscore aliases (internal use).
from .._cascade import (
    build_cascade_RX as _build_cascade_RX,
)
from .._cascade import (
    build_cascade_TX as _build_cascade_TX,
)
from .._cascade import (
    cascade_backward_RX as _cascade_backward_RX,
)
from .._cascade import (
    cascade_backward_TX as _cascade_backward_TX,
)

# Backbone base classes for mono and dual SIM.
from ._base import DualUnrolledSCA, MonoUnrolledSCA

# Disjoint base class shared by symmetric and asymmetric variants.
from ._disjoint import DisjointUnrolledSCA

# Per-layer step modules: scalar (single weight), diagonal, or convolution.
from ._step import ConvStep, DiagonalStep, ScalarStep, build_step

# Utility functions for phase computation, NMSE, normalization, and splitting.
from ._utils import _nmse, _normalize, _phase_from_logit

# Concrete disjoint runners.
from .disjoint_asym import DisjointAsymUnrolledSCA
from .disjoint_sym import DisjointSymUnrolledSCA

# Concrete dual-SIM runners (joint and dual).
from .dual import DualSIMUnrolledSCA, JointSIMUnrolledSCA

# Concrete mono-SIM runner.
from .mono import MonoSIMUnrolledSCA

__all__ = [
    # backbone base classes
    "MonoUnrolledSCA",
    "DualUnrolledSCA",
    # single-SIM
    "MonoSIMUnrolledSCA",
    # dual-SIM — joint
    "JointSIMUnrolledSCA",
    "DualSIMUnrolledSCA",  # backward-compat alias for JointSIMUnrolledSCA
    # dual-SIM — disjoint family
    "DisjointUnrolledSCA",
    "DisjointSymUnrolledSCA",
    "DisjointAsymUnrolledSCA",
    # step modules
    "ScalarStep",
    "DiagonalStep",
    "ConvStep",
    "build_step",
    # cascade helpers
    "_build_cascade_TX",
    "_build_cascade_RX",
    "_cascade_backward_TX",
    "_cascade_backward_RX",
    # utilities
    "_phase_from_logit",
    "_nmse",
    "_normalize",
]

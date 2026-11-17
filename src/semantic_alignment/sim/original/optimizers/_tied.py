"""Shared storage for the propagation matrices of a SIM stack.

All intermediate → intermediate matrices of a stack are identical (same lattice,
same spacing), so ``_build_W_list`` returns one array repeated L-2 times. Keeping
them as one tensor matters at large M: at 96x96 meta-atoms each matrix is ~650 MB
in complex64, i.e. ~31 GB for L=50 if every layer held its own copy.
"""

import numpy as np
import torch
import torch.nn as nn


class TiedBuffersModule(nn.Module):
    """``nn.Module`` whose buffers registered as the same tensor stay one tensor.

    ``Module._apply`` (used by ``.to``/``.cuda``/``.float``…) converts every buffer
    name separately, which would turn one shared tensor into L copies on the
    device. Here each distinct tensor is converted once and re-tied first; the
    base pass that follows then hits already-converted tensors and returns them
    unchanged (all ``_apply`` callers are idempotent conversions).
    """

    def register_W_buffers(self, prefix: str, W_list: list[np.ndarray]) -> None:
        """Register ``{prefix}_{i}`` buffers, one tensor per distinct array in ``W_list``."""
        tensors: dict[int, torch.Tensor] = {}
        for i, W in enumerate(W_list):
            if id(W) not in tensors:
                tensors[id(W)] = torch.tensor(W, dtype=torch.complex64)
            self.register_buffer(f"{prefix}_{i}", tensors[id(W)])

    def _apply(self, fn, recurse=True):
        """Convert each distinct buffer once, re-tie aliases, then defer to ``nn.Module``."""
        converted: dict[int, torch.Tensor] = {}
        for name, buf in self._buffers.items():
            if buf is not None:
                if id(buf) not in converted:
                    converted[id(buf)] = fn(buf)
                self._buffers[name] = converted[id(buf)]
        return super()._apply(fn, recurse)

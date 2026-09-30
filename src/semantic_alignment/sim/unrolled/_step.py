"""Learnable unrolled step-size modules.

Each module owns step-size parameters (w, mu, s) and applies one unrolled step:

    xi^{k+1} = s^k * xi^k + step^k(g^k)

where:
    - xi are the phase logits
    - g are the gradients (∂L/∂ξ)
    - w is the step-size (learning rate per iteration)
    - mu is the momentum coefficient (EMA of past gradients)
    - s is a learned scaling factor (defaults to 1 = no scaling)

Interface shared by all subclasses:
    apply(phases, g_list, v_list, k) -> (phases_new, v_new)
    init_momentum(B, sizes, device)  -> list[Tensor] | None
    reg_S()                          -> scalar Tensor
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _near_square(n: int) -> tuple[int, int]:
    """(H, W) with H*W == n and H <= W minimising |H - W|."""
    h = int(math.isqrt(n))
    while n % h != 0:
        h -= 1
    return h, n // h


def _group_layers(sizes: list[int], shapes: list[tuple[int, int]]) -> list[dict]:
    """Group consecutive layers with the same spatial shape.

    Returns a list of dicts: {indices, size, shape, count}.
    """
    segs, i = [], 0
    while i < len(sizes):
        sh, j = shapes[i], i + 1
        while j < len(sizes) and shapes[j] == sh:
            j += 1
        segs.append(dict(indices=list(range(i, j)), size=sizes[i], shape=sh, count=j - i))
        i = j
    return segs


def build_step(
    coupling: str, K: int, sizes: list[int], shapes: list[tuple[int, int]] | None = None, **kwargs
) -> "UnrolledStep":
    """Factory — return the right UnrolledStep subclass.

    Parameters
    ----------
    coupling : str
        Step module type: ``"scalar"``, ``"diagonal"``, ``"conv"``, or ``"conv2d"``.
    K : int
        Number of unrolled iterations (defines parameter dimensions).
    sizes : list[int]
        Phase parameter counts per SIM layer.
    shapes : list[tuple[int, int]] or None
        Spatial grid shapes ``(H, W)`` per layer; required for conv variants.

    Returns
    -------
    UnrolledStep
        Configured step module.
    """
    if coupling == "scalar":
        return ScalarStep(K, sizes, **kwargs)
    if coupling == "diagonal":
        return DiagonalStep(K, sizes, **kwargs)
    if coupling in ("conv", "conv2d"):
        if shapes is None:
            shapes = [_near_square(s) for s in sizes]
        per_layer = coupling == "conv2d"
        return ConvStep(K, sizes, shapes, per_layer=per_layer, **kwargs)
    raise ValueError(
        f"coupling must be 'scalar', 'diagonal', 'conv', or 'conv2d'; got {coupling!r}"
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class UnrolledStep(nn.Module):
    """Abstract base class for unrolled step modules.

    Subclasses must implement:
        - ``apply(phases, g_list, v_list, k)`` — compute one optimization step.
        - ``init_momentum(B, sizes, device)`` — initialize momentum buffers.
        - ``reg_S()`` — regularization loss for the S parameter (if learned).
    """

    def apply(
        self,
        phases: list[torch.Tensor],
        g_list: list[torch.Tensor],
        v_list: list[torch.Tensor] | None,
        k: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Compute one unrolled optimization step.

        Parameters
        ----------
        phases : list[Tensor]
            Current phase logits, one per SIM layer.
        g_list : list[Tensor]
            Gradients ∂L/∂ξ for each layer.
        v_list : list[Tensor] or None
            Momentum buffers (EMA of past gradients); None if no momentum.
        k : int
            Current unrolled iteration index.

        Returns
        -------
        tuple[list[Tensor], list[Tensor] | None]
            ``(phases_new, v_new)`` updated phases and momentum buffers.
        """
        raise NotImplementedError

    def init_momentum(
        self, B: int, sizes: list[int], device: torch.device
    ) -> list[torch.Tensor] | None:
        """Initialize momentum buffers as zero tensors.

        Parameters
        ----------
        B : int
            Batch size.
        sizes : list[int]
            Phase parameter counts per layer.
        device : torch.device
            Target device.

        Returns
        -------
        list[Tensor] or None
            Zero-initialized momentum buffers, or None if momentum is disabled.
        """
        raise NotImplementedError

    def reg_S(self) -> torch.Tensor:
        """Return regularization loss for the S (scaling) parameter.

        Returns
        -------
        torch.Tensor
            Scalar regularization loss (0 if S is not learned).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ScalarStep — one scalar w per unrolled step k
# ---------------------------------------------------------------------------


class ScalarStep(UnrolledStep):
    """Scalar unrolled step: one shared step-size w per iteration k.

    Each iteration applies:
        xi_new = s_k * xi + w_k * g_eff

    where g_eff is the momentum-smoothed gradient (if momentum is enabled).

    Parameters
    ----------
    K : int
        Number of unrolled iterations.
    sizes : list[int]
        Phase parameter counts per SIM layer.
    momentum : bool
        Enable exponential moving average of gradients.
    init_mu : float
        Initial momentum coefficient (in (0, 1)); stored as logit for stability.
    learn_S : bool
        If True, learn a per-iteration scaling factor s_k.
    init_w : float
        Initial step-size value for all iterations.
    eps : float
        Small constant for numerical stability.
    """

    def __init__(
        self,
        K: int,
        sizes: list[int],
        momentum: bool = True,
        init_mu: float = 0.9,
        learn_S: bool = False,
        init_w: float = 0.1,
        eps: float = 1e-12,
        **_,
    ):
        super().__init__()
        self.K, self.sizes = K, sizes
        self.momentum, self.learn_S, self.eps = momentum, learn_S, eps
        # Step-size: one scalar per iteration, shape (K,)
        self.w = nn.Parameter(torch.full((K,), float(init_w)))
        if momentum:
            # Momentum coefficient: stored as logit for unconstrained optimization
            self.mu = nn.Parameter(torch.full((K,), float(torch.logit(torch.tensor(init_mu)))))
        if learn_S:
            # Learned scaling factor: initialized to 1 (no scaling)
            self.s = nn.Parameter(torch.ones(K))

    def apply(
        self,
        phases: list[torch.Tensor],
        g_list: list[torch.Tensor],
        v_list: list[torch.Tensor] | None,
        k: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Apply one scalar step: xi_new = s_k * xi + w_k * g_eff.

        Parameters
        ----------
        phases : list[Tensor]
            Current phase logits.
        g_list : list[Tensor]
            Gradients for each layer.
        v_list : list[Tensor] or None
            Momentum buffers; None if momentum is disabled.
        k : int
            Current iteration index.

        Returns
        -------
        tuple[list[Tensor], list[Tensor] | None]
            Updated phases and momentum buffers.
        """
        w_k = self.w[k]
        if self.momentum:
            # Compute bias-corrected momentum: v_t / (1 - μ^(t+1))
            mu_k = torch.sigmoid(self.mu[k]).clamp(max=0.999)
            denom = (1.0 - mu_k.detach() ** (k + 1)).clamp(min=1e-4)
            v_new = [mu_k * v + (1 - mu_k) * g for v, g in zip(v_list, g_list)]
            g_eff = [v / denom for v in v_new]
        else:
            v_new, g_eff = None, g_list
        if self.learn_S:
            # Apply learned scaling: xi_new = s_k * xi + w_k * g_eff
            s_k = self.s[k]
            return [s_k * x + w_k * g for x, g in zip(phases, g_eff)], v_new
        # Standard step: xi_new = xi + w_k * g_eff
        return [x + w_k * g for x, g in zip(phases, g_eff)], v_new

    def init_momentum(
        self, B: int, sizes: list[int], device: torch.device
    ) -> list[torch.Tensor] | None:
        """Initialize momentum buffers as zero tensors.

        Parameters
        ----------
        B : int
            Batch size.
        sizes : list[int]
            Phase parameter counts per layer.
        device : torch.device
            Target device.

        Returns
        -------
        list[Tensor] or None
            Zero-initialized momentum buffers, or None if momentum is disabled.
        """
        return [torch.zeros(B, m, device=device) for m in sizes] if self.momentum else None

    def reg_S(self) -> torch.Tensor:
        """Return regularization loss for the S parameter.

        Returns
        -------
        torch.Tensor
            L2 penalty pulling s toward 1 (if learn_S), else 0.
        """
        if not self.learn_S:
            return self.w.new_zeros(())
        return (self.s - 1.0).pow(2).sum()


# ---------------------------------------------------------------------------
# DiagonalStep — one scalar per phase coordinate per step k
# ---------------------------------------------------------------------------


class DiagonalStep(UnrolledStep):
    """Diagonal unrolled step: one step-size per phase coordinate per iteration.

    Each iteration applies:
        xi_new[i] = s_k[i] * xi[i] + w_k[i] * g_eff[i]

    where w_k and s_k are per-coordinate vectors of length sum(sizes).

    Parameters
    ----------
    K : int
        Number of unrolled iterations.
    sizes : list[int]
        Phase parameter counts per SIM layer.
    momentum : bool
        Enable exponential moving average of gradients.
    init_mu : float
        Initial momentum coefficient (in (0, 1)).
    learn_S : bool
        If True, learn per-coordinate scaling factors.
    init_w : float
        Initial step-size value.
    eps : float
        Small constant for numerical stability.
    """

    def __init__(
        self,
        K: int,
        sizes: list[int],
        momentum: bool = True,
        init_mu: float = 0.9,
        learn_S: bool = False,
        init_w: float = 0.1,
        eps: float = 1e-12,
        **_,
    ):
        super().__init__()
        self.K, self.sizes = K, sizes
        self.momentum, self.learn_S, self.eps = momentum, learn_S, eps
        n = sum(sizes)
        # Step-size: one scalar per coordinate per iteration, shape (K, n)
        self.w = nn.Parameter(torch.full((K, n), float(init_w)))
        if momentum:
            self.mu = nn.Parameter(torch.full((K, n), float(torch.logit(torch.tensor(init_mu)))))
        if learn_S:
            self.s = nn.Parameter(torch.ones(K, n))

    def _split(self, t: torch.Tensor) -> list[torch.Tensor]:
        """Split a flat (K, n) tensor into per-layer lists of (1, m_i) tensors."""
        return [c[None, :] for c in torch.split(t, self.sizes)]

    def apply(
        self,
        phases: list[torch.Tensor],
        g_list: list[torch.Tensor],
        v_list: list[torch.Tensor] | None,
        k: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Apply one diagonal step: xi_new[i] = s_k[i] * xi[i] + w_k[i] * g_eff[i].

        Parameters
        ----------
        phases : list[Tensor]
            Current phase logits.
        g_list : list[Tensor]
            Gradients for each layer.
        v_list : list[Tensor] or None
            Momentum buffers; None if momentum is disabled.
        k : int
            Current iteration index.

        Returns
        -------
        tuple[list[Tensor], list[Tensor] | None]
            Updated phases and momentum buffers.
        """
        w_split = self._split(self.w[k])
        if self.momentum:
            # Per-coordinate momentum with bias correction
            mu_split = [torch.sigmoid(m).clamp(max=0.999) for m in self._split(self.mu[k])]
            denom = [(1.0 - mu.detach() ** (k + 1)).clamp(min=1e-4) for mu in mu_split]
            v_new = [mu * v + (1 - mu) * g for mu, v, g in zip(mu_split, v_list, g_list)]
            g_eff = [v / d for v, d in zip(v_new, denom)]
        else:
            v_new, g_eff = None, g_list
        if self.learn_S:
            s_split = self._split(self.s[k])
            return [s * x + w * g for s, x, w, g in zip(s_split, phases, w_split, g_eff)], v_new
        return [x + w * g for x, w, g in zip(phases, w_split, g_eff)], v_new

    def init_momentum(
        self, B: int, sizes: list[int], device: torch.device
    ) -> list[torch.Tensor] | None:
        """Initialize momentum buffers as zero tensors.

        Parameters
        ----------
        B : int
            Batch size.
        sizes : list[int]
            Phase parameter counts per layer.
        device : torch.device
            Target device.

        Returns
        -------
        list[Tensor] or None
            Zero-initialized momentum buffers, or None if momentum is disabled.
        """
        return [torch.zeros(B, m, device=device) for m in sizes] if self.momentum else None

    def reg_S(self) -> torch.Tensor:
        """Return regularization loss for the S parameter.

        Returns
        -------
        torch.Tensor
            L2 penalty pulling s toward 1 (if learn_S), else 0.
        """
        if not self.learn_S:
            return self.w.new_zeros(())
        return (self.s - 1.0).pow(2).sum()


# ---------------------------------------------------------------------------
# ConvStep — conv2d/conv3d kernel per segment per step k
# ---------------------------------------------------------------------------


class ConvStep(UnrolledStep):
    """Convolutional unrolled step.

    Consecutive SIM layers with the same spatial shape are grouped into
    segments and updated with:
      - conv3d  if the segment has more than one layer (depth = n_layers)
      - conv2d  if the segment has exactly one layer

    Parameters
    ----------
    K : int
    sizes : list[int]   phases per SIM layer
    shapes : list[tuple[int,int]]  (H, W) grid shape per SIM layer
    k_spatial : int  spatial kernel size for w kernels, default 3
    k_layer : int  depth kernel size for w conv3d, default 1
    k_spatial_s : int  spatial kernel size for S kernel (defaults to k_spatial)
    k_layer_s : int  depth kernel size for S conv3d (defaults to k_layer)
    mu_mode : str   ``"scalar"`` (K,) | ``"diagonal"`` (K, n)
    s_mode : str    ``"scalar"`` (K,) | ``"diagonal"`` (K, n) | ``"kernel"``
    reg_s_mode : str  ``"none"`` | ``"l2"`` | ``"identity"``
        Regulariser applied to S when ``learn_S=True``.
        ``"none"``     — no regularisation (default).
        ``"l2"``       — L2 on weights; prevents large values.
        ``"identity"`` — pulls toward 1 (scalar/diagonal) or identity kernel.
    init_w : float  centre-pixel init for w kernels
    """

    def __init__(
        self,
        K: int,
        sizes: list[int],
        shapes: list[tuple[int, int]],
        momentum: bool = True,
        init_mu: float = 0.9,
        learn_S: bool = False,
        init_w: float = 0.1,
        k_spatial: int = 3,
        k_layer: int = 1,
        k_spatial_s: int | None = None,
        k_layer_s: int | None = None,
        mu_mode: str = "scalar",
        s_mode: str = "scalar",
        reg_s_mode: str = "none",
        w_mode: str = "kernel",
        per_layer: bool = False,
        eps: float = 1e-12,
        **_,
    ):
        """Initialize ConvStep with convolutional kernels for step-size and scaling.

        Parameters
        ----------
        K : int
            Number of unrolled iterations.
        sizes : list[int]
            Phase parameter counts per SIM layer.
        shapes : list[tuple[int, int]]
            Spatial grid shapes ``(H, W)`` per layer.
        momentum : bool
            Enable exponential moving average of gradients.
        init_mu : float
            Initial momentum coefficient.
        learn_S : bool
            If True, learn spatial convolutional scaling factors.
        init_w : float
            Center-pixel initialization for w kernels.
        k_spatial : int
            Spatial kernel size for w kernels.
        k_layer : int
            Depth kernel size for w conv3d.
        k_spatial_s : int or None
            Spatial kernel size for S kernel (defaults to k_spatial).
        k_layer_s : int or None
            Depth kernel size for S conv3d (defaults to k_layer).
        mu_mode : str
            ``"scalar"`` for one mu per iteration, ``"diagonal"`` per coordinate.
        s_mode : str
            ``"scalar"``, ``"diagonal"``, ``"kernel"``, or ``"hybrid"``.
        reg_s_mode : str
            ``"none"``, ``"l2"``, or ``"identity"``.
        w_mode : str
            ``"kernel"`` for pure convolution, ``"hybrid"`` for conv + diagonal.
        per_layer : bool
            If True, one kernel per SIM layer (no shared conv3d).
        eps : float
            Small constant for numerical stability.
        """
        super().__init__()
        assert len(sizes) == len(shapes)
        assert mu_mode in ("scalar", "diagonal"), (
            f"mu_mode must be 'scalar' or 'diagonal'; got {mu_mode!r}"
        )
        assert s_mode in ("scalar", "diagonal", "kernel", "hybrid"), (
            f"s_mode must be 'scalar', 'diagonal', 'kernel', or 'hybrid'; got {s_mode!r}"
        )
        assert reg_s_mode in ("none", "l2", "identity"), (
            f"reg_s_mode must be 'none', 'l2', or 'identity'; got {reg_s_mode!r}"
        )
        assert w_mode in ("kernel", "hybrid"), (
            f"w_mode must be 'kernel' or 'hybrid'; got {w_mode!r}"
        )
        self.K, self.sizes = K, sizes
        self.momentum, self.learn_S, self.eps = momentum, learn_S, eps
        self.k_spatial, self.k_layer = k_spatial, k_layer
        self.k_spatial_s = k_spatial_s if k_spatial_s is not None else k_spatial
        self.k_layer_s = k_layer_s if k_layer_s is not None else k_layer
        self.mu_mode, self.s_mode, self.reg_s_mode = mu_mode, s_mode, reg_s_mode
        self.w_mode = w_mode
        self.per_layer = per_layer

        if per_layer:
            # One segment per SIM layer (no shared conv3d).
            self.segs = [
                dict(indices=[i], size=sizes[i], shape=shapes[i], count=1)
                for i in range(len(sizes))
            ]
            self._is_3d = [False] * len(sizes)
            # Map each layer index → sorted list of indices in its same-shape group.
            # Used for cross-layer neighbor gathering when k_layer > 1.
            self._per_layer_groups: list[list[int]] = []
            for group in _group_layers(sizes, shapes):
                for _ in group["indices"]:
                    self._per_layer_groups.append(group["indices"])
        else:
            self.segs = _group_layers(sizes, shapes)
            self._is_3d = []
            self._per_layer_groups = []  # unused in grouped mode

        n_total = sum(sizes)
        pad_w = k_spatial // 2
        pad_s = self.k_spatial_s // 2

        # Cumulative offsets into the flat phase vector, one entry per layer.
        self._offsets: list[int] = []
        acc = 0
        for s in sizes:
            self._offsets.append(acc)
            acc += s
        self._offsets.append(acc)

        # Diagonal component for hybrid w_mode: (K, n_total), init 0 so the conv
        # kernel alone defines the initial step (avoids doubling init_w).
        if w_mode == "hybrid":
            self.w_diag = nn.Parameter(torch.zeros(K, n_total))

        # Step-size kernels (w).
        # per_layer=False: one kernel per segment (shared across layers in group).
        # per_layer=True:  one kernel per SIM layer with kl input channels (neighbors).
        self.kernels = nn.ParameterList()
        if per_layer:
            kl = k_layer
            for seg in self.segs:
                # kernel: (K, out_ch=1, in_ch=kl, ks, ks)
                kp = nn.Parameter(torch.zeros(K, 1, kl, k_spatial, k_spatial))
                with torch.no_grad():
                    kp[:, 0, kl // 2, pad_w, pad_w] = init_w
                self.kernels.append(kp)
        else:
            for seg in self.segs:
                n = seg["count"]
                if n == 1:
                    kp = nn.Parameter(torch.zeros(K, 1, 1, k_spatial, k_spatial))
                    with torch.no_grad():
                        kp[:, 0, 0, pad_w, pad_w] = init_w
                    self._is_3d.append(False)
                else:
                    kl = min(k_layer, n)
                    kp = nn.Parameter(torch.zeros(K, 1, 1, kl, k_spatial, k_spatial))
                    with torch.no_grad():
                        kp[:, 0, 0, kl // 2, pad_w, pad_w] = init_w
                    self._is_3d.append(True)
                self.kernels.append(kp)

        # mu
        if momentum:
            mu0 = float(torch.logit(torch.tensor(init_mu)))
            self.mu = nn.Parameter(torch.full((K,) if mu_mode == "scalar" else (K, n_total), mu0))

        # S — scalar, diagonal, kernel, or hybrid (diagonal × kernel)
        if learn_S:
            if s_mode in ("scalar", "hybrid"):
                # hybrid needs a scalar or diagonal; we always use diagonal for hybrid
                self.s = nn.Parameter(
                    torch.ones(K) if s_mode == "scalar" else torch.ones(K, n_total)
                )
            elif s_mode == "diagonal":
                self.s = nn.Parameter(torch.ones(K, n_total))
            if s_mode in ("kernel", "hybrid"):
                self.s_kernels = nn.ParameterList()
                if per_layer:
                    kl_s = self.k_layer_s
                    for seg in self.segs:
                        sp = nn.Parameter(
                            torch.zeros(K, 1, kl_s, self.k_spatial_s, self.k_spatial_s)
                        )
                        with torch.no_grad():
                            sp[:, 0, kl_s // 2, pad_s, pad_s] = 1.0
                        self.s_kernels.append(sp)
                else:
                    for seg, is_3d in zip(self.segs, self._is_3d):
                        n = seg["count"]
                        if not is_3d:
                            sp = nn.Parameter(
                                torch.zeros(K, 1, 1, self.k_spatial_s, self.k_spatial_s)
                            )
                            with torch.no_grad():
                                sp[:, 0, 0, pad_s, pad_s] = 1.0
                        else:
                            kl_s = min(self.k_layer_s, n)
                            sp = nn.Parameter(
                                torch.zeros(K, 1, 1, kl_s, self.k_spatial_s, self.k_spatial_s)
                            )
                            with torch.no_grad():
                                sp[:, 0, 0, kl_s // 2, pad_s, pad_s] = 1.0
                        self.s_kernels.append(sp)

    def _conv_seg(
        self, x_seg: list[torch.Tensor], kernel_k: torch.Tensor, seg: dict, three_d: bool
    ) -> list[torch.Tensor]:
        """Apply conv2d or conv3d to a segment of phase tensors.

        Parameters
        ----------
        x_seg : list[Tensor]
            Phase tensors for layers in this segment, each shape (B, H*W).
        kernel_k : Tensor
            Convolutional kernel for iteration k.
        seg : dict
            Segment metadata with ``"shape"`` and ``"count"``.
        three_d : bool
            If True, use conv3d (depth = segment layer count).

        Returns
        -------
        list[Tensor]
            Convolved tensors, same shapes as input.
        """
        B = x_seg[0].shape[0]
        H, W = seg["shape"]
        n = seg["count"]
        # Same-padding from spatial kernel size
        pad_s = kernel_k.shape[-1] // 2
        if not three_d:
            # Single layer: reshape (B, H*W) → (B, 1, H, W) for conv2d
            out = F.conv2d(x_seg[0].reshape(B, 1, H, W), kernel_k, padding=pad_s)
            return [out.reshape(B, H * W)]
        # Multiple layers: reshape (B, n, H*W) → (B, 1, n, H, W) for conv3d
        x = torch.stack(x_seg, dim=1).reshape(B, 1, n, H, W)
        kl = kernel_k.shape[2]
        out = F.conv3d(x, kernel_k, padding=(kl // 2, pad_s, pad_s))
        return list(out.squeeze(1).reshape(B, n, H * W).unbind(dim=1))

    def _conv_per_layer(
        self, all_tensors: list[torch.Tensor], kernel_k: torch.Tensor, seg_idx: int, seg: dict
    ) -> torch.Tensor:
        """Per-layer conv2d with cross-layer neighbors.

        Gathers a window of k_layer tensors from the same-shape group (zero-
        padded at boundaries), stacks them as input channels, and applies the
        per-layer kernel (out=1, in=k_layer).

        Parameters
        ----------
        all_tensors : list[Tensor]
            All phase tensors across all layers.
        kernel_k : Tensor
            Per-layer convolutional kernel for iteration k.
        seg_idx : int
            Index of this segment in ``self.segs``.
        seg : dict
            Segment metadata with ``"shape"`` and ``"indices"``.

        Returns
        -------
        Tensor
            Convolved tensor, shape (B, H*W).
        """
        global_i = self.segs[seg_idx]["indices"][0]
        group_idxs = self._per_layer_groups[global_i]
        i_pos = group_idxs.index(global_i)
        H, W = seg["shape"]
        kl = kernel_k.shape[1]  # in_channels = k_layer
        pad_s = kernel_k.shape[-1] // 2
        B = all_tensors[global_i].shape[0]
        half = kl // 2
        # Zero-pad at boundaries of the same-shape group
        zero = torch.zeros(
            B, H * W, device=all_tensors[global_i].device, dtype=all_tensors[global_i].dtype
        )
        padded = [zero] * half + [all_tensors[j] for j in group_idxs] + [zero] * half
        # Extract window of kl tensors around current layer
        x_win = padded[i_pos : i_pos + kl]  # list of kl tensors (B, H*W)
        x = torch.stack(x_win, dim=1).reshape(B, kl, H, W)  # (B, kl, H, W)
        out = F.conv2d(x, kernel_k, padding=pad_s)  # (B, 1, H, W)
        return out.reshape(B, H * W)

    def apply(
        self,
        phases: list[torch.Tensor],
        g_list: list[torch.Tensor],
        v_list: list[torch.Tensor] | None,
        k: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Apply one convolutional step across all segments.

        For each segment, computes the step direction via convolution,
        applies momentum (if enabled), and updates phases with optional
        learned scaling S.

        Parameters
        ----------
        phases : list[Tensor]
            Current phase logits.
        g_list : list[Tensor]
            Gradients for each layer.
        v_list : list[Tensor] or None
            Momentum buffers; None if momentum is disabled.
        k : int
            Current iteration index.

        Returns
        -------
        tuple[list[Tensor], list[Tensor] | None]
            Updated phases and momentum buffers.
        """
        phases_new = [None] * len(phases)
        v_new = [None] * len(phases) if self.momentum else None

        # Scalar mu: compute once for all segments and layers
        if self.momentum and self.mu_mode == "scalar":
            mu_scalar = torch.sigmoid(self.mu[k]).clamp(max=0.999)
            denom_scalar = (1.0 - mu_scalar.detach() ** (k + 1)).clamp(min=1e-4)

        # Hybrid S: pre-scale phases by diagonal before spatial conv
        if self.learn_S and self.s_mode == "hybrid":
            phases_for_S = [
                self.s[k, self._offsets[i] : self._offsets[i + 1]] * phases[i]
                for i in range(len(phases))
            ]
        else:
            phases_for_S = phases

        for s_idx, seg in enumerate(self.segs):
            idxs = seg["indices"]

            # --- Step direction via convolution ---
            if self.per_layer:
                # One layer per segment; kernel has kl input channels (neighbors)
                u_list = [self._conv_per_layer(g_list, self.kernels[s_idx][k], s_idx, seg)]
            else:
                three_d = self._is_3d[s_idx]
                u_list = self._conv_seg(
                    [g_list[i] for i in idxs], self.kernels[s_idx][k], seg, three_d
                )

            # Hybrid w_mode: add per-position diagonal term before momentum
            if self.w_mode == "hybrid":
                u_list = [
                    u + self.w_diag[k, self._offsets[i] : self._offsets[i + 1]] * g_list[i]
                    for u, i in zip(u_list, idxs)
                ]

            # --- Momentum ---
            u_eff = []
            for i, u in zip(idxs, u_list):
                if self.momentum:
                    if self.mu_mode == "scalar":
                        mu_i, denom_i = mu_scalar, denom_scalar
                    else:  # diagonal
                        a, b = self._offsets[i], self._offsets[i + 1]
                        mu_i = torch.sigmoid(self.mu[k, a:b]).clamp(max=0.999)
                        denom_i = (1.0 - mu_i.detach() ** (k + 1)).clamp(min=1e-4)
                    # EMA update: v_t = μ * v_{t-1} + (1-μ) * u
                    v_i = mu_i * v_list[i] + (1 - mu_i) * u
                    v_new[i] = v_i
                    # Bias-corrected momentum
                    u_eff.append(v_i / denom_i)
                else:
                    u_eff.append(u)

            # --- Phase update with optional S scaling ---
            if not self.learn_S:
                for i, u in zip(idxs, u_eff):
                    phases_new[i] = phases[i] + u
            elif self.s_mode == "scalar":
                s_k = self.s[k]
                for i, u in zip(idxs, u_eff):
                    phases_new[i] = s_k * phases[i] + u
            elif self.s_mode == "diagonal":
                for i, u in zip(idxs, u_eff):
                    a, b = self._offsets[i], self._offsets[i + 1]
                    phases_new[i] = self.s[k, a:b] * phases[i] + u
            else:  # kernel or hybrid — both use s_kernels
                if self.per_layer:
                    s_xi = [
                        self._conv_per_layer(phases_for_S, self.s_kernels[s_idx][k], s_idx, seg)
                    ]
                else:
                    three_d = self._is_3d[s_idx]
                    s_xi = self._conv_seg(
                        [phases_for_S[i] for i in idxs], self.s_kernels[s_idx][k], seg, three_d
                    )
                for i, s_x, u in zip(idxs, s_xi, u_eff):
                    phases_new[i] = s_x + u

        return phases_new, v_new

    def init_momentum(
        self, B: int, sizes: list[int], device: torch.device
    ) -> list[torch.Tensor] | None:
        """Initialize momentum buffers as zero tensors.

        Parameters
        ----------
        B : int
            Batch size.
        sizes : list[int]
            Phase parameter counts per layer.
        device : torch.device
            Target device.

        Returns
        -------
        list[Tensor] or None
            Zero-initialized momentum buffers, or None if momentum is disabled.
        """
        return [torch.zeros(B, m, device=device) for m in sizes] if self.momentum else None

    def reg_S(self) -> torch.Tensor:
        """Return regularization loss for the S (scaling) parameter.

        Supports L2 penalty (``"l2"``) or identity penalty (``"identity"``)
        on both diagonal and kernel components of S.

        Returns
        -------
        torch.Tensor
            Scalar regularization loss (0 if S is not learned or reg_s_mode is "none").
        """
        if not self.learn_S or self.reg_s_mode == "none":
            return self.kernels[0].new_zeros(())
        loss = self.kernels[0].new_zeros(())
        # Regularize diagonal component (scalar, diagonal, or hybrid)
        if self.s_mode in ("scalar", "diagonal", "hybrid"):
            if self.reg_s_mode == "l2":
                loss = loss + self.s.pow(2).sum()
            else:  # identity — pull toward 1
                loss = loss + (self.s - 1.0).pow(2).sum()
        # Regularize kernel component (kernel or hybrid)
        if self.s_mode in ("kernel", "hybrid"):
            for s_idx, sp in enumerate(self.s_kernels):
                if self.reg_s_mode == "l2":
                    loss = loss + sp.pow(2).sum()
                else:  # identity — pull toward identity kernel
                    target = torch.zeros_like(sp)
                    pad_s = sp.shape[-1] // 2
                    if self.per_layer:
                        kl = sp.shape[2]
                        target[:, 0, kl // 2, pad_s, pad_s] = 1.0
                    elif self._is_3d[s_idx]:
                        kl = sp.shape[3]
                        target[:, 0, 0, kl // 2, pad_s, pad_s] = 1.0
                    else:
                        target[:, 0, 0, pad_s, pad_s] = 1.0
                    loss = loss + (sp - target).pow(2).sum()
        return loss

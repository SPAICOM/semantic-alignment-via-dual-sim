"""Alignment strategies: zero-shot PPFE and supervised ridge regression.

Both functions operate on complex tensors with shape convention ``(D, N)`` —
dimensions as rows, samples as columns — matching the SIM optimizer convention.
"""

import torch
from sklearn.cluster import KMeans


def ppfe(
    X: torch.Tensor,
    Y: torch.Tensor,
    output_real: torch.Tensor,
    n_clusters: int,
    n_proto: int,
    seed: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prototype-based Parseval Frame Equalization (zero-shot alignment).

    Clusters the RX embedding space, computes per-cluster prototypes for TX
    and RX, then finds the nearest orthonormal maps via polar decomposition.

    The TX synthesis frame ``F`` and the RX analysis frame ``G`` satisfy
    ``G @ F = A`` (the full alignment matrix).  The caller is responsible for
    computing ``A = G @ F`` when needed.

    Parameters
    ----------
    X : torch.Tensor
        Whitened TX embedding matrix, shape ``(D_tx, N)``.
    Y : torch.Tensor
        Whitened RX embedding matrix, shape ``(D_rx, N)``.
    output_real : torch.Tensor
        Real-valued RX embeddings used for KMeans clustering, shape ``(N, D_rx)``.
    n_clusters : int
        Number of prototypes / cluster centres.  Set to ``ch_dim`` (the physical
        channel dimension) when computing frames for the CSI-disjoint SIM.
    n_proto : int
        Number of samples per cluster used to compute each prototype mean.
    seed : int, optional
        Random seed for KMeans and prototype subsampling.

    Returns
    -------
    F : torch.Tensor
        TX synthesis frame, shape ``(n_clusters, D_tx)``.
    G : torch.Tensor
        RX analysis frame, shape ``(D_rx, n_clusters)``.
    """
    # Work with samples as rows for cluster/prototype operations
    X = X.H
    Y = Y.H

    data = output_real.detach().cpu().resolve_conj().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = torch.tensor(kmeans.fit_predict(data))

    n_input = []
    n_output = []

    for label in labels.unique():
        mask = labels == label
        inps = X[mask]
        outs = Y[mask]
        # Randomly sub-sample n_proto points per cluster to form the prototype
        iidx = torch.randperm(inps.size(0))[:n_proto]
        n_input.append(inps[iidx].mean(dim=0))
        n_output.append(outs[iidx].mean(dim=0))

    X = torch.stack(n_input)
    Y = torch.stack(n_output)

    # Shuffle so cluster ordering does not bias the alignment
    idx = torch.randperm(X.size(0))
    X = X[idx]
    Y = Y[idx]

    # Polar decomposition of X_proto: F = U Vt  (TX synthesis frame)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    F = U @ Vt

    # Polar decomposition of Y_proto: G = (U Vt)ᴴ  (RX analysis frame)
    U, S, Vt = torch.linalg.svd(Y, full_matrices=False)
    G = (U @ Vt).H

    return F, G


def ridge_regression(
    X: torch.Tensor,
    Y: torch.Tensor,
    weights: torch.Tensor | None = None,
    lmb: float = 0.0,
) -> torch.Tensor:
    """Solve a (weighted) ridge regression for the alignment matrix A.

    Closed-form solution of ``min_A ‖AX − Y‖²_F + λ‖A‖²_F``:

        A = Y Xᴴ (X Xᴴ + λI)⁻¹

    Parameters
    ----------
    X : torch.Tensor
        TX pilot matrix, shape ``(D_tx, N)``.
    Y : torch.Tensor
        RX pilot matrix, shape ``(D_rx, N)``.
    weights : torch.Tensor, optional
        Non-negative sample weights of shape ``(N,)``.  Equal weights if ``None``.
    lmb : float, optional
        Ridge regularisation coefficient λ.  ``0`` gives ordinary least squares.

    Returns
    -------
    torch.Tensor
        Alignment matrix A of shape ``(D_rx, D_tx)``.
    """
    d, n = X.shape
    reg = lmb * torch.eye(d, dtype=X.dtype, device=X.device)

    if weights is not None:
        W = torch.diag(weights.to(X.device, dtype=X.dtype))
        A = torch.linalg.solve(X @ W @ X.H + reg, Y @ W @ X.H, left=False)
    else:
        A = torch.linalg.solve(X @ X.H + reg, Y @ X.H, left=False)

    return A


if __name__ == "__main__":
    import torch

    X = torch.randn(8, 300, dtype=torch.complex64)
    Y = torch.randn(6, 300, dtype=torch.complex64)
    A = ridge_regression(X, Y)
    print(f"ridge_regression : A={A.shape}")
    output_real = torch.randn(300, 16)
    F_T, F_R_H = ppfe(X, Y, output_real=output_real, n_clusters=4, n_proto=20, seed=0)
    print(f"ppfe             : F_T={F_T.shape}, F_R_H={F_R_H.shape}")

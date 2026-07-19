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
    seed: int | None = None,
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
        Whitened TX embedding matrix, shape ``(D_tx, N)``, complex64.
    Y : torch.Tensor
        Whitened RX embedding matrix, shape ``(D_rx, N)``, complex64.
    output_real : torch.Tensor
        Real-valued RX embeddings used for KMeans clustering,
        shape ``(N, D_rx)``, float32.
    n_clusters : int
        Number of prototypes / cluster centres.
    n_proto : int
        Number of samples per cluster used to compute each prototype mean.
    seed : int or None, optional
        Random seed for KMeans and prototype subsampling.

    Returns
    -------
    F : torch.Tensor
        TX synthesis frame, shape ``(n_clusters, D_tx)``, complex64.
        Each row is the orthonormal prototype for one cluster.
    G : torch.Tensor
        RX analysis frame, shape ``(D_rx, n_clusters)``, complex64.
        Each column is the orthonormal prototype for one cluster.

    Notes
    -----
    The polar decomposition ``U @ Vt`` of the prototype matrix yields the
    nearest unitary matrix in Frobenius norm, which is why SVD is used here
    rather than a direct orthonormalisation.
    """
    # Seed a local RNG for reproducible prototype subsampling.
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # Transpose to (N, D) so KMeans can consume rows as samples.
    X = X.H
    Y = Y.H

    # Cluster RX embeddings in the real-valued space.
    data = output_real.detach().cpu().resolve_conj().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = torch.tensor(kmeans.fit_predict(data))

    n_input = []
    n_output = []

    # Compute per-cluster prototype means from random subsamples.
    for label in labels.unique():
        mask = labels == label
        inps = X[mask]
        outs = Y[mask]
        # Randomly select n_proto samples and average to form the prototype.
        iidx = torch.randperm(inps.size(0), generator=rng)[:n_proto]
        n_input.append(inps[iidx].mean(dim=0))
        n_output.append(outs[iidx].mean(dim=0))

    X = torch.stack(n_input)
    Y = torch.stack(n_output)

    # Shuffle prototypes to remove cluster-index ordering.
    idx = torch.randperm(X.size(0), generator=rng)
    X = X[idx]
    Y = Y[idx]

    # Polar decomposition via SVD: nearest unitary matrix to the prototype set.
    # F = nearest unitary to X (TX synthesis frame).
    U, _, Vt = torch.linalg.svd(X, full_matrices=False)
    F = U @ Vt

    # G = nearest unitary to Y, then conjugate-transpose (RX analysis frame).
    U, _, Vt = torch.linalg.svd(Y, full_matrices=False)
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
        TX pilot matrix, shape ``(D_tx, N)``, complex64.
    Y : torch.Tensor
        RX pilot matrix, shape ``(D_rx, N)``, complex64.
    weights : torch.Tensor or None, optional
        Non-negative sample weights of shape ``(N,)``.  Equal weights if
        ``None`` (default).
    lmb : float, optional
        Ridge regularisation coefficient λ.  ``0`` gives ordinary least
        squares (default ``0.0``).

    Returns
    -------
    torch.Tensor
        Alignment matrix A of shape ``(D_rx, D_tx)``, complex64.

    Notes
    -----
    The solve is performed via ``torch.linalg.solve`` rather than explicit
    inversion for better numerical stability.
    """
    d, n = X.shape
    # Regularisation term: λ I added to the Gram matrix.
    reg = lmb * torch.eye(d, dtype=X.dtype, device=X.device)

    if weights is not None:
        # Weighted case: W = diag(weights), solve (X W Xᴴ + λI) Aᴴ = Y W Xᴴ.
        W = torch.diag(weights.to(X.device, dtype=X.dtype))
        A = torch.linalg.solve(X @ W @ X.H + reg, Y @ W @ X.H, left=False)
    else:
        # Unweighted case: solve (X Xᴴ + λI) Aᴴ = Y Xᴴ.
        A = torch.linalg.solve(X @ X.H + reg, Y @ X.H, left=False)

    return A


if __name__ == "__main__":
    X = torch.randn(8, 300, dtype=torch.complex64)
    Y = torch.randn(6, 300, dtype=torch.complex64)
    A = ridge_regression(X, Y)
    print(f"ridge_regression : A={A.shape}")
    output_real = torch.randn(300, 16)
    F_T, F_R_H = ppfe(X, Y, output_real=output_real, n_clusters=4, n_proto=20, seed=0)
    print(f"ppfe             : F_T={F_T.shape}, F_R_H={F_R_H.shape}")

import torch
from sklearn.cluster import KMeans


def ppfe(
    X: torch.tensor,
    Y: torch.tensor,
    output_real: torch.tensor,
    n_clusters: int,
    n_proto: int,
    seed: int = None,
) -> torch.tensor:
    """Prototype-based Parseval Frame Equalization (zero-shot alignment)."""
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
        iidx = torch.randperm(inps.size(0))[:n_proto]
        n_input.append(inps[iidx].mean(dim=0))
        n_output.append(outs[iidx].mean(dim=0))

    X = torch.stack(n_input)
    Y = torch.stack(n_output)

    idx = torch.randperm(X.size(0))
    X = X[idx]
    Y = Y[idx]

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    F = U @ Vt

    U, S, Vt = torch.linalg.svd(Y, full_matrices=False)
    G = (U @ Vt).H

    return G @ F


def ridge_regression(
    X: torch.tensor,
    Y: torch.tensor,
    weights: torch.Tensor | None = None,
    lmb: float = 0.0,
) -> torch.tensor:
    """Solve ridge regression: A = Y X^H (X X^H + λI)^{-1}."""
    d, n = X.shape
    reg = lmb * torch.eye(d, dtype=X.dtype, device=X.device)

    if weights is not None:
        W = torch.diag(weights.to(X.device, dtype=X.dtype))
        A = torch.linalg.solve(X @ W @ X.H + reg, Y @ W @ X.H, left=False)
    else:
        A = torch.linalg.solve(X @ X.H + reg, Y @ X.H, left=False)

    return A

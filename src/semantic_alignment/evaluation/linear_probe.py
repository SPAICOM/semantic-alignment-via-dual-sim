"""Closed-form linear probing via least squares, plus unified classifier dispatch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_linear_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Fit a linear probe W via ordinary least squares.

    Solves  X W ≈ Y_onehot  where X = (N, D), W = (D, C).

    Parameters
    ----------
    embeddings  : float32 tensor of shape (N, D).
    labels      : int64  tensor of shape (N,).
    num_classes : number of target classes C.

    Returns
    -------
    W : torch.Tensor of shape (D, C) on the same device as *embeddings*.
    """
    X = embeddings.float()
    Y = F.one_hot(labels.long(), num_classes).float().to(X.device)
    W = torch.linalg.lstsq(X, Y).solution  # (D, C)
    return W


def linear_probe_predict(W: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """Return predicted class indices for each row in *embeddings*."""
    return (embeddings.float() @ W).argmax(dim=1)


def linear_probe_accuracy(
    W: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Return top-1 accuracy (0–1) of the probe on the given data."""
    preds = linear_probe_predict(W, embeddings.to(W.device))
    return (preds == labels.to(W.device)).float().mean().item()


# ---------------------------------------------------------------------------
# Unified classifier dispatch (linear probe OR neural MLP)
# ---------------------------------------------------------------------------


def clf_predict(clf: torch.Tensor | nn.Module, embeddings: torch.Tensor) -> torch.Tensor:
    """Predict class indices for *embeddings* using either a linear probe or a neural clf."""
    if isinstance(clf, nn.Module):
        clf.eval()
        with torch.no_grad():
            return clf(embeddings.float().to(next(clf.parameters()).device)).argmax(dim=1)
    return linear_probe_predict(clf, embeddings)


def clf_accuracy(
    clf: torch.Tensor | nn.Module, embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    """Top-1 accuracy (0–1) for either classifier type."""
    if isinstance(clf, nn.Module):
        device = next(clf.parameters()).device
        preds = clf_predict(clf, embeddings.to(device))
        return (preds == labels.to(device)).float().mean().item()
    return linear_probe_accuracy(clf, embeddings, labels)

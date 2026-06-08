"""Closed-form linear probing via least squares, plus unified classifier dispatch.

The linear probe is used as the primary evaluation back-end throughout the
experiments because it avoids the training variance of a neural classifier and
permits direct comparison with the reference-repo baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_linear_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Fit a linear probe W via ordinary least squares.

    Solves ``X W ≈ Y_onehot`` where ``X`` has shape ``(N, D)`` and
    ``Y_onehot`` is the one-hot encoding of shape ``(N, C)``.

    Parameters
    ----------
    embeddings : torch.Tensor
        Float32 embedding matrix of shape ``(N, D)``.
    labels : torch.Tensor
        Integer class labels of shape ``(N,)``.
    num_classes : int
        Number of target classes C.

    Returns
    -------
    torch.Tensor
        Weight matrix W of shape ``(D, C)`` on the same device as
        ``embeddings``.
    """
    X = embeddings.float()
    Y = F.one_hot(labels.long(), num_classes).float().to(X.device)
    W = torch.linalg.lstsq(X, Y).solution  # (D, C)
    return W


def linear_probe_predict(W: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """Predict class indices using a linear probe weight matrix.

    Parameters
    ----------
    W : torch.Tensor
        Weight matrix of shape ``(D, C)`` from :func:`train_linear_probe`.
    embeddings : torch.Tensor
        Input embeddings of shape ``(N, D)``.

    Returns
    -------
    torch.Tensor
        Predicted class indices of shape ``(N,)``.
    """
    return (embeddings.float() @ W).argmax(dim=1)


def linear_probe_accuracy(
    W: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute top-1 accuracy of a linear probe.

    Parameters
    ----------
    W : torch.Tensor
        Weight matrix of shape ``(D, C)``.
    embeddings : torch.Tensor
        Input embeddings of shape ``(N, D)``.
    labels : torch.Tensor
        Ground-truth integer labels of shape ``(N,)``.

    Returns
    -------
    float
        Top-1 accuracy in ``[0, 1]``.
    """
    preds = linear_probe_predict(W, embeddings.to(W.device))
    return (preds == labels.to(W.device)).float().mean().item()


# ---------------------------------------------------------------------------
# Unified classifier dispatch (linear probe OR neural MLP)
# ---------------------------------------------------------------------------


def clf_predict(clf: torch.Tensor | nn.Module, embeddings: torch.Tensor) -> torch.Tensor:
    """Predict class indices using either a linear probe or a neural classifier.

    Unified dispatch so experiment runners need no branching between the two
    classifier back-ends.

    Parameters
    ----------
    clf : torch.Tensor or nn.Module
        Either a linear probe weight matrix ``(D, C)`` or a
        :class:`~semantic_alignment.evaluation.neural_classifier.NeuralClassifier`.
    embeddings : torch.Tensor
        Input embeddings of shape ``(N, D)``.

    Returns
    -------
    torch.Tensor
        Predicted class indices of shape ``(N,)``.
    """
    if isinstance(clf, nn.Module):
        # Neural path: run in eval mode without gradient tracking
        clf.eval()
        with torch.no_grad():
            return clf(embeddings.float().to(next(clf.parameters()).device)).argmax(dim=1)
    return linear_probe_predict(clf, embeddings)


def clf_accuracy(
    clf: torch.Tensor | nn.Module, embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute top-1 accuracy for either classifier type.

    Parameters
    ----------
    clf : torch.Tensor or nn.Module
        Linear probe weight matrix or neural classifier.
    embeddings : torch.Tensor
        Input embeddings of shape ``(N, D)``.
    labels : torch.Tensor
        Ground-truth integer labels of shape ``(N,)``.

    Returns
    -------
    float
        Top-1 accuracy in ``[0, 1]``.
    """
    if isinstance(clf, nn.Module):
        device = next(clf.parameters()).device
        preds = clf_predict(clf, embeddings.to(device))
        return (preds == labels.to(device)).float().mean().item()
    return linear_probe_accuracy(clf, embeddings, labels)

"""Data loading from the Semasia HuggingFace collection via Polars.

URL template
------------
``hf://datasets/spaicom-lab/semasia-{name}/{split}/{model}/*.parquet``

Data flow
---------
1. :func:`load_split`   — lazy Parquet scan → sort by ``id`` → collect embeddings
   and labels as PyTorch tensors for a single (dataset, encoder, split) triple.
2. :func:`load_semasia` — calls :func:`load_split` four times (TX train, RX train,
   TX test, RX test) and packages the result into a :class:`SemaSIAData` container.
3. :func:`select_pilots` — subsamples the training split to produce the small
   pilot set used to estimate the alignment matrix A.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans

_HF_BASE = "hf://datasets/spaicom-lab/semasia-{name}/{split}/{model}/*.parquet"


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------


def _uri(hf_name: str, model: str, split: str) -> str:
    return _HF_BASE.format(name=hf_name, split=split, model=model)


def load_split(
    hf_name: str,
    model: str,
    split: str,
    label_col: str = "label",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load embeddings and labels for one (dataset, encoder, split) from the Hub.

    Parameters
    ----------
    hf_name : str
        Semasia collection name, e.g. ``"cifar10"``.
    model : str
        Encoder model name as used in the HuggingFace path.
    split : str
        Dataset split, e.g. ``"train"`` or ``"test"``.
    label_col : str, optional
        Name of the label column in the Parquet files.

    Returns
    -------
    embeddings : torch.Tensor
        Float32 tensor of shape ``(N, D)``.
    labels : torch.Tensor
        Int64 class-label tensor of shape ``(N,)``.
    """
    df = (
        pl.scan_parquet(_uri(hf_name, model, split))
        .select(["id", label_col, "embedding"])
        .sort("id")
        .collect()
    )
    embeddings = torch.from_numpy(np.stack(df["embedding"].to_list()))
    labels = torch.tensor(df[label_col].to_list(), dtype=torch.long)
    return embeddings, labels


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SplitData:
    """Paired TX/RX embeddings and ground-truth labels for one dataset split.

    Attributes
    ----------
    z_tx   : TX encoder embeddings, shape (N, D_tx).
    z_rx   : RX encoder embeddings, shape (N, D_rx); same sample order as z_tx.
    labels : integer class labels, shape (N,).
    """

    z_tx: torch.Tensor  # (N, D_tx)
    z_rx: torch.Tensor  # (N, D_rx)
    labels: torch.Tensor  # (N,)


@dataclass
class SemaSIAData:
    """Full train/test container returned by :func:`load_semasia`.

    Attributes
    ----------
    train : :class:`SplitData` for the training split.
    test  : :class:`SplitData` for the test split.
    """

    train: SplitData
    test: SplitData


def load_semasia(
    hf_name: str,
    tx_enc: str,
    rx_enc: str,
    label_col: str = "label",
    train_split: str = "train",
    test_split: str = "test",
) -> SemaSIAData:
    """Load paired TX/RX embeddings for both splits from the Semasia collection.

    Parameters
    ----------
    hf_name : str
        Semasia collection name, e.g. ``"cifar10"``.
    tx_enc : str
        TX-side encoder model name.
    rx_enc : str
        RX-side encoder model name (may differ from ``tx_enc``).
    label_col : str, optional
        Parquet column that contains the class labels.
    train_split : str, optional
        Name of the training split directory on the Hub.
    test_split : str, optional
        Name of the test split directory on the Hub.

    Returns
    -------
    SemaSIAData
        Container with ``train`` and ``test`` :class:`SplitData` objects.
    """
    print(f"Loading {hf_name} | TX: {tx_enc} | RX: {rx_enc}")

    z_tx_tr, labels_tr = load_split(hf_name, tx_enc, train_split, label_col)
    z_rx_tr, _ = load_split(hf_name, rx_enc, train_split, label_col)

    z_tx_te, labels_te = load_split(hf_name, tx_enc, test_split, label_col)
    z_rx_te, _ = load_split(hf_name, rx_enc, test_split, label_col)

    return SemaSIAData(
        train=SplitData(z_tx=z_tx_tr, z_rx=z_rx_tr, labels=labels_tr),
        test=SplitData(z_tx=z_tx_te, z_rx=z_rx_te, labels=labels_te),
    )


# ---------------------------------------------------------------------------
# Pilot selection
# ---------------------------------------------------------------------------


def select_pilots(
    data: SplitData,
    train_label_size: int,
    method: str = "centroid",
    grouping: str = "label",
    seed: int = 42,
) -> SplitData:
    """Select a fixed number of semantic pilots per class from a training split.

    Parameters
    ----------
    data : SplitData
        Training split to subsample.
    train_label_size : int
        Number of pilots to select per class.
    method : {"centroid", "random"}, optional
        Selection strategy.  ``"centroid"`` picks the samples geometrically
        closest to the class mean; ``"random"`` draws uniformly at random.
    grouping : {"label", "proto"}, optional
        Grouping strategy.  ``"label"`` uses ground-truth class labels;
        ``"proto"`` uses KMeans clusters on TX embeddings (unsupervised).
    seed : int, optional
        Random seed for KMeans and random permutation.

    Returns
    -------
    SplitData
        Subsampled pilot set with ``train_label_size`` samples per group.
    """
    z_tx, z_rx, labels = data.z_tx, data.z_rx, data.labels
    unique_labels = labels.unique()

    if grouping == "proto":
        # Unsupervised grouping when RX labels are unavailable at TX
        km = KMeans(n_clusters=len(unique_labels), random_state=seed)
        groups = torch.tensor(km.fit_predict(z_tx.cpu().numpy()))
    else:
        groups = labels

    idx_all = torch.tensor([], dtype=torch.long)
    for g in unique_labels:
        mask = groups == g
        sel = z_tx[mask]
        indices = mask.nonzero(as_tuple=True)[0]

        if method == "centroid":
            # Pick the samples geometrically closest to the class mean
            mean = sel.mean(dim=0)
            dists = torch.norm(sel - mean, dim=1)
            _, local_idx = torch.topk(dists, k=min(train_label_size, len(dists)), largest=False)
        else:
            # Uniform random draw without replacement
            perm = torch.randperm(len(indices))[:train_label_size]
            local_idx = perm

        idx_all = torch.cat([idx_all, indices[local_idx]])

    # Shuffle final pilot set to remove per-class ordering before use in solvers
    idx_all = idx_all[torch.randperm(len(idx_all))]
    return SplitData(z_tx=z_tx[idx_all], z_rx=z_rx[idx_all], labels=labels[idx_all])


if __name__ == "__main__":
    import torch

    z = torch.randn(50, 8)
    labels = torch.arange(50) % 5
    split = SplitData(z_tx=z, z_rx=z, labels=labels)
    pilots = select_pilots(split, train_label_size=3)
    print(f"SplitData      : z_tx={split.z_tx.shape}, labels={split.labels.shape}")
    print(f"select_pilots  : z_tx={pilots.z_tx.shape} (3 per class × 5 classes)")

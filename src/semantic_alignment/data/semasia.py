"""Data loading from the Semasia HuggingFace collection via Polars."""

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
    """
    Load embeddings and labels for one (dataset, model, split) from the Hub.

    Returns
    -------
    embeddings : torch.Tensor  shape (N, D), float32
    labels     : torch.Tensor  shape (N,),   int64
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
    z_tx: torch.Tensor  # (N, D_tx)
    z_rx: torch.Tensor  # (N, D_rx)
    labels: torch.Tensor  # (N,)


@dataclass
class SemaSIAData:
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
    """Load paired TX / RX embeddings from the Semasia collection."""
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
    """
    Select a fixed number of semantic pilots per class from ``data.train``.

    Parameters
    ----------
    data             : SplitData to subsample (usually the train split).
    train_label_size : number of pilots per class.
    method           : ``"centroid"`` (closest to class mean) or ``"random"``.
    grouping         : ``"label"`` (use true labels) or ``"proto"`` (KMeans clusters).
    seed             : random seed for reproducibility.
    """
    z_tx, z_rx, labels = data.z_tx, data.z_rx, data.labels
    unique_labels = labels.unique()

    if grouping == "proto":
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
            mean = sel.mean(dim=0)
            dists = torch.norm(sel - mean, dim=1)
            _, local_idx = torch.topk(dists, k=min(train_label_size, len(dists)), largest=False)
        else:
            perm = torch.randperm(len(indices))[:train_label_size]
            local_idx = perm

        idx_all = torch.cat([idx_all, indices[local_idx]])

    idx_all = idx_all[torch.randperm(len(idx_all))]
    return SplitData(z_tx=z_tx[idx_all], z_rx=z_rx[idx_all], labels=labels[idx_all])

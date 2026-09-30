"""Data loading from the Semasia HuggingFace collection via Polars.

URL template
------------
``hf://datasets/spaicom-lab/semasia-{name}/{split}/{model}/*.parquet``

Local cache layout
------------------
``data/latents/{name}/{split}/{model}/data.parquet``

Data flow
---------
1. :func:`load_split`   — checks local cache first; falls back to HF lazy scan;
   optionally writes to disk.  Sorts by ``id`` then returns embeddings and labels
   as PyTorch tensors for a single (dataset, encoder, split) triple.
2. :func:`load_semasia` — calls :func:`load_split` four times (TX train, RX train,
   TX test, RX test) and packages the result into a :class:`SemasiaData` container.
3. :func:`select_pilots` — subsamples the training split to produce the small
   pilot set used to estimate the alignment matrix A.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans

_HF_BASE = "hf://datasets/spaicom-lab/semasia-{name}/{split}/{model}/*.parquet"
_LATENTS_DIR = Path(__file__).resolve().parents[3] / "data" / "latents"


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------


def _hf_uri(hf_name: str, model: str, split: str) -> str:
    """Build the HuggingFace fsspec URI for a (dataset, encoder, split) triple.

    Parameters
    ----------
    hf_name : str
        Semasia collection name, e.g. ``"cifar10"``.
    model : str
        Encoder model name as used in the HuggingFace path.
    split : str
        Dataset split, e.g. ``"train"`` or ``"test"``.

    Returns
    -------
    str
        URI of the form
        ``hf://datasets/spaicom-lab/semasia-{hf_name}/{split}/{model}/*.parquet``.
    """
    return _HF_BASE.format(name=hf_name, split=split, model=model)


def _local_path(hf_name: str, model: str, split: str, cache_dir: Path) -> Path:
    """Build the local cache path for a (dataset, encoder, split) triple.

    Parameters
    ----------
    hf_name : str
        Semasia collection name.
    model : str
        Encoder model name.
    split : str
        Dataset split.
    cache_dir : Path
        Root of the local cache tree.

    Returns
    -------
    Path
        ``{cache_dir}/{hf_name}/{split}/{model}/data.parquet``.
    """
    return cache_dir / hf_name / split / model / "data.parquet"


def load_split(
    hf_name: str,
    model: str,
    split: str,
    label_col: str = "label",
    cache_dir: Path | None = None,
    save_to_disk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load embeddings and labels for one (dataset, encoder, split).

    Checks ``data/latents/{hf_name}/{split}/{model}/data.parquet`` first;
    falls back to the HuggingFace Hub if not found.  When ``save_to_disk``
    is ``True`` and the local file did not yet exist, writes the collected
    DataFrame to disk after downloading.

    Parameters
    ----------
    hf_name : str
        Semasia collection name, e.g. ``"cifar10"``.
    model : str
        Encoder model name as used in the HuggingFace path.
    split : str
        Dataset split, e.g. ``"train"`` or ``"test"``.
    label_col : str, optional
        Name of the label column in the Parquet files (default ``"label"``).
    cache_dir : Path or None, optional
        Root of the local cache tree.  Defaults to ``data/latents/`` in the
        project root.
    save_to_disk : bool, optional
        If ``True`` and the local file is absent, write the downloaded data
        to disk after collecting (default ``False``).

    Returns
    -------
    embeddings : torch.Tensor
        Float32 tensor of shape ``(N, D)`` where *N* is the number of
        samples and *D* the embedding dimension.
    labels : torch.Tensor
        Int64 class-label tensor of shape ``(N,)``.

    Raises
    ------
    FileNotFoundError
        If neither the local cache nor the HuggingFace URI is accessible.
    """
    # Resolve the local cache path; fall back to HuggingFace if absent.
    root = cache_dir if cache_dir is not None else _LATENTS_DIR
    local = _local_path(hf_name, model, split, root)

    if local.exists():
        source = str(local)
    else:
        source = _hf_uri(hf_name, model, split)

    # Lazy-scan the parquet, keep only the needed columns, sort by id so
    # that TX and RX embeddings are sample-aligned.
    df = pl.scan_parquet(source).select(["id", label_col, "embedding"]).sort("id").collect()

    # Optionally persist to disk for future runs (no-op if file already exists).
    if save_to_disk and not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(local)

    # Convert Polars list-of-arrays to a single numpy array, then to torch.
    embeddings = torch.from_numpy(np.stack(df["embedding"].to_list()).astype(np.float32))
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
    z_tx : torch.Tensor
        TX encoder embeddings, shape ``(N, D_tx)``, float32.
    z_rx : torch.Tensor
        RX encoder embeddings, shape ``(N, D_rx)``; same sample order as
        ``z_tx``.
    labels : torch.Tensor
        Integer class labels, shape ``(N,)``, int64.
    """

    z_tx: torch.Tensor  # (N, D_tx)
    z_rx: torch.Tensor  # (N, D_rx)
    labels: torch.Tensor  # (N,)


@dataclass
class SemasiaData:
    """Full train/test container returned by :func:`load_semasia`.

    Attributes
    ----------
    train : SplitData
        Training split.
    test : SplitData
        Test split.
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
    cache_dir: Path | None = None,
    save_to_disk: bool = False,
) -> SemasiaData:
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
        Parquet column that contains the class labels (default ``"label"``).
    train_split : str, optional
        Name of the training split directory on the Hub (default ``"train"``).
    test_split : str, optional
        Name of the test split directory on the Hub (default ``"test"``).
    cache_dir : Path or None, optional
        Root of the local cache tree.  Defaults to ``data/latents/`` in the
        project root.
    save_to_disk : bool, optional
        If ``True``, persist each downloaded split to
        ``data/latents/{hf_name}/{split}/{model}/data.parquet`` for future
        runs.  Splits already present on disk are never re-downloaded
        (default ``False``).

    Returns
    -------
    SemasiaData
        Container with ``train`` and ``test`` :class:`SplitData` objects.
    """
    print(f"Loading {hf_name} | TX: {tx_enc} | RX: {rx_enc}")

    # Share keyword arguments across all four load_split calls.
    kw = dict(label_col=label_col, cache_dir=cache_dir, save_to_disk=save_to_disk)

    # Load training split: TX and RX embeddings share the same labels.
    z_tx_tr, labels_tr = load_split(hf_name, tx_enc, train_split, **kw)
    z_rx_tr, _ = load_split(hf_name, rx_enc, train_split, **kw)

    # Load test split.
    z_tx_te, labels_te = load_split(hf_name, tx_enc, test_split, **kw)
    z_rx_te, _ = load_split(hf_name, rx_enc, test_split, **kw)

    # Assemble into paired containers.
    return SemasiaData(
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
        closest to the class mean; ``"random"`` draws uniformly at random
        (default ``"centroid"``).
    grouping : {"label", "proto"}, optional
        Grouping strategy.  ``"label"`` uses ground-truth class labels;
        ``"proto"`` uses KMeans clusters on TX embeddings (unsupervised)
        (default ``"label"``).
    seed : int, optional
        Random seed for KMeans and random permutation (default 42).

    Returns
    -------
    SplitData
        Subsampled pilot set with ``train_label_size`` samples per group.
        The returned samples are shuffled to remove per-class ordering.
    """
    z_tx, z_rx, labels = data.z_tx, data.z_rx, data.labels
    unique_labels = labels.unique()

    if grouping == "proto":
        # Unsupervised grouping: KMeans on TX embeddings when RX labels
        # are unavailable at the transmitter.
        km = KMeans(n_clusters=len(unique_labels), random_state=seed)
        groups = torch.tensor(km.fit_predict(z_tx.cpu().numpy()))
    else:
        groups = labels

    idx_all = torch.tensor([], dtype=torch.long)
    for g in unique_labels:
        # Select all samples belonging to group *g*.
        mask = groups == g
        sel = z_tx[mask]
        indices = mask.nonzero(as_tuple=True)[0]

        if method == "centroid":
            # Pick the samples geometrically closest to the class mean.
            mean = sel.mean(dim=0)
            dists = torch.norm(sel - mean, dim=1)
            _, local_idx = torch.topk(dists, k=min(train_label_size, len(dists)), largest=False)
        else:
            # Uniform random draw without replacement.
            perm = torch.randperm(len(indices))[:train_label_size]
            local_idx = perm

        idx_all = torch.cat([idx_all, indices[local_idx]])

    # Shuffle final pilot set to remove per-class ordering before use in solvers.
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

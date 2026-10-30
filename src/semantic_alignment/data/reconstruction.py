"""Raw-pixel data loading for image-reconstruction alignment.

Mirrors :mod:`semantic_alignment.data.semasia` but reads the *original* HF
image datasets (pixels + labels) instead of precomputed Semasia embeddings —
no equivalent precomputed-embedding collection exists for reconstruction, so
this repo trains its own TX/RX autoencoders (:mod:`semantic_alignment.models.autoencoder`)
on these pixels.

Local cache layout
------------------
``data/images/{hf_repo}/{split}/data.parquet``

Row order is not backed by an explicit ``id`` column (the upstream parquet
files don't have one) — it relies on the underlying HF parquet shards being
immutable, so repeated reads (e.g. once per TX/RX autoencoder variant) return
samples in the same order.

Data flow
---------
1. :func:`load_split`               — raw images + labels for one (dataset, split).
2. :func:`embed_split`              — run a trained :class:`ConvVAE`'s encoder over
   a set of images to get deterministic embeddings.
3. :func:`load_reconstruction_pair` — decodes each split's images once and
   embeds them with both the TX and RX autoencoder, returning the same
   :class:`~semantic_alignment.data.semasia.SemasiaData` container the SIM
   alignment pipeline consumes for classification.
4. :func:`prepare_reconstruction`   — the reconstruction analogue of
   :func:`~semantic_alignment.data.prepare.prepare_semasia`: load → embed →
   pilots → prewhiten. The alignment target is left to the caller
   (:func:`compute_alignment`), since reconstruction jobs evaluate several
   strategies on the same embeddings.
"""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image

from semantic_alignment.data.prepare import ppfe_cluster_data  # noqa: F401  (re-exported)
from semantic_alignment.data.semasia import SemasiaData, SplitData, select_pilots
from semantic_alignment.models.autoencoder import ConvVAE
from semantic_alignment.utils import pack_complex, ppfe, prewhiten, ridge_regression

_IMAGES_DIR = Path(__file__).resolve().parents[3] / "data" / "images"


def _hf_uri(hf_repo: str, hf_config: str, split: str) -> str:
    """Build the HuggingFace fsspec URI for one split's parquet shards."""
    return f"hf://datasets/{hf_repo}/{hf_config}/{split}-*.parquet"


def _local_path(hf_repo: str, split: str, cache_dir: Path) -> Path:
    """Return the local cache path ``{cache_dir}/{hf_repo}/{split}/data.parquet``."""
    safe_repo = hf_repo.replace("/", "_")
    return cache_dir / safe_repo / split / "data.parquet"


def _decode_image(raw: dict, channels: int, image_size: tuple[int, int]) -> np.ndarray:
    """Decode one HF ``{'bytes', 'path'}`` image struct into a ``(C, H, W)`` float32 array.

    Parameters
    ----------
    raw : dict
        Parquet struct with a ``bytes`` field holding the encoded image.
    channels : int
        1 for grayscale, 3 for RGB; the image is converted accordingly.
    image_size : tuple of int
        Target ``(H, W)``; the image is resized if it doesn't already match.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(C, H, W)``, pixel values in ``[0, 1]``.
    """
    img = Image.open(io.BytesIO(raw["bytes"]))
    img = img.convert("L" if channels == 1 else "RGB")
    if img.size != (image_size[1], image_size[0]):
        img = img.resize((image_size[1], image_size[0]))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, :, :] if channels == 1 else arr.transpose(2, 0, 1)


def load_split(
    hf_repo: str,
    hf_config: str,
    split: str,
    image_col: str = "image",
    label_col: str = "label",
    image_size: tuple[int, int] = (28, 28),
    channels: int = 1,
    cache_dir: Path | None = None,
    save_to_disk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load images and labels for one (dataset, split) from the raw HF parquet shards.

    Checks ``data/images/{hf_repo}/{split}/data.parquet`` first; falls back to
    the HuggingFace Hub if not found.

    Parameters
    ----------
    hf_repo : str
        Full HF dataset repo id, e.g. ``"uoft-cs/cifar10"``.
    hf_config : str
        Subfolder holding the parquet shards on the Hub, e.g. ``"plain_text"``.
    split : str
        Dataset split, e.g. ``"train"`` or ``"test"``.
    image_col : str, optional
        Parquet column holding the encoded image struct.
    label_col : str, optional
        Parquet column holding the integer label.
    image_size : tuple of int, optional
        Target ``(H, W)`` for decoded images.
    channels : int, optional
        1 (grayscale) or 3 (RGB).
    cache_dir : Path or None, optional
        Root of the local cache tree. Defaults to ``data/images/`` in the
        project root.
    save_to_disk : bool, optional
        If ``True`` and the local file is absent, write the downloaded data
        to disk after collecting.

    Returns
    -------
    images : torch.Tensor
        Float32 tensor of shape ``(N, C, H, W)``, pixel values in ``[0, 1]``.
    labels : torch.Tensor
        Int64 tensor of shape ``(N,)``.
    """
    root = cache_dir if cache_dir is not None else _IMAGES_DIR
    local = _local_path(hf_repo, split, root)

    source = str(local) if local.exists() else _hf_uri(hf_repo, hf_config, split)
    df = pl.scan_parquet(source).select([image_col, label_col]).collect()

    if save_to_disk and not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(local)

    images = torch.from_numpy(
        np.stack([_decode_image(row, channels, image_size) for row in df[image_col].to_list()])
    )
    labels = torch.tensor(df[label_col].to_list(), dtype=torch.long)
    return images, labels


# ---------------------------------------------------------------------------
# Embedding extraction (the reconstruction-task analogue of Semasia embeddings)
# ---------------------------------------------------------------------------


def embed_split(
    model: ConvVAE,
    images: torch.Tensor,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
) -> torch.Tensor:
    """Run a trained :class:`ConvVAE`'s deterministic encoder over a set of images.

    Parameters
    ----------
    model : ConvVAE
        Trained autoencoder; only its encoder half is used.
    images : torch.Tensor
        Images of shape ``(N, C, H, W)``, values in ``[0, 1]``.
    device : torch.device or str, optional
        Device to run the forward passes on.
    batch_size : int, optional
        Mini-batch size for the forward pass (avoids materialising all
        activations for large splits at once).

    Returns
    -------
    torch.Tensor
        Embeddings of shape ``(N, latent_dim)`` on the CPU, the posterior mean
        ``mu`` — never a stochastic sample (see :meth:`ConvVAE.embed`).
    """
    model = model.to(device).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(device)
            chunks.append(model.embed(batch).cpu())
    return torch.cat(chunks, dim=0)


def load_reconstruction_pair(
    hf_repo: str,
    hf_config: str,
    tx_model: ConvVAE,
    rx_model: ConvVAE,
    image_col: str = "image",
    label_col: str = "label",
    image_size: tuple[int, int] = (32, 32),
    channels: int = 3,
    train_split: str = "train",
    test_split: str = "test",
    device: torch.device | str = "cpu",
    batch_size: int = 512,
    cache_dir: Path | None = None,
    save_to_disk: bool = False,
) -> tuple[SemasiaData, torch.Tensor]:
    """Load raw images once per split and embed them with both TX/RX autoencoders.

    Both encoders see the *same* image order, giving paired ``(z_tx, z_rx)``
    for the SIM alignment pipeline in the same :class:`SemasiaData` container
    :func:`semantic_alignment.data.semasia.load_semasia` returns.

    Parameters
    ----------
    hf_repo, hf_config : str
        Identify the raw HF image dataset (see :func:`load_split`).
    tx_model, rx_model : ConvVAE
        Trained autoencoders, e.g. loaded via
        :func:`semantic_alignment.models.autoencoder.load_autoencoder`.
    image_col, label_col : str, optional
        Parquet column names.
    image_size, channels : optional
        Must match both ``tx_model`` and ``rx_model``'s expected input shape.
    train_split, test_split : str, optional
        Dataset split names.
    device : torch.device or str, optional
        Device for the encoder forward passes.
    batch_size : int, optional
        Mini-batch size for embedding extraction.
    cache_dir, save_to_disk : optional
        Passed through to :func:`load_split` for the raw-pixel cache.

    Returns
    -------
    data : SemasiaData
        Paired ``z_tx``/``z_rx`` embeddings and labels for both splits.
    images_test : torch.Tensor
        Ground-truth test images ``(N, C, H, W)`` — the reconstruction target.
    """
    kw = dict(
        image_col=image_col,
        label_col=label_col,
        image_size=image_size,
        channels=channels,
        cache_dir=cache_dir,
        save_to_disk=save_to_disk,
    )
    images_tr, labels_tr = load_split(hf_repo, hf_config, train_split, **kw)
    images_te, labels_te = load_split(hf_repo, hf_config, test_split, **kw)

    data = SemasiaData(
        train=SplitData(
            z_tx=embed_split(tx_model, images_tr, device, batch_size),
            z_rx=embed_split(rx_model, images_tr, device, batch_size),
            labels=labels_tr,
        ),
        test=SplitData(
            z_tx=embed_split(tx_model, images_te, device, batch_size),
            z_rx=embed_split(rx_model, images_te, device, batch_size),
            labels=labels_te,
        ),
    )
    return data, images_te


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


@dataclass
class PreparedReconstruction:
    """All preprocessing outputs for a reconstruction-alignment run.

    The reconstruction analogue of :class:`semantic_alignment.data.prepare.PreparedData`,
    minus the alignment target (computed per strategy by the caller) and
    plus the ground-truth test images.

    Attributes
    ----------
    data : SemasiaData
        Paired TX/RX autoencoder embeddings for both splits.
    images_test : torch.Tensor
        Ground-truth test images, shape ``(N, C, H, W)``, values in ``[0, 1]``.
    pilots : SplitData
        Subsampled pilot set used to estimate the alignment matrix.
    input_w, output_w : torch.Tensor
        Whitened TX/RX pilot embeddings, shape ``(D/2, N_pilot)``, complex64.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        TX/RX whitening statistics (Cholesky factor and mean).
    """

    data: SemasiaData
    images_test: torch.Tensor
    pilots: SplitData
    input_w: torch.Tensor
    L_in: torch.Tensor
    mu_in: torch.Tensor
    output_w: torch.Tensor
    L_out: torch.Tensor
    mu_out: torch.Tensor


def whiten_pilots(pilots: SplitData, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Pack TX/RX pilots real→complex and prewhiten both sides.

    Returns
    -------
    tuple
        ``(input_w, L_in, mu_in, output_w, L_out, mu_out)``.
    """
    input_w, L_in, mu_in = prewhiten(pack_complex(pilots.z_tx.T, device=device), device=device)
    output_w, L_out, mu_out = prewhiten(pack_complex(pilots.z_rx.T, device=device), device=device)
    return input_w, L_in, mu_in, output_w, L_out, mu_out


def compute_alignment(
    input_w: torch.Tensor,
    output_w: torch.Tensor,
    output_real: torch.Tensor,
    strategy: str,
    *,
    lmb: float = 1e-3,
    n_clusters: int | None = None,
    n_proto: int | None = None,
    seed: int = 42,
    cluster_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute the alignment target for one strategy from whitened pilots.

    Same two strategies as :func:`semantic_alignment.data.prepare.prepare_semasia`.

    Parameters
    ----------
    input_w, output_w : torch.Tensor
        Whitened TX/RX pilot embeddings, shape ``(D/2, N_pilot)``, complex64.
    output_real : torch.Tensor
        Real-valued RX pilot embeddings ``(N_pilot, D_rx)`` (PPFE clusters these).
    strategy : str
        ``"Linear"`` (ridge regression) or ``"PPFE"`` (prototype frames).
    lmb : float, optional
        Ridge regularisation (Linear only).
    n_clusters, n_proto : int or None, optional
        PPFE cluster count and prototypes per cluster (required for PPFE);
        ``n_proto=None`` averages the whole cluster.
    seed : int, optional
        PPFE random seed.
    cluster_data : tuple or None, optional
        ``(X_w, Y_w, features)`` from :func:`ppfe_cluster_data`: PPFE clusters
        ``features`` and builds the prototypes from ``X_w``/``Y_w`` instead of
        the pilots (``output_real``, ``input_w``, ``output_w``). Ignored by Linear.

    Returns
    -------
    A_target : torch.Tensor
        Alignment matrix ``(D_rx/2, D_tx/2)``, complex64.
    F_T, F_R_H : torch.Tensor or None
        PPFE TX synthesis / RX analysis frames (``A = F_R_H @ F_T``); ``None`` for Linear.
    """
    if strategy == "Linear":
        return ridge_regression(input_w, output_w, lmb=lmb), None, None
    if strategy == "PPFE":
        if n_clusters is None:
            raise ValueError("PPFE alignment requires n_clusters.")
        if cluster_data is not None:
            input_w, output_w, output_real = cluster_data
        F_T, F_R_H = ppfe(
            input_w,
            output_w,
            output_real=output_real,
            n_clusters=n_clusters,
            n_proto=n_proto,
            seed=seed,
        )
        return F_R_H @ F_T, F_T, F_R_H
    raise ValueError(f"Unknown alignment strategy: {strategy!r}. Use 'Linear' or 'PPFE'.")


def prepare_reconstruction(
    tx_model: ConvVAE,
    rx_model: ConvVAE,
    hf_repo: str,
    hf_config: str,
    train_label_size: int,
    *,
    image_col: str = "image",
    label_col: str = "label",
    image_size: tuple[int, int] = (32, 32),
    channels: int = 3,
    method: str = "centroid",
    grouping: str = "label",
    train_split: str = "train",
    test_split: str = "test",
    save_to_disk: bool = False,
    cache_dir: Path | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> PreparedReconstruction:
    """Load images, embed them with both autoencoders, select pilots, and prewhiten.

    Parameters
    ----------
    tx_model, rx_model : ConvVAE
        Trained TX/RX autoencoders.
    hf_repo, hf_config : str
        Raw HF image dataset (see :func:`load_split`).
    train_label_size : int
        Number of pilot samples per class.
    image_col, label_col, image_size, channels : optional
        Dataset layout (see :func:`load_split`).
    method, grouping : str, optional
        Pilot selection strategy (see :func:`~semantic_alignment.data.semasia.select_pilots`).
    train_split, test_split : str, optional
        Dataset split names.
    save_to_disk : bool, optional
        Cache downloaded raw images under ``data/images/``.
    cache_dir : Path or None, optional
        Root of the raw-image cache tree.
    seed : int, optional
        Random seed for pilot selection.
    device : torch.device or None, optional
        Device for embedding and whitening (default CPU).

    Returns
    -------
    PreparedReconstruction
    """
    if device is None:
        device = torch.device("cpu")

    data, images_test = load_reconstruction_pair(
        hf_repo,
        hf_config,
        tx_model,
        rx_model,
        image_col=image_col,
        label_col=label_col,
        image_size=image_size,
        channels=channels,
        train_split=train_split,
        test_split=test_split,
        device=device,
        cache_dir=cache_dir,
        save_to_disk=save_to_disk,
    )
    pilots = select_pilots(
        data.train,
        train_label_size=train_label_size,
        method=method,
        grouping=grouping,
        seed=seed,
    )
    input_w, L_in, mu_in, output_w, L_out, mu_out = whiten_pilots(pilots, device)
    return PreparedReconstruction(
        data=data,
        images_test=images_test,
        pilots=pilots,
        input_w=input_w,
        L_in=L_in,
        mu_in=mu_in,
        output_w=output_w,
        L_out=L_out,
        mu_out=mu_out,
    )


if __name__ == "__main__":
    images, labels = load_split(
        hf_repo="ylecun/mnist",
        hf_config="mnist",
        split="test",
        image_col="image",
        label_col="label",
        image_size=(28, 28),
        channels=1,
    )
    print(
        f"load_split : images={images.shape} range=[{images.min():.2f}, {images.max():.2f}] "
        f"labels={labels.shape}"
    )

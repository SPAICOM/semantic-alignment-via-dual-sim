"""Tests for data/reconstruction.py: raw-image loading, embedding, and preprocessing."""

import io

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image

from semantic_alignment.data.reconstruction import (
    PreparedReconstruction,
    _decode_image,
    compute_alignment,
    embed_split,
    load_reconstruction_pair,
    load_split,
    ppfe_cluster_data,
    prepare_reconstruction,
    whiten_pilots,
)
from semantic_alignment.data.semasia import SplitData
from semantic_alignment.models.autoencoder import ConvVAE
from semantic_alignment.utils import ppfe

N_SAMPLES = 40
N_CLASSES = 2
SIZE = (8, 8)


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _write_split(cache_dir, hf_repo="org/ds", split="train", n=N_SAMPLES, seed=0):
    """Write a synthetic RGB image parquet into the local cache layout."""
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 256, size=(n, *SIZE, 3), dtype=np.uint8)
    path = cache_dir / hf_repo.replace("/", "_") / split / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "img": [{"bytes": _png_bytes(a), "path": None} for a in imgs],
            "label": [i % N_CLASSES for i in range(n)],
        }
    ).write_parquet(path)
    return imgs


def _tiny_vae(latent_dim: int) -> ConvVAE:
    torch.manual_seed(0)
    return ConvVAE(3, SIZE, latent_dim=latent_dim, channel_widths=[4, 8]).eval()


# ---------------------------------------------------------------------------
# _decode_image
# ---------------------------------------------------------------------------


def test_decode_image_rgb_layout_and_range():
    arr = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    out = _decode_image({"bytes": _png_bytes(arr)}, channels=3, image_size=(4, 4))
    assert out.shape == (3, 4, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, arr.transpose(2, 0, 1) / 255.0)


def test_decode_image_grayscale_and_resize():
    arr = np.zeros((6, 6, 3), dtype=np.uint8)
    out = _decode_image({"bytes": _png_bytes(arr)}, channels=1, image_size=(3, 5))
    assert out.shape == (1, 3, 5)


# ---------------------------------------------------------------------------
# load_split / embed_split / load_reconstruction_pair
# ---------------------------------------------------------------------------


def test_load_split_reads_local_cache(tmp_path):
    imgs = _write_split(tmp_path)
    images, labels = load_split(
        "org/ds", "cfg", "train", "img", "label", SIZE, channels=3, cache_dir=tmp_path
    )
    assert images.shape == (N_SAMPLES, 3, *SIZE)
    assert labels.tolist() == [i % N_CLASSES for i in range(N_SAMPLES)]
    np.testing.assert_allclose(images[0].numpy(), imgs[0].transpose(2, 0, 1) / 255.0)


def test_embed_split_is_deterministic_posterior_mean():
    model = _tiny_vae(latent_dim=6)
    x = torch.rand(10, 3, *SIZE)
    z = embed_split(model, x, batch_size=3)
    assert z.shape == (10, 6)
    torch.testing.assert_close(z, model.encode(x)[0].detach())
    torch.testing.assert_close(z, embed_split(model, x, batch_size=7))


def test_load_reconstruction_pair_pairs_embeddings(tmp_path):
    _write_split(tmp_path, split="train", seed=0)
    _write_split(tmp_path, split="test", n=10, seed=1)
    tx, rx = _tiny_vae(8), _tiny_vae(4)
    data, images_test = load_reconstruction_pair(
        "org/ds", "cfg", tx, rx, image_col="img", image_size=SIZE, cache_dir=tmp_path
    )
    assert data.train.z_tx.shape == (N_SAMPLES, 8)
    assert data.train.z_rx.shape == (N_SAMPLES, 4)
    assert data.test.z_tx.shape == (10, 8)
    assert images_test.shape == (10, 3, *SIZE)


# ---------------------------------------------------------------------------
# whiten_pilots / compute_alignment / prepare_reconstruction
# ---------------------------------------------------------------------------


def _pilots(n=60, d_tx=8, d_rx=4):
    g = torch.Generator().manual_seed(0)
    return SplitData(
        z_tx=torch.randn(n, d_tx, generator=g),
        z_rx=torch.randn(n, d_rx, generator=g),
        labels=torch.arange(n) % N_CLASSES,
    )


def test_whiten_pilots_shapes():
    input_w, L_in, mu_in, output_w, L_out, mu_out = whiten_pilots(_pilots(), torch.device("cpu"))
    assert input_w.shape == (4, 60) and output_w.shape == (2, 60)
    assert L_in.shape == (4, 4) and L_out.shape == (2, 2)
    assert mu_in.shape == (4, 1) and mu_out.shape == (2, 1)
    assert input_w.is_complex()


def test_compute_alignment_linear_and_ppfe():
    p = _pilots()
    input_w, _, _, output_w, _, _ = whiten_pilots(p, torch.device("cpu"))
    A, F_T, F_R_H = compute_alignment(input_w, output_w, p.z_rx, "Linear", lmb=1e-3)
    assert A.shape == (2, 4) and F_T is None and F_R_H is None

    A, F_T, F_R_H = compute_alignment(
        input_w, output_w, p.z_rx, "PPFE", n_clusters=2, n_proto=5, seed=0
    )
    assert A.shape == (2, 4)
    torch.testing.assert_close(A, F_R_H @ F_T)


def test_ppfe_cluster_data_whitens_with_pilot_statistics():
    """Whitening the pilots themselves with their own statistics reproduces whiten_pilots."""
    p = _pilots()
    input_w, L_in, mu_in, output_w, L_out, mu_out = whiten_pilots(p, torch.device("cpu"))
    X_w, Y_w, feats = ppfe_cluster_data(p, L_in, mu_in, L_out, mu_out, torch.device("cpu"))
    torch.testing.assert_close(X_w, input_w)
    torch.testing.assert_close(Y_w, output_w)
    assert feats is p.z_tx


def test_compute_alignment_ppfe_on_train_tx():
    """With cluster_data, PPFE clusters the TX training embeddings, not the RX pilots."""
    p = _pilots()
    train = _pilots(n=200)
    input_w, L_in, mu_in, output_w, L_out, mu_out = whiten_pilots(p, torch.device("cpu"))
    data = ppfe_cluster_data(train, L_in, mu_in, L_out, mu_out, torch.device("cpu"))
    A, F_T, F_R_H = compute_alignment(
        input_w, output_w, p.z_rx, "PPFE", n_clusters=3, n_proto=None, seed=0, cluster_data=data
    )
    F_ref, G_ref = ppfe(*data[:2], output_real=train.z_tx, n_clusters=3, n_proto=None, seed=0)
    torch.testing.assert_close(F_T, F_ref)
    torch.testing.assert_close(F_R_H, G_ref)
    A_pilots, _, _ = compute_alignment(
        input_w, output_w, p.z_rx, "PPFE", n_clusters=3, n_proto=None, seed=0
    )
    assert not torch.allclose(A, A_pilots)


def test_compute_alignment_rejects_bad_input():
    p = _pilots()
    input_w, _, _, output_w, _, _ = whiten_pilots(p, torch.device("cpu"))
    with pytest.raises(ValueError, match="n_clusters"):
        compute_alignment(input_w, output_w, p.z_rx, "PPFE")
    with pytest.raises(ValueError, match="Unknown"):
        compute_alignment(input_w, output_w, p.z_rx, "Nope")


def test_prepare_reconstruction_end_to_end(tmp_path):
    _write_split(tmp_path, split="train", seed=0)
    _write_split(tmp_path, split="test", n=10, seed=1)
    prep = prepare_reconstruction(
        _tiny_vae(8),
        _tiny_vae(4),
        "org/ds",
        "cfg",
        train_label_size=5,
        image_col="img",
        image_size=SIZE,
        cache_dir=tmp_path,
    )
    assert isinstance(prep, PreparedReconstruction)
    assert len(prep.pilots.labels) == 5 * N_CLASSES
    assert prep.input_w.shape == (4, 5 * N_CLASSES)
    assert prep.images_test.shape == (10, 3, *SIZE)

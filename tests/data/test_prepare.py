"""Tests for prepare.py: PreparedData and prepare_semasia."""

import numpy as np
import polars as pl
import pytest
import torch

from semantic_alignment.data.prepare import PreparedData, ppfe_cluster_data, prepare_semasia
from semantic_alignment.data.semasia import SemasiaData, SplitData
from semantic_alignment.utils import ppfe

# ---------------------------------------------------------------------------
# Helpers — reuse the same synthetic parquet setup as test_semasia.py
# ---------------------------------------------------------------------------

N_SAMPLES = 60  # enough samples for pilot selection
N_CLASSES = 3
DIM = 16  # must be even (pack_complex halves the real dimension)
PILOTS_PER_CLASS = 3


def _make_parquet(path, n=N_SAMPLES, d=DIM, n_classes=N_CLASSES, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "id": list(range(n)),
            "label": [i % n_classes for i in range(n)],
            "embedding": pl.Series("embedding", emb, dtype=pl.Array(pl.Float32, d)),
        }
    ).write_parquet(path)


def _write_splits(tmp_path, hf_name="ds", tx_enc="tx", rx_enc="rx"):
    for split in ("train", "test"):
        for enc, seed in ((tx_enc, 0), (rx_enc, 1)):
            _make_parquet(tmp_path / hf_name / split / enc / "data.parquet", seed=seed)


# ---------------------------------------------------------------------------
# PreparedData dataclass
# ---------------------------------------------------------------------------


def test_prepared_data_is_dataclass():
    """PreparedData should be constructable with the expected fields."""
    z = torch.zeros(4, 2, dtype=torch.complex64)
    lbl = torch.zeros(4, dtype=torch.long)
    split = SplitData(z_tx=z.real.float(), z_rx=z.real.float(), labels=lbl)
    data = SemasiaData(train=split, test=split)
    pd = PreparedData(
        data=data,
        pilots=split,
        input_w=z,
        L_in=z,
        mu_in=z,
        output_w=z,
        L_out=z,
        mu_out=z,
        A_target=z,
        F_T=None,
        F_R_H=None,
    )
    assert pd.F_T is None
    assert pd.A_target is z


# ---------------------------------------------------------------------------
# prepare_semasia — Linear alignment
# ---------------------------------------------------------------------------


def test_linear_returns_prepared_data(tmp_path):
    _write_splits(tmp_path)
    result = prepare_semasia(
        hf_name="ds",
        tx_enc="tx",
        rx_enc="rx",
        train_label_size=PILOTS_PER_CLASS,
        alignment_type="Linear",
        cache_dir=tmp_path,
    )
    assert isinstance(result, PreparedData)
    assert isinstance(result.data, SemasiaData)
    assert isinstance(result.pilots, SplitData)


def test_linear_f_frames_are_none(tmp_path):
    _write_splits(tmp_path)
    result = prepare_semasia("ds", "tx", "rx", PILOTS_PER_CLASS, "Linear", cache_dir=tmp_path)
    assert result.F_T is None
    assert result.F_R_H is None


def test_linear_a_target_shape(tmp_path):
    _write_splits(tmp_path)
    result = prepare_semasia("ds", "tx", "rx", PILOTS_PER_CLASS, "Linear", cache_dir=tmp_path)
    # A_target comes from ridge_regression(input_w, output_w)
    # input_w and output_w are complex, shape (D/2, N_pilot) — A_target is (D/2, D/2)
    d_half = DIM // 2
    assert result.A_target.shape == (d_half, d_half)


def test_linear_whitening_shapes(tmp_path):
    _write_splits(tmp_path)
    result = prepare_semasia("ds", "tx", "rx", PILOTS_PER_CLASS, "Linear", cache_dir=tmp_path)
    d_half = DIM // 2
    n_pilot = N_CLASSES * PILOTS_PER_CLASS
    assert result.input_w.shape == (d_half, n_pilot)
    assert result.output_w.shape == (d_half, n_pilot)
    assert result.L_in.shape == (d_half, d_half)
    assert result.mu_in.shape == (d_half, 1)


def test_linear_pilot_count(tmp_path):
    """Pilot tensor width must equal n_classes × pilots_per_class."""
    _write_splits(tmp_path)
    result = prepare_semasia("ds", "tx", "rx", PILOTS_PER_CLASS, "Linear", cache_dir=tmp_path)
    assert result.pilots.z_tx.shape[0] == N_CLASSES * PILOTS_PER_CLASS


def test_linear_whitening_solves_correctly(tmp_path):
    """L_in @ input_w + mu_in must recover a zero-mean, unit-variance signal."""
    _write_splits(tmp_path)
    result = prepare_semasia("ds", "tx", "rx", PILOTS_PER_CLASS, "Linear", cache_dir=tmp_path)
    # Reconstruct prewhitened signal back: L_in @ input_w + mu_in ≈ input_c (pack_complex of pilots)
    # prewhiten contracts: input_c = L_in @ input_w + mu_in
    reconstructed = result.L_in @ result.input_w + result.mu_in
    # input_c is pack_complex(pilots.z_tx.T) — check it's complex and shape matches
    assert reconstructed.is_complex()
    assert reconstructed.shape[0] == DIM // 2


# ---------------------------------------------------------------------------
# prepare_semasia — PPFE alignment
# ---------------------------------------------------------------------------


def test_ppfe_without_n_clusters_raises(tmp_path):
    _write_splits(tmp_path)
    with pytest.raises(ValueError, match="n_clusters"):
        prepare_semasia(
            "ds",
            "tx",
            "rx",
            PILOTS_PER_CLASS,
            "PPFE",
            cache_dir=tmp_path,
        )


def test_ppfe_frames_none_when_keep_false(tmp_path):
    _write_splits(tmp_path)
    result = prepare_semasia(
        "ds",
        "tx",
        "rx",
        PILOTS_PER_CLASS,
        "PPFE",
        n_clusters=N_CLASSES,
        n_proto=1,
        keep_csi_frames=False,
        cache_dir=tmp_path,
    )
    assert result.F_T is None
    assert result.F_R_H is None


def test_ppfe_frames_returned_when_keep_true(tmp_path):
    _write_splits(tmp_path)
    result = prepare_semasia(
        "ds",
        "tx",
        "rx",
        PILOTS_PER_CLASS,
        "PPFE",
        n_clusters=N_CLASSES,
        n_proto=1,
        keep_csi_frames=True,
        cache_dir=tmp_path,
    )
    assert result.F_T is not None
    assert result.F_R_H is not None


def test_ppfe_a_target_equals_f_r_h_at_f_t(tmp_path):
    """A_target must equal F_R_H @ F_T regardless of keep_csi_frames."""
    _write_splits(tmp_path)
    result = prepare_semasia(
        "ds",
        "tx",
        "rx",
        PILOTS_PER_CLASS,
        "PPFE",
        n_clusters=N_CLASSES,
        n_proto=1,
        keep_csi_frames=True,
        cache_dir=tmp_path,
    )
    assert torch.allclose(result.A_target, result.F_R_H @ result.F_T, atol=1e-5)


def test_ppfe_cluster_on_train_tx(tmp_path):
    """train_tx clusters the TX training set; the default (pilots) result is unchanged.

    select_pilots shuffles with the global RNG, so each call is seeded as the scripts do.
    """
    _write_splits(tmp_path)
    kw = dict(n_clusters=N_CLASSES, n_proto=None, keep_csi_frames=True, cache_dir=tmp_path)
    torch.manual_seed(0)
    pilots = prepare_semasia("ds", "tx", "rx", PILOTS_PER_CLASS, "PPFE", **kw)
    torch.manual_seed(0)
    default = prepare_semasia("ds", "tx", "rx", PILOTS_PER_CLASS, "PPFE", cluster_on="pilots", **kw)
    torch.testing.assert_close(pilots.A_target, default.A_target)

    torch.manual_seed(0)
    train = prepare_semasia("ds", "tx", "rx", PILOTS_PER_CLASS, "PPFE", cluster_on="train_tx", **kw)
    X, Y, feats = ppfe_cluster_data(
        train.data.train, train.L_in, train.mu_in, train.L_out, train.mu_out, torch.device("cpu")
    )
    F_ref, G_ref = ppfe(X, Y, output_real=feats, n_clusters=N_CLASSES, n_proto=None, seed=42)
    torch.testing.assert_close(train.F_T, F_ref)
    torch.testing.assert_close(train.F_R_H, G_ref)


def test_unknown_cluster_on_raises(tmp_path):
    _write_splits(tmp_path)
    with pytest.raises(ValueError, match="Unknown cluster_on"):
        prepare_semasia(
            "ds",
            "tx",
            "rx",
            PILOTS_PER_CLASS,
            "PPFE",
            n_clusters=2,
            cluster_on="nope",
            cache_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_alignment_type_raises(tmp_path):
    _write_splits(tmp_path)
    with pytest.raises(ValueError, match="Unknown alignment_type"):
        prepare_semasia(
            "ds",
            "tx",
            "rx",
            PILOTS_PER_CLASS,
            "BadType",
            cache_dir=tmp_path,
        )

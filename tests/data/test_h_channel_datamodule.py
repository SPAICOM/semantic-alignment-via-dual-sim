"""Tests for HChannelDataModule — both precompute_svd=False and True modes."""

import pytest
import torch

from semantic_alignment.data.h_channel import HChannelDataModule

N_R, N_T = 4, 6
K = min(N_R, N_T)  # SVD rank
N_TOTAL = 100
BATCH_SIZE = 16


@pytest.fixture(params=[False, True], ids=["no_svd", "svd"])
def dm(request):
    dm = HChannelDataModule(
        n_R=N_R,
        n_T=N_T,
        n_total=N_TOTAL,
        batch_size=BATCH_SIZE,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=0,
        precompute_svd=request.param,
    )
    dm.setup()
    return dm


# ---------------------------------------------------------------------------
# Split sizes
# ---------------------------------------------------------------------------


def test_split_sizes(dm):
    n_train = int(N_TOTAL * 0.7)
    n_val = int(N_TOTAL * 0.15)
    n_test = N_TOTAL - n_train - n_val
    assert dm.train_H.shape[0] == n_train
    assert dm.val_H.shape[0] == n_val
    assert dm.test_H.shape[0] == n_test


def test_h_shape(dm):
    for split in (dm.train_H, dm.val_H, dm.test_H):
        assert split.shape[1:] == (N_R, N_T)
        assert split.dtype == torch.complex64


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    """Same seed → identical H matrices."""
    a = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, seed=7)
    b = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, seed=7)
    a.setup()
    b.setup()
    assert torch.equal(a.train_H, b.train_H)


def test_different_seeds_differ():
    a = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, seed=1)
    b = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, seed=2)
    a.setup()
    b.setup()
    assert not torch.equal(a.train_H, b.train_H)


# ---------------------------------------------------------------------------
# precompute_svd=False — batch is a 1-tuple (H,)
# ---------------------------------------------------------------------------


def test_no_svd_batch_length():
    dm = HChannelDataModule(
        n_R=N_R, n_T=N_T, n_total=N_TOTAL, batch_size=BATCH_SIZE, precompute_svd=False
    )
    dm.setup()
    (batch,) = next(iter(dm.train_dataloader()))
    assert batch.shape == (BATCH_SIZE, N_R, N_T)


def test_no_svd_all_loaders_return_single_tensor():
    dm = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, precompute_svd=False)
    dm.setup()
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        batch = next(iter(loader))
        assert len(batch) == 1


# ---------------------------------------------------------------------------
# precompute_svd=True — batch is (H, U, s, Vh)
# ---------------------------------------------------------------------------


def test_svd_batch_length():
    dm = HChannelDataModule(
        n_R=N_R, n_T=N_T, n_total=N_TOTAL, batch_size=BATCH_SIZE, precompute_svd=True
    )
    dm.setup()
    H_b, U_b, s_b, Vh_b = next(iter(dm.train_dataloader()))
    assert H_b.shape == (BATCH_SIZE, N_R, N_T)
    assert U_b.shape == (BATCH_SIZE, N_R, K)
    assert s_b.shape == (BATCH_SIZE, K)
    assert Vh_b.shape == (BATCH_SIZE, K, N_T)


def test_svd_dtypes():
    dm = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, precompute_svd=True)
    dm.setup()
    H_b, U_b, s_b, Vh_b = next(iter(dm.train_dataloader()))
    assert H_b.dtype == torch.complex64
    assert U_b.dtype == torch.complex64
    assert s_b.dtype == torch.float32
    assert Vh_b.dtype == torch.complex64


def test_svd_singular_values_positive():
    dm = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, precompute_svd=True)
    dm.setup()
    _, _, s_b, _ = next(iter(dm.train_dataloader()))
    assert (s_b >= 0).all()


def test_svd_reconstruction():
    """U @ diag(s) @ Vh must reconstruct H for every sample in a batch."""
    dm = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, precompute_svd=True)
    dm.setup()
    H_b, U_b, s_b, Vh_b = next(iter(dm.train_dataloader()))
    # batched reconstruction: (B, n_R, k) @ (B, k, k) @ (B, k, n_T)
    S_diag = torch.diag_embed(s_b.to(torch.complex64))
    H_rec = U_b @ S_diag @ Vh_b
    assert torch.allclose(H_b, H_rec, atol=1e-4)


def test_svd_all_loaders_return_four_tensors():
    dm = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL, precompute_svd=True)
    dm.setup()
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        batch = next(iter(loader))
        assert len(batch) == 4


# ---------------------------------------------------------------------------
# DataLoader properties
# ---------------------------------------------------------------------------


def test_train_shuffles_differently_across_epochs():
    dm = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL)
    dm.setup()
    loader = dm.train_dataloader()
    first = next(iter(loader))[0].clone()
    second = next(iter(loader))[0].clone()
    # With shuffle=True and enough data this should differ (extremely unlikely to match)
    assert not torch.equal(first, second)


def test_val_and_test_not_shuffled():
    """Consecutive iterations over val/test must return the same first batch."""
    dm = HChannelDataModule(n_R=N_R, n_T=N_T, n_total=N_TOTAL)
    dm.setup()
    for loader in (dm.val_dataloader(), dm.test_dataloader()):
        a = next(iter(loader))[0].clone()
        b = next(iter(loader))[0].clone()
        assert torch.equal(a, b)

"""Tests for the batched-cascade API of the tasks (used by the deep-unrolling pipeline)."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from semantic_alignment.data.reconstruction import compute_alignment, whiten_pilots
from semantic_alignment.data.semasia import SplitData
from semantic_alignment.evaluation.inference import cascade_accuracies, transmit_accuracy
from semantic_alignment.evaluation.linear_probe import train_linear_probe
from semantic_alignment.evaluation.reconstruction import evaluate_reconstruction
from semantic_alignment.evaluation.tasks import ClassificationTask, ReconstructionTask
from semantic_alignment.models.autoencoder import ConvVAE
from semantic_alignment.sim.link import alignment_transmit, cascade_transmit

DEVICE = torch.device("cpu")
D_TX, D_RX = 8, 4  # real latent sizes → A is (2, 4) complex
N_TEST = 12


@pytest.fixture
def setup():
    g = torch.Generator().manual_seed(0)
    pilots = SplitData(
        z_tx=torch.randn(60, D_TX, generator=g),
        z_rx=torch.randn(60, D_RX, generator=g),
        labels=torch.arange(60) % 3,
    )
    test = SplitData(
        z_tx=torch.randn(N_TEST, D_TX, generator=g),
        z_rx=torch.randn(N_TEST, D_RX, generator=g),
        labels=torch.arange(N_TEST) % 3,
    )
    white = whiten_pilots(pilots, DEVICE)
    A, _, _ = compute_alignment(white[0], white[3], pilots.z_rx, "Linear")
    return pilots, test, white[1:3] + white[4:6], A


def _cascades(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two different cascades: A itself and a perturbed copy, with their MMSE β."""
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(*A.shape, dtype=A.dtype, generator=g)
    Z = torch.stack([A, A + 0.5 * noise])
    ip = (Z.conj() * A).sum((-2, -1))
    beta = ip / (Z.conj() * Z).sum((-2, -1)).real
    return Z, beta


def _classification_task(setup, neural: bool = False) -> ClassificationTask:
    pilots, test, (L_in, mu_in, L_out, mu_out), _ = setup
    clf_neural = None
    if neural:
        torch.manual_seed(0)
        clf_neural = torch.nn.Sequential(torch.nn.Linear(D_RX, 3))
    return ClassificationTask(
        DataLoader(TensorDataset(test.z_tx, test.labels), batch_size=5),
        L_in,
        mu_in,
        L_out,
        mu_out,
        train_linear_probe(pilots.z_rx, pilots.labels, num_classes=3),
        DEVICE,
        clf_neural=clf_neural,
    )


@pytest.mark.parametrize("neural", [False, True])
def test_classification_batched_matches_single(setup, neural):
    """Batched per-cascade accuracies equal the single-cascade path, cascade by cascade."""
    task = _classification_task(setup, neural)
    Z, beta = _cascades(setup[3])
    batched = task.evaluate_cascades(Z, beta)
    for b in range(2):
        Z_test, labels = task.precompute_cascade_inputs()
        probe, neu = cascade_accuracies(
            Z[b], beta[b], Z_test, labels, task.clf, task.clf_neural, *task.whitening[2:], DEVICE
        )
        assert batched["probe"][b] == pytest.approx(probe)
        assert task.evaluate_cascade(Z[b], beta[b])["probe"] == pytest.approx(probe)
        if neural:
            assert batched["neural"][b] == pytest.approx(neu)
    if not neural:
        assert batched["neural"] is None


def test_classification_cascade_matches_transmit_accuracy(setup):
    """The noiseless cascade score equals the runners' transmit-based accuracy."""
    task = _classification_task(setup)
    Z, beta = _cascades(setup[3])
    acc = transmit_accuracy(
        cascade_transmit(Z[1], beta[1], None), task.test_loader, *task.whitening, task.clf, DEVICE
    )
    assert task.evaluate_cascades(Z, beta)["probe"][1] == pytest.approx(acc)


def _reconstruction_task(setup, n_images, batch_size=512) -> ReconstructionTask:
    _, test, (L_in, mu_in, L_out, mu_out), _ = setup
    torch.manual_seed(0)
    rx_model = ConvVAE(3, (8, 8), latent_dim=D_RX, channel_widths=[4]).eval()
    images = torch.rand(N_TEST, 3, 8, 8)
    return ReconstructionTask(
        test.z_tx,
        images,
        L_in,
        mu_in,
        L_out,
        mu_out,
        rx_model,
        DEVICE,
        strategy_name="Linear",
        reg_tag="AE",
        no_mismatch_mse=0.1,
        no_mismatch_psnr=10.0,
        original_mse=0.1,
        original_psnr=10.0,
        batch_size=batch_size,
        cascade_images=n_images,
    )


@pytest.mark.parametrize("n_images", [None, 5])
def test_reconstruction_batched_matches_evaluate_reconstruction(setup, n_images):
    """Per-cascade MSE/PSNR equal evaluate_reconstruction on the same leading images."""
    task = _reconstruction_task(setup, n_images)
    A = setup[3]
    Z, beta = _cascades(A)
    out = task.evaluate_cascades(Z, beta)
    n = n_images or N_TEST
    for b in range(2):
        mse, psnr_db, _ = evaluate_reconstruction(
            cascade_transmit(Z[b], beta[b], None),
            task.z_tx_test[:n],
            task.images_test[:n],
            *task.whitening,
            task.rx_model,
            DEVICE,
        )
        assert out["mse"][b] == pytest.approx(mse, rel=1e-5)
        assert out["psnr"][b] == pytest.approx(psnr_db, rel=1e-5)
    assert out["mse"][0] != pytest.approx(out["mse"][1])
    assert task.cascade_meta() == {"reg_tag": "AE", "n_eval_images": n}

    # β = 1, Z = A is the no-SIM "original" link.
    mse_a, _, _ = evaluate_reconstruction(
        alignment_transmit(A),
        task.z_tx_test[:n],
        task.images_test[:n],
        *task.whitening,
        task.rx_model,
        DEVICE,
    )
    single = task.evaluate_cascade(A, torch.tensor(1.0 + 0j))
    assert single["mse"] == pytest.approx(mse_a, rel=1e-5)


def test_reconstruction_decode_chunking_is_irrelevant(setup):
    """Decoding the B·n reconstructions in small chunks gives the same metrics."""
    Z, beta = _cascades(setup[3])
    big = _reconstruction_task(setup, None).evaluate_cascades(Z, beta)
    small = _reconstruction_task(setup, None, batch_size=7).evaluate_cascades(Z, beta)
    assert small["mse"] == pytest.approx(big["mse"], rel=1e-6)

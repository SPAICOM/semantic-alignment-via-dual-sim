"""End-to-end tests: every SIM runner × every downstream task, on tiny synthetic data (CPU).

The runners are task-agnostic, so each one is exercised with both a
ClassificationTask and a ReconstructionTask; the task alone decides the
result columns, the results root and the filename prefix.
"""

import math

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from semantic_alignment.data.reconstruction import compute_alignment, whiten_pilots
from semantic_alignment.data.semasia import SplitData
from semantic_alignment.evaluation.linear_probe import train_linear_probe
from semantic_alignment.evaluation.reconstruction import run_original_reconstruction
from semantic_alignment.evaluation.tasks import ClassificationTask, ReconstructionTask
from semantic_alignment.models.autoencoder import ConvVAE
from semantic_alignment.sim.original import (
    Disjoint4SIM,
    DisjointAsymSIM,
    DisjointSymSIM,
    JointSIM,
    MonoSIM,
)

DEVICE = torch.device("cpu")
D_TX, D_RX = 8, 4  # real latent sizes → A_target is (2, 4) complex
SIZE = (8, 8)
N_TEST = 6


@pytest.fixture
def setup():
    """Pilots, test split, whitening and a Linear alignment target."""
    g = torch.Generator().manual_seed(0)
    pilots = SplitData(
        z_tx=torch.randn(40, D_TX, generator=g),
        z_rx=torch.randn(40, D_RX, generator=g),
        labels=torch.arange(40) % 2,
    )
    test = SplitData(
        z_tx=torch.randn(N_TEST, D_TX, generator=g),
        z_rx=torch.randn(N_TEST, D_RX, generator=g),
        labels=torch.arange(N_TEST) % 2,
    )
    white = whiten_pilots(pilots, DEVICE)
    A, _, _ = compute_alignment(white[0], white[3], pilots.z_rx, "Linear")
    return pilots, test, white, A


def _classification_task(setup, tmp_path):
    pilots, test, white, _ = setup
    _, L_in, mu_in, _, L_out, mu_out = white
    task = ClassificationTask(
        DataLoader(TensorDataset(test.z_tx, test.labels), batch_size=4),
        L_in,
        mu_in,
        L_out,
        mu_out,
        train_linear_probe(pilots.z_rx, pilots.labels, num_classes=2),
        DEVICE,
        no_mismatch_probe=0.9,
        original_probe=0.8,
    )
    task.results_dir = tmp_path
    return task


def _reconstruction_task(setup, tmp_path):
    _, test, white, A = setup
    _, L_in, mu_in, _, L_out, mu_out = white
    torch.manual_seed(0)
    rx_model = ConvVAE(3, SIZE, latent_dim=D_RX, channel_widths=[4]).eval()
    images = torch.rand(N_TEST, 3, *SIZE)
    mse, psnr_db, preview = run_original_reconstruction(
        test.z_tx, images, A, L_in, mu_in, L_out, mu_out, rx_model, DEVICE, num_preview=3
    )
    task = ReconstructionTask(
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
        original_mse=mse,
        original_psnr=psnr_db,
        no_mismatch_preview=images[:3],
        align_preview=preview,
        num_preview=3,
    )
    task.results_dir = tmp_path
    return task


TASKS = {"classification": _classification_task, "reconstruction": _reconstruction_task}


def _base(setup, **kw):
    return dict(
        A_target=setup[3],
        device=DEVICE,
        L=1,
        M_int=4,
        seed=0,
        strategy_name="Linear",
        result_subdir="sub",
        tx_enc="tx",
        rx_enc="rx",
        dataset="synthetic",
        max_iters=2,
        k_T=1,
        k_R=1,
        lr=0.1,
        lr_tx=0.1,
        lr_rx=0.1,
        wavelength=0.005,
        slayer=0.025,
        dx=0.0025,
        epsilon=1e-6,
        **kw,
    )


def _check(task_name, root, subdir, curve=False):
    """Exactly one Parquet row with the task's columns (+ PNG for reconstruction)."""
    parquets = list(root.glob(f"{subdir}/*.parquet"))
    assert len(parquets) == 1
    p = parquets[0]
    row = pl.read_parquet(p).row(0, named=True)
    if task_name == "classification":
        assert p.name.startswith("synthetic_Linear_")
        assert 0.0 <= row["sim_accuracy_probe"] <= 1.0
        assert row["no_mismatch_probe"] == 0.9 and row["original_probe"] == 0.8
        assert "sim_psnr" not in row
        curve_col = "sim_accuracy_probe_curve"
    else:
        assert p.name.startswith("AE_synthetic_Linear_")
        assert math.isfinite(row["sim_psnr"]) and row["sim_mse"] > 0
        assert "sim_accuracy_probe" not in row
        assert p.with_suffix(".png").exists()
        curve_col = "sim_psnr_curve"
    if curve:
        assert row[curve_col] and len(row[curve_col]) > 0
    elif curve_col in row:
        assert row[curve_col] is None
    return p, row


@pytest.mark.parametrize("task_name", TASKS)
def test_joint(setup, tmp_path, task_name):
    task = TASKS[task_name](setup, tmp_path)
    H = torch.randn(4, 4, dtype=torch.complex64)
    kw = _base(setup, H_mimo=H, channel_dim=4)
    JointSIM.run_sim(task=task, **kw)
    p, _ = _check(task_name, tmp_path, "sub")
    assert "_ch4_" in p.name

    # Rerun: the result exists → nothing is rewritten.
    mtime = p.stat().st_mtime_ns
    JointSIM.run_sim(task=task, **kw)
    assert p.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("task_name", TASKS)
def test_joint_curve(setup, tmp_path, task_name):
    task = TASKS[task_name](setup, tmp_path)
    H = torch.randn(4, 4, dtype=torch.complex64)
    JointSIM.run_sim(task=task, **_base(setup, H_mimo=H, channel_dim=4, save_curve=True))
    _check(task_name, tmp_path, "sub", curve=True)


@pytest.mark.parametrize("task_name", TASKS)
@pytest.mark.parametrize(
    "constructive,side,tipo",
    [
        (False, "tx", "equalized/tx"),
        (True, "tx", "constructive/tx"),
        (True, "rx", "constructive/rx"),
    ],
)
def test_mono(setup, tmp_path, task_name, constructive, side, tipo):
    task = TASKS[task_name](setup, tmp_path)
    MonoSIM.run_sim(
        task=task, **_base(setup, snr=10.0, constructive=constructive, alignment_side=side)
    )
    _check(task_name, tmp_path, f"sub/{tipo}")


def test_mono_rx_equalized_not_supported(setup, tmp_path):
    task = _classification_task(setup, tmp_path)
    with pytest.raises(NotImplementedError):
        MonoSIM.run_sim(task=task, **_base(setup, constructive=False, alignment_side="rx"))


@pytest.mark.parametrize("task_name", TASKS)
@pytest.mark.parametrize("constructive,tipo", [(False, "equalized"), (True, "constructive")])
def test_disjoint_sym(setup, tmp_path, task_name, constructive, tipo):
    task = TASKS[task_name](setup, tmp_path)
    # Linear sym needs ch_dim <= min(A.shape) = 2.
    H = torch.randn(2, 2, dtype=torch.complex64)
    DisjointSymSIM.run_sim(
        task=task, **_base(setup, H_mimo=H, channel_dim=2, constructive=constructive)
    )
    _, row = _check(task_name, tmp_path, f"sub/{tipo}")
    assert row["beta_T"] is not None and row["rx_loss_history"]


@pytest.mark.parametrize("task_name", TASKS)
def test_disjoint_sym_curve(setup, tmp_path, task_name):
    task = TASKS[task_name](setup, tmp_path)
    H = torch.randn(2, 2, dtype=torch.complex64)
    DisjointSymSIM.run_sim(task=task, **_base(setup, H_mimo=H, channel_dim=2, save_curve=True))
    _check(task_name, tmp_path, "sub/equalized", curve=True)


@pytest.mark.parametrize("task_name", TASKS)
def test_disjoint_asym(setup, tmp_path, task_name):
    task = TASKS[task_name](setup, tmp_path)
    # TX side: ch_dim = A.shape[0] = 2.
    H = torch.randn(2, 2, dtype=torch.complex64)
    DisjointAsymSIM.run_sim(
        task=task, **_base(setup, H_mimo=H, constructive=True, alignment_side="tx")
    )
    _check(task_name, tmp_path, "sub/constructive/tx")


@pytest.mark.parametrize("task_name", TASKS)
def test_disjoint_4sim(setup, tmp_path, task_name):
    task = TASKS[task_name](setup, tmp_path)
    H = torch.randn(2, 2, dtype=torch.complex64)
    Disjoint4SIM.run_sim(task=task, **_base(setup, H_mimo=H, channel_dim=2, n_clusters=2))
    p, row = _check(task_name, tmp_path, "sub/equalized")
    assert row["beta_T1"] is not None
    assert "_k2_" in p.name


@pytest.mark.parametrize(
    "runner,kw,subdir",
    [
        (JointSIM, dict(channel_dim=4), "sub"),
        (MonoSIM, dict(), "sub/equalized/tx"),
        (DisjointSymSIM, dict(channel_dim=2), "sub/equalized"),
        (Disjoint4SIM, dict(channel_dim=2), "sub/equalized"),
    ],
)
def test_rician_filename_tag(setup, tmp_path, runner, kw, subdir):
    """Rician runs are tagged ``_rician{K}dB``; Rayleigh names carry no channel tag."""
    task = _classification_task(setup, tmp_path)
    H = torch.randn(
        4 if runner is JointSIM else 2, 4 if runner is JointSIM else 2, dtype=torch.complex64
    )
    runner.run_sim(
        task=task,
        **_base(setup, H_mimo=H, channel_type="rician", rice_factor_db=10, **kw),
    )
    p, _ = _check("classification", tmp_path, subdir)
    assert "_rician10dB_" in p.name


def test_make_fname_channel_tag():
    from semantic_alignment.sim.original._utils import make_fname

    args = ("Linear", 0, 1, 4, 4, None, "tx", "rx")
    assert make_fname(*args) == make_fname(*args, channel_type="rayleigh", rice_factor_db=3)
    assert "rician" not in make_fname(*args)
    assert make_fname(
        *args, extra_tag="_lrtx1_lrrx2", channel_type="rician", rice_factor_db=3
    ).endswith("_lrtx1_lrrx2_rician3dB_tx_rx.parquet")


def test_mono_rician_channel_differs_from_rayleigh(setup):
    A = setup[3]
    torch.manual_seed(0)
    H_ray = MonoSIM.draw_channel(A, "tx", DEVICE)
    torch.manual_seed(0)
    H_ric = MonoSIM.draw_channel(A, "tx", DEVICE, "rician", 10)
    assert H_ric.shape == H_ray.shape and not torch.allclose(H_ric, H_ray)

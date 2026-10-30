"""Downstream tasks that score a trained SIM link — the only task-specific part of a run.

SIM runners (:mod:`semantic_alignment.sim.original`) are task-agnostic: they
train the metasurfaces, build a :data:`~semantic_alignment.sim.link.Transmit`
for the resulting link, and hand it to a :class:`Task`, which owns

- the evaluation (``evaluate`` / ``evaluate_step`` for convergence curves),
- the task's result columns (baselines + SIM metrics) and console summary,
- where results go (``results_dir`` and an optional filename prefix),
- any extra artifacts written next to the Parquet row (``on_saved``).

The deep-unrolling pipeline (:mod:`semantic_alignment.sim.unrolled`) scores
whole batches of noiseless cascades ``β Z_eff`` at once — one per channel
realisation — through :meth:`Task.evaluate_cascades`; its result columns are
named by :attr:`Task.cascade_columns`.

Tasks
-----
ClassificationTask  — linear-probe / neural-classifier accuracy (``results/classification/``).
ReconstructionTask  — RX-decoder MSE/PSNR plus a comparison PNG (``results/reconstruction/``).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from semantic_alignment.evaluation.inference import cascade_accuracies, transmit_accuracy
from semantic_alignment.evaluation.reconstruction import (
    evaluate_reconstruction,
    psnr_from_mse,
    save_comparison_grid,
)
from semantic_alignment.sim.link import Transmit
from semantic_alignment.sim.original._utils import (
    RECONSTRUCTION_RESULTS_DIR,
    RESULTS_DIR,
    neural_str,
    pct,
)
from semantic_alignment.utils import pack_complex, solve, unpack_complex


class Task(ABC):
    """Scores the links produced by a SIM runner and describes the result row.

    Attributes
    ----------
    results_dir : Path
        Root under which runners place ``{result_subdir}/{tipology}/{fname}``.
    fname_prefix : str
        Prepended to every result filename (e.g. the autoencoder regularisation tag).
    cascade_columns : dict[str, str]
        Metric key of :meth:`evaluate_cascades` → its column in unrolled results.
    warm_start_columns : dict[str, str]
        Metric key → its column in the unrolled warm-start metric cache.
    """

    results_dir: Path
    fname_prefix: str = ""
    cascade_columns: dict[str, str] = {}
    warm_start_columns: dict[str, str] = {}

    @abstractmethod
    def evaluate(self, transmit: Transmit) -> dict:
        """Score the final trained link; returns the task's SIM metrics."""
        ...

    def evaluate_step(self, transmit: Transmit) -> dict:
        """Score an intermediate link for a convergence curve (default: :meth:`evaluate`)."""
        return self.evaluate(transmit)

    @abstractmethod
    def columns(self, sim: dict) -> dict:
        """Result columns for this task: baselines followed by the SIM metrics."""
        ...

    @abstractmethod
    def curve_columns(self, curve: list[dict] | None) -> dict:
        """Curve columns from the per-step :meth:`evaluate_step` results (``None``: no curve)."""
        ...

    @abstractmethod
    def summary(self, sim: dict) -> str:
        """One-line console summary of the baselines and SIM metrics."""
        ...

    def on_saved(self, out_path: Path, sim: dict) -> None:
        """Hook called after the result row is written to ``out_path``."""

    # ---- batched cascades (deep unrolling) ----

    def evaluate_cascades(
        self, Z_eff: torch.Tensor, beta: torch.Tensor
    ) -> dict[str, list[float] | None]:
        """Score a batch of noiseless links ``β_b Z_eff[b]`` — one metric value per link.

        Parameters
        ----------
        Z_eff : torch.Tensor
            Effective cascades, shape ``(B, D_rx/2, D_tx/2)``.
        beta : torch.Tensor
            Per-cascade MMSE scalars, shape ``(B,)``.

        Returns
        -------
        dict
            :attr:`cascade_columns` key → ``B`` values (``None`` if unavailable).
        """
        raise NotImplementedError(f"{type(self).__name__} does not score batched cascades.")

    def evaluate_cascade(self, Z_eff: torch.Tensor, beta: torch.Tensor) -> dict:
        """Score a single noiseless link ``β Z_eff`` (``Z_eff`` 2-D, ``beta`` scalar)."""
        out = self.evaluate_cascades(Z_eff.unsqueeze(0), beta.reshape(1))
        return {k: v[0] if v is not None else None for k, v in out.items()}

    def cascade_meta(self) -> dict:
        """Task-specific metadata columns added to every unrolled result row."""
        return {}


class ClassificationTask(Task):
    """Semantic classification: accuracy of the RX classifier(s) on the received embeddings.

    Parameters
    ----------
    test_loader : DataLoader
        Test set yielding ``(z_tx_batch, labels)`` pairs.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Pilot whitening statistics.
    clf : torch.Tensor or nn.Module
        Linear probe trained on RX embeddings.
    device : torch.device
        Inference device.
    no_mismatch_probe, original_probe : float or None
        Linear-probe baselines (RX embeddings / ``A`` applied without SIM or channel);
        not needed by the unrolled pipeline.
    clf_neural : nn.Module or None
        Optional neural classifier, evaluated alongside the probe.
    no_mismatch_neural, original_neural : float or None
        Neural-classifier baselines.
    """

    results_dir = RESULTS_DIR
    cascade_columns = {"probe": "probe_acc", "neural": "neural_acc"}
    warm_start_columns = {"probe": "ws_acc_probe"}

    def __init__(
        self,
        test_loader: DataLoader,
        L_in: torch.Tensor,
        mu_in: torch.Tensor,
        L_out: torch.Tensor,
        mu_out: torch.Tensor,
        clf: torch.Tensor | nn.Module,
        device: torch.device,
        no_mismatch_probe: float | None = None,
        original_probe: float | None = None,
        clf_neural: torch.Tensor | nn.Module | None = None,
        no_mismatch_neural: float | None = None,
        original_neural: float | None = None,
    ) -> None:
        self.test_loader = test_loader
        self.whitening = (L_in, mu_in, L_out, mu_out)
        self.clf = clf
        self.clf_neural = clf_neural
        self.device = device
        self.no_mismatch_probe = no_mismatch_probe
        self.no_mismatch_neural = no_mismatch_neural
        self.original_probe = original_probe
        self.original_neural = original_neural
        self._cascade_inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def _accuracy(self, transmit: Transmit, clf: torch.Tensor | nn.Module) -> float:
        return transmit_accuracy(transmit, self.test_loader, *self.whitening, clf, self.device)

    def evaluate(self, transmit: Transmit) -> dict:
        """``{"probe": acc, "neural": acc | None}`` — probe first, then neural (fixed RNG order)."""
        probe = self._accuracy(transmit, self.clf)
        neural = self._accuracy(transmit, self.clf_neural) if self.clf_neural is not None else None
        return {"probe": probe, "neural": neural}

    def columns(self, sim: dict) -> dict:
        return {
            "no_mismatch_probe": self.no_mismatch_probe,
            "no_mismatch_neural": self.no_mismatch_neural,
            "original_probe": self.original_probe,
            "original_neural": self.original_neural,
            "sim_accuracy_probe": sim["probe"],
            "sim_accuracy_neural": pct(sim["neural"]),
        }

    def curve_columns(self, curve: list[dict] | None) -> dict:
        if curve is None:
            return {"sim_accuracy_probe_curve": None, "sim_accuracy_neural_curve": None}
        return {
            "sim_accuracy_probe_curve": [c["probe"] for c in curve],
            "sim_accuracy_neural_curve": [c["neural"] for c in curve if c["neural"] is not None],
        }

    def precompute_cascade_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Whitened complex test embeddings ``(D_tx/2, N)`` and labels ``(N,)``, cached.

        Iterates ``test_loader`` once (which draws from the global RNG), so call it
        at a fixed point of the setup to keep later random draws reproducible.
        """
        if self._cascade_inputs is None:
            z_list, label_list = [], []
            for z, lbl in self.test_loader:
                z_list.append(z.to(self.device))
                label_list.append(lbl.to(self.device))
            Z_test = torch.cat(z_list, dim=0).T.contiguous()
            Z_test = pack_complex(Z_test, device=self.device)
            L_in, mu_in, _, _ = self.whitening
            Z_test = torch.linalg.solve_triangular(L_in, Z_test - mu_in, upper=False)
            self._cascade_inputs = (Z_test, torch.cat(label_list, dim=0))
        return self._cascade_inputs

    def evaluate_cascades(
        self, Z_eff: torch.Tensor, beta: torch.Tensor
    ) -> dict[str, list[float] | None]:
        """Per-cascade probe (and neural) accuracy on the whole test split."""
        Z_test, labels = self.precompute_cascade_inputs()
        _, _, L_out, mu_out = self.whitening
        B = Z_eff.shape[0]
        z_aw = beta[:, None, None] * (Z_eff @ Z_test)
        z_a = L_out @ z_aw + mu_out
        z_real = unpack_complex(z_a, z_a.device).transpose(-1, -2).float()

        W = self.clf.to(z_real.device)
        labels = labels.to(z_real.device)
        preds = (z_real @ W).argmax(dim=-1)
        out = {
            "probe": (preds == labels[None, :]).float().mean(dim=-1).tolist(),
            "neural": None,
        }
        if self.clf_neural is not None:
            dev = next(self.clf_neural.parameters()).device
            N_test = z_real.shape[1]
            z_flat = z_real.reshape(B * N_test, -1).to(dev)
            preds_n = self.clf_neural(z_flat).argmax(dim=-1).reshape(B, N_test)
            out["neural"] = (preds_n == labels[None, :].to(dev)).float().mean(dim=-1).tolist()
        return out

    def evaluate_cascade(self, Z_eff: torch.Tensor, beta: torch.Tensor) -> dict:
        Z_test, labels = self.precompute_cascade_inputs()
        _, _, L_out, mu_out = self.whitening
        probe, neural = cascade_accuracies(
            Z_eff, beta, Z_test, labels, self.clf, self.clf_neural, L_out, mu_out, self.device
        )
        return {"probe": probe, "neural": neural}

    def summary(self, sim: dict) -> str:
        return (
            f"nm_linear={self.no_mismatch_probe * 100:.2f}%"
            f"{neural_str('nm_neural', self.no_mismatch_neural)}"
            f" | orig_linear={self.original_probe * 100:.2f}%"
            f"{neural_str('orig_neural', self.original_neural)}"
            f" | sim_linear={sim['probe'] * 100:.2f}%{neural_str('sim_neural', pct(sim['neural']))}"
        )


class ReconstructionTask(Task):
    """Image reconstruction: the RX autoencoder decodes the received embedding, scored by PSNR.

    Writes a comparison PNG next to each result row
    (``original`` / ``no_mismatch`` / ``{strategy}_align`` / ``sim_{strategy}``).

    Parameters
    ----------
    z_tx_test : torch.Tensor
        TX embeddings of the test split, shape ``(N, D_tx)``.
    images_test : torch.Tensor
        Ground-truth test images, shape ``(N, C, H, W)``.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Pilot whitening statistics.
    rx_model : ConvVAE
        Trained RX autoencoder; only its decoder half is used.
    device : torch.device
        Inference device.
    strategy_name : str
        Alignment strategy of this job (labels the PNG columns).
    reg_tag : str
        Autoencoder regularisation tag; prefixes result filenames.
    no_mismatch_mse, no_mismatch_psnr : float
        Oracle baseline: RX decodes its own true embedding.
    original_mse, original_psnr : float
        ``A`` applied without SIM or channel.
    no_mismatch_preview, align_preview : torch.Tensor
        Reconstructions of the first ``num_preview`` test images for both baselines.
    batch_size : int, optional
        Mini-batch size for the decode + metric pass.
    num_preview : int, optional
        Number of test samples in the comparison PNG (0 disables it).
    cascade_images : int or None, optional
        Leading test images scored per cascade by :meth:`evaluate_cascades`
        (``None``: all). Deep unrolling decodes every image once per channel
        realisation and step, so it uses a subset.
    """

    results_dir = RECONSTRUCTION_RESULTS_DIR
    cascade_columns = {"psnr": "psnr", "mse": "mse"}
    warm_start_columns = {"psnr": "ws_psnr", "mse": "ws_mse"}

    def __init__(
        self,
        z_tx_test: torch.Tensor,
        images_test: torch.Tensor,
        L_in: torch.Tensor,
        mu_in: torch.Tensor,
        L_out: torch.Tensor,
        mu_out: torch.Tensor,
        rx_model: nn.Module,
        device: torch.device,
        strategy_name: str,
        reg_tag: str,
        no_mismatch_mse: float,
        no_mismatch_psnr: float,
        original_mse: float,
        original_psnr: float,
        no_mismatch_preview: torch.Tensor | None = None,
        align_preview: torch.Tensor | None = None,
        batch_size: int = 512,
        num_preview: int = 5,
        cascade_images: int | None = None,
    ) -> None:
        self.z_tx_test = z_tx_test
        self.images_test = images_test
        self.whitening = (L_in, mu_in, L_out, mu_out)
        self.rx_model = rx_model
        self.device = device
        self.strategy_name = strategy_name
        self.reg_tag = reg_tag
        self.fname_prefix = f"{reg_tag}_"
        self.no_mismatch_mse = no_mismatch_mse
        self.no_mismatch_psnr = no_mismatch_psnr
        self.original_mse = original_mse
        self.original_psnr = original_psnr
        self.no_mismatch_preview = no_mismatch_preview
        self.align_preview = align_preview
        self.batch_size = batch_size
        self.num_preview = min(num_preview, len(images_test))
        self.cascade_images = cascade_images
        self._cascade_inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def _score(self, transmit: Transmit, num_preview: int) -> dict:
        mse, psnr_db, preview = evaluate_reconstruction(
            transmit,
            self.z_tx_test,
            self.images_test,
            *self.whitening,
            self.rx_model,
            self.device,
            batch_size=self.batch_size,
            num_preview=num_preview,
        )
        return {"mse": mse, "psnr": psnr_db, "preview": preview}

    def evaluate(self, transmit: Transmit) -> dict:
        return self._score(transmit, self.num_preview)

    def evaluate_step(self, transmit: Transmit) -> dict:
        return self._score(transmit, 0)

    def columns(self, sim: dict) -> dict:
        return {
            "reg_tag": self.reg_tag,
            "no_mismatch_mse": self.no_mismatch_mse,
            "no_mismatch_psnr": self.no_mismatch_psnr,
            "original_mse": self.original_mse,
            "original_psnr": self.original_psnr,
            "sim_mse": sim["mse"],
            "sim_psnr": sim["psnr"],
        }

    def curve_columns(self, curve: list[dict] | None) -> dict:
        if curve is None:
            return {"sim_psnr_curve": None, "sim_mse_curve": None}
        return {
            "sim_psnr_curve": [c["psnr"] for c in curve],
            "sim_mse_curve": [c["mse"] for c in curve],
        }

    def summary(self, sim: dict) -> str:
        return (
            f"no_mismatch={self.no_mismatch_psnr:.2f}dB"
            f" | original={self.original_psnr:.2f}dB"
            f" | sim={sim['psnr']:.2f}dB"
        )

    def _cascade_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Whitened complex TX embeddings ``(D_tx/2, n)`` and images ``(n, C, H, W)``, cached."""
        if self._cascade_inputs is None:
            n = self.cascade_images or len(self.z_tx_test)
            L_in, mu_in, _, _ = self.whitening
            x = pack_complex(self.z_tx_test[:n].to(self.device).T, device=self.device)
            self._cascade_inputs = (solve(L_in, x - mu_in), self.images_test[:n].to(self.device))
        return self._cascade_inputs

    def evaluate_cascades(
        self, Z_eff: torch.Tensor, beta: torch.Tensor
    ) -> dict[str, list[float] | None]:
        """Per-cascade reconstruction MSE and PSNR (dB) over the first ``cascade_images``."""
        X, images = self._cascade_data()
        _, _, L_out, mu_out = self.whitening
        dev = Z_eff.device
        self.rx_model.to(dev).eval()
        # Decode in full precision even under a bf16-mixed trainer.
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
            z_aw = beta[:, None, None] * (Z_eff @ X.to(Z_eff.dtype))
            z_a = L_out.to(z_aw.dtype) @ z_aw + mu_out.to(z_aw.dtype)
            z_real = unpack_complex(z_a, dev).transpose(-1, -2).float()  # (B, n, D_rx)
            B, n, _ = z_real.shape
            flat = z_real.reshape(B * n, -1)
            se = torch.empty(B * n, device=dev)
            for i in range(0, B * n, self.batch_size):
                idx = torch.arange(i, min(i + self.batch_size, B * n), device=dev)
                recon = self.rx_model.decode(flat[idx]).float()
                se[idx] = (recon - images[idx % n]).pow(2).flatten(1).mean(dim=1)
            mse = se.reshape(B, n).mean(dim=1).tolist()
        return {"psnr": [psnr_from_mse(m) for m in mse], "mse": mse}

    def cascade_meta(self) -> dict:
        n = self.cascade_images or len(self.z_tx_test)
        return {"reg_tag": self.reg_tag, "n_eval_images": min(n, len(self.z_tx_test))}

    def on_saved(self, out_path: Path, sim: dict) -> None:
        """Write the comparison PNG grid next to the Parquet row."""
        n = self.num_preview
        if n == 0 or sim["preview"] is None:
            return
        grid = {"original": self.images_test[:n]}
        if self.no_mismatch_preview is not None:
            grid["no_mismatch"] = self.no_mismatch_preview[:n].cpu()
        if self.align_preview is not None:
            grid[f"{self.strategy_name}_align"] = self.align_preview[:n].cpu()
        grid[f"sim_{self.strategy_name}"] = sim["preview"].cpu()
        png_path = save_comparison_grid(grid, out_path.with_suffix(".png"))
        print(f" Saved comparison grid: {png_path.name}")

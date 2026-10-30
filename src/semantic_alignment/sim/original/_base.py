"""Abstract base class for original (non-unrolled) SIM experiment runners."""

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ._utils import save_params, save_result

if TYPE_CHECKING:
    from semantic_alignment.evaluation.tasks import Task


class SIMRunner(ABC):
    """Abstract base class for all original (non-unrolled) SIM experiment runners.

    Runners are task-agnostic. Concrete subclasses (e.g. ``JointSIM``,
    ``MonoSIM``) implement ``run()``: build the cascades, optimize the phases,
    turn the trained link into a :data:`~semantic_alignment.sim.link.Transmit`,
    and let the :class:`~semantic_alignment.evaluation.tasks.Task` score it.
    The task also decides where the result row is written.
    """

    @abstractmethod
    def run(self, task: "Task", **kwargs):
        """Execute the experiment and write results to disk.

        Parameters
        ----------
        task : Task
            Downstream task scoring the trained link (classification, reconstruction, ...).
        **kwargs
            Experiment-specific configuration (seeds, SNR, dataset, etc.).
        """
        ...

    @staticmethod
    def seed_everything(seed: int) -> None:
        """Seed torch, NumPy and ``random`` with the same value."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def out_path(task: "Task", result_subdir: str, tipology: str, fname: str) -> Path:
        """``{task.results_dir}/{result_subdir}[/{tipology}]/{task.fname_prefix}{fname}``."""
        out_dir = task.results_dir / result_subdir
        if tipology:
            out_dir = out_dir / tipology
        return out_dir / f"{task.fname_prefix}{fname}"

    @staticmethod
    def save(
        task: "Task",
        out_path: Path,
        head: dict,
        sim: dict,
        tail: dict,
        params: dict | None = None,
    ) -> None:
        """Write ``head`` + task columns + ``tail`` as one Parquet row, then run the task hook.

        ``params`` (trained phases, cascades, H, β) goes to the ``.pt`` next to
        the Parquet, with ``head`` attached as ``meta`` (see :func:`save_params`).
        """
        save_result(out_path, {**head, **task.columns(sim), **tail})
        if params is not None:
            save_params(out_path, {**params, "meta": head})
        task.on_saved(out_path, sim)

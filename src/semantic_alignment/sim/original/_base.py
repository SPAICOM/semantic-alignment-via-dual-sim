"""Abstract base class for original (non-unrolled) SIM experiment runners."""

from abc import ABC, abstractmethod


class SIMRunner(ABC):
    """Abstract base class for all original (non-unrolled) SIM experiment runners.

    Concrete subclasses (e.g. ``JointSIM``, ``MonoSIM``) implement ``run()``
    to execute a full simulation pipeline — build cascades, optimize phases,
    evaluate channel quality — and write results to disk.
    """

    @abstractmethod
    def run(self, **kwargs):
        """Execute the experiment and write results to disk.

        Parameters
        ----------
        **kwargs
            Experiment-specific configuration (seeds, SNR, dataset, etc.).
        """
        ...

"""Experiments sub-package: single-job runner functions for parameter sweeps.

Each ``run_single_*`` function executes one configuration point of a sweep and
returns a results dict suitable for aggregation by the calling script:

- :func:`run_single_layers_config`      — dual-SIM, vary number of SIM layers
- :func:`run_single_snr_config`         — dual-SIM, vary receive SNR
- :func:`run_single_slayer_config`      — dual-SIM, vary the s-layer index
- :func:`run_single_mono_layers_config` — mono-SIM (TX only), vary layers
- :func:`run_single_disjoint_layers`    — disjoint TX/RX encoders, vary layers
- :func:`run_single_disjoint_snr`       — disjoint TX/RX encoders, vary SNR
- :func:`run_single_rx_depth`           — vary RX encoder depth (mono-SIM)
"""

from semantic_alignment.experiments.experiment_runner import (
    run_single_disjoint_layers,
    run_single_disjoint_snr,
    run_single_layers_config,
    run_single_mono_layers_config,
    run_single_rx_depth,
    run_single_slayer_config,
    run_single_snr_config,
)

__all__ = [
    "run_single_layers_config",
    "run_single_snr_config",
    "run_single_slayer_config",
    "run_single_mono_layers_config",
    "run_single_disjoint_layers",
    "run_single_disjoint_snr",
    "run_single_rx_depth",
]

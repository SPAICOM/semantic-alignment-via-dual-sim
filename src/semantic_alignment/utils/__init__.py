"""Utils sub-package: signal-processing primitives and alignment strategies.

Signal-processing utilities (``utils.py``)
    Complex tensor helpers, AWGN noise injection (:func:`awgn`), SNR / sigma
    conversion, pre-whitening (:func:`prewhiten`), MMSE-SVD equalization
    (:func:`mmse_svd_equalizer`), and Cholesky-based linear solves
    (:func:`a_inv_times_b`).

Alignment strategies (``alignment_utils.py``)
    :func:`ppfe`             — zero-shot Prototype-based Parseval Frame Equalization.
    :func:`ridge_regression` — supervised linear alignment via weighted ridge regression.
"""

from semantic_alignment.utils.alignment_utils import ppfe, ridge_regression
from semantic_alignment.utils.utils import (
    a_inv_times_b,
    awgn,
    complex_compressed_tensor,
    complex_gaussian_matrix,
    complex_tensor,
    decompress_complex_tensor,
    get_rx_equalizer,
    mmse_svd_equalizer,
    prewhiten,
    remove_non_empty_dir,
    sigma_given_snr,
    snr,
)

__all__ = [
    "complex_compressed_tensor",
    "decompress_complex_tensor",
    "complex_tensor",
    "complex_gaussian_matrix",
    "awgn",
    "snr",
    "sigma_given_snr",
    "a_inv_times_b",
    "prewhiten",
    "remove_non_empty_dir",
    "mmse_svd_equalizer",
    "get_rx_equalizer",
    "ppfe",
    "ridge_regression",
]

"""Utils sub-package: channel, signal-processing, alignment, and I/O utilities.

Channel and signal processing (``channel.py``)
    Complex Gaussian matrices (:func:`cn_matrix`), AWGN noise (:func:`awgn`),
    SNR conversion (:func:`snr`, :func:`sigma_from_snr`), MIMO equalization
    (:func:`svd_equalizer`, :func:`rx_equalizer`), complex tensor encoding
    (:func:`pack_complex`, :func:`unpack_complex`, :func:`to_complex`),
    Cholesky linear solve (:func:`solve`), and pre-whitening (:func:`prewhiten`).

Alignment strategies (``alignment.py``)
    :func:`ppfe`             — zero-shot Prototype-based Parseval Frame Equalization.
    :func:`ridge_regression` — supervised linear alignment via weighted ridge regression.

Filesystem I/O (``io.py``)
    :func:`remove_dir` — recursive directory removal.
"""

# Alignment strategies: zero-shot PPFE and supervised ridge regression.
from semantic_alignment.utils.alignment import ppfe, ridge_regression

# Channel generation, noise, SNR, equalisation, complex encoding, whitening.
from semantic_alignment.utils.channel import (
    awgn,
    channel,
    cn_matrix,
    pack_complex,
    prewhiten,
    rician_los,
    rician_matrix,
    rx_equalizer,
    sigma_from_snr,
    snr,
    solve,
    standardize_channel,
    svd_equalizer,
    to_complex,
    unpack_complex,
)

# Filesystem helpers (recursive directory removal).
from semantic_alignment.utils.io import remove_dir

__all__ = [
    "awgn",
    "channel",
    "cn_matrix",
    "pack_complex",
    "prewhiten",
    "rician_los",
    "rician_matrix",
    "rx_equalizer",
    "standardize_channel",
    "sigma_from_snr",
    "snr",
    "solve",
    "svd_equalizer",
    "to_complex",
    "unpack_complex",
    "remove_dir",
    "ppfe",
    "ridge_regression",
]

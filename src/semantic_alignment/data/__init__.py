"""Data sub-package: loading and preparing semantic embeddings and channel data.

Fetches pre-computed encoder embeddings from the HuggingFace *spaicom-lab/semasia-*
collection via Polars lazy Parquet scanning.  Provides :class:`SemasiaData` /
:class:`SplitData` containers for paired TX/RX embedding tensors, and
:func:`select_pilots` for subsampling semantic pilots (centroid or random) used
to calibrate alignment matrices.

The high-level :func:`prepare_semasia` function runs the full preprocessing
pipeline (load → pilots → prewhiten → alignment target) and returns a
:class:`PreparedData` container.
"""

# Random MIMO channel generator for training unrolled models.
from semantic_alignment.data.h_channel import HChannelDataModule

# Full preprocessing pipeline: load → pilots → prewhiten → A_target.
from semantic_alignment.data.prepare import PreparedData, prepare_semasia

# Raw embedding I/O and pilot selection from the Semasia HuggingFace collection.
from semantic_alignment.data.semasia import (
    SemasiaData,
    SplitData,
    load_semasia,
    load_split,
    select_pilots,
)

__all__ = [
    "HChannelDataModule",
    "PreparedData",
    "prepare_semasia",
    "SplitData",
    "SemasiaData",
    "load_split",
    "load_semasia",
    "select_pilots",
]

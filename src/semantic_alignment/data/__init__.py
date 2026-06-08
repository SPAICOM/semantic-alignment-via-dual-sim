"""Data sub-package: loading and preparing semantic embeddings.

Fetches pre-computed encoder embeddings from the HuggingFace *spaicom-lab/semasia-*
collection via Polars lazy Parquet scanning.  Provides :class:`SemaSIAData` /
:class:`SplitData` containers for paired TX/RX embedding tensors, and
:func:`select_pilots` for subsampling semantic pilots (centroid or random) used
to calibrate alignment matrices.
"""

from semantic_alignment.data.semasia import (
    SemaSIAData,
    SplitData,
    load_semasia,
    load_split,
    select_pilots,
)

__all__ = [
    "SplitData",
    "SemaSIAData",
    "load_split",
    "load_semasia",
    "select_pilots",
]

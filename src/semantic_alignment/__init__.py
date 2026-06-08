"""Semantic Alignment via Dual Stacked Intelligent Metasurfaces.

Sub-packages
------------
data        : loading paired TX/RX embeddings from the HuggingFace Semasia collection
              and pilot selection utilities.
evaluation  : linear probe, neural MLP classifier, inference routines, and the
              no-MIMO baseline accuracy.
experiments : single-job runner functions for all sweep axes
              (layers, SNR, s-layer, mono, disjoint).
sim         : DualSIM and MonoSIM phase-profile optimizers (NumPy and Torch variants).
utils       : signal-processing primitives (AWGN, MMSE, pre-whitening) and
              alignment strategies (PPFE, ridge regression).
"""

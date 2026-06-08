"""Evaluation sub-package: classifiers, inference loops, and baselines.

Provides two classifier back-ends — a closed-form :func:`train_linear_probe`
(least-squares) and a trained :class:`NeuralClassifier` MLP — together with a
unified dispatch layer (:func:`clf_predict`, :func:`clf_accuracy`).

Inference routines (:func:`run_evaluation`, :func:`run_evaluation_mmse`,
:func:`run_evaluation_disjoint_fixed`) simulate the full TX → SIM → channel →
SIM → RX pipeline and return top-1 accuracy.

:func:`run_original_accuracy` implements the *Accuracy Original No MIMO*
baseline: alignment applied in software with no channel or noise.
"""

from semantic_alignment.evaluation.inference import (
    run_evaluation,
    run_evaluation_disjoint_fixed,
    run_evaluation_mmse,
)
from semantic_alignment.evaluation.linear_probe import (
    linear_probe_accuracy,
    linear_probe_predict,
    train_linear_probe,
)
from semantic_alignment.evaluation.original_accuracy import run_original_accuracy

__all__ = [
    "run_evaluation",
    "run_evaluation_mmse",
    "run_evaluation_disjoint_fixed",
    "run_original_accuracy",
    "train_linear_probe",
    "linear_probe_predict",
    "linear_probe_accuracy",
]

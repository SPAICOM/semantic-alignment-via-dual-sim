"""Models sub-package: locally trained encoders for tasks without a Semasia collection.

:class:`ConvVAE`
    Convolutional beta-VAE whose bottleneck embeddings feed the SIM alignment
    pipeline for image reconstruction. Train two variants (``base``/``small``)
    per dataset with :func:`train_autoencoder` to get a TX/RX encoder pair.

Checkpoints live under ``models/autoencoders/{dataset}/{variant}/seed{seed}.pt``
(see :func:`save_autoencoder` / :func:`load_autoencoder`).
"""

from semantic_alignment.models.autoencoder import (
    ConvVAE,
    load_autoencoder,
    load_autoencoder_training_meta,
    regularization_tag,
    save_autoencoder,
    train_autoencoder,
)

__all__ = [
    "ConvVAE",
    "load_autoencoder",
    "load_autoencoder_training_meta",
    "regularization_tag",
    "save_autoencoder",
    "train_autoencoder",
]

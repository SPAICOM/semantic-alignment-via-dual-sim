"""Train and save a neural MLP classifier on RX embeddings.

One checkpoint per (dataset, rx_enc, seed) — saved to models/classifiers/.
Training metrics are streamed to Weights & Biases.

Examples
--------
    just train-clf                                   # full sweep from config
    just train-clf datasets=cifar100
    just train-clf "training.seed=27,42"
    uv run python scripts/train_classifier.py training.seed=42
"""

import random

import hydra
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from semantic_alignment.data.semasia import load_split
from semantic_alignment.evaluation.linear_probe import clf_accuracy
from semantic_alignment.evaluation.neural_classifier import (
    load_neural_clf,
    save_neural_clf,
    train_neural_classifier,
)


@hydra.main(version_base=None, config_path="../config", config_name="config_train_clf")
def main(cfg: DictConfig) -> None:
    """Train and persist a neural MLP classifier for one (dataset, rx_enc, seed) triple.

    Skips training if a checkpoint already exists for this configuration.
    Logs training metrics to Weights & Biases and writes the final checkpoint
    to ``models/classifiers/``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration composed from ``config_train_clf``.
    """
    seed = cfg.training.seed
    dataset = cfg.datasets.hf_name
    rx_enc = cfg.data.rx_enc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f" seed={seed} | dataset={dataset} | rx_enc={rx_enc}")

    # Skip if already trained
    try:
        load_neural_clf(dataset, rx_enc, seed, device)
        print(" Already trained — skipping.")
        return
    except FileNotFoundError:
        pass

    print(f" Loading RX embeddings: {rx_enc}")
    z_rx_train, labels_train = load_split(
        dataset, rx_enc, cfg.datasets.train_split, cfg.datasets.label_col
    )
    z_rx_test, labels_test = load_split(
        dataset, rx_enc, cfg.datasets.test_split, cfg.datasets.label_col
    )

    clf_cfg = cfg.classifier
    logger = WandbLogger(
        project="dual-sim-project-classifier",
        name=f"{dataset}_{rx_enc}_seed{seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )

    print(
        f" Training NeuralClassifier(hidden={clf_cfg.hidden_dim}, lr={clf_cfg.lr}, "
        f"epochs={clf_cfg.max_epochs}, patience={clf_cfg.patience}) ..."
    )

    model = train_neural_classifier(
        embeddings=z_rx_train,
        labels=labels_train,
        num_classes=cfg.datasets.num_classes,
        hidden_dim=clf_cfg.hidden_dim,
        lr=clf_cfg.lr,
        max_epochs=clf_cfg.max_epochs,
        patience=clf_cfg.patience,
        batch_size=clf_cfg.batch_size,
        val_fraction=clf_cfg.val_fraction,
        device=device,
        seed=seed,
        logger=logger,
    )

    test_acc = clf_accuracy(model, z_rx_test.to(device), labels_test.to(device))
    print(f" Test accuracy: {test_acc * 100:.2f}%")
    logger.log_metrics({"test_acc": test_acc})
    logger.experiment.finish()

    path = save_neural_clf(model, dataset, rx_enc, seed)
    print(f" Saved to {path}")


if __name__ == "__main__":
    main()

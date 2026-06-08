"""MLP classifier trained on RX embeddings — drop-in alternative to the linear probe."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

MODELS_DIR = Path("models/classifiers")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class NeuralClassifier(nn.Module):
    """Two-layer MLP with LayerNorm, matching the reference-repo architecture."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class NeuralClassifierModule(L.LightningModule):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuralClassifier(input_dim, num_classes, hidden_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _ckpt_path(dataset: str, rx_enc: str, seed: int) -> Path:
    safe_enc = rx_enc.replace("/", "_").replace(".", "_")
    return MODELS_DIR / dataset / safe_enc / f"seed{seed}.pt"


def save_neural_clf(model: NeuralClassifier, dataset: str, rx_enc: str, seed: int) -> Path:
    path = _ckpt_path(dataset, rx_enc, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.net[1].in_features,
            "num_classes": model.net[-1].out_features,
            "hidden_dim": model.net[1].out_features,
        },
        path,
    )
    return path


def load_neural_clf(dataset: str, rx_enc: str, seed: int, device: torch.device) -> NeuralClassifier:
    path = _ckpt_path(dataset, rx_enc, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"No neural classifier at {path}. "
            f"Run: just train-clf datasets={dataset} data.rx_enc={rx_enc} training.seed={seed}"
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = NeuralClassifier(ckpt["input_dim"], ckpt["num_classes"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_neural_classifier(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    hidden_dim: int = 512,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    device: torch.device | str = "cpu",
    seed: int = 42,
    logger=None,
) -> NeuralClassifier:
    L.seed_everything(seed)

    dataset = TensorDataset(embeddings.float(), labels.long())
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = embeddings.shape[1]
    module = NeuralClassifierModule(input_dim, num_classes, hidden_dim, lr)

    callbacks = [
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.fit(module, train_loader, val_loader)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        module = NeuralClassifierModule.load_from_checkpoint(best_ckpt)

    return module.model.eval()

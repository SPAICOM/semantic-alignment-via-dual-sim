"""Lightning DataModule for random MIMO channel matrices H ~ CN(0, I/2)."""

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class HChannelDataModule(LightningDataModule):
    def __init__(
        self,
        n_R: int,
        n_T: int,
        n_total: int = 10000,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.n_R = n_R
        self.n_T = n_T
        self.n_total = n_total
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def setup(self, stage: str | None = None):
        g = torch.Generator().manual_seed(self.seed)
        all_H = torch.randn(
            self.n_total, self.n_R, self.n_T, dtype=torch.complex64, generator=g
        ) / (2**0.5)

        n_train = int(self.n_total * self.train_ratio)
        n_val = int(self.n_total * self.val_ratio)

        self.train_H = all_H[:n_train]
        self.val_H = all_H[n_train : n_train + n_val]
        self.test_H = all_H[n_train + n_val :]

    def train_dataloader(self):
        ds = TensorDataset(self.train_H)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        ds = TensorDataset(self.val_H)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        ds = TensorDataset(self.test_H)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

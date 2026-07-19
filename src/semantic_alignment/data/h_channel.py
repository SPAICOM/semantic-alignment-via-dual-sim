"""Lightning DataModule for i.i.d. complex-Gaussian MIMO channels.

Generates ``n_total`` channel matrices ``H ~ CN(0, 1)`` (complex standard
normal) of shape ``(n_R, n_T)`` and splits them into train / val / test sets
according to the given ratios.  When ``precompute_svd=True`` the thin SVD
of every matrix is cached so that downstream code can skip repeated SVD
decompositions during training.
"""

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from semantic_alignment.utils import cn_matrix


class HChannelDataModule(LightningDataModule):
    """DataModule that yields batches of random MIMO channel matrices.

    Each sample is an independent draw ``H ~ CN(0, 1)`` of shape
    ``(n_R, n_T)`` where ``n_R`` is the number of receive antennas and
    ``n_T`` the number of transmit antennas.

    Parameters
    ----------
    n_R : int
        Number of receive antennas (rows of H).
    n_T : int
        Number of transmit antennas (columns of H).
    n_total : int, optional
        Total number of channel realisations to generate (default 10 000).
    batch_size : int, optional
        Batch size for every dataloader (default 32).
    train_ratio : float, optional
        Fraction of samples used for training (default 0.7).
    val_ratio : float, optional
        Fraction of samples used for validation (default 0.15).
        The remaining ``1 - train_ratio - val_ratio`` is the test set.
    seed : int, optional
        RNG seed for reproducibility (default 42).
    precompute_svd : bool, optional
        If ``True``, the thin SVD ``H = U @ diag(s) @ Vh`` is computed once
        during :meth:`setup` and stored alongside each split
        (default ``False``).

    Attributes
    ----------
    train_H : Tensor, shape ``(n_train, n_R, n_T)``, complex64
        Training channel matrices.
    val_H : Tensor, shape ``(n_val, n_R, n_T)``, complex64
        Validation channel matrices.
    test_H : Tensor, shape ``(n_test, n_R, n_T)``, complex64
        Test channel matrices.
    train_svd : tuple[Tensor, Tensor, Tensor] or None
        ``(U, s, Vh)`` for the training set, only when
        ``precompute_svd=True``.  Shapes: ``U (n_train, n_R, k)``,
        ``s (n_train, k)``, ``Vh (n_train, k, n_T)`` where
        ``k = min(n_R, n_T)``.
    val_svd : tuple or None
        Same as ``train_svd`` for the validation set.
    test_svd : tuple or None
        Same as ``train_svd`` for the test set.

    Notes
    -----
    Batch layout returned by each dataloader:

    * ``precompute_svd=False`` — ``(H,)`` with ``H.shape == (B, n_R, n_T)``
    * ``precompute_svd=True``  — ``(H, U, s, Vh)`` with shapes
      ``(B, n_R, n_T)``, ``(B, n_R, k)``, ``(B, k)``, ``(B, k, n_T)``

    Examples
    --------
    >>> dm = HChannelDataModule(n_R=4, n_T=6, n_total=5000)
    >>> dm.setup()
    >>> batch, = next(iter(dm.train_dataloader()))  # (H,)
    >>> batch.shape
    torch.Size([32, 4, 6])
    """

    def __init__(
        self,
        n_R: int,
        n_T: int,
        n_total: int = 10000,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        precompute_svd: bool = False,
    ):
        """Initialize the H channel dataset.

        Parameters
        ----------
        n_R : int
            Number of RX antennas.
        n_T : int
            Number of TX antennas.
        n_total : int, optional
            Total number of channel realizations.
        batch_size : int, optional
            Batch size for DataLoaders.
        train_ratio : float, optional
            Fraction of data for training.
        val_ratio : float, optional
            Fraction of data for validation.
        seed : int, optional
            Random seed for reproducibility.
        precompute_svd : bool, optional
            If True, precompute SVD decompositions of all H matrices.
        """
        super().__init__()
        # Persist all constructor args so Lightning can serialise them.
        self.save_hyperparameters(ignore=[])
        self.n_R = n_R
        self.n_T = n_T
        self.n_total = n_total
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.precompute_svd = precompute_svd

    def setup(self, stage: str | None = None):
        """Generate all channel matrices and split into train / val / test.

        Parameters
        ----------
        stage : str or None, optional
            Ignored — the full dataset is always materialised regardless of
            the stage string passed by the Lightning Trainer.
        """
        # Seed a local generator so the global RNG is unaffected.
        g = torch.Generator().manual_seed(self.seed)

        # Draw n_total i.i.d. CN(0,1) matrices of shape (n_R, n_T).
        all_H = cn_matrix(0, 1, (self.n_total, self.n_R, self.n_T), generator=g)

        # Compute split sizes; test gets whatever remains after train + val.
        n_train = int(self.n_total * self.train_ratio)
        n_val = int(self.n_total * self.val_ratio)

        # Slice contiguous blocks (no shuffle here — shuffling happens in
        # the DataLoader for the training set).
        splits = [
            all_H[:n_train],
            all_H[n_train : n_train + n_val],
            all_H[n_train + n_val :],
        ]
        self.train_H, self.val_H, self.test_H = splits

        if self.precompute_svd:
            # Thin SVD: U (N, n_R, k), s (N, k), Vh (N, k, n_T)
            # with k = min(n_R, n_T).  full_matrices=False avoids
            # materialising the full unitary matrices.
            svds = [torch.linalg.svd(H, full_matrices=False) for H in splits]
            self.train_svd = svds[0]
            self.val_svd = svds[1]
            self.test_svd = svds[2]

    def _make_dataset(
        self,
        H: torch.Tensor,
        svd: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> TensorDataset:
        """Wrap a split's tensors into a :class:`TensorDataset`.

        Parameters
        ----------
        H : Tensor, shape ``(N, n_R, n_T)``
            Channel matrices for one split.
        svd : tuple[Tensor, Tensor, Tensor] or None, optional
            Precomputed ``(U, s, Vh)`` for the same split.  When *None*
            the dataset contains only *H*.

        Returns
        -------
        TensorDataset
            ``(H,)`` if *svd* is *None*, otherwise ``(H, U, s, Vh)``.
        """
        if svd is not None:
            U, s, Vh = svd
            return TensorDataset(H, U, s, Vh)
        return TensorDataset(H)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader (shuffled).

        Returns
        -------
        DataLoader
            Yields ``(H,)`` or ``(H, U, s, Vh)`` batches of size
            ``self.batch_size``.  ``H`` is complex64.
        """
        ds = self._make_dataset(self.train_H, self.train_svd if self.precompute_svd else None)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader (not shuffled).

        Returns
        -------
        DataLoader
            Yields ``(H,)`` or ``(H, U, s, Vh)`` batches of size
            ``self.batch_size``.  ``H`` is complex64.
        """
        ds = self._make_dataset(self.val_H, self.val_svd if self.precompute_svd else None)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader (not shuffled).

        Returns
        -------
        DataLoader
            Yields ``(H,)`` or ``(H, U, s, Vh)`` batches of size
            ``self.batch_size``.  ``H`` is complex64.
        """
        ds = self._make_dataset(self.test_H, self.test_svd if self.precompute_svd else None)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

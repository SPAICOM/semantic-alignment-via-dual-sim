"""Lightning DataModule for random MIMO channel matrices (Rayleigh or Rician).

Generates ``n_total`` channel matrices of shape ``(n_R, n_T)`` and splits
them into train / val / test sets according to the given ratios.
When ``precompute_svd=True`` the thin SVD of every matrix is cached so that
downstream code can skip repeated SVD decompositions during training.

Channel models (``channel_type``):

* ``"rayleigh"`` (default) — i.i.d. CN(0, 1) entries.
* ``"rician"`` — Rician fading: ``H_i = sqrt(K/(K+1))*H_los + sqrt(1/(K+1))*H_nlos_i``
  where ``K = 10^(rice_factor_db/10)`` and ``H_los`` is a fixed rank-1 LoS
  component (outer product of random phase-shift steering vectors) shared
  across all samples.  Requires ``rice_factor_db`` to be set.

Rayleigh perturbation (``h0_var``, Rayleigh only):
  A single base channel ``H_0 ~ CN(0, 1)`` is drawn first; each realisation
  is ``H_i = H_0 + sqrt(h0_var) * eps_i`` where ``eps_i ~ CN(0, 1)``.
  ``h0_var`` is the per-element perturbation variance; with ``h0_var=1``
  each entry has the same variance as a pure Rayleigh channel.
  Limit cases: ``h0_var=0`` → all realisations equal ``H_0`` (deterministic);
  ``h0_var=inf`` → pure i.i.d. Rayleigh (same as ``h0_var=None``).
  Not compatible with Rician.
"""

import math

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from semantic_alignment.utils import cn_matrix
from semantic_alignment.utils.channel import channel as _channel


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
        h0_var: float | None = None,
        channel_type: str = "rayleigh",
        rice_factor_db: float | None = None,
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
        self.h0_var = h0_var
        self.channel_type = channel_type
        self.rice_factor_db = rice_factor_db

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

        if self.h0_var is not None and not math.isinf(self.h0_var):
            if self.channel_type != "rayleigh":
                raise ValueError(
                    "h0_var perturbation is only supported with channel_type='rayleigh'; "
                    f"got channel_type={self.channel_type!r}"
                )
            # Draw a single base channel H_0 ~ CN(0,1), then perturb.
            # h0_var=0 → scale=0 → all realisations equal H_0 (deterministic).
            H0 = cn_matrix(0, 1, (self.n_R, self.n_T), generator=g)
            noise = cn_matrix(0, 1, (self.n_total, self.n_R, self.n_T), generator=g)
            scale = self.h0_var**0.5
            all_H = H0.unsqueeze(0) + scale * noise
        else:
            # h0_var=None or h0_var=inf → pure i.i.d. Rayleigh
            all_H = _channel(
                self.n_R,
                self.n_T,
                n_samples=self.n_total,
                channel_type=self.channel_type,
                rice_factor_db=self.rice_factor_db,
                generator=g,
            )

        # For Rician channels: per-sample standardise each matrix so entries
        # have zero mean and unit variance, matching the CN(0,1) assumption of
        # the SIM optimizer.  Raw Rician has Var[H_ij]=1/(K+1) which shrinks
        # gradients by sqrt(K+1) and breaks the learned step sizes.
        if self.channel_type == "rician":
            flat = all_H.reshape(self.n_total, -1)
            mu = flat.mean(-1, keepdim=True)
            sigma = (flat - mu).abs().pow(2).mean(-1, keepdim=True).sqrt()
            all_H = ((all_H.reshape(self.n_total, -1) - mu) / (sigma + 1e-8)).reshape(
                self.n_total, self.n_R, self.n_T
            )

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

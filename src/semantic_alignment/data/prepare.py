"""Feature preprocessing and alignment-target computation for SIM experiments.

Encapsulates the pipeline shared by all experiment entry points:
    load_semasia → select_pilots → pack/prewhiten → A_target (Linear or PPFE)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from semantic_alignment.data.semasia import SemasiaData, SplitData, load_semasia, select_pilots
from semantic_alignment.utils import pack_complex, ppfe, prewhiten, ridge_regression


@dataclass
class PreparedData:
    """All preprocessing outputs for a Semasia experiment run.

    Attributes
    ----------
    data : SemasiaData
        Full train/test embedding container.
    pilots : SplitData
        Subsampled pilot set used to estimate the alignment matrix.
    input_w : torch.Tensor
        Whitened TX pilot embeddings, shape ``(D/2, N_pilot)``, complex64.
    L_in : torch.Tensor
        Lower-triangular Cholesky factor of the TX covariance,
        shape ``(D/2, D/2)``, complex64.
    mu_in : torch.Tensor
        TX pilot column mean, shape ``(D/2, 1)``, complex64.
    output_w : torch.Tensor
        Whitened RX pilot embeddings, shape ``(D/2, N_pilot)``, complex64.
    L_out : torch.Tensor
        Lower-triangular Cholesky factor of the RX covariance,
        shape ``(D/2, D/2)``, complex64.
    mu_out : torch.Tensor
        RX pilot column mean, shape ``(D/2, 1)``, complex64.
    A_target : torch.Tensor
        Alignment target matrix, shape ``(D/2, D/2)``, complex64.
        For Linear alignment this is the ridge-regression solution;
        for PPFE it is ``F_R_H @ F_T``.
    F_T : torch.Tensor or None
        TX prototype frame matrix from PPFE, or ``None`` when
        ``alignment_type="Linear"`` or ``keep_csi_frames=False``.
    F_R_H : torch.Tensor or None
        Conjugate-transpose of the RX prototype frame matrix from PPFE,
        or ``None`` under the same conditions as ``F_T``.
    """

    data: SemasiaData
    pilots: SplitData
    # Whitened pilot tensors — shape (D/2, N_pilot), complex64
    input_w: torch.Tensor
    L_in: torch.Tensor
    mu_in: torch.Tensor
    output_w: torch.Tensor
    L_out: torch.Tensor
    mu_out: torch.Tensor
    # Alignment target
    A_target: torch.Tensor
    F_T: torch.Tensor | None  # None for Linear alignment or non-CSI PPFE
    F_R_H: torch.Tensor | None


def ppfe_cluster_data(
    train: SplitData,
    L_in: torch.Tensor,
    mu_in: torch.Tensor,
    L_out: torch.Tensor,
    mu_out: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPFE inputs over the whole training latent space, clustered on the TX side.

    The TX clusters its full training latent space and shares the
    sample → cluster map; the RX builds its own prototypes from its embeddings
    of the same samples with that map, so the rows of ``F_T`` and the columns
    of ``F_R_H`` correspond. Both sides are whitened with the pilot statistics.

    Returns
    -------
    tuple
        ``(X_w, Y_w, z_tx)``: whitened TX/RX training embeddings ``(D/2, N)``,
        complex64, and the real TX embeddings ``(N, D_tx)`` that KMeans clusters.
    """
    X_w = pack_complex(train.z_tx.T, device=device)
    X_w = torch.linalg.solve_triangular(L_in, X_w - mu_in, upper=False)
    Y_w = pack_complex(train.z_rx.T, device=device)
    Y_w = torch.linalg.solve_triangular(L_out, Y_w - mu_out, upper=False)
    return X_w, Y_w, train.z_tx


def prepare_semasia(
    hf_name: str,
    tx_enc: str,
    rx_enc: str,
    train_label_size: int,
    alignment_type: str,
    *,
    lmb: float = 1e-3,
    n_clusters: int | None = None,
    n_proto: int | None = None,
    keep_csi_frames: bool = False,
    cluster_on: str = "pilots",
    method: str = "centroid",
    grouping: str = "label",
    label_col: str = "label",
    train_split: str = "train",
    test_split: str = "test",
    save_to_disk: bool = False,
    cache_dir: Path | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> PreparedData:
    """Load Semasia, select pilots, prewhiten, and compute the alignment target.

    Parameters
    ----------
    hf_name : str
        Semasia dataset name (e.g. ``"cifar10"``).
    tx_enc : str
        TX-side encoder model name.
    rx_enc : str
        RX-side encoder model name (may differ from ``tx_enc``).
    train_label_size : int
        Number of pilot samples per class.
    alignment_type : str
        ``"Linear"`` (ridge regression) or ``"PPFE"`` (prototype frames).
    lmb : float, optional
        Ridge-regression regularisation (only used when
        ``alignment_type="Linear"``, default ``1e-3``).
    n_clusters : int or None, optional
        Number of PPFE clusters (required when ``alignment_type="PPFE"``).
    n_proto : int or None, optional
        Prototypes per cluster for PPFE.
    keep_csi_frames : bool, optional
        When ``True``, ``F_T`` and ``F_R_H`` are returned in the result
        even for non-CSI setups (PPFE only, default ``False``).
    cluster_on : {"pilots", "train_tx"}, optional
        Where PPFE's KMeans runs: ``"pilots"`` clusters the RX pilots and builds
        the prototypes from them (default); ``"train_tx"`` clusters the whole TX
        training latent space and builds both sides' prototypes from all
        training samples with that map (see :func:`ppfe_cluster_data`).
    method : str, optional
        Pilot selection strategy forwarded to :func:`select_pilots`
        (default ``"centroid"``).
    grouping : str, optional
        Pilot grouping strategy forwarded to :func:`select_pilots`
        (default ``"label"``).
    label_col : str, optional
        Parquet label column forwarded to :func:`load_semasia`
        (default ``"label"``).
    train_split : str, optional
        Training split name forwarded to :func:`load_semasia`
        (default ``"train"``).
    test_split : str, optional
        Test split name forwarded to :func:`load_semasia`
        (default ``"test"``).
    save_to_disk : bool, optional
        Persist downloaded splits to disk (default ``False``).
    cache_dir : Path or None, optional
        Root of the local cache tree forwarded to :func:`load_semasia`.
    seed : int, optional
        Random seed for pilot selection and PPFE (default 42).
    device : torch.device or None, optional
        Torch device for all output tensors (default CPU).

    Returns
    -------
    PreparedData
        Container with all preprocessed tensors and the alignment target.

    Raises
    ------
    ValueError
        If ``alignment_type`` is not ``"Linear"`` or ``"PPFE"``, or if
        ``alignment_type="PPFE"`` but ``n_clusters`` is ``None``.
    """
    if device is None:
        device = torch.device("cpu")

    # --- 1. Load paired TX/RX embeddings from HuggingFace / local cache. ---
    data = load_semasia(
        hf_name=hf_name,
        tx_enc=tx_enc,
        rx_enc=rx_enc,
        label_col=label_col,
        train_split=train_split,
        test_split=test_split,
        cache_dir=cache_dir,
        save_to_disk=save_to_disk,
    )

    # --- 2. Subsample semantic pilots (centroid or random, per class). ---
    pilots = select_pilots(
        data.train,
        train_label_size=train_label_size,
        method=method,
        grouping=grouping,
        seed=seed,
    )

    # --- 3. Pack real→complex and prewhiten both TX and RX pilots. ---
    # pack_complex: (D, N) real → (D/2, N) complex by splitting features.
    # prewhiten:    z = L^{-1} (x - mu), returns whitened data + Cholesky L + mean.
    input_c = pack_complex(pilots.z_tx.T, device=device)
    output_c = pack_complex(pilots.z_rx.T, device=device)
    input_w, L_in, mu_in = prewhiten(input_c, device=device)
    output_w, L_out, mu_out = prewhiten(output_c, device=device)

    # --- 4. Compute the alignment target A_target. ---
    if alignment_type == "Linear":
        # Ridge regression: A = output_w @ input_w^H (input_w @ input_w^H + lmb I)^{-1}
        A_target = ridge_regression(input_w, output_w, lmb=lmb)
        F_T, F_R_H = None, None
    elif alignment_type == "PPFE":
        if n_clusters is None:
            raise ValueError("PPFE alignment requires n_clusters.")
        # Prototype Frame Expansion: cluster (RX pilots | TX training set), build
        # frame matrices F_T and F_R_H, then A_target = F_R_H @ F_T.
        if cluster_on == "pilots":
            X, Y, feats = input_w, output_w, pilots.z_rx
        elif cluster_on == "train_tx":
            X, Y, feats = ppfe_cluster_data(data.train, L_in, mu_in, L_out, mu_out, device)
        else:
            raise ValueError(f"Unknown cluster_on: {cluster_on!r}. Use 'pilots' or 'train_tx'.")
        F_T, F_R_H = ppfe(
            X,
            Y,
            output_real=feats,
            n_clusters=n_clusters,
            n_proto=n_proto,
            seed=seed,
        )
        A_target = F_R_H @ F_T
        if not keep_csi_frames:
            F_T, F_R_H = None, None
    else:
        raise ValueError(f"Unknown alignment_type: {alignment_type!r}. Use 'Linear' or 'PPFE'.")

    return PreparedData(
        data=data,
        pilots=pilots,
        input_w=input_w,
        L_in=L_in,
        mu_in=mu_in,
        output_w=output_w,
        L_out=L_out,
        mu_out=mu_out,
        A_target=A_target,
        F_T=F_T,
        F_R_H=F_R_H,
    )

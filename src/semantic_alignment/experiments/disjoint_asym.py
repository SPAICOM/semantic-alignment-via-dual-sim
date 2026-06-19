"""Asymmetric disjoint Dual-SIM: target computation and public entry point."""

import torch

from semantic_alignment.utils.utils import get_rx_equalizer


def compute_asym_targets(
    A_target: torch.Tensor,
    H_mimo: torch.Tensor,
    ch_dim: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (target_tx, target_rx) for the asymmetric disjoint strategy.

    TX-SIM emulates A (or its truncated SVD factor when the channel is a
    bottleneck), RX-SIM emulates Q = H† (zero-forcing equalizer).

    Parameters
    ----------
    A_target : torch.Tensor
        Alignment matrix of shape ``(D_rx/2, D_tx/2)``, complex64.
    H_mimo : torch.Tensor
        MIMO channel matrix, complex64.
    ch_dim : int
        Physical channel dimension.  When smaller than ``A_target.shape[0]``
        the alignment matrix is split via truncated SVD.
    device :
        Target device for the returned tensors.

    Returns
    -------
    target_tx, target_rx : torch.Tensor
        Pre-computed complex64 target matrices for the TX and RX SIMs.
    """
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    with torch.no_grad():
        Q = get_rx_equalizer(H_torch, snr_db=None)
        if ch_dim < A_target.shape[0]:
            U_svd, s_svd, Vh_svd = torch.linalg.svd(A_torch, full_matrices=False)
            k = min(ch_dim, s_svd.shape[0])
            sqrt_s = torch.sqrt(s_svd[:k]).to(torch.complex64)
            target_tx = torch.diag(sqrt_s) @ Vh_svd[:k, :]
            target_rx = U_svd[:, :k] @ torch.diag(sqrt_s) @ Q
        else:
            target_tx = A_torch
            target_rx = Q

    return target_tx, target_rx


def run_disjoint_asym(
    L: int,
    M_int: int,
    A_target,
    H_mimo,
    test_loader,
    clf,
    L_in,
    mu_in,
    L_out,
    mu_out,
    device,
    result_subdir: str,
    strategy_name: str = "Linear",
    seed: int = 42,
    max_iters: int = 3000,
    clf_neural=None,
    no_mismatch_neural: float | None = None,
    no_mismatch_probe: float = 0.0,
    original_neural: float | None = None,
    original_probe: float = 0.0,
    tx_enc: str = "",
    rx_enc: str = "",
    optimizer: str = "adam",
    scheduler: str = "none",
    warmup_frac: float = 0.05,
    channel_dim: int | None = None,
    wavelength: float = 0.005,
    slayer: float = 0.025,
    dx: float = 0.0025,
    snr: float | None = None,
    n_clusters: int | None = None,
    lr_tx: float = 0.1,
    lr_rx: float = 0.1,
    **_,
) -> None:
    """Entry point for the asymmetric disjoint Dual-SIM experiment.

    Computes TX/RX targets via :func:`compute_asym_targets` (TX→A, RX→H†),
    then delegates to
    :func:`~semantic_alignment.experiments.disjoint_2sim.run_disjoint_2sim`.
    Accepts ``**_`` so unknown YAML metadata fields (``name``, ``sim_type``,
    ``_target_``, etc.) are silently ignored.

    Parameters
    ----------
    L : int
        Number of metasurface layers per SIM.
    M_int : int
        Side length of the square intermediate aperture.
    A_target : array-like
        Alignment matrix of shape ``(D_rx/2, D_tx/2)``.
    H_mimo : array-like
        MIMO channel matrix.
    test_loader : DataLoader
        Test set yielding ``(z_tx, labels)`` pairs.
    clf :
        Linear probe weight matrix.
    L_in, mu_in, L_out, mu_out : torch.Tensor
        Whitening statistics (Cholesky factor and mean for TX and RX).
    device : torch.device
        Inference and optimization device.
    result_subdir : str
        Subdirectory under ``results/`` where the Parquet file is written.
    strategy_name : str, optional
        Alignment strategy label (``"Linear"`` or ``"PPFE"``).
    seed : int, optional
        Global random seed.
    max_iters : int, optional
        Optimization iterations per SIM.
    lr_tx : float, optional
        TX-SIM learning rate.
    lr_rx : float, optional
        RX-SIM learning rate.
    clf_neural : optional
        Optional neural classifier.
    no_mismatch_neural, no_mismatch_probe : float or None, optional
        No-mismatch accuracy baselines.
    original_neural, original_probe : float or None, optional
        Oracle linear-alignment baselines (no SIM, no channel).
    tx_enc, rx_enc : str, optional
        Encoder names embedded in the output filename.
    optimizer : str, optional
        ``"adam"`` or ``"sgd"``.
    scheduler : str, optional
        ``"none"`` or ``"cosine"``.
    warmup_frac : float, optional
        Cosine-scheduler warm-up fraction.
    channel_dim : int or None, optional
        Physical channel dimension override; defaults to ``max(A_target.shape)``.
    wavelength : float, optional
        Operating wavelength λ (metres).
    slayer : float, optional
        Default inter-layer spacing in wavelengths.
    dx : float, optional
        Meta-atom spacing (metres).
    snr : float or None, optional
        AWGN SNR in dB; ``None`` means noiseless.
    """
    from semantic_alignment.experiments.disjoint_2sim import run_disjoint_2sim

    ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
    target_tx, target_rx = compute_asym_targets(A_target, H_mimo, ch_dim, device)
    run_disjoint_2sim(
        target_tx,
        target_rx,
        L=L,
        M_int=M_int,
        A_target=A_target,
        H_mimo=H_mimo,
        test_loader=test_loader,
        clf=clf,
        L_in=L_in,
        mu_in=mu_in,
        L_out=L_out,
        mu_out=mu_out,
        device=device,
        result_subdir=result_subdir,
        strategy_name=strategy_name,
        seed=seed,
        max_iters=max_iters,
        clf_neural=clf_neural,
        no_mismatch_neural=no_mismatch_neural,
        no_mismatch_probe=no_mismatch_probe,
        original_neural=original_neural,
        original_probe=original_probe,
        tx_enc=tx_enc,
        rx_enc=rx_enc,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_frac=warmup_frac,
        channel_dim=ch_dim,
        wavelength=wavelength,
        slayer=slayer,
        dx=dx,
        snr=snr,
        n_clusters=n_clusters,
        lr_tx=lr_tx,
        lr_rx=lr_rx,
    )


if __name__ == "__main__":
    print("run_disjoint_asym: asymmetric disjoint Dual-SIM runner (TX→A, RX→H†).")
    print("Invoke via: just run-disjoint-asym")
    print(
        "         or: uv run python scripts/classification.py --config-name config_disjoint_asym_layers"
    )

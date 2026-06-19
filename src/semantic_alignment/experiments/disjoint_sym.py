"""Target computation for the symmetric (full-CSI) disjoint Dual-SIM strategy."""

import torch


def compute_sym_targets(
    A_target: torch.Tensor,
    H_mimo: torch.Tensor,
    ch_dim: int,
    strategy_name: str,
    device,
    F_T=None,
    F_R_H=None,
    epsilon: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (target_tx, target_rx) for the symmetric CSI disjoint strategy.

    Both SIMs exploit the SVD of H via a symmetric regularized split:
    ``B_T = V diag(sqrt(sigma/(sigma^2+eps)))`` and
    ``B_R = diag(sqrt(sigma/(sigma^2+eps))) U^H``,
    so that ``B_R H B_T = diag(sigma^2/(sigma^2+eps)) -> I`` as ``eps -> 0``.

    Targets are strategy-specific:

    - **PPFE**: ``target_tx = B_T @ F_T``, ``target_rx = F_R^H @ B_R``
      (requires pre-computed PPFE frames ``F_T`` and ``F_R_H``).
    - **Linear**: ``target_tx = B_T @ Sigma_A^{1/2} V_A^H``,
      ``target_rx = U_A Sigma_A^{1/2} @ B_R`` (from SVD of A).

    Parameters
    ----------
    A_target : torch.Tensor
        Alignment matrix, complex64.
    H_mimo : torch.Tensor
        MIMO channel matrix, complex64.
    ch_dim : int
        Physical channel dimension (must equal the frame/equalizer dimension).
    strategy_name : str
        ``"PPFE"`` or ``"Linear"``.
    device :
        Target device.
    F_T : array-like or None
        PPFE TX synthesis frame (required when ``strategy_name="PPFE"``).
    F_R_H : array-like or None
        PPFE RX analysis frame (required when ``strategy_name="PPFE"``).
    epsilon : float, optional
        Tikhonov regularization: each side applies ``sqrt(sigma/(sigma^2+eps))``
        instead of ``1/sqrt(sigma)``.  Together they yield
        ``sigma^2/(sigma^2+eps)`` on the cascade diagonal.
        Defaults to ``0.0`` (pure square-root inversion).

    Returns
    -------
    target_tx, target_rx : torch.Tensor
        Pre-computed complex64 target matrices for the TX and RX SIMs.
    """
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    with torch.no_grad():
        U, s, Vh = torch.linalg.svd(H_torch, full_matrices=False)
        reg_inv_sqrt_s = torch.sqrt(s / (s**2 + epsilon)).to(H_torch.dtype)
        B_T = Vh.H @ torch.diag(reg_inv_sqrt_s)
        B_R = torch.diag(reg_inv_sqrt_s) @ U.H

        if ch_dim < B_T.shape[0]:
            B_T = B_T[:ch_dim, :ch_dim]
            B_R = B_R[:ch_dim, :ch_dim]

        match strategy_name:
            case "PPFE":
                if F_T is None or F_R_H is None:
                    raise ValueError(
                        "PPFE CSI disjoint requires F_T and F_R_H (call ppfe with "
                        "n_clusters=ch_dim and pass the frames here)."
                    )
                F_T_t = torch.as_tensor(F_T, dtype=torch.complex64, device=device)
                F_R_H_t = torch.as_tensor(F_R_H, dtype=torch.complex64, device=device)
                target_tx = (B_T @ F_T_t).detach()
                target_rx = (F_R_H_t @ B_R).detach()
            case "Linear":
                if ch_dim > min(A_torch.shape):
                    raise ValueError(
                        f"Linear CSI requires channel_dim <= {min(A_torch.shape)}, got {ch_dim}"
                    )
                U_A, s_A, Vh_A = torch.linalg.svd(A_torch, full_matrices=False)
                sqrt_sA = torch.sqrt(s_A[:ch_dim]).to(A_torch.dtype)
                A_T = torch.diag(sqrt_sA) @ Vh_A[:ch_dim, :]
                A_R = U_A[:, :ch_dim] @ torch.diag(sqrt_sA)
                target_tx = (B_T @ A_T).detach()
                target_rx = (A_R @ B_R).detach()
            case _:
                raise NotImplementedError(
                    f"Sym disjoint not implemented for strategy '{strategy_name}'."
                )

    return target_tx, target_rx


def run_disjoint_sym(
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
    strategy_name: str = "PPFE",
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
    F_T=None,
    F_R_H=None,
    lr_tx: float = 0.1,
    lr_rx: float = 0.1,
    epsilon: float = 0.0,
    **_,
) -> None:
    """Entry point for the symmetric (full-CSI) disjoint Dual-SIM experiment.

    Computes TX/RX targets via :func:`compute_sym_targets` (SVD-based channel
    inversion with symmetric regularized split), then delegates to
    :func:`~semantic_alignment.experiments.disjoint_2sim.run_disjoint_2sim`.
    Accepts ``**_`` so unknown YAML metadata fields are silently ignored.

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
        ``"PPFE"`` or ``"Linear"``.
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
        Operating wavelength lambda (metres).
    slayer : float, optional
        Default inter-layer spacing in wavelengths.
    dx : float, optional
        Meta-atom spacing (metres).
    snr : float or None, optional
        AWGN SNR in dB; ``None`` means noiseless.
    F_T : array-like or None, optional
        PPFE TX synthesis frame (required when ``strategy_name="PPFE"``).
    F_R_H : array-like or None, optional
        PPFE RX analysis frame (required when ``strategy_name="PPFE"``).
    epsilon : float, optional
        Tikhonov regularization for the symmetric channel split:
        ``sqrt(sigma/(sigma^2+eps))`` on each side.  Defaults to ``0.0``.
    """
    from semantic_alignment.experiments.disjoint_2sim import run_disjoint_2sim

    ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
    target_tx, target_rx = compute_sym_targets(
        A_target, H_mimo, ch_dim, strategy_name, device, F_T=F_T, F_R_H=F_R_H, epsilon=epsilon
    )
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
        lr_tx=lr_tx,
        lr_rx=lr_rx,
    )


if __name__ == "__main__":
    print("run_disjoint_sym: symmetric (full-CSI) disjoint Dual-SIM runner.")
    print("Invoke via: just run-disjoint-sym")
    print(
        "         or: uv run python scripts/classification.py --config-name config_disjoint_sym_layers"
    )

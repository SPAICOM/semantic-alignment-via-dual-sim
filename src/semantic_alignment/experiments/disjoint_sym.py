"""Target computation for the symmetric (full-CSI) disjoint Dual-SIM strategy."""

import torch

from semantic_alignment.experiments._utils import tipology_tag


def compute_sym_targets(
    A_target: torch.Tensor,
    H_mimo: torch.Tensor,
    ch_dim: int,
    strategy_name: str,
    device,
    F_T=None,
    F_R_H=None,
    epsilon: float = 0.0,
    constructive: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Compute the 6 matrices for the symmetric disjoint strategy.

    Returns ``(target_tx, tx_left, tx_right, target_rx, rx_left, rx_right)``.
    The SIM loss for each side is ``‖β (L G R) − T‖²``; ``None`` means identity.

    ``constructive=False`` (default):
        Channel inversion is baked into the targets:
        TX: ``T = B_T T_T``, RX: ``T = T_R B_R`` where ``B_T/B_R`` are the
        regularized SVD-split factors of H.

    ``constructive=True``:
        Alignment-only targets; H's SVD factors sit in the loss matrices:
        TX: ``T = T_T``, ``L = Σ¹ᐟ² Vᴴ``;
        RX: ``T = T_R``, ``R = U Σ¹ᐟ²``.
        The SIM learns a better-conditioned target (no Σ¹ in the target itself).
    """
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    with torch.no_grad():
        U, s, Vh = torch.linalg.svd(H_torch, full_matrices=False)

        if strategy_name == "PPFE":
            if F_T is None or F_R_H is None:
                raise ValueError("PPFE requires F_T and F_R_H (n_clusters=ch_dim).")
            T_T = torch.as_tensor(F_T, dtype=torch.complex64, device=device)
            T_R = torch.as_tensor(F_R_H, dtype=torch.complex64, device=device)
        elif strategy_name == "Linear":
            if ch_dim > min(A_torch.shape):
                raise ValueError(
                    f"Linear sym requires channel_dim <= {min(A_torch.shape)}, got {ch_dim}"
                )
            U_A, s_A, Vh_A = torch.linalg.svd(A_torch, full_matrices=False)
            sqrt_sA = torch.sqrt(s_A[:ch_dim]).to(A_torch.dtype)
            T_T = torch.diag(sqrt_sA) @ Vh_A[:ch_dim, :]
            T_R = U_A[:, :ch_dim] @ torch.diag(sqrt_sA)
        else:
            raise NotImplementedError(f"Sym disjoint not implemented for '{strategy_name}'.")

        reg_inv_sqrt_s = torch.sqrt(s / (s**2 + epsilon)).to(H_torch.dtype)

        if not constructive:
            B_T = Vh.H @ torch.diag(reg_inv_sqrt_s)
            B_R = torch.diag(reg_inv_sqrt_s) @ U.H
            if ch_dim < B_T.shape[0]:
                B_T = B_T[:ch_dim, :ch_dim]
                B_R = B_R[:ch_dim, :ch_dim]
            return (B_T @ T_T).detach(), None, None, (T_R @ B_R).detach(), None, None

        # constructive=True: regularised Σ^{1/2} factors in the loss.
        # reg_sqrt_s = sqrt(σ²/(σ+ε)) = σ/sqrt(σ+ε)  →  σ^{1/2} for large σ, 0 for σ→0.
        # Ideal cascade: G_R H G_T = T_R Σ^{-1/2} Σ Σ^{-1/2} T_T = T_R T_T = A  ✓
        reg_sqrt_s = torch.sqrt(s**2 / (s + epsilon)).to(H_torch.dtype)
        tx_left = torch.diag(reg_sqrt_s) @ Vh  # Σ^{1/2} Vᴴ  (regularised)
        rx_right = U @ torch.diag(reg_sqrt_s)  # U Σ^{1/2}   (regularised)
        return T_T.detach(), tx_left, None, T_R.detach(), None, rx_right


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
    constructive: bool = False,
    dataset: str = "cifar10",
    **_,
) -> None:
    """Entry point for the symmetric (full-CSI) disjoint Dual-SIM experiment.

    See :func:`compute_sym_targets` for the effect of ``constructive``.
    """
    from semantic_alignment.experiments.disjoint_2sim import run_disjoint_2sim

    ch_dim = channel_dim if channel_dim is not None else int(max(A_target.shape))
    target_tx, tx_left, tx_right, target_rx, rx_left, rx_right = compute_sym_targets(
        A_target,
        H_mimo,
        ch_dim,
        strategy_name,
        device,
        F_T=F_T,
        F_R_H=F_R_H,
        epsilon=epsilon,
        constructive=constructive,
    )
    run_disjoint_2sim(
        target_tx,
        target_rx,
        tx_left=tx_left,
        tx_right=tx_right,
        rx_left=rx_left,
        rx_right=rx_right,
        tipology_subdir=tipology_tag("disjoint_sym", constructive),
        constructive=constructive,
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
        dataset=dataset,
    )


if __name__ == "__main__":
    print("run_disjoint_sym: symmetric (full-CSI) disjoint Dual-SIM runner.")
    print("Invoke via: just run-disjoint-sym")
    print(
        "         or: uv run python scripts/classification.py --config-name config_disjoint_sym_layers"
    )

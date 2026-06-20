"""Asymmetric disjoint Dual-SIM: target computation and public entry point."""

import torch

from semantic_alignment.experiments._utils import tipology_tag
from semantic_alignment.utils.utils import get_rx_equalizer


def compute_asym_targets(
    A_target: torch.Tensor,
    H_mimo: torch.Tensor,
    ch_dim: int,
    device,
    epsilon: float = 0.0,
    constructive: bool = False,
    alignment_side: str = "tx",
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Compute the 6 matrices for the asymmetric disjoint strategy.

    Returns ``(target_tx, tx_left, tx_right, target_rx, rx_left, rx_right)``.
    The SIM loss for each side is ``‖β (L G R) − T‖²``; ``None`` means identity.

    ``alignment_side`` chooses which SIM carries the alignment matrix A:

    - ``"tx"`` — alignment on TX, channel equalization on RX
    - ``"rx"`` — alignment on RX, channel equalization on TX

    ``constructive`` determines how the channel enters:

    - ``False`` — H inverted analytically and baked into the target (H†)
    - ``True``  — H placed as left/right matrix in the loss (gradient flows through H)

    Supported combinations (rx equalized is not implemented):

    +----------------+------------------+----------------------------------------------+
    | alignment_side | constructive     | TX loss / RX loss                            |
    +================+==================+==============================================+
    | tx             | False (standard) | ‖β G_T − A‖²      / ‖β G_R − H†‖²          |
    +----------------+------------------+----------------------------------------------+
    | tx             | True             | ‖β G_T − A‖²      / ‖β G_R H − I‖²         |
    +----------------+------------------+----------------------------------------------+
    | rx             | True             | ‖β H G_T − I‖²    / ‖β G_R − A‖²           |
    +----------------+------------------+----------------------------------------------+
    """
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
    H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)

    with torch.no_grad():
        I_ch = torch.eye(ch_dim, dtype=torch.complex64, device=device)

        if not constructive:
            if alignment_side == "rx":
                raise NotImplementedError(
                    "Disjoint Asym RX equalized (non-constructive) is not supported. "
                    "Use alignment_side='rx' with constructive=True instead."
                )
            # TX: ‖β G_T − A‖²,  RX: ‖β G_R − H†‖²
            Q = get_rx_equalizer(H_torch, snr_db=None, epsilon=epsilon)
            return A_torch, None, None, Q, None, None

        # constructive=True: H lives in the loss, target is I on the equalization side
        if alignment_side == "tx":
            # TX: ‖β G_T − A‖²,    RX: ‖β G_R H − I‖²
            return A_torch, None, None, I_ch, None, H_torch
        else:
            # TX: ‖β H G_T − I‖²,  RX: ‖β G_R − A‖²
            return I_ch, H_torch, None, A_torch, None, None


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
    epsilon: float = 0.0,
    constructive: bool = False,
    alignment_side: str = "tx",
    **_,
) -> None:
    """Entry point for the asymmetric disjoint Dual-SIM experiment.

    See :func:`compute_asym_targets` for the effect of ``constructive`` and
    ``alignment_side``.
    """
    from semantic_alignment.experiments.disjoint_2sim import run_disjoint_2sim

    match alignment_side:
        case "rx":
            ch_dim = int(A_target.shape[1])  # tx latent dim — G_R must approximate A

        case "tx":
            ch_dim = int(A_target.shape[0])  # rx latent dim — G_T must approximate A

        case _:
            raise ValueError(f"Unknown alignment_side '{alignment_side}'. Expected 'tx' or 'rx'.")

    target_tx, tx_left, tx_right, target_rx, rx_left, rx_right = compute_asym_targets(
        A_target,
        H_mimo,
        ch_dim,
        device,
        epsilon=epsilon,
        constructive=constructive,
        alignment_side=alignment_side,
    )
    run_disjoint_2sim(
        target_tx,
        target_rx,
        tx_left=tx_left,
        tx_right=tx_right,
        rx_left=rx_left,
        rx_right=rx_right,
        tipology_subdir=tipology_tag("disjoint_asym", constructive, alignment_side),
        constructive=constructive,
        alignment_side=alignment_side,
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
    print("run_disjoint_asym: asymmetric disjoint Dual-SIM runner.")
    print("Invoke via: just run-disjoint-asym")
    print(
        "         or: uv run python scripts/classification.py --config-name config_disjoint_asym_layers"
    )

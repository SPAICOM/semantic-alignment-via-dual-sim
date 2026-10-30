"""Publication-ready reconstruction figure: decoded test images per architecture and strategy.

For each dataset, one figure of 2×5 image panels (one test image per class):

                     Original       No mismatch
    Linear    Mono   Disjoint Sym   Joint
    PPFE      Mono   Disjoint Sym   Joint

The SIM panels are decoded through the trained links saved next to the final
runs (``.pt`` files: ``G_T``, ``H``, ``G_R``, ``β``, or ``Z_eff = H·G_T`` for
constructive Mono-SIM) — nothing is retrained.
The embedding / whitening pipeline is rebuilt exactly as in ``reconstruction.py``
(same config, same seed). Decoded images are cached in the plot directory;
delete ``ReconstructionGrid_images.pt`` to recompute them.

Plot conventions follow ``plot_results_classification.py`` (mplstyle, LaTeX,
PDF + PNG).

Usage
-----
    just plot-rec
    uv run python scripts/plot_results_reconstruction.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from matplotlib.gridspec import GridSpec
from plot_results_classification import _apply_style
from reconstruction import _tags

from semantic_alignment.data.reconstruction import prepare_reconstruction
from semantic_alignment.evaluation.reconstruction import evaluate_reconstruction
from semantic_alignment.models.autoencoder import load_autoencoder
from semantic_alignment.sim.link import cascade_transmit, dual_transmit
from semantic_alignment.sim.original._utils import RECONSTRUCTION_RESULTS_DIR, load_params

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
RESULTS = RECONSTRUCTION_RESULTS_DIR

DATASETS = ["mnist", "fashion_mnist", "cifar10"]
STRATEGIES = ["Linear", "PPFE"]
# Architecture → result directory of its final run (M64, L50, cosine).
ARCHS = {
    "Mono": "mono/constructive/tx",
    "Disjoint Sym": "disjoint_sym/constructive",
    "Joint": "joint",
}
RUN_TAG = "L50_M64x64"
GRID_ROWS, GRID_COLS = 2, 5
# Per dataset, {class: k} shows the k-th test image of that class (0-based) instead of the first.
SAMPLE_OVERRIDES = {"mnist": {3: 1, 5: 2, 6: 1, 9: 1}}


def _cfg(dataset: str):
    """Hydra config of the final runs for ``dataset`` (AE pair, pilots, seed)."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="config_reconstruction_joint",
            overrides=[f"datasets/reconstruction={dataset}"],
        )


def _params_file(arch_dir: str, reg_tag: str, dataset: str, strategy: str, seed: int) -> Path:
    """The single ``.pt`` of the final run for (architecture, dataset, strategy)."""
    pattern = f"{reg_tag}_{dataset}_{strategy}_seed{seed}_{RUN_TAG}_*.pt"
    matches = sorted((RESULTS / arch_dir).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {arch_dir}, found {len(matches)}")
    return matches[0]


def _compute_images(dataset: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Decode one test image per class for the references and every (strategy, arch) link."""
    cfg = _cfg(dataset)
    rc = cfg.reconstruction
    seed = cfg.training.seed
    _, _, reg_tag = _tags(cfg)
    tx_model = load_autoencoder(dataset, rc.tx_variant, rc.tx_seed, device)
    rx_model = load_autoencoder(dataset, rc.rx_variant, rc.rx_seed, device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    prep = prepare_reconstruction(
        tx_model,
        rx_model,
        cfg.datasets.hf_repo,
        cfg.datasets.hf_config,
        rc.train_label_size,
        image_col=cfg.datasets.image_col,
        label_col=cfg.datasets.label_col,
        image_size=tuple(cfg.datasets.image_size),
        channels=cfg.datasets.channels,
        method=rc.method,
        grouping=rc.grouping,
        train_split=cfg.datasets.train_split,
        test_split=cfg.datasets.test_split,
        save_to_disk=rc.save_to_disk,
        seed=seed,
        device=device,
    )
    test = prep.data.test
    # First test image of each class (unless overridden), in class order.
    overrides = SAMPLE_OVERRIDES.get(dataset, {})
    idx = torch.stack(
        [
            torch.nonzero(test.labels == c)[overrides.get(c, 0), 0]
            for c in range(GRID_ROWS * GRID_COLS)
        ]
    )
    z_tx, images = test.z_tx[idx], prep.images_test[idx]

    with torch.no_grad():
        out = {"original": images, "no_mismatch": rx_model.decode(test.z_rx[idx].to(device)).cpu()}
    for strategy in STRATEGIES:
        for arch, arch_dir in ARCHS.items():
            p = load_params(_params_file(arch_dir, reg_tag, dataset, strategy, seed), device)
            if "Z_eff" in p:  # constructive Mono-SIM: Z_eff = H·G_T already includes the channel
                transmit = cascade_transmit(p["Z_eff"], p["beta"], None)
            else:
                H = p["H"].to(torch.complex64)
                transmit = dual_transmit(p["G_T"], H, p["G_R"], p["beta"], None)
            _, _, recon = evaluate_reconstruction(
                transmit,
                z_tx,
                images,
                prep.L_in,
                prep.mu_in,
                prep.L_out,
                prep.mu_out,
                rx_model,
                device,
                num_preview=len(idx),
            )
            out[f"{arch}/{strategy}"] = recon.cpu()
    out["overrides"] = overrides
    return out


def _mosaic(images: torch.Tensor, pad: int = 1) -> np.ndarray:
    """Tile ``(10, C, H, W)`` images into a ``GRID_ROWS × GRID_COLS`` array with white gaps."""
    n, c, h, w = images.shape
    canvas = np.ones(
        (GRID_ROWS * h + (GRID_ROWS - 1) * pad, GRID_COLS * w + (GRID_COLS - 1) * pad, c)
    )
    arr = images.clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    for i in range(n):
        r, col = divmod(i, GRID_COLS)
        canvas[r * (h + pad) : r * (h + pad) + h, col * (w + pad) : col * (w + pad) + w] = arr[i]
    return canvas[..., 0] if c == 1 else canvas


def _panel(ax, images: torch.Tensor, title: str) -> None:
    """One 2×5 image panel with its title; no ticks or grid."""
    ax.imshow(_mosaic(images), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def reconstruction_grid(images: dict[str, torch.Tensor], out_dir: Path) -> None:
    """References centred on the top row; (Linear, PPFE) rows × ``ARCHS`` columns below."""
    fig = plt.gcf()
    n = len(ARCHS)
    # The mplstyle's 16x8 would leave large gaps around the wide 2x5 panels.
    fig.set_size_inches(5.5 * n, 8)
    # Each panel spans two grid columns so the two references can sit centred above
    # the n architecture columns; a narrow spacer row separates them.
    gs = GridSpec(4, 2 * n, figure=fig, height_ratios=[1, 0.1, 1, 1], wspace=0.1, hspace=0.2)
    _panel(fig.add_subplot(gs[0, n - 2 : n]), images["original"], "Original")
    _panel(fig.add_subplot(gs[0, n : n + 2]), images["no_mismatch"], "No mismatch")
    for row, strategy in enumerate(STRATEGIES):
        for col, arch in enumerate(ARCHS):
            ax = fig.add_subplot(gs[2 + row, 2 * col : 2 * col + 2])
            # Architecture names head the columns once, above the first strategy row.
            _panel(ax, images[f"{arch}/{strategy}"], arch if row == 0 else "")
            if col == 0:
                ax.set_ylabel(strategy)
    _save("ReconstructionGrid", out_dir)


def _save(name: str, out_dir: Path) -> None:
    """Save the current figure as both PDF and PNG under ``out_dir``, then close it."""
    plt.savefig(str(out_dir / f"{name}.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(str(out_dir / f"{name}.png"), bbox_inches="tight")
    plt.close()
    print(f" {name}.pdf / .png")


def main() -> None:
    """Build the reconstruction figure for every dataset in ``DATASETS``."""
    _apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in DATASETS:
        out_dir = RESULTS / "plots" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n── dataset: {ds} → {out_dir} ──")
        cache = out_dir / "ReconstructionGrid_images.pt"
        images = torch.load(cache, weights_only=True) if cache.exists() else {}
        if not all(f"{a}/{s}" in images for a in ARCHS for s in STRATEGIES) or (
            images.get("overrides") != SAMPLE_OVERRIDES.get(ds, {})
        ):
            images = _compute_images(ds, device)
            torch.save(images, cache)
        reconstruction_grid(images, out_dir)


if __name__ == "__main__":
    main()

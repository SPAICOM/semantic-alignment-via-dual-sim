"""Sweep alignment hyperparameters (Linear lmb; PPFE n_clusters/n_proto) fast.

Embeds the dataset once with a fixed TX/RX autoencoder pair, then evaluates
many (strategy, hyperparameter) combinations against the "original" (no-SIM,
no-channel) baseline — this skips the expensive SIM training entirely, so a
combination that takes minutes via ``reconstruction.py`` takes seconds here.
Use this to narrow down good settings, then confirm the winners through the
full physical channel with ``reconstruction.py``.

Preview grids and the sweep table (Parquet) are written to
``results/reconstruction/alignment_sweep/``. PPFE is swept over where KMeans
runs (``cluster_on_values``: ``train_tx`` = the whole TX training latent space,
``pilots`` = the RX pilots), ``n_clusters``, ``n_proto`` and ``ppfe_seeds``.

Examples
--------
    just sweep-align
    uv run python scripts/sweep_alignment.py "n_clusters_values=[500,2000]" reconstruction.rx_variant=small_mmd1
"""

import hydra
import polars as pl
import torch
from omegaconf import DictConfig

from semantic_alignment.data.reconstruction import (
    compute_alignment,
    ppfe_cluster_data,
    prepare_reconstruction,
    whiten_pilots,
)
from semantic_alignment.data.semasia import select_pilots
from semantic_alignment.evaluation.reconstruction import (
    run_original_reconstruction,
    save_comparison_grid,
)
from semantic_alignment.models.autoencoder import load_autoencoder
from semantic_alignment.sim.original._utils import RECONSTRUCTION_RESULTS_DIR

_OUT_DIR = RECONSTRUCTION_RESULTS_DIR / "alignment_sweep"


@hydra.main(version_base=None, config_path="../config", config_name="config_sweep_alignment")
def main(cfg: DictConfig) -> None:
    """Embed once, then fast-sweep Linear/PPFE hyperparameters against the no-channel baseline.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration composed from ``config_sweep_alignment``.
    """
    rc = cfg.reconstruction
    dataset = cfg.datasets.name
    seed = cfg.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f" dataset={dataset} | tx={rc.tx_variant}(seed{rc.tx_seed}) | "
        f"rx={rc.rx_variant}(seed{rc.rx_seed}) | pilots/class={rc.train_label_size}"
    )
    tx_model = load_autoencoder(dataset, rc.tx_variant, rc.tx_seed, device)
    rx_model = load_autoencoder(dataset, rc.rx_variant, rc.rx_seed, device)

    print(" Loading + embedding train/test splits (once) ...")
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
    z_tx_test, images_test = prep.data.test.z_tx, prep.images_test
    n = rc.num_preview

    def _original(A, white):
        """No-channel MSE/PSNR/preview for one alignment matrix and whitening set."""
        _, L_in, mu_in, _, L_out, mu_out = white
        return run_original_reconstruction(
            z_tx_test,
            images_test,
            A,
            L_in,
            mu_in,
            L_out,
            mu_out,
            rx_model,
            device,
            batch_size=rc.batch_size,
            num_preview=n,
        )

    white = (prep.input_w, prep.L_in, prep.mu_in, prep.output_w, prep.L_out, prep.mu_out)
    align_kw = dict(output_real=prep.pilots.z_rx, seed=seed)
    # PPFE over the whole training latent space, clustered on the TX side.
    train_tx = (
        ppfe_cluster_data(prep.data.train, prep.L_in, prep.mu_in, prep.L_out, prep.mu_out, device)
        if "train_tx" in cfg.cluster_on_values
        else None
    )
    results = []
    tx_enc, rx_enc = f"{rc.tx_variant}seed{rc.tx_seed}", f"{rc.rx_variant}seed{rc.rx_seed}"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = (
        _OUT_DIR / f"{dataset}_{tx_enc}_{rx_enc}_p{rc.train_label_size}_alignment_sweep.parquet"
    )

    def _save() -> None:
        """Rewrite the sweep table — after every combination, so a long sweep keeps partial results."""
        pl.DataFrame(results, infer_schema_length=None).with_columns(
            pl.lit(dataset).alias("dataset"),
            pl.lit(tx_enc).alias("tx_enc"),
            pl.lit(rx_enc).alias("rx_enc"),
            pl.lit(rc.train_label_size).alias("pilots_per_class"),
        ).write_parquet(table_path)

    print("\n--- Linear (ridge regression) ---")
    for lmb in cfg.lmb_values:
        A, _, _ = compute_alignment(
            prep.input_w, prep.output_w, strategy="Linear", lmb=lmb, **align_kw
        )
        mse, psnr_db, _ = _original(A, white)
        print(f" lmb={lmb:<8} : original_psnr={psnr_db:.2f} dB (MSE={mse:.4f})")
        results.append({"strategy": "Linear", "lmb": lmb, "psnr": psnr_db})
        _save()

    print("\n--- PPFE (zero-shot, prototype-based) ---")
    for cluster_on in cfg.cluster_on_values:
        for n_clusters in cfg.n_clusters_values:
            for n_proto in cfg.n_proto_values:
                for ppfe_seed in cfg.ppfe_seeds:
                    A, _, _ = compute_alignment(
                        prep.input_w,
                        prep.output_w,
                        prep.pilots.z_rx,
                        "PPFE",
                        n_clusters=n_clusters,
                        n_proto=n_proto,
                        seed=ppfe_seed,
                        cluster_data=train_tx if cluster_on == "train_tx" else None,
                    )
                    mse, psnr_db, _ = _original(A, white)
                    print(
                        f" {cluster_on:<8} n_clusters={n_clusters:<4} n_proto={n_proto!s:<5} "
                        f"seed={ppfe_seed:<4}: original_psnr={psnr_db:.2f} dB (MSE={mse:.4f})"
                    )
                    results.append(
                        {
                            "strategy": "PPFE",
                            "cluster_on": cluster_on,
                            "n_clusters": n_clusters,
                            "n_proto": n_proto,
                            "seed": ppfe_seed,
                            "mse": mse,
                            "psnr": psnr_db,
                        }
                    )
                    _save()

    print(f" Saved sweep table: {table_path}")

    best_linear = max((r for r in results if r["strategy"] == "Linear"), key=lambda r: r["psnr"])
    best_ppfe = max((r for r in results if r["strategy"] == "PPFE"), key=lambda r: r["psnr"])
    print("\n=== BEST ===")
    print(f" Linear: lmb={best_linear['lmb']} -> {best_linear['psnr']:.2f} dB")
    print(
        f" PPFE  : {best_ppfe['cluster_on']} n_clusters={best_ppfe['n_clusters']}"
        f" n_proto={best_ppfe['n_proto']}"
        f" -> {best_ppfe['psnr']:.2f} dB"
    )

    print(f"\n--- PPFE_align preview grid (n_clusters={list(cfg.preview_clusters)}) ---")
    columns = {"original": images_test[:n]}
    for n_clusters in cfg.preview_clusters:
        A, _, _ = compute_alignment(
            prep.input_w,
            prep.output_w,
            strategy="PPFE",
            n_clusters=n_clusters,
            n_proto=cfg.preview_n_proto,
            cluster_data=train_tx if cfg.cluster_on_values[0] == "train_tx" else None,
            **align_kw,
        )
        columns[f"PPFE_k{n_clusters}"] = _original(A, white)[2].cpu()
    out_path = save_comparison_grid(
        columns, _OUT_DIR / f"{dataset}_ppfe_cluster_sweep_p{rc.train_label_size}.png"
    )
    print(f" Saved preview grid: {out_path}")

    print(f"\n--- Linear: pilot-count sweep ({list(cfg.preview_pilots)}) ---")
    pilot_columns = {"original": images_test[:n]}
    for ps in cfg.preview_pilots:
        pilots_i = select_pilots(
            prep.data.train,
            train_label_size=ps,
            method=rc.method,
            grouping=rc.grouping,
            seed=seed,
        )
        white_i = whiten_pilots(pilots_i, device)
        A_i, _, _ = compute_alignment(
            white_i[0], white_i[3], pilots_i.z_rx, "Linear", lmb=0.0, seed=seed
        )
        mse_i, psnr_i, prev_i = _original(A_i, white_i)
        print(f" pilot_size={ps:<5}: original_psnr={psnr_i:.2f} dB (MSE={mse_i:.4f})")
        pilot_columns[f"Linear_p{ps}"] = prev_i.cpu()
    out_path2 = save_comparison_grid(
        pilot_columns, _OUT_DIR / f"{dataset}_linear_pilot_sweep_p{rc.train_label_size}.png"
    )
    print(f" Saved preview grid: {out_path2}")


if __name__ == "__main__":
    main()

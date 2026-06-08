"""Extract per-seed Mono-SIM results from detailed Parquet files."""

from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"
SEEDS = [27, 42, 123]
STRATEGIES = ["PPFE", "Linear"]


def extract_layers_monosim():
    """
    Read the detailed layers Parquet, aggregate by (Seed, M_int, L),
    and write per-seed Mono-SIM Parquets.
    """
    for strategy in STRATEGIES:
        src = RESULTS / f"results_layers_{strategy}.parquet"
        if not src.exists():
            print(f"Skipping: {src.name} not found.")
            continue

        df_all = pl.read_parquet(src)
        valid_atoms = [16, 32, 64]

        for seed in SEEDS:
            df = df_all.filter((pl.col("Seed") == seed) & (pl.col("Alignment Type") == strategy))
            if df.is_empty():
                continue

            agg = (
                df.group_by(["SIM Meta Atoms Intermediate X", "SIM Layers"])
                .agg(pl.col("Accuracy SIM Mimo").mean())
                .filter(pl.col("SIM Meta Atoms Intermediate X").is_in(valid_atoms))
                .with_columns(
                    pl.concat_str(
                        pl.col("SIM Meta Atoms Intermediate X").cast(pl.Int64).cast(pl.Utf8),
                        pl.lit("x"),
                        pl.col("SIM Meta Atoms Intermediate X").cast(pl.Int64).cast(pl.Utf8),
                    ).alias("M_int"),
                    pl.col("SIM Layers").alias("L"),
                    (
                        pl.when(pl.col("Accuracy SIM Mimo") <= 1.0)
                        .then(pl.col("Accuracy SIM Mimo") * 100)
                        .otherwise(pl.col("Accuracy SIM Mimo"))
                    ).alias("accuracy"),
                )
                .select(["M_int", "L", "accuracy"])
                .sort(["M_int", "L"])
            )

            out = RESULTS / f"results_layers_monosim_{strategy}_seed{seed}.parquet"
            agg.write_parquet(out)
            print(f" {out.name}")


def extract_snr_monosim():
    """
    Read the detailed SNR Parquet, aggregate by (Seed, M_int, SNR),
    and write per-seed Mono-SIM SNR Parquets.
    """
    for strategy in STRATEGIES:
        src = RESULTS / f"results_snr_{strategy}.parquet"
        if not src.exists():
            print(f"Skipping: {src.name} not found.")
            continue

        df_all = pl.read_parquet(src)
        valid_atoms = [16, 32, 64]

        for seed in SEEDS:
            df = df_all.filter(
                (pl.col("Seed") == seed)
                & (pl.col("Alignment Type") == strategy)
                & (pl.col("SIM Layers") == 10)
            )
            if df.is_empty():
                continue

            agg = (
                df.group_by(["SIM Meta Atoms Intermediate X", "SNR [dB]"])
                .agg(pl.col("Accuracy SIM Mimo").mean())
                .filter(pl.col("SIM Meta Atoms Intermediate X").is_in(valid_atoms))
                .with_columns(
                    pl.concat_str(
                        pl.col("SIM Meta Atoms Intermediate X").cast(pl.Int64).cast(pl.Utf8),
                        pl.lit("x"),
                        pl.col("SIM Meta Atoms Intermediate X").cast(pl.Int64).cast(pl.Utf8),
                    ).alias("M_int"),
                    pl.col("SNR [dB]").alias("snr"),
                    (
                        pl.when(pl.col("Accuracy SIM Mimo") <= 1.0)
                        .then(pl.col("Accuracy SIM Mimo") * 100)
                        .otherwise(pl.col("Accuracy SIM Mimo"))
                    ).alias("accuracy"),
                )
                .select(["M_int", "snr", "accuracy"])
                .sort(["M_int", "snr"])
            )

            out = RESULTS / f"results_snr_monosim_{strategy}_seed{seed}.parquet"
            agg.write_parquet(out)
            print(f" {out.name}")


if __name__ == "__main__":
    extract_layers_monosim()
    extract_snr_monosim()

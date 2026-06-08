"""Publication-ready result plots.

Follows the plotting conventions of the semantic-alignment-via-sim reference repo:
  - ``sns.set_style('whitegrid')`` first, then ``plt.style.use(mplstyle)``
  - No explicit ``plt.figure()`` calls — figure size comes from the mplstyle
  - Save both PDF and PNG
  - ``plt.clf(); plt.cla()`` after every plot
  - LaTeX strings for axis labels (requires a working TeX installation)

Usage
-----
    just plot
    uv run python scripts/plot_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"
IMG_PATH = RESULTS / "plots"
IMG_PATH.mkdir(parents=True, exist_ok=True)

_STYLE = BASE_DIR / "config" / "plotting" / "plt.mplstyle"

STRATEGIES = ["Linear", "PPFE"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_style() -> None:
    sns.set_style("whitegrid")
    if _STYLE.exists():
        plt.style.use(str(_STYLE))


def _load_parquet_glob(pattern: str) -> pl.DataFrame | None:
    files = list(RESULTS.glob(pattern))
    if not files:
        return None
    return pl.read_parquet(files)


def _save(name: str) -> None:
    from matplotlib.legend import Legend

    extra = plt.gcf().findobj(Legend)
    plt.savefig(
        str(IMG_PATH / f"{name}.pdf"), format="pdf", bbox_inches="tight", bbox_extra_artists=extra
    )
    plt.savefig(str(IMG_PATH / f"{name}.png"), bbox_inches="tight", bbox_extra_artists=extra)
    plt.clf()
    plt.cla()
    print(f" {name}.pdf / .png")


def _add_baseline_hlines(df: pl.DataFrame) -> list[str]:
    """Horizontal lines for No Mismatch (solid) and Original per strategy (dashed).
    Returns the list of labels added, for use in a legend."""
    added = []

    nm = df["no_mismatch_neural"].drop_nulls().mean()
    if nm is not None:
        plt.axhline(y=nm / 100, color="black", linestyle="-", linewidth=2.5, label="No Mismatch")
        added.append("No Mismatch")

    for strat, linestyle, label in [
        ("Linear", "-.", "Original Linear"),
        ("PPFE", ":", "Original PPFE"),
    ]:
        vals = df.filter(pl.col("strategy") == strat)["original_neural"].drop_nulls()
        if vals.len() > 0:
            plt.axhline(
                y=vals.mean() / 100, color="gray", linestyle=linestyle, linewidth=2.5, label=label
            )
            added.append(label)

    return added


# ---------------------------------------------------------------------------
# Accuracy vs SIM Layers
# ---------------------------------------------------------------------------


def accuracy_vs_simlayers() -> None:
    """Accuracy vs $L$ — Joint architecture, both alignment strategies."""
    df = _load_parquet_glob("dual_layers/*.parquet")
    if df is None:
        print("No layers results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
        .with_columns(
            (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
            pl.col("M_int").cast(pl.Utf8).alias("Intermediate Atoms"),
        )
        .sort(["Intermediate Atoms", "strategy"])
    )

    ticks = df["L"].unique().sort().to_list()

    ax = sns.lineplot(
        df.to_pandas(),
        x="L",
        y="accuracy",
        hue="Intermediate Atoms",
        style="strategy",
        markers=True,
        dashes=True,
    )
    baseline_labels = _add_baseline_hlines(df)

    handles, labels = ax.get_legend_handles_labels()

    atoms_labels = sorted(df["Intermediate Atoms"].unique().to_list())
    strat_labels = sorted(df["strategy"].unique().to_list())

    def _make_legend(lbl_list, title, **kwargs):
        hdls = [handles[labels.index(lbl)] for lbl in lbl_list if lbl in labels]
        return ax.legend(hdls, lbl_list, title=title, frameon=True, framealpha=1, **kwargs)

    leg1 = _make_legend(
        strat_labels, "Alignment Type", ncol=2, loc="upper center", bbox_to_anchor=(0.37, 0.18)
    )
    leg2 = _make_legend(
        baseline_labels,
        "",
        ncol=len(baseline_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    leg3 = _make_legend(
        atoms_labels, "Intermediate Atoms", ncol=3, loc="upper center", bbox_to_anchor=(0.76, 0.18)
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    ax.add_artist(leg3)

    plt.xlabel(r"SIM Layers $L$")
    plt.ylabel("Accuracy")
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    _save("AccuracyVsSIMLayers")


# ---------------------------------------------------------------------------
# Accuracy vs SNR
# ---------------------------------------------------------------------------


def accuracy_vs_snr() -> None:
    """Accuracy vs SNR [dB] — Joint architecture."""
    df = _load_parquet_glob("dual_snr/*.parquet")
    if df is None:
        print("No SNR results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES) & pl.col("snr").is_not_null())
        .with_columns(
            (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
            pl.col("M_int").cast(pl.Utf8).alias("Intermediate Atoms"),
        )
        .sort(["Intermediate Atoms", "strategy", "snr"])
    )

    ticks = df["snr"].unique().sort().to_list()

    ax = sns.lineplot(
        df.to_pandas(),
        x="snr",
        y="accuracy",
        hue="Intermediate Atoms",
        style="strategy",
        markers=True,
        dashes=True,
    )
    baseline_labels = _add_baseline_hlines(df)

    handles, labels = ax.get_legend_handles_labels()
    atoms_labels = sorted(df["Intermediate Atoms"].unique().to_list())
    strat_labels = sorted(df["strategy"].unique().to_list())

    def _make_legend(lbl_list, title, **kwargs):
        hdls = [handles[labels.index(lbl)] for lbl in lbl_list if lbl in labels]
        return ax.legend(hdls, lbl_list, title=title, frameon=True, framealpha=1, **kwargs)

    leg1 = _make_legend(
        strat_labels, "Alignment Type", ncol=2, loc="upper center", bbox_to_anchor=(0.16, 0.88)
    )
    leg2 = _make_legend(
        baseline_labels,
        "",
        ncol=len(baseline_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    leg3 = _make_legend(
        atoms_labels, "Intermediate Atoms", ncol=3, loc="upper center", bbox_to_anchor=(0.76, 0.18)
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    ax.add_artist(leg3)

    plt.xlabel(r"Signal-to-Noise Ratio [dB]")
    plt.ylabel("Accuracy")
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    _save("AccuracyVsSNR")


# ---------------------------------------------------------------------------
# Accuracy vs Disjoint Layers
# ---------------------------------------------------------------------------


def accuracy_vs_disjoint_layers() -> None:
    """Accuracy vs $L$ — Disjoint architecture."""
    df = _load_parquet_glob("disjoint_layers/*.parquet")
    if df is None:
        print("No disjoint layers results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
        .with_columns(
            (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
            pl.col("M_int").cast(pl.Utf8).alias("Intermediate Atoms"),
        )
        .sort(["Intermediate Atoms", "strategy"])
    )

    ticks = df["L"].unique().sort().to_list()

    ax = sns.lineplot(
        df.to_pandas(),
        x="L",
        y="accuracy",
        hue="Intermediate Atoms",
        style="strategy",
        markers=True,
        dashes=True,
    )
    baseline_labels = _add_baseline_hlines(df)

    handles, labels = ax.get_legend_handles_labels()
    atoms_labels = sorted(df["Intermediate Atoms"].unique().to_list())
    strat_labels = sorted(df["strategy"].unique().to_list())

    def _make_legend(lbl_list, title, **kwargs):
        hdls = [handles[labels.index(lbl)] for lbl in lbl_list if lbl in labels]
        return ax.legend(hdls, lbl_list, title=title, frameon=True, framealpha=1, **kwargs)

    leg1 = _make_legend(
        strat_labels, "Alignment Type", ncol=2, loc="upper center", bbox_to_anchor=(0.37, 0.18)
    )
    leg2 = _make_legend(
        baseline_labels,
        "",
        ncol=len(baseline_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    leg3 = _make_legend(
        atoms_labels, "Intermediate Atoms", ncol=2, loc="upper center", bbox_to_anchor=(0.76, 0.18)
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    ax.add_artist(leg3)

    plt.xlabel(r"SIM Layers $L$")
    plt.ylabel("Accuracy")
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    _save("AccuracyVsSIMLayers_Disjoint")


# ---------------------------------------------------------------------------
# Accuracy vs inter-layer spacing (s_layer)
# ---------------------------------------------------------------------------


def accuracy_vs_thickness() -> None:
    """Accuracy vs $s_{\\mathrm{layer}}$ at TX, for three RX spacing values."""
    df = _load_parquet_glob("dual_slayer/*.parquet")
    if df is None:
        print("No slayer results found.")
        return

    df = df.with_columns(
        (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
        pl.col("s_rx").cast(pl.Utf8).str.concat(r"$\lambda$").alias(r"$s_{\rm RX}$"),
    ).sort([r"$s_{\rm RX}$", "s_tx"])

    ticks = df["s_tx"].unique().sort().to_list()
    ticks_labels = [rf"${t}\lambda$" for t in ticks]

    ax = sns.lineplot(
        df.to_pandas(),
        x="s_tx",
        y="accuracy",
        hue=r"$s_{\rm RX}$",
        style=r"$s_{\rm RX}$",
        markers=True,
        dashes=False,
    )
    baseline_labels = _add_baseline_hlines(df)

    handles, labels = ax.get_legend_handles_labels()
    srx_labels = sorted(df[r"$s_{\rm RX}$"].unique().to_list())

    def _make_legend(lbl_list, title, **kwargs):
        hdls = [handles[labels.index(lbl)] for lbl in lbl_list if lbl in labels]
        return ax.legend(hdls, lbl_list, title=title, frameon=True, framealpha=1, **kwargs)

    leg1 = _make_legend(
        srx_labels, r"$s_{\rm layer}$ (RX)", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12)
    )
    leg2 = _make_legend(
        baseline_labels,
        "",
        ncol=len(baseline_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    ax.tick_params(axis="y", pad=10)
    plt.xlabel(r"$s_{\mathrm{layer}}$ (TX)")
    plt.ylabel("Accuracy")
    plt.xticks(ticks, labels=ticks_labels)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    _save("AccuracyVsThickness")


# ---------------------------------------------------------------------------
# Accuracy vs Layers — Mono-SIM (TX-only ablation)
# ---------------------------------------------------------------------------


def accuracy_vs_mono_layers() -> None:
    """Accuracy vs $L$ — Mono-SIM (TX-only, no channel)."""
    df = _load_parquet_glob("mono/*.parquet")
    if df is None:
        print("No mono-SIM results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
        .with_columns(
            (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
            pl.col("M_int").alias("Intermediate Atoms"),
        )
        .sort(["Intermediate Atoms", "strategy", "L"])
    )
    if df.is_empty():
        print("No mono-SIM (no-channel) results found.")
        return

    ticks = df["L"].unique().sort().to_list()

    ax = sns.lineplot(
        df.to_pandas(),
        x="L",
        y="accuracy",
        hue="Intermediate Atoms",
        style="strategy",
        markers=True,
        dashes=True,
    )
    baseline_labels = _add_baseline_hlines(df)

    handles, labels = ax.get_legend_handles_labels()
    atoms_labels = sorted(df["Intermediate Atoms"].unique().to_list())
    strat_labels = sorted(df["strategy"].unique().to_list())

    def _make_legend(lbl_list, title, **kwargs):
        hdls = [handles[labels.index(lbl)] for lbl in lbl_list if lbl in labels]
        return ax.legend(hdls, lbl_list, title=title, frameon=True, framealpha=1, **kwargs)

    leg1 = _make_legend(
        strat_labels, "Alignment Type", ncol=2, loc="upper center", bbox_to_anchor=(0.37, 0.18)
    )
    leg2 = _make_legend(
        baseline_labels,
        "",
        ncol=len(baseline_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    leg3 = _make_legend(
        atoms_labels, "Intermediate Atoms", ncol=3, loc="upper center", bbox_to_anchor=(0.76, 0.18)
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    ax.add_artist(leg3)

    plt.xlabel(r"SIM Layers $L$")
    plt.ylabel("Accuracy")
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    _save("AccuracyVsSIMLayers_Mono")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _apply_style()

    accuracy_vs_simlayers()
    accuracy_vs_snr()
    accuracy_vs_disjoint_layers()
    accuracy_vs_thickness()
    accuracy_vs_mono_layers()

    print(f"\n All plots saved to {IMG_PATH}")


if __name__ == "__main__":
    main()

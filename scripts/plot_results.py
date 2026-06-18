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

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"
IMG_PATH = RESULTS / "plots"
IMG_PATH.mkdir(parents=True, exist_ok=True)

_STYLE = BASE_DIR / "config" / "plotting" / "plt.mplstyle"

STRATEGIES = ["Linear", "PPFE"]

_ARCH_ORDER = ["Joint", "Disjoint Asym", "Disjoint Sym", "Disjoint 4-SIM", "Mono"]
_ARCH_PALETTE = dict(zip(_ARCH_ORDER, sns.color_palette("tab10", len(_ARCH_ORDER))))
_ARCH_MARKERS = {
    "Joint": "o",
    "Disjoint Asym": "P",
    "Disjoint Sym": "X",
    "Disjoint 4-SIM": "^",
    "Mono": "v",
}


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
    plt.close()
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
        ("Linear", "-.", "Oracle Linear"),
        ("PPFE", ":", "Oracle PPFE"),
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


def accuracy_vs_disjoint_asym_layers() -> None:
    """Accuracy vs $L$ — Disjoint Asym architecture."""
    df = _load_parquet_glob("disjoint_asym_layers/*.parquet")
    if df is None:
        print("No disjoint asym layers results found.")
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
    _save("AccuracyVsSIMLayers_DisjointAsym")


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


def accuracy_vs_mono_snr() -> None:
    """Accuracy vs SNR [dB] — Mono-SIM (TX-only) over the MIMO channel."""
    df = _load_parquet_glob("mono_snr/*.parquet")
    if df is None:
        print("No mono-SIM SNR results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES) & pl.col("snr").is_not_null())
        .with_columns(
            (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
            pl.col("M_int").alias("Intermediate Atoms"),
        )
        .sort(["Intermediate Atoms", "strategy", "snr"])
    )
    if df.is_empty():
        print("No mono-SIM (channel) results found.")
        return

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
    _save("AccuracyVsSNR_Mono")


# ---------------------------------------------------------------------------
# Accuracy vs SNR — Disjoint standard
# ---------------------------------------------------------------------------


def accuracy_vs_disjoint_asym_snr() -> None:
    """Accuracy vs SNR [dB] — Disjoint Asym architecture."""
    df = _load_parquet_glob("disjoint_asym_snr/*.parquet")
    if df is None:
        print("No disjoint asym SNR results found.")
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
    _save("AccuracyVsSNR_DisjointAsym")


# ---------------------------------------------------------------------------
# Accuracy vs Layers / SNR — Disjoint Sym
# ---------------------------------------------------------------------------


def accuracy_vs_disjoint_sym_layers() -> None:
    """Accuracy vs $L$ — Disjoint Sym architecture (PPFE only)."""
    df = _load_parquet_glob("disjoint_sym_layers/*.parquet")
    if df is None:
        print("No disjoint sym layers results found.")
        return

    df = df.with_columns(
        (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
        pl.col("M_int").cast(pl.Utf8).alias("Intermediate Atoms"),
    ).sort(["Intermediate Atoms", "L"])

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
    _save("AccuracyVsSIMLayers_DisjointSym")


def accuracy_vs_disjoint_sym_snr() -> None:
    """Accuracy vs SNR [dB] — Disjoint Sym architecture (PPFE only)."""
    df = _load_parquet_glob("disjoint_sym_snr/*.parquet")
    if df is None:
        print("No disjoint sym SNR results found.")
        return

    df = (
        df.filter(pl.col("snr").is_not_null())
        .with_columns(
            (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
            pl.col("M_int").cast(pl.Utf8).alias("Intermediate Atoms"),
        )
        .sort(["Intermediate Atoms", "snr"])
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
    _save("AccuracyVsSNR_DisjointSym")


# ---------------------------------------------------------------------------
# Accuracy vs Layers / SNR — Disjoint 4-SIM
# ---------------------------------------------------------------------------


def accuracy_vs_disjoint_4sim_layers() -> None:
    """Accuracy vs $L$ — Disjoint 4-SIM architecture (PPFE only).

    L is per-SIM depth; hardware equivalent to 2L layers in the 2-SIM case.
    """
    df = _load_parquet_glob("disjoint_4sim_layers/*.parquet")
    if df is None:
        print("No disjoint 4-SIM layers results found.")
        return

    df = df.with_columns(
        (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
        pl.col("M_int").cast(pl.Utf8).alias("Intermediate Atoms"),
    ).sort(["Intermediate Atoms", "L"])

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

    plt.xlabel(r"SIM Layers $L$ per SIM (equiv. $2L$ in 2-SIM)")
    plt.ylabel("Accuracy")
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    _save("AccuracyVsSIMLayers_Disjoint4SIM")


def accuracy_vs_disjoint_4sim_snr() -> None:
    """Accuracy vs SNR [dB] — Disjoint 4-SIM architecture (PPFE only)."""
    df = _load_parquet_glob("disjoint_4sim_snr/*.parquet")
    if df is None:
        print("No disjoint 4-SIM SNR results found.")
        return

    df = (
        df.filter(pl.col("snr").is_not_null())
        .with_columns(
            (pl.col("sim_accuracy_neural") / 100).alias("accuracy"),
            pl.col("M_int").cast(pl.Utf8).alias("Intermediate Atoms"),
        )
        .sort(["Intermediate Atoms", "snr"])
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
    _save("AccuracyVsSNR_Disjoint4SIM")


# ---------------------------------------------------------------------------
# Architecture comparison — Layers (M_int = 32×32)
# ---------------------------------------------------------------------------


def arch_comparison_layers() -> None:
    """Accuracy vs $L$ — all architectures, M_int=32×32. Grid: cols = strategy.

    4-SIM L values are multiplied by 2 to match hardware-equivalent depth
    (each 4-SIM run uses L layers per SIM = 2L total layers per side).
    """
    M_FILTER = "32x32"

    frames = {}
    for key, pattern, arch_label in [
        ("joint", "dual_layers/*.parquet", "Joint"),
        ("disjoint_asym", "disjoint_asym_layers/*.parquet", "Disjoint Asym"),
        ("disjoint_sym", "disjoint_sym_layers/*.parquet", "Disjoint Sym"),
        ("disjoint_4sim", "disjoint_4sim_layers/*.parquet", "Disjoint 4-SIM"),
        ("mono", "mono/*.parquet", "Mono"),
    ]:
        df = _load_parquet_glob(pattern)
        if df is None:
            continue
        df = df.filter(pl.col("M_int") == M_FILTER).with_columns(
            pl.lit(arch_label).alias("Architecture")
        )
        if key == "disjoint_4sim":
            df = df.with_columns((pl.col("L") * 2).alias("L"))
        if not df.is_empty():
            frames[key] = df

    if not frames:
        print("No layers data found for architecture comparison.")
        return

    acc_col = "sim_accuracy_neural"
    df_all = pl.concat(
        [
            f.select(
                [
                    "L",
                    "strategy",
                    "Architecture",
                    acc_col,
                    "no_mismatch_probe",
                    "no_mismatch_neural",
                    "original_probe",
                    "original_neural",
                ]
            )
            for f in frames.values()
        ],
        how="diagonal_relaxed",
    ).with_columns((pl.col(acc_col) / 100).alias("accuracy"))

    ticks = df_all["L"].unique().sort().to_list()
    strat_list = sorted(df_all["strategy"].unique().to_list())
    arch_labels = sorted(df_all["Architecture"].unique().to_list())

    fig, axes = plt.subplots(
        1,
        len(strat_list),
        sharey=True,
        sharex=True,
        squeeze=False,
        figsize=(8 * len(strat_list), 6),
        layout="constrained",
    )

    arch_handles, arch_handle_labels = [], []
    baseline_handles_dict: dict[str, object] = {}

    for c, strategy in enumerate(strat_list):
        ax = axes[0][c]
        df_s = df_all.filter(pl.col("strategy") == strategy)

        sns.lineplot(
            df_s.to_pandas(),
            x="L",
            y="accuracy",
            hue="Architecture",
            hue_order=[a for a in _ARCH_ORDER if a in arch_labels],
            palette=_ARCH_PALETTE,
            style="Architecture",
            markers={a: _ARCH_MARKERS[a] for a in _ARCH_ORDER if a in arch_labels},
            dashes=False,
            markersize=12,
            ax=ax,
        )

        # Capture arch handles from the first subplot only
        if c == 0:
            arch_handle_labels, arch_handles = (
                zip(
                    *[
                        (label, h)
                        for h, label in zip(*ax.get_legend_handles_labels())
                        if label in arch_labels
                    ]
                )
                if arch_labels
                else ([], [])
            )

        plt.sca(ax)
        bl_added = _add_baseline_hlines(df_s)

        # Collect baseline handles (each subplot contributes its own Oracle)
        h_all, l_all = ax.get_legend_handles_labels()
        for hi, li in zip(h_all, l_all):
            if li in bl_added and li not in baseline_handles_dict:
                baseline_handles_dict[li] = hi

        if ax.get_legend():
            ax.get_legend().remove()

        ax.set_title(strategy)
        ax.set_xlabel(r"SIM Layers $L$")
        ax.set_ylabel("Accuracy" if c == 0 else "")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlim(min(ticks), max(ticks))
        ax.set_ylim(None, 1.0)

    bl_order = ["No Mismatch", "Oracle Linear", "Oracle PPFE"]
    bl_labels = [lbl for lbl in bl_order if lbl in baseline_handles_dict]
    fig.legend(
        [baseline_handles_dict[lbl] for lbl in bl_labels],
        bl_labels,
        ncol=len(bl_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=True,
        framealpha=1,
    )
    fig.legend(
        list(arch_handles),
        list(arch_handle_labels),
        ncol=len(arch_labels),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
        framealpha=1,
    )

    _save("ArchComparison_Layers_M32")


# ---------------------------------------------------------------------------
# Architecture comparison — SNR (M_int = 32×32)
# ---------------------------------------------------------------------------


def arch_comparison_snr() -> None:
    """Accuracy vs SNR — all architectures, M_int=32×32. Grid: cols = strategy."""
    M_FILTER = "32x32"

    frames = {}
    for key, pattern, arch_label in [
        ("joint", "dual_snr/*.parquet", "Joint"),
        ("disjoint_asym", "disjoint_asym_snr/*.parquet", "Disjoint Asym"),
        ("disjoint_sym", "disjoint_sym_snr/*.parquet", "Disjoint Sym"),
        ("disjoint_4sim", "disjoint_4sim_snr/*.parquet", "Disjoint 4-SIM"),
        ("mono", "mono_snr/*.parquet", "Mono"),
    ]:
        df = _load_parquet_glob(pattern)
        if df is None:
            continue
        df = df.filter((pl.col("M_int") == M_FILTER) & pl.col("snr").is_not_null()).with_columns(
            pl.lit(arch_label).alias("Architecture")
        )
        if not df.is_empty():
            frames[key] = df

    if not frames:
        print("No SNR data found for architecture comparison.")
        return

    acc_col = "sim_accuracy_neural"
    df_all = pl.concat(
        [
            f.select(
                [
                    "snr",
                    "strategy",
                    "Architecture",
                    acc_col,
                    "no_mismatch_probe",
                    "no_mismatch_neural",
                    "original_probe",
                    "original_neural",
                ]
            )
            for f in frames.values()
        ],
        how="diagonal_relaxed",
    ).with_columns((pl.col(acc_col) / 100).alias("accuracy"))

    ticks = df_all["snr"].unique().sort().to_list()
    strat_list = sorted(df_all["strategy"].unique().to_list())
    arch_labels = sorted(df_all["Architecture"].unique().to_list())

    fig, axes = plt.subplots(
        1,
        len(strat_list),
        sharey=True,
        sharex=True,
        squeeze=False,
        figsize=(8 * len(strat_list), 6),
        layout="constrained",
    )

    arch_handles, arch_handle_labels = [], []
    baseline_handles_dict: dict[str, object] = {}

    for c, strategy in enumerate(strat_list):
        ax = axes[0][c]
        df_s = df_all.filter(pl.col("strategy") == strategy)

        sns.lineplot(
            df_s.to_pandas(),
            x="snr",
            y="accuracy",
            hue="Architecture",
            hue_order=[a for a in _ARCH_ORDER if a in arch_labels],
            palette=_ARCH_PALETTE,
            style="Architecture",
            markers={a: _ARCH_MARKERS[a] for a in _ARCH_ORDER if a in arch_labels},
            dashes=False,
            markersize=12,
            ax=ax,
        )

        if c == 0:
            arch_handle_labels, arch_handles = (
                zip(
                    *[
                        (lbl, h)
                        for h, lbl in zip(*ax.get_legend_handles_labels())
                        if lbl in arch_labels
                    ]
                )
                if arch_labels
                else ([], [])
            )

        plt.sca(ax)
        bl_added = _add_baseline_hlines(df_s)

        h_all, l_all = ax.get_legend_handles_labels()
        for hi, li in zip(h_all, l_all):
            if li in bl_added and li not in baseline_handles_dict:
                baseline_handles_dict[li] = hi

        if ax.get_legend():
            ax.get_legend().remove()

        ax.set_title(strategy)
        ax.set_xlabel(r"Signal-to-Noise Ratio [dB]")
        ax.set_ylabel("Accuracy" if c == 0 else "")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlim(min(ticks), max(ticks))
        ax.set_ylim(None, 1.0)

    bl_order = ["No Mismatch", "Oracle Linear", "Oracle PPFE"]
    bl_labels = [lbl for lbl in bl_order if lbl in baseline_handles_dict]
    fig.legend(
        [baseline_handles_dict[lbl] for lbl in bl_labels],
        bl_labels,
        ncol=len(bl_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=True,
        framealpha=1,
    )
    fig.legend(
        list(arch_handles),
        list(arch_handle_labels),
        ncol=len(arch_labels),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
        framealpha=1,
    )

    _save("ArchComparison_SNR_M32")


# ---------------------------------------------------------------------------
# Training loss curves — Joint Dual-SIM
# ---------------------------------------------------------------------------


def loss_curves_dual_layers() -> None:
    """Training loss curves — Joint Dual-SIM.

    Grid: rows = alignment strategy (Linear / PPFE),
          cols = M_int,
          color = SIM depth L (averaged across seeds).
    """
    df = _load_parquet_glob("dual_layers/*.parquet")
    if df is None:
        print("No layers results found.")
        return

    df = df.filter(pl.col("strategy").is_in(STRATEGIES))
    if df.is_empty():
        print("No dual_layers loss data found.")
        return

    m_ints = sorted(df["M_int"].unique().to_list())
    L_values = sorted(df["L"].unique().to_list())

    n_rows = len(STRATEGIES)
    n_cols = len(m_ints)

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(L_values), vmax=max(L_values))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey=False,
        sharex=True,
        squeeze=False,
        layout="constrained",
    )

    for r, strategy in enumerate(STRATEGIES):
        for c, m_int in enumerate(m_ints):
            ax = axes[r][c]
            sub = df.filter((pl.col("strategy") == strategy) & (pl.col("M_int") == m_int))

            for L in L_values:
                rows = sub.filter(pl.col("L") == L)
                if rows.is_empty():
                    continue
                histories = [list(h) for h in rows["loss_history"]]
                avg = np.mean(np.array(histories), axis=0)
                ax.plot(
                    np.arange(len(avg)),
                    avg,
                    color=cmap(norm(L)),
                    linewidth=0.9,
                    alpha=0.85,
                )

            ax.set_title(rf"$M_{{\rm int}}$ = {m_int}")
            if c == 0:
                ax.set_ylabel(f"{strategy}\nLoss")
            if r == n_rows - 1:
                ax.set_xlabel("Iteration")

    cbar_ticks = [L for L in L_values if L in (1, 5, 10, 15, 20, 25)]
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label(r"SIM Layers $L$")
    cbar.set_ticks(cbar_ticks)

    fig.suptitle("Training Loss — Joint Dual-SIM")
    _save("LossCurves_DualLayers")


# ---------------------------------------------------------------------------
# Architecture comparison — Channel Dimension
# ---------------------------------------------------------------------------


def arch_comparison_channel_dim() -> None:
    """Accuracy vs channel dimension $H$ — all architectures, L=10, M_int=32×32. Grid: cols = strategy."""
    M_FILTER = "32x32"

    frames = {}
    for key, subdir, arch_label in [
        ("joint", "dual_channel_dim", "Joint"),
        ("disjoint_asym", "disjoint_asym_channel_dim", "Disjoint Asym"),
        ("disjoint_sym", "disjoint_sym_channel_dim", "Disjoint Sym"),
        ("disjoint_4sim", "disjoint_4sim_channel_dim", "Disjoint 4-SIM"),
    ]:
        df = _load_parquet_glob(f"{subdir}/*.parquet")
        if df is None:
            continue
        df = df.filter(pl.col("M_int") == M_FILTER).with_columns(
            pl.lit(arch_label).alias("Architecture")
        )
        if not df.is_empty():
            frames[key] = df

    if not frames:
        print("No channel_dim data found — run 'just channel-dim' first.")
        return

    acc_col = "sim_accuracy_neural"
    if next(iter(frames.values()))[acc_col].drop_nulls().is_empty():
        acc_col = "sim_accuracy_probe"

    df_all = pl.concat(
        [f.select(["channel_dim", "strategy", "Architecture", acc_col]) for f in frames.values()],
        how="diagonal_relaxed",
    ).with_columns((pl.col(acc_col) / 100).alias("accuracy"))

    channel_dims = sorted(df_all["channel_dim"].drop_nulls().unique().to_list())
    strat_list = sorted(df_all["strategy"].unique().to_list())
    arch_labels = sorted(df_all["Architecture"].unique().to_list())

    fig, axes = plt.subplots(
        1,
        len(strat_list),
        sharey=True,
        sharex=True,
        squeeze=False,
        figsize=(8 * len(strat_list), 6),
        layout="constrained",
    )

    arch_handles, arch_handle_labels = [], []

    for c, strategy in enumerate(strat_list):
        ax = axes[0][c]
        df_s = df_all.filter(pl.col("strategy") == strategy)

        sns.lineplot(
            df_s.to_pandas(),
            x="channel_dim",
            y="accuracy",
            hue="Architecture",
            hue_order=[a for a in _ARCH_ORDER if a in arch_labels],
            palette=_ARCH_PALETTE,
            style="Architecture",
            markers={a: _ARCH_MARKERS[a] for a in _ARCH_ORDER if a in arch_labels},
            dashes=False,
            markersize=12,
            ax=ax,
        )

        if c == 0:
            arch_handle_labels, arch_handles = (
                zip(
                    *[
                        (label, h)
                        for h, label in zip(*ax.get_legend_handles_labels())
                        if label in arch_labels
                    ]
                )
                if arch_labels
                else ([], [])
            )

        if ax.get_legend():
            ax.get_legend().remove()

        ax.set_title(strategy)
        ax.set_xscale("log")
        ax.set_xticks(channel_dims)
        ax.set_xticklabels([str(c) for c in channel_dims])
        ax.set_xlim(min(channel_dims), max(channel_dims))
        ax.set_xlabel(r"$\mathbf{H}$")
        ax.set_ylabel("Accuracy" if c == 0 else "")
        ax.set_ylim(None, 1.0)

    fig.legend(
        list(arch_handles),
        list(arch_handle_labels),
        ncol=len(arch_labels),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
        framealpha=1,
    )

    _save("ArchComparison_ChannelDim")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _apply_style()

    accuracy_vs_simlayers()
    accuracy_vs_snr()
    accuracy_vs_disjoint_asym_layers()
    accuracy_vs_disjoint_asym_snr()
    accuracy_vs_disjoint_sym_layers()
    accuracy_vs_disjoint_sym_snr()
    accuracy_vs_disjoint_4sim_layers()
    accuracy_vs_disjoint_4sim_snr()
    accuracy_vs_thickness()
    accuracy_vs_mono_layers()
    accuracy_vs_mono_snr()
    loss_curves_dual_layers()
    arch_comparison_layers()
    arch_comparison_snr()
    arch_comparison_channel_dim()

    print(f"\n All plots saved to {IMG_PATH}")


if __name__ == "__main__":
    main()

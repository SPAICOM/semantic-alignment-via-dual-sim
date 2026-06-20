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

# Variants for the asym-vs-sym comparison plots.
# Each entry: (arch_key, tipology, display_label, marker, color, linestyle)
_DISJOINT_COMPARE = [
    ("disjoint_asym", "tx", "Asym", "P", "#1f77b4", "-"),
    ("disjoint_asym", "tx_c", "Asym (C)", "P", "#aec7e8", "--"),
    ("disjoint_sym", "standard", "Sym", "X", "#ff7f0e", "-"),
    ("disjoint_sym", "constructive", "Sym (C)", "X", "#ffbb78", "--"),
]

# Default tipology per sim_type for single-arch plots
_DEFAULT_TIPOLOGY = {
    "disjoint_asym": "tx",
    "disjoint_sym": "standard",
    "disjoint_4sim": "standard",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_style() -> None:
    """Apply the project's seaborn + mplstyle theme to all subsequent figures."""
    sns.set_style("whitegrid")
    if _STYLE.exists():
        plt.style.use(str(_STYLE))


_cache: pl.DataFrame | None = None


def _load_all() -> pl.DataFrame | None:
    """Load every result Parquet under RESULTS once and cache.

    Uses ``diagonal_relaxed`` so files where ``snr=None`` (Null type) and
    files where ``snr`` is numeric (Int64/Float64) merge without schema errors.
    """
    global _cache
    if _cache is None:
        files = [f for f in RESULTS.rglob("*.parquet") if f.parent.name != "plots"]
        if not files:
            return None
        _cache = pl.concat(
            [pl.read_parquet(f) for f in files],
            how="diagonal_relaxed",
        )
    return _cache


def _get(result_subdir: str) -> pl.DataFrame | None:
    """Return all rows whose ``result_subdir`` column matches, or ``None``."""
    df = _load_all()
    if df is None:
        return None
    out = df.filter(pl.col("result_subdir") == result_subdir)
    return out if not out.is_empty() else None


def _get_tipology(result_subdir: str, tipology: str) -> pl.DataFrame | None:
    """Return rows matching ``result_subdir`` and ``tipology``.

    Falls back to all rows for that subdir when the ``tipology`` column is
    absent (backward-compatible with results saved before tipology subfolders).
    """
    df = _get(result_subdir)
    if df is None:
        return None
    if "tipology" not in df.columns:
        return df
    out = df.filter(pl.col("tipology") == tipology)
    return out if not out.is_empty() else None


def _save(name: str) -> None:
    """Save the current figure as both PDF and PNG under ``IMG_PATH``, then close it."""
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
# Loss-curve grid helpers
# ---------------------------------------------------------------------------


def _setup_loss_grid(strategies, m_ints, L_values):
    """Create figure/axes/colormap for a loss-curve grid (rows=strategies, cols=M_int)."""
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(L_values), vmax=max(L_values))
    fig, axes = plt.subplots(
        len(strategies),
        len(m_ints),
        figsize=(5 * len(m_ints), 4 * len(strategies)),
        sharey="row",
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    return fig, axes, cmap, norm


def _add_loss_colorbar(fig, axes, cmap, norm, L_values):
    """Attach a vertical colorbar for SIM layers L; ticks auto-sampled to ≤8."""
    tick_step = max(1, len(L_values) // 8)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label(r"SIM Layers $L$")
    cbar.set_ticks(L_values[::tick_step])


def _plot_loss_col(ax, sub, L_values, cmap, norm, col, linestyle="-"):
    """Plot averaged loss curves for one (strategy, M_int) cell."""
    for L in L_values:
        rows = sub.filter(pl.col("L") == L)
        if rows.is_empty():
            continue
        valid = [list(h) for h in rows[col] if h is not None]
        if not valid:
            continue
        avg = np.mean(np.array(valid), axis=0)
        ax.plot(
            np.arange(len(avg)),
            avg,
            color=cmap(norm(L)),
            linestyle=linestyle,
            linewidth=0.9,
            alpha=0.85,
        )


# ---------------------------------------------------------------------------
# Accuracy vs SIM Layers
# ---------------------------------------------------------------------------


def accuracy_vs_simlayers() -> None:
    """Accuracy vs $L$ — Joint architecture, both alignment strategies."""
    df = _get("joint_layers")
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
    _save("AccuracyVsSIMLayers_Joint")


# ---------------------------------------------------------------------------
# Accuracy vs SNR
# ---------------------------------------------------------------------------


def accuracy_vs_snr() -> None:
    """Accuracy vs SNR [dB] — Joint architecture."""
    df = _get("joint_snr")
    if df is None:
        print("No SNR results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
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
    _save("AccuracyVsSNR_Joint")


# ---------------------------------------------------------------------------
# Accuracy vs Disjoint Layers
# ---------------------------------------------------------------------------


def accuracy_vs_disjoint_asym_layers() -> None:
    """Accuracy vs $L$ — Disjoint Asym architecture, standard tipology (tx)."""
    df = _get_tipology("disjoint_asym_layers", "tx")
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
    df = _get("joint_slayer")
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
    """Accuracy vs $L$ — Mono-SIM, standard tipology."""
    df = _get_tipology("mono", "standard")
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
    """Accuracy vs SNR [dB] — Mono-SIM, standard tipology."""
    df = _get_tipology("mono_snr", "standard")
    if df is None:
        print("No mono-SIM SNR results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
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
    """Accuracy vs SNR [dB] — Disjoint Asym architecture, standard tipology (tx)."""
    df = _get_tipology("disjoint_asym_snr", "tx")
    if df is None:
        print("No disjoint asym SNR results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
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
    """Accuracy vs $L$ — Disjoint Sym architecture, standard tipology."""
    df = _get_tipology("disjoint_sym_layers", "standard")
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
    """Accuracy vs SNR [dB] — Disjoint Sym architecture, standard tipology."""
    df = _get_tipology("disjoint_sym_snr", "standard")
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
    """Accuracy vs $L$ — Disjoint 4-SIM architecture, standard tipology."""
    df = _get_tipology("disjoint_4sim_layers", "standard")
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
    """Accuracy vs SNR [dB] — Disjoint 4-SIM architecture, standard tipology."""
    df = _get_tipology("disjoint_4sim_snr", "standard")
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
    for key, result_subdir, arch_label, topo in [
        ("joint", "joint_layers", "Joint", None),
        ("disjoint_asym", "disjoint_asym_layers", "Disjoint Asym", "tx"),
        ("disjoint_sym", "disjoint_sym_layers", "Disjoint Sym", "standard"),
        ("disjoint_4sim", "disjoint_4sim_layers", "Disjoint 4-SIM", "standard"),
        ("mono", "mono", "Mono", "standard"),
    ]:
        df = _get_tipology(result_subdir, topo) if topo else _get(result_subdir)
        if df is None:
            continue
        df = df.filter((pl.col("M_int") == M_FILTER)).with_columns(
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
    for key, result_subdir, arch_label, topo in [
        ("joint", "joint_snr", "Joint", None),
        ("disjoint_asym", "disjoint_asym_snr", "Disjoint Asym", "tx"),
        ("disjoint_sym", "disjoint_sym_snr", "Disjoint Sym", "standard"),
        ("disjoint_4sim", "disjoint_4sim_snr", "Disjoint 4-SIM", "standard"),
        ("mono", "mono_snr", "Mono", "standard"),
    ]:
        df = _get_tipology(result_subdir, topo) if topo else _get(result_subdir)
        if df is None:
            continue
        df = df.filter((pl.col("M_int") == M_FILTER)).with_columns(
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


def loss_curves_joint_layers() -> None:
    """Training loss curves — Joint Dual-SIM.

    Grid: rows = alignment strategy (Linear / PPFE),
          cols = M_int, color = SIM depth L (averaged across seeds).
    """
    df = _get("joint_layers")
    if df is None:
        print("No joint_layers results found.")
        return
    df = df.filter(pl.col("strategy").is_in(STRATEGIES))
    if df.is_empty():
        return

    m_ints = sorted(df["M_int"].unique().to_list())
    L_values = sorted(df["L"].unique().to_list())
    fig, axes, cmap, norm = _setup_loss_grid(STRATEGIES, m_ints, L_values)

    for r, strategy in enumerate(STRATEGIES):
        for c, m_int in enumerate(m_ints):
            ax = axes[r][c]
            sub = df.filter((pl.col("strategy") == strategy) & (pl.col("M_int") == m_int))
            _plot_loss_col(ax, sub, L_values, cmap, norm, "loss_history")
            if r == 0:
                ax.set_title(rf"$M_{{\rm int}}$ = {m_int}")
            if c == 0:
                ax.set_ylabel(f"{strategy}\nLoss")
            if r == len(STRATEGIES) - 1:
                ax.set_xlabel("Iteration")

    _add_loss_colorbar(fig, axes, cmap, norm, L_values)
    _save("LossCurves_Joint")


def loss_curves_mono_layers() -> None:
    """Training loss curves — Mono-SIM (TX-only).

    Grid: rows = alignment strategy, cols = M_int,
          color = L (averaged across seeds).
    """
    df = _get("mono")
    if df is None:
        print("No mono results found.")
        return
    df = df.filter(pl.col("strategy").is_in(STRATEGIES))
    if df.is_empty():
        return

    m_ints = sorted(df["M_int"].unique().to_list())
    L_values = sorted(df["L"].unique().to_list())
    fig, axes, cmap, norm = _setup_loss_grid(STRATEGIES, m_ints, L_values)

    for r, strategy in enumerate(STRATEGIES):
        for c, m_int in enumerate(m_ints):
            ax = axes[r][c]
            sub = df.filter((pl.col("strategy") == strategy) & (pl.col("M_int") == m_int))
            _plot_loss_col(ax, sub, L_values, cmap, norm, "loss_history")
            if r == 0:
                ax.set_title(rf"$M_{{\rm int}}$ = {m_int}")
            if c == 0:
                ax.set_ylabel(f"{strategy}\nLoss")
            if r == len(STRATEGIES) - 1:
                ax.set_xlabel("Iteration")

    _add_loss_colorbar(fig, axes, cmap, norm, L_values)
    _save("LossCurves_Mono")


def _loss_curves_single_col(result_subdir: str, col: str, fname: str) -> None:
    """Single-column loss grid: rows=strategies, cols=M_int, color=L."""
    df = _get(result_subdir)
    if df is None:
        print(f"No {result_subdir} results found.")
        return
    df = df.filter(pl.col("strategy").is_in(STRATEGIES))
    if df.is_empty():
        return

    m_ints = sorted(df["M_int"].unique().to_list())
    L_values = sorted(df["L"].unique().to_list())
    fig, axes, cmap, norm = _setup_loss_grid(STRATEGIES, m_ints, L_values)

    for r, strategy in enumerate(STRATEGIES):
        for c, m_int in enumerate(m_ints):
            ax = axes[r][c]
            sub = df.filter((pl.col("strategy") == strategy) & (pl.col("M_int") == m_int))
            _plot_loss_col(ax, sub, L_values, cmap, norm, col)
            if r == 0:
                ax.set_title(rf"$M_{{\rm int}}$ = {m_int}")
            if c == 0:
                ax.set_ylabel(f"{strategy}\nLoss")
            if r == len(STRATEGIES) - 1:
                ax.set_xlabel("Iteration")

    _add_loss_colorbar(fig, axes, cmap, norm, L_values)
    _save(fname)


def loss_curves_disjoint_asym_layers() -> None:
    """Training loss curves — Disjoint Asym (standard tipology), TX and RX."""
    _loss_curves_single_col("disjoint_asym_layers", "loss_history", "LossCurves_DisjointAsym_TX")
    _loss_curves_single_col("disjoint_asym_layers", "rx_loss_history", "LossCurves_DisjointAsym_RX")


def loss_curves_disjoint_sym_layers() -> None:
    """Training loss curves — Disjoint Sym (standard tipology), TX and RX."""
    _loss_curves_single_col("disjoint_sym_layers", "loss_history", "LossCurves_DisjointSym_TX")
    _loss_curves_single_col("disjoint_sym_layers", "rx_loss_history", "LossCurves_DisjointSym_RX")


def loss_curves_disjoint_4sim_layers() -> None:
    """Training loss curves — Disjoint 4-SIM (T1/T2/R2/R1, all separate).

    Grid: rows = strategy, cols = M_int, color = L.
    Line styles: T1 solid, T2 dotted, R2 dash-dot, R1 dashed.
    """
    from matplotlib.lines import Line2D

    df = _get("disjoint_4sim_layers")
    if df is None:
        print("No disjoint_4sim_layers results found.")
        return
    df = df.filter(pl.col("strategy").is_in(STRATEGIES))
    if df.is_empty():
        return

    m_ints = sorted(df["M_int"].unique().to_list())
    L_values = sorted(df["L"].unique().to_list())
    fig, axes, cmap, norm = _setup_loss_grid(STRATEGIES, m_ints, L_values)

    curve_specs = [
        ("loss_history", "-", "T1"),
        ("t2_loss_history", ":", "T2"),
        ("r2_loss_history", "-.", "R2"),
        ("rx_loss_history", "--", "R1"),
    ]

    for r, strategy in enumerate(STRATEGIES):
        for c, m_int in enumerate(m_ints):
            ax = axes[r][c]
            sub = df.filter((pl.col("strategy") == strategy) & (pl.col("M_int") == m_int))
            for col, ls, _ in curve_specs:
                if col in sub.columns:
                    _plot_loss_col(ax, sub, L_values, cmap, norm, col, linestyle=ls)
            if r == 0:
                ax.set_title(rf"$M_{{\rm int}}$ = {m_int}")
            if c == 0:
                ax.set_ylabel(f"{strategy}\nLoss")
            if r == len(STRATEGIES) - 1:
                ax.set_xlabel("Iteration")

    legend_elems = [
        Line2D([0], [0], color="gray", ls=ls, lw=1.5, label=label) for _, ls, label in curve_specs
    ]
    axes[0][0].legend(handles=legend_elems, loc="upper right", framealpha=1)
    _add_loss_colorbar(fig, axes, cmap, norm, L_values)
    _save("LossCurves_Disjoint4SIM")


# ---------------------------------------------------------------------------
# Architecture comparison — Channel Dimension
# ---------------------------------------------------------------------------


def arch_comparison_channel_dim() -> None:
    """Accuracy vs channel dimension $H$ — all architectures, L=10, M_int=32×32. Grid: cols = strategy."""
    M_FILTER = "32x32"

    frames = {}
    for key, subdir, arch_label, topo in [
        ("joint", "joint_channel_dim", "Joint", None),
        ("disjoint_asym", "disjoint_asym_channel_dim", "Disjoint Asym", "tx"),
        ("disjoint_sym", "disjoint_sym_channel_dim", "Disjoint Sym", "standard"),
        ("disjoint_4sim", "disjoint_4sim_channel_dim", "Disjoint 4-SIM", "standard"),
    ]:
        df = _get_tipology(subdir, topo) if topo else _get(subdir)
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
# Accuracy vs n_proto (PPFE) — Joint / Disjoint Asym / Mono
# ---------------------------------------------------------------------------


def accuracy_vs_n_proto() -> None:
    """Accuracy vs PPFE n_clusters — Joint, Disjoint Asym, and Mono (PPFE only).

    Fixed: L=10, M_int=32×32, channel_dim=20, no noise.
    Architecture comparison averaged across seeds.
    """
    frames = {}
    for key, subdir, arch_label in [
        ("joint", "n_proto_joint", "Joint"),
        ("disjoint_asym", "n_proto_disjoint_asym", "Disjoint Asym"),
        ("mono", "n_proto_mono", "Mono"),
    ]:
        df = _get(subdir)
        if df is None:
            continue
        df = df.with_columns(pl.lit(arch_label).alias("Architecture"))
        if not df.is_empty():
            frames[key] = df

    if not frames:
        print("No n_proto results found — run 'just n-proto' first.")
        return

    acc_col = "sim_accuracy_neural"
    df_all = pl.concat(
        [
            f.select(
                ["n_clusters", "Architecture", acc_col, "no_mismatch_neural", "original_neural"]
            )
            for f in frames.values()
        ],
        how="diagonal_relaxed",
    ).with_columns((pl.col(acc_col) / 100).alias("accuracy"))

    if df_all["n_clusters"].drop_nulls().is_empty():
        print("n_clusters column missing — re-run experiments with the updated code.")
        return

    arch_labels = sorted(df_all["Architecture"].unique().to_list())
    ticks = sorted(df_all["n_clusters"].drop_nulls().unique().to_list())

    ax = sns.lineplot(
        df_all.to_pandas(),
        x="n_clusters",
        y="accuracy",
        hue="Architecture",
        hue_order=[a for a in _ARCH_ORDER if a in arch_labels],
        palette=_ARCH_PALETTE,
        style="Architecture",
        markers={a: _ARCH_MARKERS[a] for a in _ARCH_ORDER if a in arch_labels},
        dashes=False,
        markersize=10,
    )
    baseline_labels = _add_baseline_hlines(df_all)

    handles, labels = ax.get_legend_handles_labels()

    def _make_legend(lbl_list, title, **kwargs):
        hdls = [handles[labels.index(lbl)] for lbl in lbl_list if lbl in labels]
        return ax.legend(hdls, lbl_list, title=title, frameon=True, framealpha=1, **kwargs)

    leg1 = _make_legend(
        arch_labels,
        "Architecture",
        ncol=len(arch_labels),
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
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

    plt.xscale("log")
    plt.xticks(ticks, labels=[str(k) for k in ticks])
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    plt.xlabel(r"PPFE $n_{\rm clusters}$")
    plt.ylabel("Accuracy")
    _save("AccuracyVsNProto")


# ---------------------------------------------------------------------------
# Loss curves by tipology
# ---------------------------------------------------------------------------

# Sym: color=L, linestyle=tipology
_SYM_TOPO_STYLES = [
    ("standard", "-", "standard"),
    ("constructive", "--", "constructive"),
]

# Asym: color=tipology-family, linestyle=constructive flag
_ASYM_TOPO_STYLES = [
    ("tx", "#1f77b4", "-", "tx"),
    ("tx_c", "#aec7e8", "--", "tx (C)"),
    ("rx", "#ff7f0e", "-", "rx"),
    ("rx_c", "#ffbb78", "--", "rx (C)"),
]


def _loss_curves_sym_tipology(col: str, loss_label: str, fname: str) -> None:
    """Loss curves for Disjoint Sym — standard vs constructive.

    Grid: rows = strategies, cols = M_int.
    Color = L (viridis colorbar), linestyle = tipology (solid=standard, dashed=constructive).
    """
    from matplotlib.lines import Line2D

    frames = {}
    for topo, ls, label in _SYM_TOPO_STYLES:
        df = _get_tipology("disjoint_sym_layers", topo)
        if df is not None and not df.is_empty():
            frames[topo] = df.filter(pl.col("strategy").is_in(STRATEGIES))

    if not frames:
        print(f"No disjoint_sym data for loss curves ({loss_label}).")
        return

    all_df = pl.concat(list(frames.values()), how="diagonal_relaxed")
    m_ints = sorted(all_df["M_int"].unique().to_list())
    L_values = sorted(all_df["L"].unique().to_list())
    fig, axes, cmap, norm = _setup_loss_grid(STRATEGIES, m_ints, L_values)

    for r, strategy in enumerate(STRATEGIES):
        for c, m_int in enumerate(m_ints):
            ax = axes[r][c]
            for topo, ls, _ in _SYM_TOPO_STYLES:
                if topo not in frames:
                    continue
                sub = frames[topo].filter(
                    (pl.col("strategy") == strategy) & (pl.col("M_int") == m_int)
                )
                _plot_loss_col(ax, sub, L_values, cmap, norm, col, linestyle=ls)
            if r == 0:
                ax.set_title(rf"$M_{{\rm int}}$ = {m_int}")
            if c == 0:
                ax.set_ylabel(f"{strategy}\n{loss_label}")
            if r == len(STRATEGIES) - 1:
                ax.set_xlabel("Iteration")

    topo_legend = [
        Line2D([0], [0], color="gray", ls=ls, lw=1.5, label=label)
        for _, ls, label in _SYM_TOPO_STYLES
    ]
    axes[0][0].legend(handles=topo_legend, loc="upper right", framealpha=1, title="Tipology")
    _add_loss_colorbar(fig, axes, cmap, norm, L_values)
    _save(fname)


def loss_curves_disjoint_sym_tipology() -> None:
    """TX and RX loss curves — Disjoint Sym, standard vs constructive."""
    _loss_curves_sym_tipology("loss_history", "TX Loss", "LossCurves_DisjointSym_Tipology_TX")
    _loss_curves_sym_tipology("rx_loss_history", "RX Loss", "LossCurves_DisjointSym_Tipology_RX")


def _loss_curves_asym_tipology(col: str, loss_label: str, fname: str) -> None:
    """Loss curves for Disjoint Asym — all 4 tipologies (tx, tx_c, rx, rx_c).

    Grid: rows = strategies, cols = M_int.
    Curves are averaged over seeds; color and linestyle encode tipology.
    L dimension is averaged to keep the plot readable.
    """
    from matplotlib.lines import Line2D

    frames = {}
    for topo, color, ls, label in _ASYM_TOPO_STYLES:
        df = _get_tipology("disjoint_asym_layers", topo)
        if df is not None and not df.is_empty():
            frames[topo] = df.filter(pl.col("strategy").is_in(STRATEGIES))

    if not frames:
        print(f"No disjoint_asym data for loss curves ({loss_label}).")
        return

    all_df = pl.concat(list(frames.values()), how="diagonal_relaxed")
    m_ints = sorted(all_df["M_int"].unique().to_list())

    fig, axes = plt.subplots(
        len(STRATEGIES),
        len(m_ints),
        figsize=(5 * len(m_ints), 4 * len(STRATEGIES)),
        sharey="row",
        sharex=True,
        squeeze=False,
        layout="constrained",
    )

    for r, strategy in enumerate(STRATEGIES):
        for c, m_int in enumerate(m_ints):
            ax = axes[r][c]
            for topo, color, ls, label in _ASYM_TOPO_STYLES:
                if topo not in frames:
                    continue
                sub = frames[topo].filter(
                    (pl.col("strategy") == strategy) & (pl.col("M_int") == m_int)
                )
                if sub.is_empty():
                    continue
                valid = [list(h) for h in sub[col] if h is not None]
                if not valid:
                    continue
                avg = np.mean(np.array(valid), axis=0)
                ax.plot(
                    np.arange(len(avg)),
                    avg,
                    color=color,
                    linestyle=ls,
                    linewidth=1.2,
                    alpha=0.9,
                    label=label,
                )
            if r == 0:
                ax.set_title(rf"$M_{{\rm int}}$ = {m_int}")
            if c == 0:
                ax.set_ylabel(f"{strategy}\n{loss_label}")
            if r == len(STRATEGIES) - 1:
                ax.set_xlabel("Iteration")

    topo_legend = [
        Line2D([0], [0], color=color, ls=ls, lw=1.5, label=label)
        for _, color, ls, label in _ASYM_TOPO_STYLES
    ]
    axes[0][0].legend(handles=topo_legend, loc="upper right", framealpha=1, title="Tipology")
    _save(fname)


def loss_curves_disjoint_asym_tipology() -> None:
    """TX and RX loss curves — Disjoint Asym, all 4 tipologies."""
    _loss_curves_asym_tipology("loss_history", "TX Loss", "LossCurves_DisjointAsym_Tipology_TX")
    _loss_curves_asym_tipology("rx_loss_history", "RX Loss", "LossCurves_DisjointAsym_Tipology_RX")


# ---------------------------------------------------------------------------
# Asym vs Sym comparison — all tipology variants
# ---------------------------------------------------------------------------


def _arch_comparison_asym_vs_sym(
    sweep: str,
    x_col: str,
    x_label: str,
    fname_suffix: str,
    M_FILTER: str = "32x32",
    x_log: bool = False,
) -> None:
    """Generic asym-vs-sym comparison across tipology variants.

    ``sweep`` is the result_subdir suffix, e.g. ``"layers"`` → loads
    ``disjoint_asym_layers`` and ``disjoint_sym_layers``.
    """
    frames = []
    for arch_key, topo, label, marker, color, ls in _DISJOINT_COMPARE:
        subdir = f"{arch_key}_{sweep}"
        df = _get_tipology(subdir, topo)
        if df is None:
            continue
        df = df.filter(pl.col("M_int") == M_FILTER).with_columns(
            pl.lit(label).alias("Variant"),
            pl.lit(color).alias("_color"),
            pl.lit(ls).alias("_ls"),
            pl.lit(marker).alias("_marker"),
        )
        if not df.is_empty():
            frames.append(df)

    if not frames:
        print(f"No asym/sym data found for sweep='{sweep}'.")
        return

    acc_col = "sim_accuracy_neural"
    needed = [
        x_col,
        "strategy",
        "Variant",
        "_color",
        "_ls",
        "_marker",
        acc_col,
        "no_mismatch_probe",
        "no_mismatch_neural",
        "original_probe",
        "original_neural",
    ]
    df_all = pl.concat(
        [f.select([c for c in needed if c in f.columns]) for f in frames],
        how="diagonal_relaxed",
    ).with_columns((pl.col(acc_col) / 100).alias("accuracy"))

    strat_list = sorted(df_all["strategy"].unique().to_list())

    fig, axes = plt.subplots(
        1,
        len(strat_list),
        sharey=True,
        sharex=True,
        squeeze=False,
        figsize=(8 * len(strat_list), 6),
        layout="constrained",
    )

    variant_handles: dict[str, object] = {}
    baseline_handles_dict: dict[str, object] = {}

    for c, strategy in enumerate(strat_list):
        ax = axes[0][c]
        df_s = df_all.filter(pl.col("strategy") == strategy)

        for _, topo, label, marker, color, ls in _DISJOINT_COMPARE:
            sub = df_s.filter(pl.col("Variant") == label)
            if sub.is_empty():
                continue
            sub_pd = sub.to_pandas()
            grp = sub_pd.groupby(x_col)["accuracy"]
            xs = sorted(grp.groups.keys())
            ys = [grp.get_group(x).mean() for x in xs]
            (line,) = ax.plot(
                xs, ys, color=color, linestyle=ls, marker=marker, markersize=10, label=label
            )
            if label not in variant_handles:
                variant_handles[label] = line

        plt.sca(ax)
        bl_added = _add_baseline_hlines(df_s)
        h_all, l_all = ax.get_legend_handles_labels()
        for hi, li in zip(h_all, l_all):
            if li in bl_added and li not in baseline_handles_dict:
                baseline_handles_dict[li] = hi

        if ax.get_legend():
            ax.get_legend().remove()

        ax.set_title(strategy)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy" if c == 0 else "")
        if x_log:
            ax.set_xscale("log")
        ticks = sorted(df_s[x_col].drop_nulls().unique().to_list())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(min(ticks), max(ticks))
        ax.set_ylim(None, 1.0)

    bl_order = ["No Mismatch", "Oracle Linear", "Oracle PPFE"]
    bl_labels = [lbl for lbl in bl_order if lbl in baseline_handles_dict]
    if bl_labels:
        fig.legend(
            [baseline_handles_dict[lbl] for lbl in bl_labels],
            bl_labels,
            ncol=len(bl_labels),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            frameon=True,
            framealpha=1,
        )
    ordered_variants = [v[2] for v in _DISJOINT_COMPARE if v[2] in variant_handles]
    fig.legend(
        [variant_handles[v] for v in ordered_variants],
        ordered_variants,
        ncol=len(ordered_variants),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=True,
        framealpha=1,
    )

    _save(f"AsymVsSym_{fname_suffix}_M32")


def arch_comparison_asym_vs_sym_layers() -> None:
    """Accuracy vs $L$ — Asym vs Sym, all tipology variants, M_int=32×32."""
    _arch_comparison_asym_vs_sym(
        sweep="layers",
        x_col="L",
        x_label=r"SIM Layers $L$",
        fname_suffix="Layers",
    )


def arch_comparison_asym_vs_sym_snr() -> None:
    """Accuracy vs SNR — Asym vs Sym, all tipology variants, M_int=32×32."""
    _arch_comparison_asym_vs_sym(
        sweep="snr",
        x_col="snr",
        x_label=r"Signal-to-Noise Ratio [dB]",
        fname_suffix="SNR",
    )


def arch_comparison_asym_vs_sym_channel_dim() -> None:
    """Accuracy vs channel_dim — Asym vs Sym, all tipology variants, M_int=32×32."""
    _arch_comparison_asym_vs_sym(
        sweep="channel_dim",
        x_col="channel_dim",
        x_label=r"Channel dimension $d_H$",
        fname_suffix="ChannelDim",
        x_log=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all publication plots from the cached result Parquet files.

    Applies the project style, then calls every individual plot function in
    sequence.  Silently skips any plot whose result subdirectory has no data.
    All output is written to ``results/plots/``.
    """
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
    loss_curves_joint_layers()
    loss_curves_mono_layers()
    loss_curves_disjoint_asym_layers()
    loss_curves_disjoint_sym_layers()
    loss_curves_disjoint_4sim_layers()
    loss_curves_disjoint_sym_tipology()
    loss_curves_disjoint_asym_tipology()
    arch_comparison_layers()
    arch_comparison_snr()
    arch_comparison_channel_dim()
    arch_comparison_asym_vs_sym_layers()
    arch_comparison_asym_vs_sym_snr()
    arch_comparison_asym_vs_sym_channel_dim()
    accuracy_vs_n_proto()

    print(f"\n All plots saved to {IMG_PATH}")


if __name__ == "__main__":
    main()

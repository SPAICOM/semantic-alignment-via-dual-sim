"""Publication-ready result plots.

Follows the plotting conventions of the semantic-alignment-via-sim reference repo:
  - ``sns.set_style('whitegrid')`` first, then ``plt.style.use(mplstyle)``
  - No explicit ``plt.figure()`` calls — figure size comes from the mplstyle
  - Save both PDF and PNG
  - ``plt.clf(); plt.cla()`` after every plot
  - LaTeX strings for axis labels — rendered by TeX when ``latex`` is on PATH,
    otherwise by matplotlib's mathtext with STIX (Times-like) fonts

Usage
-----
    just plot-cls
    uv run python scripts/plot_results_classification.py
"""

import shutil
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results" / "classification"


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

# ── Arch comparison (all tipologies) ─────────────────────────────────────────
# Families: color encodes arch, linestyle encodes constructive (solid) vs standard (dashed).
# Joint is constructive by design. Asym TX and Asym RX are separate families.
_ARCH_FAMILIES = [
    "Joint",
    "Disjoint Asym TX",
    "Disjoint Asym RX",
    "Disjoint Sym",
    "4-SIM",
    "Mono TX",
    "Mono RX",
]
_ARCH_FAMILY_COLORS = dict(zip(_ARCH_FAMILIES, sns.color_palette("tab10", len(_ARCH_FAMILIES))))
_ARCH_FAMILY_MARKERS = {
    "Joint": "o",
    "Disjoint Asym TX": "P",
    "Disjoint Asym RX": "s",
    "Disjoint Sym": "X",
    "4-SIM": "^",
    "Mono TX": "v",
    "Mono RX": "D",
}

# Each entry: (arch_key, tipology, family, constructive, subdir_layers, subdir_snr, subdir_ch, double_L)
_ARCH_COMPARE_VARIANTS = [
    ("joint", None, "Joint", True, "joint_layers", "joint_snr", "joint_channel_dim", False),
    (
        "disjoint_asym",
        "equalized/tx",
        "Disjoint Asym TX",
        False,
        "disjoint_asym_layers",
        "disjoint_asym_snr",
        None,
        False,
    ),
    (
        "disjoint_asym",
        "constructive/tx",
        "Disjoint Asym TX",
        True,
        "disjoint_asym_layers",
        "disjoint_asym_snr",
        None,
        False,
    ),
    (
        "disjoint_asym",
        "constructive/rx",
        "Disjoint Asym RX",
        True,
        "disjoint_asym_layers",
        "disjoint_asym_snr",
        None,
        False,
    ),
    (
        "disjoint_sym",
        "equalized",
        "Disjoint Sym",
        False,
        "disjoint_sym_layers",
        "disjoint_sym_snr",
        "disjoint_sym_channel_dim",
        False,
    ),
    (
        "disjoint_sym",
        "constructive",
        "Disjoint Sym",
        True,
        "disjoint_sym_layers",
        "disjoint_sym_snr",
        "disjoint_sym_channel_dim",
        False,
    ),
    (
        "disjoint_4sim",
        "equalized",
        "4-SIM",
        False,
        "disjoint_4sim_layers",
        "disjoint_4sim_snr",
        "disjoint_4sim_channel_dim",
        True,
    ),
    (
        "disjoint_4sim",
        "constructive",
        "4-SIM",
        True,
        "disjoint_4sim_layers",
        "disjoint_4sim_snr",
        "disjoint_4sim_channel_dim",
        True,
    ),
    (
        "mono",
        ["equalized", "equalized/tx"],
        "Mono TX",
        False,
        "mono_layers",
        "mono_snr",
        None,
        False,
    ),
    (
        "mono",
        ["constructive", "constructive/tx"],
        "Mono TX",
        True,
        "mono_layers",
        "mono_snr",
        None,
        False,
    ),
    ("mono", "constructive/rx", "Mono RX", True, "mono_layers", "mono_snr", None, False),
]
_SUBDIR_IDX = {"layers": 4, "snr": 5, "channel_dim": 6}

# Variants for the asym-vs-sym comparison plots.
# Each entry: (arch_key, tipology, display_label, marker, color, linestyle)
_DISJOINT_COMPARE = [
    ("disjoint_asym", "equalized/tx", "Disjoint Asym", "P", "#1f77b4", "-"),
    ("disjoint_asym", "constructive/tx", "Disjoint Asym", "P", "#1f77b4", "--"),
    ("disjoint_sym", "equalized", "Disjoint Sym", "X", "#ff7f0e", "-"),
    ("disjoint_sym", "constructive", "Disjoint Sym", "X", "#ff7f0e", "--"),
]

# Default tipology per sim_type for single-arch plots
_DEFAULT_TIPOLOGY = {
    "disjoint_asym": "tx",
    "disjoint_sym": "equalized",
    "disjoint_4sim": "equalized",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_style() -> None:
    """Apply the project's seaborn + mplstyle theme to all subsequent figures."""
    sns.set_style("whitegrid")
    if _STYLE.exists():
        plt.style.use(str(_STYLE))
    if plt.rcParams["text.usetex"] and shutil.which("latex") is None:
        # No TeX installation: render the same $...$ labels with mathtext instead.
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.serif"] = ["STIXGeneral", *plt.rcParams["font.serif"]]


_cache: pl.DataFrame | None = None


def _load_all() -> pl.DataFrame | None:
    """Load every result Parquet under RESULTS once and cache, filtered to ``_DATASET``.

    Uses ``diagonal_relaxed`` so files where ``snr=None`` (Null type) and
    files where ``snr`` is numeric (Int64/Float64) merge without schema errors.
    The global ``_DATASET`` is set per-dataset run in ``main()``.
    """
    global _cache
    if _cache is None:
        files = [f for f in RESULTS.rglob("*.parquet") if "plots" not in f.parts]
        if not files:
            return None
        _cache = pl.concat(
            [pl.read_parquet(f) for f in files],
            how="diagonal_relaxed",
        )
    df = _cache
    if "dataset" in df.columns:
        df = df.filter(pl.col("dataset") == _DATASET)
    return df if not df.is_empty() else None


def _get(result_subdir: str) -> pl.DataFrame | None:
    """Return all rows whose ``result_subdir`` column matches, or ``None``."""
    df = _load_all()
    if df is None:
        return None
    out = df.filter(pl.col("result_subdir") == result_subdir)
    return out if not out.is_empty() else None


def _get_tipology(result_subdir: str, tipology: str | list[str]) -> pl.DataFrame | None:
    """Return rows matching ``result_subdir`` and ``tipology``.

    ``tipology`` may be a single string or a list of accepted strings (for
    backward-compatibility when the same logical variant has multiple tag values).
    Falls back to all rows for that subdir when the ``tipology`` column is absent.
    """
    df = _get(result_subdir)
    if df is None:
        return None
    if "tipology" not in df.columns:
        return df
    tips = [tipology] if isinstance(tipology, str) else tipology
    out = df.filter(pl.col("tipology").is_in(tips))
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


def _add_baseline_hlines(df: pl.DataFrame, kind: str = "neural") -> list[str]:
    """Horizontal lines for No Mismatch (solid) and Original per strategy (dashed).
    Returns the list of labels added, for use in a legend.
    ``kind`` is ``"neural"`` or ``"probe"``.
    """
    added = []

    nm_col = f"no_mismatch_{kind}"
    orig_col = f"original_{kind}"

    if nm_col in df.columns:
        nm = df[nm_col].drop_nulls().mean()
        if nm is not None:
            plt.axhline(y=nm, color="black", linestyle="-", linewidth=2.5, label="No Mismatch")
            added.append("No Mismatch")

    if orig_col in df.columns:
        oracle_added = False
        for strat in ["Linear", "PPFE"]:
            vals = df.filter(pl.col("strategy") == strat)[orig_col].drop_nulls()
            if vals.len() > 0:
                plt.axhline(
                    y=vals.mean(), color="gray", linestyle="-.", linewidth=2.5, label="Oracle"
                )
                if not oracle_added:
                    added.append("Oracle")
                    oracle_added = True

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
            pl.col("sim_accuracy_neural").alias("accuracy"),
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
            pl.col("sim_accuracy_neural").alias("accuracy"),
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
    df = _get_tipology("disjoint_asym_layers", "equalized/tx")
    if df is None:
        print("No disjoint asym layers results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
        .with_columns(
            pl.col("sim_accuracy_neural").alias("accuracy"),
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
        pl.col("sim_accuracy_neural").alias("accuracy"),
        (pl.col("s_rx").cast(pl.Utf8) + pl.lit(r"$\lambda$")).alias(r"$s_{\rm layer}^{\rm RX}$"),
    ).sort([r"$s_{\rm layer}^{\rm RX}$", "s_tx"])

    ticks = df["s_tx"].unique().sort().to_list()
    ticks_labels = [rf"${t}\lambda$" for t in ticks]

    ax = sns.lineplot(
        df.to_pandas(),
        x="s_tx",
        y="accuracy",
        hue=r"$s_{\rm layer}^{\rm RX}$",
        style=r"$s_{\rm layer}^{\rm RX}$",
        markers=True,
        dashes=False,
    )
    baseline_labels = _add_baseline_hlines(df)

    handles, labels = ax.get_legend_handles_labels()
    srx_labels = sorted(df[r"$s_{\rm layer}^{\rm RX}$"].unique().to_list())

    def _make_legend(lbl_list, title, **kwargs):
        hdls = [handles[labels.index(lbl)] for lbl in lbl_list if lbl in labels]
        return ax.legend(hdls, lbl_list, title=title, frameon=True, framealpha=1, **kwargs)

    leg1 = _make_legend(
        srx_labels,
        r"$s_{\rm layer}^{\rm RX}$",
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
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
    plt.xlabel(r"$s_{\rm layer}^{\rm TX}$")
    plt.ylabel("Accuracy")
    plt.xticks(ticks, labels=ticks_labels)
    if len(ticks) > 1:
        plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    _save("AccuracyVsThickness")


def accuracy_vs_thickness_grid() -> None:
    """Grid plot: accuracy vs s_TX — Neural and Probe versions."""
    for acc_col, kind, suffix in [
        ("sim_accuracy_neural", "neural", "Neural"),
        ("sim_accuracy_probe", "probe", "Probe"),
    ]:
        _accuracy_vs_thickness_grid_one(acc_col, kind, suffix)


def _accuracy_vs_thickness_grid_one(acc_col: str, kind: str, suffix: str) -> None:
    """Inner: one pass of the slayer grid plot for a given accuracy column."""
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle as _Rect

    df_joint = _get("joint_slayer")
    df_dsym = _get("disjoint_sym_slayer")

    if df_joint is None and df_dsym is None:
        print("No slayer grid results found.")
        return

    frames = []
    if df_joint is not None:
        frames.append(df_joint.with_columns(pl.lit("Joint").alias("SIM")))
    if df_dsym is not None:
        frames.append(df_dsym.with_columns(pl.lit("Disjoint Sym").alias("SIM")))

    df = pl.concat(frames, how="diagonal")
    if acc_col not in df.columns:
        print(f"No {acc_col} column in slayer data.")
        return

    df = df.filter(pl.col("strategy").is_in(STRATEGIES)).with_columns(
        pl.col(acc_col).alias("accuracy"),
        (pl.col("s_rx").cast(pl.Utf8) + pl.lit(r"$\lambda$")).alias("s_RX"),
    )

    if df.is_empty():
        print("No slayer grid data after filtering.")
        return

    strat_list = [s for s in STRATEGIES if s in df["strategy"].unique().to_list()]
    sim_order = [s for s in ["Joint", "Disjoint Sym"] if s in df["SIM"].unique().to_list()]
    srx_vals = sorted(
        df["s_RX"].unique().to_list(),
        key=lambda s: float(s.replace(r"$\lambda$", "").strip()),
    )
    ticks = sorted(df["s_tx"].unique().to_list())
    ticks_labels = [rf"${t}\lambda$" for t in ticks]

    _srx_ls = ["-", "--", ":"]
    _srx_mk = ["o", "s", "^"]
    srx_linestyles = {v: _srx_ls[i % len(_srx_ls)] for i, v in enumerate(srx_vals)}
    srx_markers = {v: _srx_mk[i % len(_srx_mk)] for i, v in enumerate(srx_vals)}

    n_cols = len(strat_list)
    fig, axes = plt.subplots(
        1, n_cols, sharey=True, sharex=True, squeeze=False, figsize=(10 * n_cols, 6.5)
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.06, right=0.98, wspace=0.06)

    sim_handles: dict[str, object] = {}
    srx_handles: dict[str, object] = {}
    baseline_handles_dict: dict[str, object] = {}

    for c, strategy in enumerate(strat_list):
        ax = axes[0][c]
        df_s = df.filter(pl.col("strategy") == strategy)

        for sim in sim_order:
            color = _ARCH_FAMILY_COLORS.get(sim, "gray")
            df_sim = df_s.filter(pl.col("SIM") == sim)

            for srx in srx_vals:
                ls = srx_linestyles[srx]
                mk = srx_markers[srx]
                df_cur = df_sim.filter(pl.col("s_RX") == srx)
                if df_cur.is_empty():
                    continue

                xs = sorted(df_cur["s_tx"].unique().to_list())
                ys, ys_se = [], []
                for x in xs:
                    vals = df_cur.filter(pl.col("s_tx") == x)["accuracy"]
                    n = vals.len()
                    ys.append(vals.mean())
                    ys_se.append(vals.std() / (n**0.5) if n > 1 else 0.0)

                (line,) = ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=ls,
                    marker=mk,
                    markersize=10,
                    linewidth=1.5,
                    alpha=0.9,
                )
                ax.fill_between(
                    xs,
                    [y - 1.96 * se for y, se in zip(ys, ys_se)],
                    [y + 1.96 * se for y, se in zip(ys, ys_se)],
                    color=color,
                    alpha=0.12,
                )
                if sim not in sim_handles:
                    sim_handles[sim] = line
                if srx not in srx_handles:
                    srx_handles[srx] = Line2D(
                        [0], [0], color="gray", ls=ls, marker=mk, lw=1.5, markersize=8
                    )

        plt.sca(ax)
        bl_added = _add_baseline_hlines(df_s, kind=kind)
        h_all, l_all = ax.get_legend_handles_labels()
        for hi, li in zip(h_all, l_all):
            if li in bl_added and li not in baseline_handles_dict:
                baseline_handles_dict[li] = hi

        ax.set_title(strategy)
        ax.set_xlabel(r"$s_{\rm layer}^{\rm TX}$")
        ax.set_ylabel("Accuracy" if c == 0 else "")
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks_labels)
            ax.set_xlim(min(ticks), max(ticks))
    # Compute subplot centers in figure coordinates for legend anchoring
    _pos0 = axes[0][0].get_position()
    _cx0 = (_pos0.x0 + _pos0.x1) / 2
    _pos1 = axes[0][-1].get_position()
    _cx1 = (_pos1.x0 + _pos1.x1) / 2

    # Baseline legend — left-anchored at left edge of first subplot
    bl_order = ["No Mismatch", "Oracle"]
    bl_labels = [lbl for lbl in bl_order if lbl in baseline_handles_dict]
    if bl_labels:
        fig.legend(
            [baseline_handles_dict[lbl] for lbl in bl_labels],
            bl_labels,
            ncol=len(bl_labels),
            loc="upper center",
            bbox_to_anchor=(_pos0.x0 + 5 * (_pos0.x1 - _pos0.x0) / 12, 1.00),
            frameon=True,
            framealpha=1,
        )

    # SIM family legend — centered in figure
    family_legend_handles = [
        Line2D(
            [0],
            [0],
            color=_ARCH_FAMILY_COLORS[s],
            marker=_ARCH_FAMILY_MARKERS[s],
            ls="none",
            markersize=10,
        )
        for s in sim_order
        if s in sim_handles
    ]
    fig.legend(
        family_legend_handles,
        [s for s in sim_order if s in sim_handles],
        ncol=len(sim_order),
        loc="upper center",
        bbox_to_anchor=((_cx0 + _cx1) / 2, 1.00),
        frameon=True,
        framealpha=1,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    # s_RX legend — centered above last subplot
    class _HandlerEmpty(HandlerBase):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            patch = _Rect((0, 0), 0, 0, fill=False, edgecolor="none", linewidth=0)
            handlebox.add_artist(patch)
            return patch

    ordered_srx = sorted(
        srx_handles.keys(),
        key=lambda s: float(s.replace(r"$\lambda$", "").strip()),
    )
    _invis = Line2D([], [], alpha=0)
    _all_handles = [_invis] + [srx_handles[s] for s in ordered_srx]
    _all_labels = [r"$s_{\rm layer}^{\rm RX}$"] + ordered_srx
    fig.legend(
        _all_handles,
        _all_labels,
        handler_map={_invis: _HandlerEmpty()},
        ncol=len(ordered_srx) + 1,
        loc="upper right",
        bbox_to_anchor=(_pos1.x1, 1.00),
        frameon=True,
        framealpha=1,
        handletextpad=0.4,
    )

    _save(f"AccuracyVsThicknessGrid_{suffix}")


# ---------------------------------------------------------------------------
# Accuracy vs Layers — Mono-SIM (TX-only ablation)
# ---------------------------------------------------------------------------


_MONO_TIPOLOGY_LABELS = {
    "equalized/tx": "TX Equalized",
    "constructive/tx": "TX Constructive",
    "constructive/rx": "RX Constructive",
}


def _get_mono_all(result_subdir: str) -> pl.DataFrame | None:
    """Load all mono tipology variants for a given subdir, adding a 'Variant' column.

    Accepts both old tipology strings ("equalized", "constructive") and new ones
    ("equalized/tx", "constructive/tx") for TX side so migrated results are included.
    """
    df_all = _get(result_subdir)
    if df_all is None:
        return None
    if "tipology" not in df_all.columns:
        return df_all.with_columns(pl.lit("TX Equalized").alias("Variant"))

    frames = []
    # TX Equalized: old "equalized" or new "equalized/tx"
    part = df_all.filter(pl.col("tipology").is_in(["equalized", "equalized/tx"]))
    if not part.is_empty():
        frames.append(part.with_columns(pl.lit("TX Equalized").alias("Variant")))
    # TX Constructive: old "constructive" or new "constructive/tx"
    part = df_all.filter(pl.col("tipology").is_in(["constructive", "constructive/tx"]))
    if not part.is_empty():
        frames.append(part.with_columns(pl.lit("TX Constructive").alias("Variant")))
    # RX Constructive: new string only
    part = df_all.filter(pl.col("tipology") == "constructive/rx")
    if not part.is_empty():
        frames.append(part.with_columns(pl.lit("RX Constructive").alias("Variant")))

    if not frames:
        return None
    return pl.concat(frames, how="diagonal")


def accuracy_vs_mono_layers() -> None:
    """Accuracy vs $L$ — Mono-SIM, all tipologies (TX Eq, TX C, RX C)."""
    df = _get_mono_all("mono_layers")
    if df is None:
        print("No mono-SIM results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
        .with_columns(
            pl.col("sim_accuracy_neural").alias("accuracy"),
            pl.col("M_int").alias("Intermediate Atoms"),
        )
        .sort(["Variant", "strategy", "L"])
    )
    if df.is_empty():
        print("No mono-SIM (no-channel) results found.")
        return

    ticks = df["L"].unique().sort().to_list()

    ax = sns.lineplot(
        df.to_pandas(),
        x="L",
        y="accuracy",
        hue="Variant",
        style="strategy",
        markers=True,
        dashes=True,
    )
    baseline_labels = _add_baseline_hlines(df)

    handles, labels = ax.get_legend_handles_labels()
    variant_labels = list(_MONO_TIPOLOGY_LABELS.values())
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
        variant_labels, "SIM Variant", ncol=1, loc="upper center", bbox_to_anchor=(0.76, 0.22)
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
    """Accuracy vs SNR [dB] — Mono-SIM, all tipologies (TX Eq, TX C, RX C)."""
    df = _get_mono_all("mono_snr")
    if df is None:
        print("No mono-SIM SNR results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
        .with_columns(
            pl.col("sim_accuracy_neural").alias("accuracy"),
            pl.col("M_int").alias("Intermediate Atoms"),
        )
        .sort(["Variant", "strategy", "snr"])
    )
    if df.is_empty():
        print("No mono-SIM (channel) results found.")
        return

    ticks = df["snr"].unique().sort().to_list()

    ax = sns.lineplot(
        df.to_pandas(),
        x="snr",
        y="accuracy",
        hue="Variant",
        style="strategy",
        markers=True,
        dashes=True,
    )
    baseline_labels = _add_baseline_hlines(df)

    handles, labels = ax.get_legend_handles_labels()
    variant_labels = list(_MONO_TIPOLOGY_LABELS.values())
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
        variant_labels, "SIM Variant", ncol=1, loc="upper center", bbox_to_anchor=(0.76, 0.22)
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
    df = _get_tipology("disjoint_asym_snr", "equalized/tx")
    if df is None:
        print("No disjoint asym SNR results found.")
        return

    df = (
        df.filter(pl.col("strategy").is_in(STRATEGIES))
        .with_columns(
            pl.col("sim_accuracy_neural").alias("accuracy"),
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
    df = _get_tipology("disjoint_sym_layers", "equalized")
    if df is None:
        print("No disjoint sym layers results found.")
        return

    df = df.with_columns(
        pl.col("sim_accuracy_neural").alias("accuracy"),
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
    df = _get_tipology("disjoint_sym_snr", "equalized")
    if df is None:
        print("No disjoint sym SNR results found.")
        return

    df = (
        df.filter(pl.col("snr").is_not_null())
        .with_columns(
            pl.col("sim_accuracy_neural").alias("accuracy"),
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
    df = _get_tipology("disjoint_4sim_layers", "equalized")
    if df is None:
        print("No disjoint 4-SIM layers results found.")
        return

    df = df.with_columns(
        pl.col("sim_accuracy_neural").alias("accuracy"),
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
    df = _get_tipology("disjoint_4sim_snr", "equalized")
    if df is None:
        print("No disjoint 4-SIM SNR results found.")
        return

    df = (
        df.filter(pl.col("snr").is_not_null())
        .with_columns(
            pl.col("sim_accuracy_neural").alias("accuracy"),
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


def _arch_comparison_generic(
    sweep: str,
    x_col: str,
    x_label: str,
    fname: str,
    M_FILTER: str = "32x32",
    x_log: bool = False,
    show_baselines: bool = True,
    y_min: float | None = None,
    acc_col: str = "sim_accuracy_neural",
    baseline_kind: str = "neural",
) -> None:
    """Generic arch comparison: color=family, solid=constructive, dashed=standard.

    ``sweep`` ∈ ``{"layers", "snr", "channel_dim"}``.
    ``acc_col`` selects the accuracy column (``sim_accuracy_neural`` or ``sim_accuracy_probe``).
    ``baseline_kind`` selects the baseline columns (``"neural"`` or ``"probe"``).
    """
    from matplotlib.lines import Line2D

    sweep_idx = {"layers": 4, "snr": 5, "channel_dim": 6}[sweep]

    frames = []
    for arch_key, tipology, family, is_constructive, *subdirs, double_L in _ARCH_COMPARE_VARIANTS:
        subdir = subdirs[sweep_idx - 4]
        if subdir is None:
            continue
        df = _get_tipology(subdir, tipology) if tipology is not None else _get(subdir)
        if df is None:
            continue
        df = df.filter(pl.col("M_int") == M_FILTER)
        if double_L and "L" in df.columns:
            df = df.with_columns((pl.col("L") * 2).alias("L"))
        if df.is_empty():
            continue
        df = df.with_columns(
            pl.lit(family).alias("_family"),
            pl.lit(is_constructive).alias("_constructive"),
        )
        frames.append(df)

    if not frames:
        print(f"No data for arch comparison sweep='{sweep}'.")
        return

    needed = [
        x_col,
        "strategy",
        "_family",
        "_constructive",
        acc_col,
        f"no_mismatch_{baseline_kind}",
        f"original_{baseline_kind}",
    ]
    df_all = pl.concat(
        [f.select([c for c in needed if c in f.columns]) for f in frames],
        how="diagonal_relaxed",
    ).with_columns(pl.col(acc_col).alias("accuracy"))

    strat_list = sorted(df_all["strategy"].unique().to_list())

    fig, axes = plt.subplots(
        1,
        len(strat_list),
        sharey=True,
        sharex=True,
        squeeze=False,
        figsize=(10 * len(strat_list), 6.5),
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.06, right=0.98, wspace=0.06)

    family_handles: dict[str, object] = {}
    baseline_handles_dict: dict[str, object] = {}

    for c, strategy in enumerate(strat_list):
        ax = axes[0][c]
        df_s = df_all.filter(pl.col("strategy") == strategy)

        for (
            arch_key,
            tipology,
            family,
            is_constructive,
            *subdirs,
            double_L,
        ) in _ARCH_COMPARE_VARIANTS:
            color = _ARCH_FAMILY_COLORS.get(family, "gray")
            marker = _ARCH_FAMILY_MARKERS.get(family, "o")
            ls = "-" if is_constructive else "--"

            subdf = df_s.filter(
                (pl.col("_family") == family) & (pl.col("_constructive") == is_constructive)
            )
            if subdf.is_empty():
                continue

            xs = sorted(subdf[x_col].drop_nulls().unique().to_list())
            ys, ys_se = [], []
            for x in xs:
                col_vals = subdf.filter(pl.col(x_col) == x)["accuracy"]
                n = col_vals.len()
                ys.append(col_vals.mean())
                ys_se.append(col_vals.std() / (n**0.5) if n > 1 else 0.0)
            (line,) = ax.plot(
                xs,
                ys,
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=10,
                linewidth=1.5,
                alpha=0.9,
            )
            ax.fill_between(
                xs,
                [y - 1.96 * se for y, se in zip(ys, ys_se)],
                [y + 1.96 * se for y, se in zip(ys, ys_se)],
                color=color,
                alpha=0.12,
            )
            if family not in family_handles:
                family_handles[family] = line

        if show_baselines:
            plt.sca(ax)
            bl_added = _add_baseline_hlines(df_s, kind=baseline_kind)
            h_all, l_all = ax.get_legend_handles_labels()
            for hi, li in zip(h_all, l_all):
                if li in bl_added and li not in baseline_handles_dict:
                    baseline_handles_dict[li] = hi

        ax.set_title(strategy)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy" if c == 0 else "")
        if x_log:
            ax.set_xscale("log")
        ticks = sorted(df_s[x_col].drop_nulls().unique().to_list())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(min(ticks), max(ticks))
        ax.set_ylim(y_min, 1.0)

    # Compute subplot centers in figure coordinates for legend anchoring
    _pos0 = axes[0][0].get_position()
    _cx0 = (_pos0.x0 + _pos0.x1) / 2
    _pos1 = axes[0][-1].get_position()
    _cx1 = (_pos1.x0 + _pos1.x1) / 2

    # Baseline legend (top) — centered above first subplot
    if show_baselines:
        bl_order = ["No Mismatch", "Oracle"]
        bl_labels = [lbl for lbl in bl_order if lbl in baseline_handles_dict]
        if bl_labels:
            fig.legend(
                [baseline_handles_dict[lbl] for lbl in bl_labels],
                bl_labels,
                ncol=len(bl_labels),
                loc="upper center",
                bbox_to_anchor=(_cx0, 1.00),
                frameon=True,
                framealpha=1,
            )

    # Constructive/Equalized linestyle legend — centered above last subplot
    ls_handles = [
        Line2D([0], [0], color="gray", ls="-", lw=1.5),
        Line2D([0], [0], color="gray", ls="--", lw=1.5),
    ]
    fig.legend(
        ls_handles,
        [r"$\mathbf{H}$ Constructive", r"$\mathbf{H}$ Equalized"],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(_cx1, 1.00),
        frameon=True,
        framealpha=1,
    )

    # Family legend (bottom-center): color + marker with solid line, no linestyle encoding
    ordered_families = [f for f in _ARCH_FAMILIES if f in family_handles]
    family_legend_handles = [
        Line2D(
            [0],
            [0],
            color=_ARCH_FAMILY_COLORS[f],
            marker=_ARCH_FAMILY_MARKERS[f],
            ls="none",
            markersize=10,
        )
        for f in ordered_families
    ]
    fig.legend(
        family_legend_handles,
        ordered_families,
        ncol=len(ordered_families),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        framealpha=1,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    _save(fname)


def arch_comparison_layers() -> None:
    """Accuracy vs $L$ — all architectures and tipologies, M_int=32×32. Both accuracy types."""
    for acc_col, kind, suffix in [
        ("sim_accuracy_neural", "neural", "Neural"),
        ("sim_accuracy_probe", "probe", "Probe"),
    ]:
        _arch_comparison_generic(
            sweep="layers",
            x_col="L",
            x_label=r"SIM Layers $L$",
            fname=f"ArchComparison_Layers_M32_{suffix}",
            acc_col=acc_col,
            baseline_kind=kind,
        )


# ---------------------------------------------------------------------------
# Architecture comparison — SNR (M_int = 32×32)
# ---------------------------------------------------------------------------


def arch_comparison_snr() -> None:
    """Accuracy vs SNR — all architectures and tipologies, M_int=32×32. Both accuracy types."""
    for acc_col, kind, suffix in [
        ("sim_accuracy_neural", "neural", "Neural"),
        ("sim_accuracy_probe", "probe", "Probe"),
    ]:
        _arch_comparison_generic(
            sweep="snr",
            x_col="snr",
            x_label=r"Signal-to-Noise Ratio [dB]",
            fname=f"ArchComparison_SNR_M32_{suffix}",
            acc_col=acc_col,
            baseline_kind=kind,
        )


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
    """Accuracy vs channel dimension — all architectures and tipologies, M_int=32×32. Both accuracy types."""
    for acc_col, kind, suffix in [
        ("sim_accuracy_neural", "neural", "Neural"),
        ("sim_accuracy_probe", "probe", "Probe"),
    ]:
        _arch_comparison_generic(
            sweep="channel_dim",
            x_col="channel_dim",
            x_label=r"Channel dimension $d_H$",
            fname=f"ArchComparison_ChannelDim_{suffix}",
            x_log=True,
            show_baselines=False,
            y_min=0.0,
            acc_col=acc_col,
            baseline_kind=kind,
        )


# ---------------------------------------------------------------------------
# Accuracy vs n_proto (PPFE) — all architectures
# ---------------------------------------------------------------------------

# (subdir, tipology_filter, family, is_constructive, x_col)
_N_PROTO_VARIANTS = [
    ("n_proto_joint", None, "Joint", True, "n_clusters"),
    ("n_proto_disjoint_asym", "equalized/tx", "Disjoint Asym TX", False, "n_clusters"),
    ("n_proto_disjoint_asym", "constructive/tx", "Disjoint Asym TX", True, "n_clusters"),
    ("n_proto_disjoint_asym", "constructive/rx", "Disjoint Asym RX", True, "n_clusters"),
    ("n_proto_mono", ["equalized", "equalized/tx"], "Mono TX", False, "n_clusters"),
    ("n_proto_mono", ["constructive", "constructive/tx"], "Mono TX", True, "n_clusters"),
    ("n_proto_mono", "constructive/rx", "Mono RX", True, "n_clusters"),
    ("n_proto_disjoint_sym", "equalized", "Disjoint Sym", False, "channel_dim"),
    ("n_proto_disjoint_sym", "constructive", "Disjoint Sym", True, "channel_dim"),
]


def accuracy_vs_n_proto() -> None:
    """Accuracy vs PPFE n_clusters — all architectures, subplots per strategy.

    SIM accuracy: color = family, linestyle = constructive (solid) / equalized (dashed).
    Oracle: same color, dotted, no marker.
    No Mismatch: black horizontal line.
    """
    from matplotlib.lines import Line2D

    acc_col = "sim_accuracy_neural"
    oracle_col = "original_neural"

    frames = []
    for subdir, topo, family, is_constructive, x_col in _N_PROTO_VARIANTS:
        df = _get_tipology(subdir, topo) if topo is not None else _get(subdir)
        if df is None:
            continue
        df = df.with_columns(
            pl.col(x_col).alias("_n_proto"),
            pl.lit(family).alias("_family"),
            pl.lit(is_constructive).alias("_constructive"),
        )
        frames.append(df)

    if not frames:
        print("No n_proto results found — run 'just n-proto' first.")
        return

    needed = [
        "_n_proto",
        "strategy",
        "_family",
        "_constructive",
        acc_col,
        oracle_col,
        "no_mismatch_neural",
    ]
    df_all = pl.concat(
        [f.select([c for c in needed if c in f.columns]) for f in frames],
        how="diagonal_relaxed",
    ).with_columns(
        pl.col(acc_col).alias("accuracy"),
        pl.col(oracle_col).alias("oracle"),
    )

    strat_list = sorted(df_all["strategy"].unique().to_list())

    fig, axes = plt.subplots(
        1,
        len(strat_list),
        sharey=True,
        squeeze=False,
        figsize=(8 * len(strat_list), 6),
        layout="constrained",
    )

    family_handles: dict[str, object] = {}

    for c, strategy in enumerate(strat_list):
        ax = axes[0][c]
        df_s = df_all.filter(pl.col("strategy") == strategy)

        for _, topo, family, is_constructive, _ in _N_PROTO_VARIANTS:
            color = _ARCH_FAMILY_COLORS.get(family, "gray")
            marker = _ARCH_FAMILY_MARKERS.get(family, "o")
            ls = "-" if is_constructive else "--"

            subdf = df_s.filter(
                (pl.col("_family") == family) & (pl.col("_constructive") == is_constructive)
            )
            if subdf.is_empty():
                continue

            xs = sorted(subdf["_n_proto"].drop_nulls().unique().to_list())
            ys, ys_se, ys_or = [], [], []
            for x in xs:
                vals = subdf.filter(pl.col("_n_proto") == x)["accuracy"].drop_nulls()
                n = vals.len()
                ys.append(vals.mean() if n > 0 else None)
                ys_se.append(vals.std() / n**0.5 if n > 1 else 0.0)
                ov = subdf.filter(pl.col("_n_proto") == x)["oracle"].drop_nulls()
                ys_or.append(ov.mean() if ov.len() > 0 else None)

            xs_clean = [x for x, y in zip(xs, ys) if y is not None]
            ys_clean = [y for y in ys if y is not None]
            se_clean = [se for y, se in zip(ys, ys_se) if y is not None]
            if xs_clean:
                (line,) = ax.plot(
                    xs_clean,
                    ys_clean,
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markersize=10,
                    linewidth=1.5,
                    alpha=0.9,
                )
                ax.fill_between(
                    xs_clean,
                    [y - 1.96 * se for y, se in zip(ys_clean, se_clean)],
                    [y + 1.96 * se for y, se in zip(ys_clean, se_clean)],
                    color=color,
                    alpha=0.12,
                )
                if family not in family_handles:
                    family_handles[family] = line

            xs_or = [x for x, y in zip(xs, ys_or) if y is not None]
            ys_or_clean = [y for y in ys_or if y is not None]
            if xs_or:
                ax.plot(xs_or, ys_or_clean, color=color, linestyle=":", linewidth=1.5, alpha=0.7)

        nm = df_s["no_mismatch_neural"].drop_nulls().mean()
        if nm is not None:
            ax.axhline(y=nm, color="black", linestyle="-", linewidth=2.5)

        ax.set_title(strategy)
        ax.set_xlabel(r"PPFE $n_{\rm clusters}$")
        ax.set_ylabel("Accuracy" if c == 0 else "")
        ax.set_xscale("log")
        ticks = sorted(df_s["_n_proto"].drop_nulls().unique().to_list())
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.set_xlim(min(ticks), max(ticks))
        ax.set_ylim(None, 1.0)

    # No Mismatch + Oracle legend (top)
    fig.legend(
        [
            Line2D([0], [0], color="black", ls="-", lw=2.5),
            Line2D([0], [0], color="gray", ls=":", lw=1.5),
        ],
        ["No Mismatch", "Oracle"],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=True,
        framealpha=1,
    )

    # Family legend (bottom left)
    ordered_families = [f for f in _ARCH_FAMILIES if f in family_handles]
    fig.legend(
        [
            Line2D(
                [0],
                [0],
                color=_ARCH_FAMILY_COLORS[f],
                marker=_ARCH_FAMILY_MARKERS[f],
                ls="none",
                markersize=10,
            )
            for f in ordered_families
        ],
        ordered_families,
        ncol=len(ordered_families),
        loc="lower left",
        bbox_to_anchor=(0.0, -0.12),
        frameon=True,
        framealpha=1,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    # Constructive / Equalized legend (bottom right)
    fig.legend(
        [
            Line2D([0], [0], color="gray", ls="-", lw=1.5),
            Line2D([0], [0], color="gray", ls="--", lw=1.5),
        ],
        [r"$\mathbf{H}$ Constructive", r"$\mathbf{H}$ Equalized"],
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.12),
        frameon=True,
        framealpha=1,
    )

    _save("AccuracyVsNProto")


# ---------------------------------------------------------------------------
# Loss curves by tipology
# ---------------------------------------------------------------------------

# Sym: color=L, linestyle=tipology
_SYM_TOPO_STYLES = [
    ("equalized", "-", "equalized"),
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
    show_baselines: bool = True,
    y_min: float | None = None,
) -> None:
    """Generic asym-vs-sym comparison across tipology variants.

    ``sweep`` is the result_subdir suffix, e.g. ``"layers"`` → loads
    ``disjoint_asym_layers`` and ``disjoint_sym_layers``.
    """
    from matplotlib.lines import Line2D

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
    ).with_columns(pl.col(acc_col).alias("accuracy"))

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
            ys_se = [
                grp.get_group(x).std() / (len(grp.get_group(x)) ** 0.5)
                if len(grp.get_group(x)) > 1
                else 0.0
                for x in xs
            ]
            (line,) = ax.plot(
                xs, ys, color=color, linestyle=ls, marker=marker, markersize=10, label=label
            )
            ax.fill_between(
                xs,
                [y - 1.96 * se for y, se in zip(ys, ys_se)],
                [y + 1.96 * se for y, se in zip(ys, ys_se)],
                color=color,
                alpha=0.12,
            )
            if label not in variant_handles:
                variant_handles[label] = line

        if show_baselines:
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
        ax.set_ylim(y_min, 1.0)

    if show_baselines:
        bl_order = ["No Mismatch", "Oracle"]
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
    ordered_variants = sorted(
        set(v[2] for v in _DISJOINT_COMPARE if v[2] in variant_handles),
        key=lambda v: [x[2] for x in _DISJOINT_COMPARE].index(v),
    )
    fig.legend(
        [variant_handles[v] for v in ordered_variants],
        ordered_variants,
        ncol=len(ordered_variants),
        loc="lower left",
        bbox_to_anchor=(0.0, -0.14),
        frameon=True,
        framealpha=1,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    ls_handles = [
        Line2D([0], [0], color="gray", ls="-", lw=1.5),
        Line2D([0], [0], color="gray", ls="--", lw=1.5),
    ]
    fig.legend(
        ls_handles,
        [r"$\mathbf{H}$ Constructive", r"$\mathbf{H}$ Equalized"],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.75, -0.14),
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
        show_baselines=False,
        y_min=0.0,
    )


# ---------------------------------------------------------------------------
# Unrolled (diagonal) vs Classic — per-layer accuracy grid
# ---------------------------------------------------------------------------


_WS_LABELS = {
    "none": "No Warm-Start",
    "only_semantic": "With Warm-Start",
    "average": "Average",
}
_WS_ORDER = ["none", "only_semantic", "average"]
_H0V_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def unrolled_accuracy_grid() -> None:
    """Probe and Neural accuracy vs K grids — one file each."""
    for acc_col, curve_col, kind, suffix in [
        ("probe_acc", "probe_curve", "probe", "Probe"),
        ("neural_acc", "neural_curve", "neural", "Neural"),
    ]:
        _unrolled_accuracy_grid_one(acc_col, curve_col, suffix)


def _unrolled_accuracy_grid_one(acc_col: str, curve_col: str, suffix: str) -> None:
    """Accuracy vs K grid.

    Columns: warm_start_type (Semantic Only / Average).
    Hue + marker: h0_var.
    Linestyle: solid = Unrolled (colored), dashed black = Classic Adam (no marker).
    Classic baseline replicated across all warm_start columns (it has no warm_start).

    Unrolled parquets: results/classification/unrolled/**/*.parquet (excluding baseline/).
    Baseline parquets: results/classification/unrolled/baseline/**/*.parquet
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    unrolled_root = RESULTS / "unrolled"
    baseline_root = unrolled_root / "baseline"

    unrolled_files = [
        f
        for f in unrolled_root.rglob("*.parquet")
        if "baseline" not in f.parts and "warmstart" not in f.parts
    ]
    # For each ws_type: prefer _wa/ variant over cold if available.
    # Only show ws types in _WS_SHOW.
    _WS_SHOW = {"none", "only_semantic"}

    def _pick_files(cold_root, wa_root, ws_types):
        files = []
        for ws in ws_types:
            src = wa_root / ws if (wa_root / ws).exists() else cold_root / ws
            if src.exists():
                files += list(src.rglob("*.parquet"))
        return files

    oracle_files = _pick_files(baseline_root / "oracle", baseline_root / "oracle_wa", _WS_SHOW)
    h0t_files = _pick_files(
        baseline_root / "h0_transfer", baseline_root / "h0_transfer_wa", _WS_SHOW
    )

    # warm-start init accuracy (written by _ensure_ws_accuracy, warm_start_cache.py)
    _ws_acc_path = unrolled_root / "joint" / "warmstart" / "ws_accuracy.parquet"
    df_ws_acc: pl.DataFrame | None = (
        pl.read_parquet(_ws_acc_path) if _ws_acc_path.exists() else None
    )

    if not unrolled_files and not oracle_files and not h0t_files:
        print(f"No unrolled results found for {suffix}.")
        return

    # ── Unrolled data ─────────────────────────────────────────────────────────
    df_u: pl.DataFrame | None = None
    if unrolled_files:
        raw = pl.concat([pl.read_parquet(f) for f in unrolled_files], how="diagonal_relaxed")
        needed = {"channel_type", "h0_var", "warm_start_type", "K", "variant", acc_col}
        missing = needed - set(raw.columns)
        if missing:
            print(f"Unrolled parquets missing columns {missing} — re-run unrolled.")
        else:
            df_u = (
                raw.filter(pl.col("channel_type") == "rayleigh")
                .with_columns(pl.col(acc_col).cast(pl.Float64).alias("acc"))
                .select(["warm_start_type", "K", "h0_var", "variant", "acc"])
            )
            if df_u.is_empty():
                df_u = None

    # ── Oracle baseline — lazy scan + streaming aggregation ──────────────────
    # Pre-aggregate to (K, warm_start_type) → mean/std/n/lo/hi so we never
    # materialise the full N_test × K_max exploded frame (can be millions of rows).
    df_b: pl.DataFrame | None = None
    if oracle_files:
        _sample = pl.read_parquet(oracle_files[0])
        if curve_col in _sample.columns:
            _curve_len = len(_sample[curve_col][0].to_list())
            _has_ws = "warm_start_type" in _sample.columns
            _has_ch = "channel_type" in _sample.columns
            lf = pl.scan_parquet(oracle_files, hive_partitioning=False)
            lf = lf.with_columns(
                pl.col("warm_start_type").fill_null("none")
                if _has_ws
                else pl.lit("none").alias("warm_start_type")
            )
            if _has_ch:
                lf = lf.filter(pl.col("channel_type") == "rayleigh")
            df_b = (
                lf.select([curve_col, "warm_start_type"])
                .explode(curve_col)
                .with_row_index("_gi")
                .with_columns(
                    (pl.col("_gi") % _curve_len + 1).alias("K"),
                    pl.col(curve_col).cast(pl.Float64).alias("acc"),
                )
                .group_by(["K", "warm_start_type"])
                .agg(
                    pl.col("acc").mean().alias("mean"),
                    pl.col("acc").std().alias("std"),
                    pl.col("acc").count().alias("n"),
                )
                .sort("K")
                .with_columns(
                    (pl.col("mean") - 1.96 * pl.col("std") / pl.col("n").sqrt()).alias("lo"),
                    (pl.col("mean") + 1.96 * pl.col("std") / pl.col("n").sqrt()).alias("hi"),
                )
                .collect(engine="streaming")
            )
            if df_b.is_empty():
                df_b = None

    # ── H_0-transfer baseline — one curve.parquet per (ws_type, h0_var) ──────
    df_bt: pl.DataFrame | None = None
    if h0t_files:
        dfs_bt = []
        for f in h0t_files:
            df = pl.read_parquet(f)
            if curve_col in df.columns:
                # explode curve → (K, acc, acc_std, warm_start_type, h0_var)
                curve = df[curve_col][0].to_list()
                std_col = curve_col.replace("_curve", "_std_curve")
                std_vals = (
                    df[std_col][0].to_list()
                    if std_col in df.columns
                    else [float("nan")] * len(curve)
                )
                n_test = int(df["n_test"][0]) if "n_test" in df.columns else None
                Ks = [i + 1 for i in range(len(curve))]
                h0v_val = df["h0_var"][0]
                rows = pl.DataFrame(
                    {
                        "K": Ks,
                        "acc": [float(v) if v is not None else float("nan") for v in curve],
                        "acc_std": [float(v) if v is not None else float("nan") for v in std_vals],
                        "n_test": [float(n_test) if n_test is not None else float("nan")]
                        * len(curve),
                        "warm_start_type": [df["warm_start_type"][0]] * len(curve),
                        "h0_var": pl.Series([h0v_val] * len(curve), dtype=pl.Float64),
                    }
                )
                dfs_bt.append(rows)
        if dfs_bt:
            df_bt = pl.concat(dfs_bt).sort("K")
            if df_bt.is_empty():
                df_bt = None

    if df_u is None and df_b is None:
        print(f"No data to plot for {suffix}.")
        return

    _SHOW_KS = {1, 5, 10, 15, 30}

    # x-axis range from actually displayed K values
    k_mins, k_maxs = [], []
    if df_u is not None:
        k_mins.append(int(df_u["K"].min()))
        k_maxs.append(int(df_u["K"].max()))
    k_mins.append(min(_SHOW_KS))
    k_maxs.append(max(_SHOW_KS))
    x_min, x_max = min(k_mins), max(k_maxs)
    if df_u is not None:
        x_max = int(df_u["K"].max())
        if df_b is not None:
            df_b = df_b.filter(pl.col("K") <= x_max)
        if df_bt is not None:
            df_bt = df_bt.filter(pl.col("K") <= x_max)
    # warm-start point sits slightly left of K_min in linear scale
    ws_x = -3

    # Columns = warm_start_types present in data, restricted to _WS_SHOW
    _ws_present: set[str] = set()
    if df_u is not None:
        _ws_present |= set(df_u["warm_start_type"].unique().to_list())
    if df_b is not None:
        _ws_present |= set(df_b["warm_start_type"].unique().to_list())
    ws_vals = [w for w in _WS_ORDER if w in _ws_present and w in _WS_SHOW]
    if not ws_vals:
        ws_vals = [w for w in _WS_ORDER if w in _WS_SHOW]

    _has_ws_acc = df_ws_acc is not None and any(
        ws in df_ws_acc["warm_start_type"].unique().to_list() for ws in ws_vals
    )

    # h0_var values and visual encoding (None = full Rayleigh, sorted last)
    h0_src = df_u if df_u is not None else df_b
    _h0_raw = h0_src["h0_var"].unique().to_list()  # type: ignore[union-attr]
    h0_vals = sorted([v for v in _h0_raw if v is not None]) + ([None] if None in _h0_raw else [])
    palette = sns.color_palette("viridis", len(h0_vals))
    color_map = {v: palette[i] for i, v in enumerate(h0_vals)}
    marker_map = {v: _H0V_MARKERS[i % len(_H0V_MARKERS)] for i, v in enumerate(h0_vals)}

    n_cols = len(ws_vals)
    fig, axes = plt.subplots(
        1,
        n_cols,
        sharey=True,
        sharex=False,
        figsize=(10 * n_cols, 6.5),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.06, right=0.98, wspace=0.06)

    for col_idx, ws in enumerate(ws_vals):
        ax = axes[0][col_idx]
        ax.set_title(_WS_LABELS.get(ws, ws))

        for h0v in h0_vals:
            color = color_map[h0v]
            mk = marker_map[h0v]

            # Unrolled line: solid, colored, with marker
            if df_u is not None:
                h0_expr = pl.col("h0_var").is_null() if h0v is None else (pl.col("h0_var") == h0v)
                sub = df_u.filter(
                    (pl.col("warm_start_type") == ws)
                    & h0_expr
                    & pl.col("variant").str.contains("Unrolled")
                ).to_pandas()
                if ws == "only_semantic":
                    sub = sub[sub["K"].isin({1, 5, 10, 15, 30})]
                if not sub.empty:
                    sns.lineplot(
                        data=sub,
                        x="K",
                        y="acc",
                        ax=ax,
                        color=color,
                        linestyle="-",
                        linewidth=2,
                        marker=mk,
                        markersize=12,
                        errorbar=("ci", 95),
                        n_boot=200,
                    )

        # H_0-transfer baseline (source already selected per ws_type at load time)
        if df_bt is not None:
            for h0v in h0_vals:
                color = color_map[h0v]
                mk = marker_map[h0v]
                h0_expr = pl.col("h0_var").is_null() if h0v is None else (pl.col("h0_var") == h0v)
                sub_bt = df_bt.filter((pl.col("warm_start_type") == ws) & h0_expr)
                if not sub_bt.is_empty():
                    import numpy as _np2

                    sub_bt_pd = (
                        sub_bt.to_pandas()
                        .dropna(subset=["acc"])
                        .pipe(lambda d: d[d["K"].isin(_SHOW_KS)])
                        .drop_duplicates("K", keep="last")
                        .sort_values("K")
                        .reset_index(drop=True)
                    )
                    if sub_bt_pd.empty:
                        continue
                    # WS → K=1 dotted connection for H0-transfer baseline
                    if _has_ws_acc and ws in df_ws_acc["warm_start_type"].unique().to_list():
                        _acc_col_name = (
                            "ws_acc_probe" if "ws_acc_probe" in df_ws_acc.columns else "ws_acc"
                        )
                        _h0v_expr_ws = (
                            pl.col("h0_var").is_null() if h0v is None else (pl.col("h0_var") == h0v)
                        )
                        _row_ws = df_ws_acc.filter((pl.col("warm_start_type") == ws) & _h0v_expr_ws)
                        _k1_bt = sub_bt_pd[sub_bt_pd["K"] == 1]
                        if not _row_ws.is_empty() and not _k1_bt.empty:
                            ax.plot(
                                [ws_x, 1],
                                [
                                    float(_row_ws[_acc_col_name].mean()),
                                    float(_k1_bt["acc"].iloc[0]),
                                ],
                                color=color,
                                linestyle=":",
                                linewidth=1.5,
                                zorder=2,
                            )
                    ax.plot(
                        sub_bt_pd["K"],
                        sub_bt_pd["acc"],
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        marker=mk,
                        markersize=10,
                        zorder=3,
                    )
                    # 95% CI band (1.96 * SEM) if std is available
                    if "acc_std" in sub_bt_pd.columns and "n_test" in sub_bt_pd.columns:
                        _std = sub_bt_pd["acc_std"].to_numpy()
                        _n = sub_bt_pd["n_test"].to_numpy()
                        _sem = _std / _np2.sqrt(_np2.where(_n > 0, _n, _np2.nan))
                        if not _np2.all(_np2.isnan(_sem)):
                            ax.fill_between(
                                sub_bt_pd["K"],
                                sub_bt_pd["acc"] - 1.96 * _sem,
                                sub_bt_pd["acc"] + 1.96 * _sem,
                                color=color,
                                alpha=0.15,
                                zorder=2,
                            )

        # Classic Adam Oracle: black, dashdot, no marker
        if df_b is not None and ws in df_b["warm_start_type"].unique().to_list():
            sub_b = df_b.filter(pl.col("warm_start_type") == ws).sort("K")
            if not sub_b.is_empty():
                agg_b = sub_b.filter(pl.col("K").is_in(list(_SHOW_KS))).sort("K").to_pandas()
                # WS → K=1 dotted connection for oracle (black)
                if _has_ws_acc and ws in df_ws_acc["warm_start_type"].unique().to_list():
                    _acc_col_name = (
                        "ws_acc_probe" if "ws_acc_probe" in df_ws_acc.columns else "ws_acc"
                    )
                    _row_ws_or = df_ws_acc.filter(
                        (pl.col("warm_start_type") == ws) & pl.col("h0_var").is_null()
                    )
                    _k1_or = agg_b[agg_b["K"] == 1]
                    if not _row_ws_or.is_empty() and not _k1_or.empty:
                        ax.plot(
                            [ws_x, 1],
                            [
                                float(_row_ws_or[_acc_col_name].mean()),
                                float(_k1_or["mean"].iloc[0]),
                            ],
                            color="black",
                            linestyle=":",
                            linewidth=1.5,
                            zorder=2,
                        )
                ax.plot(
                    agg_b["K"],
                    agg_b["mean"],
                    color="black",
                    linestyle="-.",
                    linewidth=1.5,
                    zorder=3,
                )
                ax.fill_between(agg_b["K"], agg_b["lo"], agg_b["hi"], color="black", alpha=0.15)

        # Warm-start init: marker + dotted line to K=x_min, same marker as h_var line
        if _has_ws_acc and ws in df_ws_acc["warm_start_type"].unique().to_list():
            acc_col_name = "ws_acc_probe" if "ws_acc_probe" in df_ws_acc.columns else "ws_acc"
            for h0v in h0_vals:
                h0v_expr = pl.col("h0_var").is_null() if h0v is None else (pl.col("h0_var") == h0v)
                row = df_ws_acc.filter((pl.col("warm_start_type") == ws) & h0v_expr)
                if row.is_empty():
                    continue
                ws_acc = float(row[acc_col_name].mean())
                # dotted line to K=x_min (unrolled)
                if df_u is not None:
                    sub_k1 = df_u.filter(
                        (pl.col("warm_start_type") == ws)
                        & h0v_expr
                        & (pl.col("K") == x_min)
                        & pl.col("variant").str.contains("Unrolled")
                    )
                    if not sub_k1.is_empty():
                        acc_k1 = float(sub_k1["acc"].mean())
                        ax.plot(
                            [ws_x, x_min],
                            [ws_acc, acc_k1],
                            color=color_map[h0v],
                            linestyle=":",
                            linewidth=1.5,
                            zorder=2,
                        )
                ax.plot(
                    [ws_x],
                    [ws_acc],
                    marker=marker_map[h0v],
                    color=color_map[h0v],
                    markersize=10,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    zorder=10,
                    linestyle="none",
                    clip_on=False,
                )

        # Bring all markers to the front: strip them from their original Line2D
        # objects (which share zorder with the line) and re-draw them as pure
        # marker artists at zorder=20, so they always sit above every line/band.
        for _ln in list(ax.get_lines()):
            _mk = _ln.get_marker()
            if not _mk or _mk in ("", "None", "none"):
                continue
            _xd, _yd = _ln.get_xdata(), _ln.get_ydata()
            if not len(_xd):
                continue
            ax.plot(
                _xd,
                _yd,
                marker=_mk,
                markersize=_ln.get_markersize(),
                color=_ln.get_color(),
                markerfacecolor=_ln.get_markerfacecolor(),
                markeredgecolor=_ln.get_markeredgecolor(),
                markeredgewidth=_ln.get_markeredgewidth(),
                linestyle="none",
                zorder=20,
                clip_on=_ln.get_clip_on(),
            )
            _ln.set_marker("none")

        _ws_col = _has_ws_acc and ws in df_ws_acc["warm_start_type"].unique().to_list()
        ax.set_xlim(ws_x if _ws_col else x_min, x_max)
        from matplotlib.ticker import FixedLocator

        _step = max(1, x_max // 6)
        _base_ticks = sorted(set([x_min, x_max] + list(range(_step, x_max + 1, _step))))
        _k_ticks = sorted(set([ws_x] + _base_ticks)) if _ws_col else _base_ticks
        ax.xaxis.set_major_locator(FixedLocator(_k_ticks))

        def _ws_fmt(x, pos, _wx=ws_x, _wc=_ws_col):
            if _wc and x == _wx:
                return "WS"
            return str(int(x))

        ax.xaxis.set_major_formatter(FuncFormatter(_ws_fmt))
        ax.set_xlabel("K (layers / iterations)")
        if col_idx == 0:
            ax.set_ylabel("Accuracy")

    # ── Legends ───────────────────────────────────────────────────────────────
    # ── Legends — linestyle left, h_var right, anchored to subplot positions ──
    _pos_l = axes[0][0].get_position()
    _pos_r = axes[0][-1].get_position()
    _cx_l = (_pos_l.x0 + _pos_l.x1) / 2
    _cx_r = (_pos_r.x0 + _pos_r.x1) / 2

    ls_handles = [
        Line2D([0], [0], color="gray", ls="-", lw=2, label="Deep Unrolling"),
        Line2D([0], [0], color="black", ls="-.", lw=1.5, label="Classic Adam Oracle"),
        Line2D([0], [0], color="gray", ls="--", lw=1.5, label=r"Classic Adam on $\mathbf{H}_0$"),
    ]
    fig.legend(
        ls_handles,
        [h.get_label() for h in ls_handles],
        ncol=len(ls_handles),
        loc="upper left",
        bbox_to_anchor=(_pos_l.x0 / 2, 1),
        frameon=True,
        framealpha=1,
    )

    h0v_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[v],
            ls="-",
            lw=2,
            marker=marker_map[v],
            markersize=10,
            label="Full Rayleigh"
            if v is None
            else (r"Fixed $\mathbf{H}_0$" if v == 0 else rf"$h_{{\mathrm{{var}}}}={v}$"),
        )
        for v in h0_vals
    ]
    fig.legend(
        h0v_handles,
        [h.get_label() for h in h0v_handles],
        ncol=len(h0v_handles),
        loc="upper center",
        bbox_to_anchor=(_cx_r, 1),
        frameon=True,
        framealpha=1,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    _save(f"Unrolled_AccuracyGrid_{suffix}")


# ---------------------------------------------------------------------------
# Joint LR grid-search heatmap
# ---------------------------------------------------------------------------


def joint_lr_search_heatmap() -> None:
    """Accuracy heatmap over lr_tx × lr_rx — Joint Dual-SIM LR search.

    Rows: alignment strategies (Linear / PPFE).
    Cols: probe accuracy / neural accuracy.
    Each cell is a 5×5 heatmap with annotated values.
    """
    df = _get("joint_lr_search")
    if df is None:
        print("No joint_lr_search results found.")
        return

    acc_specs = []
    if "sim_accuracy_probe" in df.columns and df["sim_accuracy_probe"].drop_nulls().len() > 0:
        acc_specs.append(("sim_accuracy_probe", "Probe Accuracy"))
    if "sim_accuracy_neural" in df.columns and df["sim_accuracy_neural"].drop_nulls().len() > 0:
        acc_specs.append(("sim_accuracy_neural", "Neural Accuracy"))
    if not acc_specs:
        print("joint_lr_search: no accuracy columns found.")
        return

    strats = [s for s in STRATEGIES if s in df["strategy"].unique().to_list()]
    if not strats:
        return

    n_rows, n_cols = len(strats), len(acc_specs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
        layout="constrained",
    )

    for r, strat in enumerate(strats):
        df_s = df.filter(pl.col("strategy") == strat)
        for c, (acc_col, acc_label) in enumerate(acc_specs):
            ax = axes[r][c]
            lr_txs = sorted(df_s["lr_tx"].drop_nulls().unique().to_list())
            lr_rxs = sorted(df_s["lr_rx"].drop_nulls().unique().to_list())
            mat = np.full((len(lr_rxs), len(lr_txs)), np.nan)
            for i, lrx in enumerate(lr_rxs):
                for j, ltx in enumerate(lr_txs):
                    vals = df_s.filter((pl.col("lr_rx") == lrx) & (pl.col("lr_tx") == ltx))[
                        acc_col
                    ].drop_nulls()
                    if vals.len() > 0:
                        mat[i, j] = float(vals.mean())

            import numpy.ma as ma

            mat_masked = ma.masked_invalid(mat)
            vmin = float(np.nanmin(mat)) if not np.all(np.isnan(mat)) else 0.0
            vmax = float(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 1.0

            im = ax.imshow(
                mat_masked,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            for i in range(len(lr_rxs)):
                for j in range(len(lr_txs)):
                    v = mat[i, j]
                    if not np.isnan(v):
                        ax.text(
                            j,
                            i,
                            f"{v:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if v < (vmin + vmax) / 2 else "black",
                        )

            ax.set_xticks(range(len(lr_txs)))
            ax.set_xticklabels([str(v) for v in lr_txs], rotation=90)
            ax.set_yticks(range(len(lr_rxs)))
            ax.set_yticklabels([str(v) for v in lr_rxs])
            ax.set_xlabel(r"$\eta_{\rm TX}$")
            ax.set_ylabel(r"$\eta_{\rm RX}$")
            ax.set_title(f"{strat} — {acc_label}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")

    _save("JointLRSearch_Heatmap")


# ---------------------------------------------------------------------------
# Joint LR grid-search — loss and accuracy curves
# ---------------------------------------------------------------------------

_LR_STRAT_COLORS = {"Linear": "#1f77b4", "PPFE": "#ff7f0e"}


def _joint_lr_search_curve_grid(col: str, y_label: str, fname: str) -> None:
    """Generic 5×5 curve grid for the joint LR search.

    Rows = lr_rx values, cols = lr_tx values.
    Each panel plots the requested ``col`` (list-valued) for every strategy.
    Missing or empty curves are silently skipped.
    """
    df = _get("joint_lr_search")
    if df is None:
        print("No joint_lr_search results found.")
        return
    if col not in df.columns:
        print(f"joint_lr_search: column '{col}' not found.")
        return
    # keep only rows that actually have the curve
    df = df.filter(pl.col(col).is_not_null())
    if df.is_empty():
        print(f"joint_lr_search: no non-null '{col}' found.")
        return

    lr_txs = sorted(df["lr_tx"].drop_nulls().unique().to_list())
    lr_rxs = sorted(df["lr_rx"].drop_nulls().unique().to_list())
    strats = [s for s in STRATEGIES if s in df["strategy"].unique().to_list()]

    n_rows, n_cols = len(lr_rxs), len(lr_txs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
        sharex=False,
        sharey=True,
        layout="constrained",
    )

    for r, lrx in enumerate(lr_rxs):
        for c, ltx in enumerate(lr_txs):
            ax = axes[r][c]
            sub = df.filter((pl.col("lr_rx") == lrx) & (pl.col("lr_tx") == ltx))
            for strat in strats:
                color = _LR_STRAT_COLORS.get(strat, "gray")
                rows_s = sub.filter(pl.col("strategy") == strat)
                curves = [list(v) for v in rows_s[col] if v is not None and len(v) > 0]
                if not curves:
                    continue
                min_len = min(len(c_) for c_ in curves)
                arr = np.array([c_[:min_len] for c_ in curves])
                mean = arr.mean(axis=0)
                se = arr.std(axis=0) / max(np.sqrt(len(arr)), 1)
                x = np.arange(len(mean))
                ax.plot(x, mean, color=color, linewidth=1.3, label=strat)
                ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.15)

            if r == 0:
                ax.set_title(rf"$\eta_{{TX}}={ltx}$", fontsize=9)
            if c == 0:
                ax.set_ylabel(rf"$\eta_{{RX}}={lrx}$" + f"\n{y_label}", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("Cycle", fontsize=8)
            ax.tick_params(labelsize=7)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=_LR_STRAT_COLORS.get(s, "gray"), lw=1.5, label=s) for s in strats
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        ncol=len(strats),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        frameon=True,
        framealpha=1,
    )
    _save(fname)


def joint_lr_search_curves() -> None:
    """Loss and accuracy curves grid — Joint LR search (lr_tx × lr_rx).

    Produces three plots:
      - JointLRSearch_LossCurves   — training loss
      - JointLRSearch_ProbeCurves  — probe accuracy over cycles (save_curve)
      - JointLRSearch_NeuralCurves — neural accuracy over cycles (save_curve)
    """
    _joint_lr_search_curve_grid("loss_history", "Loss", "JointLRSearch_LossCurves")
    _joint_lr_search_curve_grid(
        "sim_accuracy_probe_curve", "Probe Acc.", "JointLRSearch_ProbeCurves"
    )
    _joint_lr_search_curve_grid(
        "sim_accuracy_neural_curve", "Neural Acc.", "JointLRSearch_NeuralCurves"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _plot_all() -> None:
    """Run every plot function for the currently active ``_DATASET`` / ``IMG_PATH``."""
    arch_comparison_layers()
    arch_comparison_snr()
    arch_comparison_channel_dim()
    accuracy_vs_thickness_grid()
    unrolled_accuracy_grid()
    joint_lr_search_heatmap()
    joint_lr_search_curves()


def main() -> None:
    """Generate all publication plots for every dataset found in the results.

    Iterates over all distinct ``dataset`` values in the Parquet files and
    saves plots to ``results/classification/plots/{dataset}/`` for each one.
    Falls back to a single run with ``_DATASET="unknown"`` when the column is absent.
    """
    global _DATASET, IMG_PATH, _cache

    _apply_style()

    # Load everything once to discover which datasets are present.
    if _cache is None:
        files = [f for f in RESULTS.rglob("*.parquet") if "plots" not in f.parts]
        if not files:
            print("No result files found.")
            return
        _cache = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    all_df = _cache

    if "dataset" in all_df.columns:
        datasets = sorted(all_df["dataset"].drop_nulls().unique().to_list())
    else:
        datasets = ["unknown"]

    for ds in datasets:
        _DATASET = ds
        IMG_PATH = RESULTS / "plots" / ds
        IMG_PATH.mkdir(parents=True, exist_ok=True)
        print(f"\n── dataset: {ds} → {IMG_PATH.relative_to(BASE_DIR)} ──")
        _plot_all()
        print(f" All plots saved to {IMG_PATH}")


if __name__ == "__main__":
    main()

"""Small matplotlib helpers used by the notebooks (consistent, colorblind-safe styling).

Categorical colors are assigned in a fixed order (never cycled), one hue per feature,
so the same line has the same color in every figure.
"""

from __future__ import annotations

import numpy as np

#: Categorical palette (validated order); slot i is always used for series i.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
#: Sequential single-hue ramp (light -> dark) for magnitudes (heatmaps).
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"

#: Fixed colors for the most used line features (so Ha is always blue, etc.).
FEATURE_COLORS = {
    "Ha": CATEGORICAL[0],
    "OIII_5007": CATEGORICAL[1],
    "OII_3727": CATEGORICAL[2],
    "Hb": CATEGORICAL[3],
    "OIII_4959": CATEGORICAL[4],
    "SIII_9532": CATEGORICAL[5],
    "SIII_9069": CATEGORICAL[6],
    "OIII_4363": CATEGORICAL[7],
    "SIII_6312": TEXT_SECONDARY,
}

OUTCOME_COLORS = {
    "MULTI_LINE": CATEGORICAL[0],
    "SINGLE_CORRECT": CATEGORICAL[2],
    "SINGLE_MISID": CATEGORICAL[1],
    "SINGLE_FAILED": CATEGORICAL[3],
    "NOT_DETECTED": TEXT_SECONDARY,
}


def feature_color(name: str) -> str:
    return FEATURE_COLORS.get(name, TEXT_SECONDARY)


def use_style():
    """Apply a quiet style: thin marks, recessive grid, no top/right spines."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.figsize": (7.5, 4.2),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "font.size": 10,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
    })


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("romanmock_seq", SEQUENTIAL)


def bar(ax, labels, values, colors=None, gap=0.15, **kwargs):
    """Thin bars with a visible gap; values labelled directly (no y-axis clutter)."""
    x = np.arange(len(labels))
    colors = colors or [CATEGORICAL[0]] * len(labels)
    bars = ax.bar(x, values, width=1 - gap, color=colors, edgecolor="none", **kwargs)
    ax.set_xticks(x, labels)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:,.0f}" if v >= 10 else f"{v:.2g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY, xytext=(0, 2), textcoords="offset points")
    ax.grid(axis="x", visible=False)
    return bars


def shade_band(ax, lo, hi, label=None, color=GRID):
    ax.axvspan(lo, hi, color=color, alpha=0.9, zorder=0)
    if label:
        ax.text(lo, ax.get_ylim()[1], label, color=TEXT_SECONDARY, fontsize=8, va="top", ha="left")

"""Report figures in the house exhibit style: black and white, no fills, line styles and
open markers to tell series apart, Times New Roman, unfilled step histograms.

Each function returns PNG bytes.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 3))]
MARKERS = ["o", "s", "^", "D", "v", "x"]


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10, "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    return buf.getvalue()


def _x(index: pd.PeriodIndex) -> pd.DatetimeIndex:
    return index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index


def cumulative_growth(returns: pd.DataFrame, title_units: str = "Growth of $1 (log scale)") -> bytes:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    for i, c in enumerate(returns.columns):
        w = (1 + returns[c].fillna(0)).cumprod()
        ax.plot(_x(returns.index), w, color="black", linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=1.4 if i == 0 else 0.9, label=c)
    ax.set_yscale("log")
    ax.set_ylabel(title_units)
    ax.legend(loc="upper left", fontsize=8)
    return _png(fig)


def rolling_sharpe(series: pd.Series, months: int = 36) -> bytes:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.plot(_x(series.index), series, color="black", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylabel(f"Rolling {months}-month Sharpe")
    return _png(fig)


def placebo_histogram(placebo: np.ndarray, student: float) -> bytes:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.hist(placebo, bins=40, histtype="step", color="black", linewidth=0.9)
    ax.axvline(student, color="black", linewidth=1.6, linestyle="--")
    ax.annotate("Your strategy", xy=(student, ax.get_ylim()[1] * 0.9), xytext=(5, 0), textcoords="offset points",
                fontsize=8)
    ax.set_xlabel("Annualized Sharpe ratio of random combinations of published predictors")
    ax.set_ylabel("Number of draws")
    return _png(fig)


def best_of_n(best: dict[int, float], student: float) -> bytes:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    n = list(best)
    ax.plot(n, [best[k] for k in n], color="black", marker="o", markerfacecolor="white", linewidth=0.9,
            label="Best of N random combinations")
    ax.axhline(student, color="black", linestyle="--", linewidth=1.2, label="Your strategy")
    ax.set_xscale("log")
    ax.set_xlabel("Number of random combinations tried (N)")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(fontsize=8, loc="upper left")
    return _png(fig)


def shapley_bars(share: pd.Series, risk: pd.Series) -> bytes:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    y = np.arange(len(share))
    ax.barh(y + 0.18, share.values, height=0.32, fill=False, edgecolor="black", hatch="", label="Sharpe contribution")
    ax.barh(y - 0.18, risk.values, height=0.32, fill=False, edgecolor="black", hatch="////", label="Risk share")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y, share.index)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="lower right")
    return _png(fig)


def factor_map(coords: pd.DataFrame, highlight: str = "YOUR STRATEGY", family: list[str] | None = None) -> bytes:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fam = set(family or [])
    other = coords.drop(index=[i for i in coords.index if i == highlight or i in fam])
    ax.scatter(other["x"], other["y"], s=10, facecolors="none", edgecolors="black", linewidths=0.4)
    f = coords.loc[[i for i in coords.index if i in fam]]
    ax.scatter(f["x"], f["y"], s=14, color="black")
    me = coords.loc[highlight]
    ax.scatter([me["x"]], [me["y"]], s=90, marker="*", facecolors="white", edgecolors="black", linewidths=1.2)
    ax.annotate("Your strategy", (me["x"], me["y"]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel("Dimension 1"), ax.set_ylabel("Dimension 2")
    return _png(fig)

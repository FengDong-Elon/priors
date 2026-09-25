"""Performance statistics. Every number the user sees is computed here, never by an LLM."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = {"monthly": 12, "weekly": 52}


def summarize(r: pd.Series, periods_per_year: int) -> dict[str, float]:
    """Summary statistics for a periodic return series.

    The Sharpe ratio is mean / std of the series as given; subtract the risk-free
    rate first when the series is not already an excess return. Its 95% CI uses
    the i.i.d. standard error of Lo (2002): se(SR) = sqrt((1 + SR^2 / 2) / T).
    """
    r = r.dropna()
    t = len(r)
    if t < 2:
        raise ValueError("need at least 2 periods")
    ppy = periods_per_year
    mean, sd = r.mean(), r.std(ddof=1)
    sr = mean / sd if sd > 0 else np.nan
    se = math.sqrt((1 + 0.5 * sr**2) / t) if np.isfinite(sr) else np.nan
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {
        "periods": t,
        "years": t / ppy,
        "mean_ann": mean * ppy,
        "vol_ann": sd * math.sqrt(ppy),
        "cagr": wealth.iloc[-1] ** (ppy / t) - 1,
        "sharpe": sr * math.sqrt(ppy),
        "sharpe_ci_low": (sr - 1.96 * se) * math.sqrt(ppy),
        "sharpe_ci_high": (sr + 1.96 * se) * math.sqrt(ppy),
        "t_stat_mean": mean / (sd / math.sqrt(t)) if sd > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "max_drawdown_periods": _longest_underwater(drawdown),
        "hit_rate": (r > 0).mean(),
        "skew": r.skew(),
        "excess_kurtosis": r.kurt(),
        "best": r.max(),
        "worst": r.min(),
    }


def _longest_underwater(drawdown: pd.Series) -> int:
    longest = run = 0
    for under in (drawdown < 0).to_numpy():
        run = run + 1 if under else 0
        longest = max(longest, run)
    return longest

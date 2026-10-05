"""Stability over time (ARCHITECTURE §7.5): decades, halves, and a rolling Sharpe ratio."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..composite import FactorResult
from ..engine.metrics import summarize

ROLLING_MONTHS = 36
MIN_SUBPERIOD = 24


@dataclass
class Stability:
    by_decade: pd.DataFrame        # rows: decade; columns: months, mean_ann, sharpe, t_stat_mean_nw
    halves: pd.DataFrame
    rolling_sharpe: pd.Series
    share_positive_rolling: float
    flags: list[str] = field(default_factory=list)


def _table(groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for label, r in groups.items():
        r = r.dropna()
        if len(r) < MIN_SUBPERIOD:
            continue
        m = summarize(r, 12)
        rows[label] = {"start": str(r.index[0]), "end": str(r.index[-1]), "months": len(r),
                       "mean_ann": m["mean_ann"], "sharpe": m["sharpe"], "t_stat_mean_nw": m["t_stat_mean_nw"],
                       "max_drawdown": m["max_drawdown"]}
    return pd.DataFrame(rows).T


def stability(result: FactorResult) -> Stability:
    r = result.returns["composite"].dropna()
    decade = (r.index.year // 10) * 10
    by_decade = _table({f"{d}s": r[decade == d] for d in sorted(set(decade))})
    mid = len(r) // 2
    halves = _table({"first half": r.iloc[:mid], "second half": r.iloc[mid:]})
    roll = r.rolling(ROLLING_MONTHS).apply(lambda x: x.mean() / x.std(ddof=1) * math.sqrt(12), raw=True).dropna()
    share_pos = float((roll > 0).mean()) if len(roll) else float("nan")

    flags = []
    if len(by_decade):
        neg = by_decade.index[by_decade["mean_ann"] < 0].tolist()
        if neg:
            flags.append(f"Average returns were negative in: {', '.join(neg)}.")
    if len(halves) == 2:
        a, b = halves["sharpe"].iloc[0], halves["sharpe"].iloc[1]
        if b < a / 2:
            flags.append(
                f"The Sharpe ratio fell from {a:.2f} in the first half of the sample to {b:.2f} in the second, "
                "consistent with decay after publication or a change in the market."
            )
    if len(roll):
        flags.append(f"The rolling {ROLLING_MONTHS}-month Sharpe ratio was positive {share_pos:.0%} of the time.")
    return Stability(by_decade=by_decade, halves=halves, rolling_sharpe=roll,
                     share_positive_rolling=share_pos, flags=flags)

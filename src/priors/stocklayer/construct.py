"""Filters, signal composite, and top-N selection (ARCHITECTURE §10.2, §10.4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data.base import Panel
from ..spec.strategy import StockFilters


def microcap_cutoffs(signal_dates: pd.DatetimeIndex, breakpoints: pd.DataFrame | None) -> pd.Series:
    """NYSE 20th-percentile market cap in USD for each signal date (latest available month carried forward)."""
    if breakpoints is None or breakpoints.empty:
        return pd.Series(np.nan, index=signal_dates)
    p20 = breakpoints["p20"] * 1e6
    p20.index = p20.index.to_timestamp(how="end").normalize()
    return p20.reindex(signal_dates, method="ffill")


def eligible(panel: Panel, f: StockFilters, breakpoints: pd.DataFrame | None,
             universe_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Investable stocks at each signal date, using only information known at that date.

    ``universe_mask`` restricts to the universe's members at each date (point-in-time membership).
    """
    mask = panel["price"].notna() & panel["mcap"].notna()
    if universe_mask is not None:
        mask &= universe_mask.reindex(index=mask.index, columns=mask.columns, fill_value=False).astype(bool)
    if f.min_price is not None:
        mask &= panel["price"] >= f.min_price
    if f.min_adv_usd is not None:
        mask &= panel["dollar_volume"] >= f.min_adv_usd
    if f.exclude_microcaps:
        cut = microcap_cutoffs(panel.signal_dates, breakpoints)
        if cut.notna().any():
            mask &= panel["mcap"].ge(cut, axis=0) | cut.isna().to_numpy()[:, None]
    return mask


def composite_score(signals: dict[str, pd.DataFrame], mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average percentile rank across signals among eligible stocks.

    A stock needs at least half of the signals, and at least two when there are
    two or more, to get a score. Otherwise a stock with a single extreme value
    (for example a new listing with no momentum history) can top the list on one
    signal alone. Returns the score and the number of signals per stock.
    """
    ranks = [s.where(mask).rank(axis=1, pct=True) for s in signals.values()]
    stack = np.stack([r.to_numpy() for r in ranks])
    count = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.nansum(stack, axis=0) / np.where(count > 0, count, np.nan)
    need = max(math.ceil(len(ranks) / 2), min(len(ranks), 2))
    score = np.where(count >= need, score, np.nan)
    ref = ranks[0]
    return (pd.DataFrame(score, index=ref.index, columns=ref.columns),
            pd.DataFrame(count, index=ref.index, columns=ref.columns))


def top_n_weights(score: pd.DataFrame, n: int, weighting: str, volatility: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hold the n highest-scoring stocks; equal weights or weights proportional to 1 / volatility."""
    rank = score.rank(axis=1, ascending=False, method="first")
    held = rank <= n
    if weighting == "inverse_vol":
        if volatility is None:
            raise ValueError("inverse_vol weighting needs a volatility field")
        base = (1.0 / volatility.where(volatility > 0)).where(held)
    else:
        base = held.astype(float).where(held)
    return base.div(base.sum(axis=1), axis=0).fillna(0.0)

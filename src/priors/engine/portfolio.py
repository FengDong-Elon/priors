"""Universe filters and portfolio weights. All operations are per signal date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.base import Panel
from ..spec.schema import Portfolio, Universe


def eligible(panel: Panel, u: Universe) -> pd.DataFrame:
    """Boolean mask of investable securities at each signal date.

    Uses only information known at the signal date. In particular it must not
    require ret_next to exist, because that would drop stocks that are about to
    delist.
    """
    mask = panel["price"].notna()
    if u.min_price is not None:
        mask &= panel["price"] >= u.min_price
    if u.min_dollar_volume is not None:
        mask &= panel["dollar_volume"] >= u.min_dollar_volume
    if u.min_mcap_pctile is not None:
        mcap = panel["mcap"]
        ref = mcap.where(nyse_mask(panel)) if u.mcap_breakpoints == "nyse" else mcap
        cut = ref.quantile(u.min_mcap_pctile / 100, axis=1)
        mask &= mcap.ge(cut, axis=0)
    return mask


def nyse_mask(panel: Panel) -> pd.DataFrame:
    """Boolean frame, True for NYSE-listed securities (broadcast over dates)."""
    is_nyse = panel.nyse.reindex(panel.ids, fill_value=False).astype(bool).to_numpy()
    return pd.DataFrame(np.broadcast_to(is_nyse, (len(panel.signal_dates), len(is_nyse))),
                        index=panel.signal_dates, columns=panel.ids)


def quantile_buckets(signal: pd.DataFrame, mask: pd.DataFrame, q: int,
                     breakpoint_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bucket 1 (lowest) .. q (highest) for eligible securities; NaN elsewhere.

    With ``breakpoint_mask`` (e.g. NYSE stocks), the q-1 cutoffs are computed from
    that subset only and then applied to every eligible security, as in Fama-French.
    """
    s = signal.where(mask)
    if breakpoint_mask is None:
        pct = s.rank(axis=1, pct=True, method="first")
        return np.ceil(pct * q).clip(1, q)
    ref = s.where(breakpoint_mask)
    buckets = pd.DataFrame(1.0, index=s.index, columns=s.columns)
    for j in range(1, q):
        cut = ref.quantile(j / q, axis=1)
        buckets += s.gt(cut, axis=0).astype(float)
    return buckets.where(s.notna())


def weights(panel: Panel, signal: pd.DataFrame, mask: pd.DataFrame, p: Portfolio, direction: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (long_weights, short_weights). Each leg sums to 1 per date; short is None for long-only."""
    base = _base_weights(panel, p.weighting)
    # Sort only among stocks that have both a signal and a usable weight.
    mask = mask & signal.notna() & base.notna() & (base > 0)
    bp = nyse_mask(panel) if p.breakpoints == "nyse" else None
    buckets = quantile_buckets(signal, mask, p.quantiles, bp)
    top, bottom = (p.quantiles, 1) if direction == "high" else (1, p.quantiles)
    long = _normalize(base.where(buckets == top))
    short = _normalize(base.where(buckets == bottom)) if p.side == "long_short" else None
    return long, short


def _base_weights(panel: Panel, weighting: str) -> pd.DataFrame:
    if weighting == "equal":
        return pd.DataFrame(1.0, index=panel.signal_dates, columns=panel.ids)
    if weighting == "value":
        return panel["mcap"]
    if weighting == "inverse_vol":
        v = panel["volatility"]
        return 1.0 / v.where(v > 0)
    raise ValueError(f"Unknown weighting {weighting!r}")


def _normalize(w: pd.DataFrame) -> pd.DataFrame:
    w = w.where(w > 0)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)

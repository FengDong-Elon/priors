"""Rebalance calendars built from a list of trading days."""

from __future__ import annotations

import pandas as pd

FREQS = ("monthly", "weekly")


def rebalance_dates(trading_days: pd.DatetimeIndex, freq: str, lag_days: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return (signal_dates, exec_dates).

    signal_dates are the last trading day of each month or week. exec_dates are
    ``lag_days`` trading days later. The final period, whose execution date would
    fall after the last trading day, is dropped, and so is an incomplete trailing
    period.
    """
    if freq not in FREQS:
        raise ValueError(f"freq must be one of {FREQS}, got {freq!r}")
    if lag_days < 0:
        raise ValueError("lag_days must be >= 0")
    days = pd.DatetimeIndex(sorted(set(trading_days)))
    period = days.to_period("M" if freq == "monthly" else "W-FRI")
    s = pd.Series(days, index=days)
    signal = pd.DatetimeIndex(s.groupby(period).max().values)

    # Drop the trailing period if business days remain in it (conservative: a
    # holiday at the very end of the data can drop a period that was complete).
    remaining = pd.bdate_range(days[-1] + pd.Timedelta(days=1), period[-1].end_time.normalize())
    if len(remaining) > 0:
        signal = signal[:-1]

    pos = days.get_indexer(signal) + lag_days
    keep = pos < len(days)
    return signal[keep], days[pos[keep]]


def assign_periods(trading_days: pd.DatetimeIndex, boundaries: pd.DatetimeIndex) -> pd.Series:
    """Map each trading day d to k such that boundaries[k-1] < d <= boundaries[k].

    Days after the last boundary get -1.
    """
    k = boundaries.searchsorted(trading_days, side="left")
    k = pd.Series(k, index=trading_days)
    k[k >= len(boundaries)] = -1
    return k

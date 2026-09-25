"""Data-layer contract.

Every data source is wrapped in a ``DataProvider`` that produces a ``Panel``: a set
of wide DataFrames (rows = rebalance periods, columns = security ids) on a common
rebalance calendar. The engine only ever talks to this interface.

Timing convention
-----------------
For rebalance period k there are two dates:

* ``signal_dates[k]``: the last trading day of the period. Signals may use any
  field labelled k, because those fields are all known at the close of this day.
* ``exec_dates[k]``: ``lag_days`` trading days later. Trades happen at this close.

``ret_next[k]`` is the return from ``exec_dates[k]`` to ``exec_dates[k+1]``. It is
the only forward-looking field, and the engine uses it for P&L only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import date

import pandas as pd

FIELD_DOCS: dict[str, str] = {
    "ret_past": "Total return over the period ending at signal_dates[k]. Known at k.",
    "ret_next": "Total return from exec_dates[k] to exec_dates[k+1]. FORWARD-LOOKING: P&L only.",
    "price": "Unadjusted close at signal_dates[k].",
    "mcap": "Market capitalization in USD at signal_dates[k].",
    "dollar_volume": "Mean daily dollar volume over the period ending at signal_dates[k].",
    "volatility": "Std. dev. of daily returns over the period ending at signal_dates[k] (not annualized).",
}

FORWARD_FIELDS = frozenset({"ret_next"})


@dataclass(frozen=True)
class Capabilities:
    """What a data source can and cannot support. The Referee reads these flags."""

    universe: str
    history_start: date
    includes_delisted: bool
    delisting_returns: bool
    point_in_time_fundamentals: bool
    fundamentals: bool
    institutional_holdings: bool
    historical_exchange: bool

    def warnings(self) -> list[str]:
        out = []
        if not self.includes_delisted:
            out.append(
                "Survivorship bias: this source excludes delisted securities, so "
                "backtested returns are likely overstated."
            )
        elif not self.delisting_returns:
            out.append(
                "No delisting returns: a security's final return is measured to its "
                "last traded price, so losses at delisting are understated."
            )
        if self.fundamentals and not self.point_in_time_fundamentals:
            out.append(
                "Fundamentals are not point-in-time: values may have been restated "
                "after the fact, which creates look-ahead bias."
            )
        if not self.historical_exchange:
            out.append(
                "Exchange listing is today's, not historical: NYSE breakpoints are "
                "approximate."
            )
        return out


@dataclass(frozen=True)
class Panel:
    freq: str
    lag_days: int
    signal_dates: pd.DatetimeIndex
    exec_dates: pd.DatetimeIndex
    fields: dict[str, pd.DataFrame]
    nyse: pd.Series
    source: str
    capabilities: Capabilities

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self.fields:
            raise KeyError(f"Panel has no field {name!r}. Available: {sorted(self.fields)}")
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    @property
    def ids(self) -> pd.Index:
        return next(iter(self.fields.values())).columns

    def truncate(self, end: pd.Timestamp) -> "Panel":
        """Keep only periods with signal date <= end. Used to test for look-ahead."""
        mask = self.signal_dates <= pd.Timestamp(end)
        fields = {k: v.loc[mask] for k, v in self.fields.items()}
        return replace(
            self,
            signal_dates=self.signal_dates[mask],
            exec_dates=self.exec_dates[mask],
            fields=fields,
        )


class DataProvider(ABC):
    name: str

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def panel(self, freq: str = "monthly", lag_days: int = 1) -> Panel:
        """Build (or load from cache) the panel for a rebalance frequency and execution lag."""

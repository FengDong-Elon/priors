"""Stock-level signals (ARCHITECTURE §10.3).

Convention: for every signal, **higher values are better** (more attractive to
buy). Signals where the literature buys the low end (volatility, beta, size,
asset growth, last month's return) are negated.

Two kinds:

* ``price``: computed from the price panel; available historically, so they
  can be backtested (with the data source's survivorship caveats).
* ``fundamental``: computed from the latest statements; available only for the
  current holdings list unless the data source has point-in-time fundamentals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

from ..data.base import Panel
from ..engine.signals import past_return


@dataclass(frozen=True)
class StockSignal:
    name: str
    kind: Literal["price", "fundamental"]
    description: str
    source: str
    history: Callable[[Panel], pd.DataFrame] | None = None
    current: Callable[[pd.DataFrame, pd.Series], pd.Series] | None = None   # (fundamentals, price) -> values


def _fund(col_fn):
    return lambda f, price: col_fn(f, price).replace([np.inf, -np.inf], np.nan)


SIGNALS: dict[str, StockSignal] = {s.name: s for s in [
    StockSignal("mom_12_1", "price", "Return from 12 months ago to 1 month ago (skips the latest month).",
                "Jegadeesh & Titman (1993)", history=lambda p: past_return(p, 12, 1)),
    StockSignal("st_reversal", "price", "Last month's return, negated: last month's losers rank highest.",
                "Jegadeesh (1990)", history=lambda p: -p["ret_past"]),
    StockSignal("low_vol", "price", "Average daily volatility over the past 12 months, negated.",
                "Ang, Hodrick, Xing & Zhang (2006)",
                history=lambda p: -p["volatility"].rolling(12, min_periods=6).mean()),
    StockSignal("low_beta", "price", "Market beta from the past year of daily returns, negated.",
                "Frazzini & Pedersen (2014)", history=lambda p: -p["beta"]),
    StockSignal("high52", "price", "Price relative to its 52-week high: stocks near their high rank highest.",
                "George & Hwang (2004)", history=lambda p: p["high52"]),
    StockSignal("size_small", "price", "Market capitalization (log), negated: smaller firms rank highest.",
                "Banz (1981)", history=lambda p: -np.log(p["mcap"].where(p["mcap"] > 0))),
    StockSignal("book_to_market", "fundamental", "Book value per share divided by price (negative book excluded).",
                "Fama & French (1992)",
                current=_fund(lambda f, px: (f["book_value_ps"] / px).where(f["book_value_ps"] > 0))),
    StockSignal("earnings_yield", "fundamental", "Trailing earnings per share divided by price.",
                "Basu (1977)", current=_fund(lambda f, px: f["trailing_eps"] / px)),
    StockSignal("roe", "fundamental", "Return on equity (trailing twelve months).",
                "Hou, Xue & Zhang (2015)", current=_fund(lambda f, px: f["roe"])),
    StockSignal("gross_profitability", "fundamental", "Gross profit divided by total assets (latest fiscal year).",
                "Novy-Marx (2013)", current=_fund(lambda f, px: f["gross_profit"] / f["total_assets"])),
    StockSignal("asset_growth_low", "fundamental", "Growth in total assets over the latest fiscal year, negated.",
                "Cooper, Gulen & Schill (2008)",
                current=_fund(lambda f, px: -(f["total_assets"] / f["total_assets_prev"] - 1))),
]}

# Default stock-level signal for each factor in the library (None: factor layer only).
FACTOR_SIGNAL: dict[str, str] = {
    "UMD": "mom_12_1", "Mom12m": "mom_12_1",
    "STreversal": "st_reversal",
    "IdioVol3F": "low_vol", "RealizedVol": "low_vol",
    "BetaFP": "low_beta", "Beta": "low_beta",
    "High52": "high52",
    "SMB": "size_small", "SMB_FF3": "size_small", "Size": "size_small", "Q_ME": "size_small",
    "HML": "book_to_market", "BM": "book_to_market", "BMdec": "book_to_market",
    "EP": "earnings_yield",
    "RMW": "roe", "RoE": "roe", "Q_ROE": "roe",
    "GP": "gross_profitability",
    "CMA": "asset_growth_low", "AssetGrowth": "asset_growth_low", "Q_IA": "asset_growth_low",
}


def default_signal(factor: str) -> str | None:
    return FACTOR_SIGNAL.get(factor.removesuffix("_VW"))


def resolve(name: str) -> StockSignal:
    if name not in SIGNALS:
        raise KeyError(f"Unknown stock signal {name!r}. Available: {sorted(SIGNALS)}")
    return SIGNALS[name]

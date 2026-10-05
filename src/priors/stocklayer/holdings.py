"""Current holdings list (ARCHITECTURE §10.6).

The latest portfolio under the student's rules: price signals from the latest
month-end in the price panel, fundamental signals from the latest statements.
Survivorship bias does not affect a list built from today's data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.base import Panel
from ..spec.strategy import StrategySpec
from .construct import composite_score, eligible, top_n_weights
from .signals import resolve

LABEL = "Mechanical output of your rules, for education. Not a recommendation to buy or sell any security."


@dataclass
class Holdings:
    as_of: str                       # signal date of the price data
    table: pd.DataFrame              # ticker-indexed: name, sector, weight, score, one rank column per signal
    sector_weights: pd.Series
    summary: dict
    universe_label: str
    label: str = LABEL
    notes: list[str] = field(default_factory=list)


def current_holdings(spec: StrategySpec, panel: Panel, universe: pd.DataFrame,
                     fundamentals: pd.DataFrame | None = None, breakpoints: pd.DataFrame | None = None,
                     universe_mask: pd.DataFrame | None = None) -> Holdings:
    sl = spec.stock_layer
    last = panel.signal_dates[-1:]
    mask = eligible(panel, sl.filters, breakpoints, universe_mask).loc[last]
    price = panel["price"].loc[last].iloc[0]
    signals, notes = {}, []
    for c in spec.components:
        if c.stock_signal is None:
            notes.append(f"{c.factor} has no stock-level signal and does not affect the holdings.")
            continue
        s = resolve(c.stock_signal)
        if s.kind == "price":
            try:
                signals[c.stock_signal] = s.history(panel).loc[last]
            except KeyError as e:
                raise ValueError(f"{c.stock_signal} needs data that {panel.source} does not provide: {e}") from e
        else:
            if fundamentals is None:
                raise ValueError(
                    f"{c.stock_signal} needs company financial data, which this installation does not have (a hosted "
                    "deployment downloads prices only). The factor-layer analysis is unaffected. Running `priors refresh` "
                    "on a computer adds financial data (it takes about half an hour)."
                )
            vals = s.current(fundamentals.reindex(price.index), price)
            signals[c.stock_signal] = pd.DataFrame([vals.to_numpy()], index=last, columns=price.index)
    if not signals:
        raise ValueError("No component has a stock-level signal.")

    score, count = composite_score(signals, mask)
    w = top_n_weights(score, sl.holdings, sl.weighting, panel["volatility"].loc[last]).iloc[0]
    held = w[w > 0].index

    meta = universe.set_index("ticker").reindex(held)
    ranks = {name: s.loc[last].where(mask).rank(axis=1, pct=True).iloc[0].reindex(held) for name, s in signals.items()}
    table = pd.DataFrame({
        "name": meta["name"], "sector": meta["sector"], "weight": w[held],
        "score": score.iloc[0].reindex(held),
        **{f"rank_{k}": v for k, v in ranks.items()},
        "mcap": panel["mcap"].loc[last].iloc[0].reindex(held),
    }).sort_values("score", ascending=False)

    sectors = table.groupby(table["sector"].fillna("Unknown"))["weight"].sum().sort_values(ascending=False)
    n_elig = int(mask.iloc[0].sum())
    scored = int(score.iloc[0].notna().sum())
    coverage = {k: int(v.where(mask).iloc[0].notna().sum()) for k, v in signals.items()}
    summary = {
        "holdings": len(held), "eligible_stocks": n_elig, "scored_stocks": scored,
        "signal_coverage": coverage,
        "mean_mcap": float(table["mcap"].mean()), "median_mcap": float(table["mcap"].median()),
        "top_sector": sectors.index[0], "top_sector_weight": float(sectors.iloc[0]),
    }
    for k, v in coverage.items():
        if n_elig and v / n_elig < 0.8:
            notes.append(f"Only {v} of {n_elig} eligible stocks have a value for {k}.")
    if sectors.iloc[0] > 0.4:
        notes.append(f"{sectors.index[0]} is {sectors.iloc[0]:.0%} of the portfolio: a concentrated sector bet.")
    return Holdings(
        as_of=str(last[0].date()), table=table, sector_weights=sectors, summary=summary,
        universe_label=universe.attrs.get("label", ""), notes=notes,
    )

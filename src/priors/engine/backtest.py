"""Deterministic, vectorized cross-sectional backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import __version__
from ..data.base import DataProvider, Panel
from ..spec.stock import StockSpec
from .metrics import PERIODS_PER_YEAR, summarize
from .portfolio import eligible, weights
from .signals import compute_signal


@dataclass
class BacktestResult:
    spec: StockSpec
    returns: pd.DataFrame          # gross, cost, net, long, [short], benchmark
    turnover: pd.Series            # one-way turnover per period, averaged across legs
    n_long: pd.Series
    n_short: pd.Series | None
    weights_long: pd.DataFrame
    weights_short: pd.DataFrame | None
    periods_per_year: int
    missing_return_share: float    # share of held positions with no next-period return (set to 0)
    warnings: list[str] = field(default_factory=list)
    source: str = ""
    engine_version: str = __version__

    @property
    def metrics(self) -> dict[str, dict[str, float]]:
        return {c: summarize(self.returns[c], self.periods_per_year) for c in self.returns.columns if c != "cost"}

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.metrics)
        extra = pd.Series(
            {"turnover_ann": self.turnover.mean() * self.periods_per_year,
             "avg_holdings_long": self.n_long.mean(),
             "cost_ann": self.returns["cost"].mean() * self.periods_per_year},
            name="net",
        )
        return pd.concat([df, extra.to_frame()]).loc[:, list(df.columns)]


def run_backtest(spec: StockSpec, data: DataProvider | Panel) -> BacktestResult:
    panel = data if isinstance(data, Panel) else data.panel(spec.rebalance, spec.execution.lag_days)
    if panel.freq != spec.rebalance:
        raise ValueError(f"Panel frequency {panel.freq!r} does not match spec rebalance {spec.rebalance!r}")

    # Signals and weights use the full history, so the sample start does not
    # truncate lookback windows. Each date only sees data up to itself.
    signal = compute_signal(panel, spec.signal.name, spec.signal.params)
    mask = eligible(panel, spec.universe)
    w_long, w_short = weights(panel, signal, mask, spec.portfolio, spec.signal.direction)

    ret = panel["ret_next"]
    active = (w_long.sum(axis=1) > 0) & ret.notna().any(axis=1)
    if w_short is not None:
        active &= w_short.sum(axis=1) > 0
    if spec.sample.start:
        active &= panel.signal_dates >= pd.Timestamp(spec.sample.start)
    if spec.sample.end:
        active &= panel.signal_dates <= pd.Timestamp(spec.sample.end)
    if active.sum() < 2:
        raise ValueError("Fewer than 2 active periods: check the sample window and signal warm-up.")

    ret, w_long = ret.loc[active], w_long.loc[active]
    w_short = w_short.loc[active] if w_short is not None else None

    held = w_long > 0 if w_short is None else (w_long > 0) | (w_short > 0)
    missing_share = float((held & ret.isna()).to_numpy().sum() / max(held.to_numpy().sum(), 1))
    r = ret.fillna(0.0)

    long_ret = (w_long * r).sum(axis=1)
    traded = _traded(w_long, r)
    cols = {"long": long_ret}
    gross = long_ret
    if w_short is not None:
        short_ret = (w_short * r).sum(axis=1)
        traded = traded + _traded(w_short, r)
        cols["short"] = short_ret
        gross = long_ret - short_ret

    n_legs = 1 if w_short is None else 2
    cost = traded * spec.execution.cost_bps_one_way / 1e4
    returns = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, **cols})
    returns["benchmark"] = _market(panel, active)

    warnings = list(panel.capabilities.warnings())
    if missing_share > 0.001:
        warnings.append(
            f"{missing_share:.2%} of held positions had no next-period price and were assigned a 0% return."
        )
    if spec.portfolio.side == "long_only":
        warnings.append("Risk-free rate not subtracted: Sharpe ratios for long-only returns are overstated.")

    return BacktestResult(
        spec=spec,
        returns=returns,
        turnover=traded / 2 / n_legs,
        n_long=(w_long > 0).sum(axis=1),
        n_short=(w_short > 0).sum(axis=1) if w_short is not None else None,
        weights_long=w_long,
        weights_short=w_short,
        periods_per_year=PERIODS_PER_YEAR[spec.rebalance],
        missing_return_share=missing_share,
        warnings=warnings,
        source=panel.source,
    )


def _traded(w: pd.DataFrame, r: pd.DataFrame) -> pd.Series:
    """Sum of |trade| per period: target weights minus drifted weights from the previous period."""
    grown = w * (1 + r)
    drifted = grown.div(grown.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    prev = drifted.shift(1).fillna(0.0)
    return (w - prev).abs().sum(axis=1)


def _market(panel: Panel, active: pd.Series) -> pd.Series:
    """Value-weighted return of every stock with a market cap and a next-period return."""
    mcap = panel["mcap"].loc[active]
    ret = panel["ret_next"].loc[active]
    ok = mcap.notna() & ret.notna() & (mcap > 0)
    w = mcap.where(ok)
    return (w * ret).sum(axis=1) / w.sum(axis=1)

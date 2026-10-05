"""Stock-layer backtest (ARCHITECTURE §10.5, §10.7).

Long-only, top-N portfolio rebuilt monthly or quarterly from the composite of
the components' stock signals. Signals are known at the close of the signal
date; trades happen ``lag_days`` later. Between quarterly rebalances, positions
drift with prices and are not traded.

Only price signals can be backtested on a source without point-in-time
fundamentals. The sealed holdout applies: by default the backtest ends before
the holdout starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import __version__
from ..data.base import Panel
from ..engine.metrics import summarize
from ..holdout import holdout_start as default_holdout_start
from ..spec.strategy import StrategySpec
from .construct import composite_score, eligible, top_n_weights
from .signals import resolve

WARMUP_MONTHS = 12


class FundamentalHistoryUnavailable(ValueError):
    pass


MCAP_NOTE = (
    "Market caps are approximate: today's share count times the historical split-adjusted price. "
    "Share issuance and buybacks are ignored, which affects the size filter and the size signal."
)


@dataclass
class StockResult:
    spec: StrategySpec
    returns: pd.DataFrame          # by return month: gross, cost, net, benchmark (EW eligible universe), excess
    weights: pd.DataFrame          # by signal date
    turnover: pd.Series            # one-way turnover per month
    n_holdings: pd.Series
    holding_mcap: pd.DataFrame     # mean and median market cap of holdings per month
    signals_used: list[str]
    sample: dict
    holdout_start: pd.Period
    warnings: list[str] = field(default_factory=list)
    source: str = ""
    engine_version: str = __version__

    @property
    def metrics(self) -> dict[str, dict[str, float]]:
        return {c: summarize(self.returns[c], 12) for c in ("gross", "net", "benchmark", "excess")}

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.metrics)
        extra = pd.DataFrame({
            "net": {"turnover_ann": self.turnover.mean() * 12, "cost_ann": self.returns["cost"].mean() * 12,
                    "avg_holdings": self.n_holdings.mean(),
                    "median_holding_mcap_bn": self.holding_mcap["median"].median() / 1e9},
        })
        return pd.concat([df, extra])


def stock_signals(spec: StrategySpec, panel: Panel) -> tuple[dict[str, pd.DataFrame], list[str]]:
    used, out, warnings = [], {}, []
    for c in spec.components:
        if c.stock_signal is None:
            warnings.append(f"{c.factor} has no stock-level signal and is used at the factor layer only.")
            continue
        s = resolve(c.stock_signal)
        if s.kind == "fundamental" and not panel.capabilities.point_in_time_fundamentals:
            raise FundamentalHistoryUnavailable(
                f"{c.stock_signal} ({c.factor}) is a fundamental signal. {panel.source} has no point-in-time "
                "fundamentals, so it cannot be backtested here; it is used for the current holdings list only. "
                "A data source with point-in-time fundamentals (for example Sharadar) removes this limit."
            )
        if s.history is None:
            raise FundamentalHistoryUnavailable(
                f"{c.stock_signal} ({c.factor}) cannot be backtested yet: monthly point-in-time fundamental fields "
                "are not built for any data source in this version. It is used for the current holdings list only."
            )
        try:
            out[c.stock_signal] = s.history(panel)
        except KeyError as e:
            raise ValueError(f"{c.stock_signal} needs data that {panel.source} does not provide: {e}") from e
        used.append(c.stock_signal)
    if not out:
        raise ValueError("No component has a stock-level signal.")
    return out, warnings


def _drift(w: pd.DataFrame, r: pd.DataFrame, rebalance: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Held weights each month and one-way turnover, trading only in rebalance months."""
    held = np.zeros(w.shape)
    turnover = np.zeros(len(w))
    prev_end = np.zeros(w.shape[1])
    W, R = w.to_numpy(), r.fillna(0.0).to_numpy()
    for i in range(len(w)):
        target = W[i] if rebalance.iloc[i] else prev_end
        turnover[i] = np.abs(target - prev_end).sum() / 2
        held[i] = target
        grown = target * (1 + R[i])
        prev_end = grown / grown.sum() if grown.sum() > 0 else np.zeros_like(grown)
    return pd.DataFrame(held, index=w.index, columns=w.columns), pd.Series(turnover, index=w.index)


def run_stock_backtest(spec: StrategySpec, panel: Panel, breakpoints: pd.DataFrame | None = None,
                       holdout_from: pd.Period | str | None = None,
                       universe_mask: pd.DataFrame | None = None) -> StockResult:
    sl = spec.stock_layer
    hs = pd.Period(holdout_from, "M") if holdout_from is not None else default_holdout_start()
    signals, warnings = stock_signals(spec, panel)
    mask = eligible(panel, sl.filters, breakpoints, universe_mask)
    score, _ = composite_score(signals, mask)
    target = top_n_weights(score, sl.holdings, sl.weighting, panel["volatility"])

    # A portfolio formed at the end of month k earns its return in month k + 1.
    return_month = pd.Series(panel.signal_dates.to_period("M") + 1, index=panel.signal_dates)
    active = (target.sum(axis=1) > 0) & (return_month < hs)
    active.iloc[:WARMUP_MONTHS] = False
    idx = active[active].index
    if len(idx) < 24:
        raise ValueError(f"Only {len(idx)} months with a full portfolio before the holdout; at least 24 are needed.")
    idx = target.loc[idx[0]: idx[-1]].index      # contiguous window

    r = panel["ret_next"].loc[idx]
    reb = pd.Series(True, index=idx)
    if sl.rebalance == "quarterly":
        reb = pd.Series(idx.month.isin([3, 6, 9, 12]), index=idx)
        reb.iloc[0] = True
    held, turnover = _drift(target.loc[idx], r, reb)

    gross = (held * r.fillna(0.0)).sum(axis=1)
    cost = turnover * 2 * sl.cost_bps_one_way / 1e4      # buys and sells both pay the one-way cost
    elig = mask.loc[idx] & r.notna()
    bench = r.where(elig).mean(axis=1)
    net = gross - cost
    returns = pd.DataFrame({"gross": gross, "cost": cost, "net": net, "benchmark": bench, "excess": net - bench})
    returns.index = pd.PeriodIndex(return_month.loc[idx].values, name="month")

    mc = panel["mcap"].loc[idx].where(held > 0)
    holding_mcap = pd.DataFrame({"mean": mc.mean(axis=1), "median": mc.median(axis=1)})

    # Only the data limitations that affect this backtest: no fundamentals are used here, and the
    # microcap cutoff comes from Ken French's NYSE breakpoints, not from listing exchanges.
    relevant = [w for w in panel.capabilities.warnings()
                if not w.startswith(("Fundamentals are not point-in-time", "Exchange listing"))]
    warnings = relevant + warnings
    if panel.source == "yfinance":
        warnings.append(MCAP_NOTE)
    warnings.append(
        "Long-only returns include the market's return. Compare with the benchmark (an equal-weighted "
        "portfolio of every eligible stock) and look at the excess return."
    )
    if not sl.filters.exclude_microcaps:
        warnings.append(
            "Portfolio includes microcaps; the cost assumption may substantially understate real trading costs."
        )
    missing = float((held > 0).where(r.isna(), False).to_numpy().sum() / max((held > 0).to_numpy().sum(), 1))
    if missing > 0.001:
        warnings.append(f"{missing:.2%} of held positions had no next-month return and were counted as 0%.")

    return StockResult(
        spec=spec, returns=returns, weights=held, turnover=turnover, n_holdings=(held > 0).sum(axis=1),
        holding_mcap=holding_mcap, signals_used=list(signals), holdout_start=hs, warnings=warnings,
        sample={"start": returns.index[0], "end": returns.index[-1], "months": len(idx)},
        source=panel.source,
    )

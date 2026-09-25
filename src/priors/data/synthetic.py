"""Synthetic data with a known, planted effect. Used by tests and demos."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .base import Capabilities, DataProvider, Panel


class SyntheticProvider(DataProvider):
    """Stock returns with an optional planted momentum effect.

    Each period, a stock's next return is
    ``market + momentum_strength * zscore(past 11-period return, skipping 1) + noise``.
    With ``momentum_strength=0`` there is no effect to find.
    """

    name = "synthetic"

    def __init__(self, n_stocks: int = 400, n_periods: int = 240, momentum_strength: float = 0.004,
                 noise: float = 0.08, seed: int = 0, start: str = "2000-01-31"):
        self.n_stocks = n_stocks
        self.n_periods = n_periods
        self.momentum_strength = momentum_strength
        self.noise = noise
        self.seed = seed
        self.start = start

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            universe="synthetic",
            history_start=date.fromisoformat(self.start),
            includes_delisted=True,
            delisting_returns=True,
            point_in_time_fundamentals=True,
            fundamentals=False,
            institutional_holdings=False,
            historical_exchange=True,
        )

    def panel(self, freq: str = "monthly", lag_days: int = 0) -> Panel:
        if freq != "monthly":
            raise ValueError("SyntheticProvider only supports monthly data")
        rng = np.random.default_rng(self.seed)
        n, t = self.n_stocks, self.n_periods
        r = np.full((t, n), np.nan)  # r[k] = return over period k (ending at signal date k)
        for k in range(t):
            market = rng.normal(0.006, 0.04)
            mom = np.zeros(n)
            if k >= 12:
                mom = np.log1p(r[k - 12:k - 1]).sum(axis=0)
                sd = mom.std()
                mom = (mom - mom.mean()) / sd if sd > 0 else np.zeros(n)
            r[k] = market + self.momentum_strength * mom + rng.normal(0, self.noise, n)
        ids = [f"S{i:04d}" for i in range(n)]
        dates = pd.date_range(self.start, periods=t, freq="ME")
        ret_past = pd.DataFrame(r, index=dates, columns=ids)
        # ret_next[k] is next period's return: with lag 0, holdings formed at k earn r[k+1].
        ret_next = ret_past.shift(-1)
        mcap0 = np.exp(rng.normal(21, 1.5, n))
        mcap = pd.DataFrame(mcap0 * np.cumprod(1 + r, axis=0), index=dates, columns=ids)
        price = pd.DataFrame(20 * np.cumprod(1 + r, axis=0), index=dates, columns=ids)
        vol = pd.DataFrame(np.abs(rng.normal(0.02, 0.005, (t, n))), index=dates, columns=ids)
        fields = {
            "ret_past": ret_past,
            "ret_next": ret_next,
            "price": price,
            "mcap": mcap,
            "dollar_volume": mcap * 0.004,
            "volatility": vol,
        }
        nyse = pd.Series(rng.random(n) < 0.4, index=ids)
        return Panel(freq=freq, lag_days=0, signal_dates=dates, exec_dates=dates, fields=fields,
                     nyse=nyse, source=self.name, capabilities=self.capabilities)

"""Factor-layer combination weights (ARCHITECTURE §9.2).

Every function returns a DataFrame of weights (months x components). The weight
in month t may use returns up to month t-1 only; the composite return in month t
is the weighted sum of the components' month-t returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MIN_VOL_MONTHS = 12


def equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    k = returns.shape[1]
    return pd.DataFrame(1.0 / k, index=returns.index, columns=returns.columns)


def fixed_weights(returns: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    row = pd.Series(weights, dtype=float).reindex(returns.columns)
    if row.isna().any():
        raise ValueError(f"missing weights for {list(row.index[row.isna()])}")
    return pd.DataFrame(np.tile(row.to_numpy(), (len(returns), 1)), index=returns.index, columns=returns.columns)


def inverse_vol(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Weights proportional to 1 / trailing volatility, estimated from months t-lookback..t-1.

    Months with fewer than MIN_VOL_MONTHS of history get no weights (NaN rows).
    """
    vol = returns.rolling(lookback, min_periods=MIN_VOL_MONTHS).std().shift(1)
    inv = 1.0 / vol
    w = inv.div(inv.sum(axis=1), axis=0)
    w[vol.isna().any(axis=1)] = np.nan
    return w


def max_sharpe_long_only(returns: pd.DataFrame, ridge: float = 1e-6) -> pd.Series:
    """Long-only weights (summing to 1) that maximize the in-sample Sharpe ratio of ``returns``."""
    k = returns.shape[1]
    if k == 1:
        return pd.Series(1.0, index=returns.columns)
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy() + ridge * np.eye(k)

    def neg_sharpe(w: np.ndarray) -> float:
        return -(w @ mu) / np.sqrt(w @ cov @ w)

    res = minimize(
        neg_sharpe,
        x0=np.full(k, 1.0 / k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"weight optimization failed: {res.message}")
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=returns.columns)

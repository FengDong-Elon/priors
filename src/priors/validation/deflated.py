"""Multiple-testing adjustments (ARCHITECTURE §7.2).

* Probabilistic Sharpe ratio (PSR) and deflated Sharpe ratio (DSR), from
  Bailey & López de Prado (2014), "The Deflated Sharpe Ratio," Journal of
  Portfolio Management, doi:10.3905/jpm.2014.40.5.094.
* The Harvey, Liu & Zhu (2016) t > 3.0 hurdle, shown for reference.

All Sharpe ratios inside the formulas are per period (monthly), not annualized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..engine.metrics import newey_west_t

EULER_GAMMA = 0.5772156649015329
HLZ_T = 3.0


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum per-period Sharpe among n_trials unskilled strategies (the DSR benchmark SR0)."""
    if n_trials < 2 or var_sharpe <= 0:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(var_sharpe) * (
        (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e))
    )


def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark_sr), adjusting for sample length, skewness, and kurtosis."""
    x = r.dropna().to_numpy(dtype=float)
    t = len(x)
    sr = x.mean() / x.std(ddof=1)
    skew = pd.Series(x).skew()
    kurt = pd.Series(x).kurt() + 3  # non-excess kurtosis
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - benchmark_sr) * math.sqrt(t - 1) / math.sqrt(denom)))


@dataclass
class DeflatedSharpe:
    sharpe_ann: float
    n_trials: int
    trial_sharpes_ann: list[float]
    sr0_ann: float                   # Sharpe the best of n_trials unskilled strategies would reach
    psr: float                       # P(true Sharpe > 0)
    dsr: float                       # P(true Sharpe > sr0)
    t_stat_nw: float
    passes_hlz: bool
    flags: list[str] = field(default_factory=list)


def deflated_sharpe(r: pd.Series, trial_sharpes_ann: list[float], periods_per_year: int = 12) -> DeflatedSharpe:
    """DSR of ``r``, given the annualized Sharpe ratios of every trial in the ledger (including this one)."""
    x = r.dropna()
    root = math.sqrt(periods_per_year)
    sr_ann = float(x.mean() / x.std(ddof=1) * root)
    trials = [s for s in trial_sharpes_ann if s is not None and np.isfinite(s)] or [sr_ann]
    n = len(trials)
    var = float(np.var(np.array(trials) / root, ddof=1)) if n > 1 else 0.0
    sr0 = expected_max_sharpe(n, var)
    psr = probabilistic_sharpe(x, 0.0)
    dsr = probabilistic_sharpe(x, sr0)
    t = newey_west_t(x)

    flags = []
    if n > 1:
        flags.append(
            f"{n} different specifications have been tested in this project. The best of {n} strategies with "
            f"no real edge would be expected to reach a Sharpe ratio of about {sr0 * root:.2f} by luck alone."
        )
    if dsr < 0.95:
        flags.append(
            f"After accounting for the number of trials, the probability that the true Sharpe ratio exceeds "
            f"that luck benchmark is {dsr:.0%}, below the conventional 95%."
        )
    if not abs(t) >= HLZ_T:
        flags.append(f"The Newey-West t-statistic is {t:.2f}, below the Harvey-Liu-Zhu (2016) hurdle of 3.0.")
    return DeflatedSharpe(
        sharpe_ann=sr_ann, n_trials=n, trial_sharpes_ann=trials, sr0_ann=sr0 * root,
        psr=psr, dsr=dsr, t_stat_nw=t, passes_hlz=abs(t) >= HLZ_T, flags=flags,
    )

"""Random-factor placebo (ARCHITECTURE §7.3).

Question: is the student's combination better than combining *published*
factors at random? Each draw picks k Chen-Zimmermann predictors (k = number of
components), combines them with the same method, and computes the Sharpe ratio
over the same window. The student's percentile among the draws is reported.

A second view, "best of N," shows how good the *best* of N random combinations
looks as N grows: it illustrates what trying many combinations and keeping the
winner does to an in-sample Sharpe ratio.

The pool matches the student's construction. If every component is
value-weighted (Ken French, Hou-Xue-Zhang, or ``_VW`` series), the pool is the
value-weighted decile versions of the published predictors. Otherwise it is the
predictors as built in the original papers, most of which are equal-weighted
and earn higher gross Sharpe ratios.

Only predictors with data in every month of the window are eligible. If the
student's window starts before enough predictors exist, the placebo window
starts later, and the student's Sharpe is recomputed on that window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..composite import FactorResult
from ..composite.weights import MIN_VOL_MONTHS
from ..factors import FactorLibrary

MIN_POOL = 100
N_DRAWS = 1000
BEST_OF = (1, 2, 5, 10, 20, 50, 100, 200)


@dataclass
class Placebo:
    window_start: str
    window_end: str
    months: int
    pool_size: int
    pool: str                             # "value-weighted" or "original construction"
    k: int
    method: str
    student_sharpe: float
    placebo_sharpes: np.ndarray
    percentile: float                     # share of draws with a lower Sharpe than the student's
    best_of_n: dict[int, float]           # N -> expected best Sharpe among N random combinations
    seed: int
    flags: list[str] = field(default_factory=list)


def _sharpe(x: np.ndarray) -> np.ndarray:
    """Annualized Sharpe of each column."""
    return x.mean(axis=0) / x.std(axis=0, ddof=1) * math.sqrt(12)


def _window(pool: pd.DataFrame, start: pd.Period, end: pd.Period) -> pd.Period:
    """Earliest start >= ``start`` at which at least MIN_POOL predictors cover every month to ``end``."""
    first = pool.apply(lambda s: s.first_valid_index())
    gaps = pool.loc[:end].isna()
    ok_from = {}
    for c in pool.columns:
        g = gaps[c]
        missing = g[g].index
        ok_from[c] = (missing.max() + 1) if len(missing) and missing.max() >= first[c] else first[c]
    starts = pd.Series(ok_from).sort_values()
    if len(starts) < MIN_POOL:
        raise ValueError("Fewer than MIN_POOL predictors in the library.")
    return max(start, starts.iloc[MIN_POOL - 1])


def random_factor_placebo(
    result: FactorResult, lib: FactorLibrary, n_draws: int = N_DRAWS, seed: int = 20261004,
) -> Placebo:
    spec = result.spec
    method = spec.factor_layer.combination
    k = len(spec.components)
    all_vw = all(lib.is_value_weighted(f) for f in spec.factor_ids)
    pool_source = "cz_vw" if all_vw and (lib.info["source"] == "cz_vw").any() else "cz"
    pool_label = "value-weighted" if pool_source == "cz_vw" else "original construction"
    own = {f.removesuffix("_VW") for f in spec.factor_ids}
    own |= {f + "_VW" for f in own}
    cz_ids = [i for i in lib.info.index[lib.info["source"] == pool_source] if i not in own]
    pool_all = lib.returns[cz_ids]
    start = _window(pool_all, result.sample["start"], result.sample["end"])
    end = result.sample["end"]
    pool = pool_all.loc[start:end].dropna(axis=1, how="any")
    m = pool.shape[1]
    if m < max(MIN_POOL, k):
        raise ValueError(f"Only {m} predictors cover {start} to {end}; the placebo needs at least {MIN_POOL}.")

    flags = []
    eff_method = method
    if method == "optimized":
        eff_method = "equal_weight"
        flags.append("Placebo draws use equal weights; re-optimizing 1,000 random combinations is not informative.")

    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(m, size=k, replace=False) for _ in range(n_draws)])
    R = pool.to_numpy()                                         # T x m
    if eff_method == "inverse_vol":
        lookback = spec.factor_layer.vol_lookback_months
        hist = pool_all[pool.columns].loc[: end]
        vol = hist.rolling(lookback, min_periods=MIN_VOL_MONTHS).std().shift(1).loc[start:end].to_numpy()
        inv = 1 / vol[:, draws]                                 # T x n_draws x k
        w = inv / inv.sum(axis=2, keepdims=True)
        comp = np.nansum(R[:, draws] * w, axis=2)
        valid = ~np.isnan(vol[:, draws]).any(axis=(1, 2))
        comp = comp[valid]
    elif eff_method == "theory_weights":
        wv = np.array([c.weight for c in spec.components])
        comp = (R[:, draws] * wv).sum(axis=2)
    else:
        comp = R[:, draws].mean(axis=2)
    sharpes = _sharpe(comp)

    student = result.returns["composite"].loc[start:end]
    if eff_method == "inverse_vol":
        student = student.iloc[-comp.shape[0]:]
    student_sr = float(student.mean() / student.std(ddof=1) * math.sqrt(12))
    pct = float((sharpes < student_sr).mean())

    best = {}
    for n in BEST_OF:
        if n <= n_draws:
            idx = rng.integers(0, n_draws, size=(2000, n))
            best[n] = float(sharpes[idx].max(axis=1).mean())

    if start > result.sample["start"]:
        flags.append(
            f"The placebo covers {start} to {end}, where at least {MIN_POOL} published predictors have data. "
            f"Your composite's Sharpe ratio on this window is {student_sr:.2f}."
        )
    flags.append(
        f"Your combination beats {pct:.0%} of {n_draws:,} random combinations of {k} published "
        f"predictor{'s' if k > 1 else ''} ({pool_label} versions, to match your construction)."
    )
    n_beaten = max((n for n, v in best.items() if v <= student_sr), default=None)
    if n_beaten is None:
        flags.append(
            "A single random combination of published predictors does about as well or better, on average."
        )
    elif n_beaten < max(best):
        nxt = min(n for n in best if n > n_beaten)
        flags.append(
            f"Picking the best of {nxt} random combinations would typically match or beat it: "
            "a high in-sample Sharpe ratio is easy to find by trying many combinations."
        )
    return Placebo(
        window_start=str(start), window_end=str(end), months=int(comp.shape[0]), pool_size=m, pool=pool_label, k=k,
        method=eff_method, student_sharpe=student_sr, placebo_sharpes=sharpes, percentile=pct,
        best_of_n=best, seed=seed, flags=flags,
    )

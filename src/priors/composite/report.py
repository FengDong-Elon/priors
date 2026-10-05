"""The multi-factor composite report (ARCHITECTURE §9.3, §9.4).

Every section is computed on the same months as the composite result. Subsets
of components (for leave-one-out and Shapley values) are recombined with the
student's method:

* equal weight and inverse volatility: recomputed on the subset;
* theory and optimized weights: the fixed weights, rescaled to sum to 1.

All contributions are in-sample and descriptive. Acting on them (dropping or
re-weighting a factor) creates a new specification, which the ledger counts as a
new trial and which can only be judged on the sealed holdout (§9.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..factors import FactorLibrary
from . import weights as W
from .engine import FactorResult

GUARD_RAIL = (
    "These contributions are measured in-sample. Dropping or re-weighting a factor because of them creates a new "
    "specification: the ledger counts it as a new trial, and only the sealed holdout can judge it."
)
MIN_SUBPERIOD = 60


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std(ddof=1) * math.sqrt(12)) if len(r) > 2 and r.std() > 0 else 0.0


def nw_lags(t: int) -> int:
    return int(math.floor(4 * (t / 100) ** (2 / 9)))


def ols_nw(y: pd.Series, X: pd.DataFrame) -> dict:
    """OLS with a constant and Newey-West standard errors. Alpha is annualized (x 12)."""
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    fit = sm.OLS(data["y"], sm.add_constant(data[X.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(data))}
    )
    return {
        "alpha_ann": float(fit.params["const"] * 12),
        "alpha_t": float(fit.tvalues["const"]),
        "betas": {c: float(fit.params[c]) for c in X.columns},
        "beta_t": {c: float(fit.tvalues[c]) for c in X.columns},
        "r2": float(fit.rsquared),
        "months": int(len(data)),
    }


class Recombiner:
    """Rebuilds the composite for any subset of the components, on the result's months."""

    def __init__(self, result: FactorResult, lib: FactorLibrary):
        self.result = result
        self.months = result.returns.index
        self.ids = result.spec.factor_ids
        self.method = result.spec.factor_layer.combination
        self.lookback = result.spec.factor_layer.vol_lookback_months
        self.history = lib.get(self.ids, how="common")       # for inverse-vol warm-up
        self.fixed = result.weights.iloc[0] if self.method in ("theory_weights", "optimized") else None
        self._cache: dict[frozenset, pd.Series] = {}

    def composite(self, subset) -> pd.Series:
        key = frozenset(subset)
        if key not in self._cache:
            cols = [c for c in self.ids if c in key]
            r = self.result.returns[cols]
            if self.method == "equal_weight":
                w = W.equal_weight(r)
            elif self.method == "inverse_vol":
                w = W.inverse_vol(self.history[cols], self.lookback).loc[self.months]
            else:
                f = self.fixed[cols]
                w = W.fixed_weights(r, (f / f.sum()).to_dict()) if f.sum() > 0 else W.equal_weight(r)
            self._cache[key] = (w * r).sum(axis=1)
        return self._cache[key]

    def sharpe(self, subset) -> float:
        return _sharpe(self.composite(subset)) if subset else 0.0


def shapley(values: dict[frozenset, float], players: list[str]) -> dict[str, float]:
    """Exact Shapley values for a value function given on every subset (empty set included)."""
    k = len(players)
    phi = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for size in range(k):
            wgt = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for s in combinations(others, size):
                s = frozenset(s)
                phi[p] += wgt * (values[s | {p}] - values[s])
    return phi


def shapley_sharpe(rc: Recombiner, months: pd.Index | None = None) -> dict[str, float]:
    players = rc.ids
    vals = {}
    for size in range(len(players) + 1):
        for s in combinations(players, size):
            s = frozenset(s)
            if not s:
                vals[s] = 0.0
            else:
                c = rc.composite(s)
                vals[s] = _sharpe(c.loc[months] if months is not None else c)
    return shapley(vals, players)


@dataclass
class CompositeReport:
    composite_sharpe: float
    correlation: pd.DataFrame
    leave_one_out: pd.DataFrame          # per factor: sharpe_without, change
    incremental_alpha: pd.DataFrame      # per factor: alpha vs the other components
    shapley: pd.Series                   # per factor: Sharpe contribution (sums to the composite Sharpe)
    shapley_share: pd.Series
    shapley_by_decade: pd.DataFrame
    risk_contribution: pd.Series         # shares of composite variance (sum to 1)
    priority: pd.DataFrame               # theory priority vs empirical rank
    flags: list[str] = field(default_factory=list)
    guard_rail: str = GUARD_RAIL


def composite_report(result: FactorResult, lib: FactorLibrary) -> CompositeReport:
    spec = result.spec
    ids = spec.factor_ids
    if len(ids) < 2:
        raise ValueError("The composite report needs at least two components.")
    rc = Recombiner(result, lib)
    r = result.returns[ids]
    full_sr = rc.sharpe(ids)

    corr = r.corr()

    loo = pd.DataFrame(
        {f: {"sharpe_without": rc.sharpe([g for g in ids if g != f])} for f in ids}
    ).T
    loo["change_if_removed"] = loo["sharpe_without"] - full_sr

    inc = {}
    for f in ids:
        others = [g for g in ids if g != f]
        fit = ols_nw(r[f], r[others])
        inc[f] = {"alpha_ann": fit["alpha_ann"], "alpha_t": fit["alpha_t"], "r2": fit["r2"]}
    inc = pd.DataFrame(inc).T

    phi = pd.Series(shapley_sharpe(rc), name="shapley_sharpe")
    share = (phi / phi.sum()).rename("share") if phi.sum() != 0 else phi * np.nan

    decades = (result.returns.index.year // 10) * 10
    by_dec = {}
    for d in sorted(set(decades)):
        m = result.returns.index[decades == d]
        if len(m) >= MIN_SUBPERIOD:
            by_dec[f"{d}s"] = shapley_sharpe(rc, m)
    by_dec = pd.DataFrame(by_dec).T

    w = result.weights.mean()
    cov = r.cov()
    marginal = cov.to_numpy() @ w.to_numpy()
    total = float(w.to_numpy() @ marginal)
    risk = pd.Series(w.to_numpy() * marginal / total, index=ids, name="risk_share")

    theory = pd.Series({c.factor: c.theory_priority for c in spec.components}, name="theory_priority")
    emp_rank = phi.rank(ascending=False, method="min").astype(int).rename("empirical_rank")
    prio = pd.concat([theory, emp_rank, share], axis=1)

    flags = [GUARD_RAIL]
    hi_corr = [(a, b, corr.loc[a, b]) for a, b in combinations(ids, 2) if corr.loc[a, b] > 0.7]
    for a, b, c in hi_corr:
        flags.append(f"{a} and {b} are highly correlated ({c:.2f}): they may be repeating the same bet.")
    for f in ids:
        if loo.loc[f, "change_if_removed"] > 0:
            flags.append(
                f"Removing {f} would raise the composite Sharpe ratio from {full_sr:.2f} to "
                f"{loo.loc[f, 'sharpe_without']:.2f} in this sample."
            )
        if abs(inc.loc[f, "alpha_t"]) < 2:
            flags.append(
                f"{f} adds little that the other components do not already capture "
                f"(alpha {inc.loc[f, 'alpha_ann']:.1%} per year, t = {inc.loc[f, 'alpha_t']:.2f})."
            )
    if theory.notna().all():
        top_theory = theory.idxmin()
        if share[top_theory] <= 1 / len(ids) / 2:
            flags.append(
                f"{top_theory} has the strongest theoretical support but contributes only "
                f"{share[top_theory]:.0%} of the composite Sharpe ratio. Possible reasons: it overlaps with "
                "other components, or its premium has decayed."
            )
        for f in ids:
            if theory[f] == theory.max() and theory.max() > theory.min() and share[f] >= 1.5 / len(ids):
                flags.append(
                    f"{f} contributes {share[f]:.0%} of the composite Sharpe ratio despite the weakest "
                    "theoretical support. Treat this in-sample contribution with caution."
                )
    return CompositeReport(
        composite_sharpe=full_sr, correlation=corr, leave_one_out=loo, incremental_alpha=inc,
        shapley=phi, shapley_share=share, shapley_by_decade=by_dec, risk_contribution=risk,
        priority=prio, flags=flags,
    )

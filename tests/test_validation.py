"""Validation suite: deflated Sharpe, benchmark check, random-factor placebo, stability."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from priors.composite import run_factor_backtest
from priors.factors import FactorLibrary
from priors.registry import Registry
from priors.spec import StrategySpec
from priors.validation import validate
from priors.validation.deflated import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe

HS = "2015-01"


def placebo_lib(seed: int = 3, n_pool: int = 120) -> FactorLibrary:
    """Two 'French' factors with a real premium plus a pool of noise predictors (half EW, half VW)."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1960-01", "2020-12", freq="M", name="month")
    n = len(idx)
    cols = {"GOOD1": 0.008 + 0.02 * rng.standard_normal(n), "GOOD2": 0.008 + 0.02 * rng.standard_normal(n)}
    for i in range(n_pool):
        cols[f"X{i}"] = 0.0 + 0.03 * rng.standard_normal(n)
        cols[f"X{i}_VW"] = 0.0 + 0.03 * rng.standard_normal(n)
    r = pd.DataFrame(cols, index=idx)
    info = pd.DataFrame(index=r.columns)
    info["source"] = ["french", "french"] + ["cz", "cz_vw"] * n_pool
    info["name"] = info.index
    info["year"] = 1993
    info["sample_start"] = [np.nan, np.nan] + [1965, 1965] * n_pool
    info["sample_end"] = [np.nan, np.nan] + [1985, 1985] * n_pool
    info["op_weighting"] = [np.nan, np.nan] + ["EW", "VW"] * n_pool
    return FactorLibrary(returns=r, rf=pd.Series(0.0, index=idx), info=info)


def spec(*ids):
    return StrategySpec.model_validate({"name": "t", "components": [{"factor": f} for f in ids]})


# ------------------------------------------------------- deflated Sharpe

def test_expected_max_sharpe_grows_with_trials():
    v = 0.01
    vals = [expected_max_sharpe(n, v) for n in (1, 2, 10, 100)]
    assert vals[0] == 0 and vals == sorted(vals)
    g = 0.5772156649015329
    hand = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / 10) + g * norm.ppf(1 - 1 / (10 * math.e)))
    assert vals[2] == pytest.approx(hand)


def test_psr_matches_formula_for_normal_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(0.01 + 0.04 * rng.standard_normal(600))
    x = r.to_numpy()
    sr = x.mean() / x.std(ddof=1)
    sk, ku = r.skew(), r.kurt() + 3
    hand = norm.cdf(sr * math.sqrt(599) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr**2))
    assert probabilistic_sharpe(r) == pytest.approx(hand)


def test_more_trials_lower_the_deflated_sharpe():
    rng = np.random.default_rng(1)
    r = pd.Series(0.006 + 0.04 * rng.standard_normal(480))
    one = deflated_sharpe(r, [])
    many = deflated_sharpe(r, list(rng.normal(0.3, 0.3, 50)))
    assert one.n_trials == 1 and one.sr0_ann == 0 and one.dsr == pytest.approx(one.psr)
    assert many.n_trials == 50 and many.sr0_ann > 0 and many.dsr < one.dsr
    assert any("50 different specifications" in f for f in many.flags)


# ----------------------------------------------------------- benchmark

def test_benchmark_flags_far_above_and_consistent_reference(lib):
    lib.info["year"] = 2000
    lib.info["source"] = "french"
    boosted = lib.returns.copy()
    boosted.loc["2000-01":, "A"] += 0.05           # implausibly strong after the reference window
    lib2 = FactorLibrary(returns=boosted, rf=lib.rf, info=lib.info)
    res = run_factor_backtest(spec("A", "B"), lib2, holdout_from=HS)
    from priors.validation.benchmark import benchmark_check
    b = benchmark_check(res, lib2)
    ref = [c for c in b.components if c.label.startswith("A: reference")][0]
    assert ref.verdict == "consistent"
    after = [c for c in b.components if c.label.startswith("A: after")][0]
    assert after.verdict == "far above"
    assert b.composite.verdict == "far above"
    assert any("look-ahead bias" in f for f in b.flags)


# -------------------------------------------------------------- placebo

def test_placebo_matches_construction_and_ranks_real_premium_high():
    lib = placebo_lib()
    res = run_factor_backtest(spec("GOOD1", "GOOD2"), lib, holdout_from=HS)
    v = validate(res, lib, placebo_draws=500)
    p = v.placebo
    assert p.pool == "value-weighted" and p.pool_size >= 100
    assert p.percentile > 0.99
    vals = list(p.best_of_n.values())
    assert vals == sorted(vals)                       # best-of-N rises with N
    assert v.errors == {}


def test_placebo_uses_original_pool_for_equal_weighted_components():
    lib = placebo_lib()
    res = run_factor_backtest(spec("GOOD1", "X0"), lib, holdout_from=HS)
    p = validate(res, lib, placebo_draws=200).placebo
    assert p.pool == "original construction"
    a = validate(res, lib, placebo_draws=200).placebo
    np.testing.assert_array_equal(a.placebo_sharpes, p.placebo_sharpes)   # seeded, reproducible


def test_placebo_window_starts_when_enough_predictors_exist():
    lib = placebo_lib()
    r = lib.returns.copy()
    r.loc[:"1969-12", [c for c in r.columns if c.endswith("_VW")][:60]] = np.nan
    lib2 = FactorLibrary(returns=r, rf=lib.rf, info=lib.info)
    res = run_factor_backtest(spec("GOOD1", "GOOD2"), lib2, holdout_from=HS)
    p = validate(res, lib2, placebo_draws=100).placebo
    assert p.window_start == "1970-01"
    assert any("The placebo covers 1970-01" in f for f in p.flags)


# ------------------------------------------------------------ stability

def test_stability_tables_and_decay_flag():
    lib = placebo_lib()
    r = lib.returns.copy()
    r.loc["1990-01":, ["GOOD1", "GOOD2"]] -= 0.008   # premium disappears in the second half
    lib2 = FactorLibrary(returns=r, rf=lib.rf, info=lib.info)
    res = run_factor_backtest(spec("GOOD1", "GOOD2"), lib2, holdout_from=HS)
    s = validate(res, lib2, placebo_draws=100).stability
    assert list(s.by_decade.index[:2]) == ["1960s", "1970s"]
    assert s.halves.loc["first half", "sharpe"] > s.halves.loc["second half", "sharpe"]
    assert any("fell from" in f for f in s.flags)
    assert len(s.rolling_sharpe) == res.sample["months"] - 35


def test_validate_reads_trial_count_from_ledger(tmp_path):
    lib = placebo_lib()
    reg = Registry(tmp_path / "p", require_evidence=False)
    for f in ("X1", "X2", "X3"):
        reg.run_factor(spec("GOOD1", f), lib)
    out = reg.run_factor(spec("GOOD1", "GOOD2"), lib)
    v = validate(out.result, lib, reg.ledger, placebo_draws=100)
    assert v.deflated.n_trials == 4 and v.deflated.sr0_ann > 0

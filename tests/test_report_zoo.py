"""Composite report (Shapley, leave-one-out, incremental alpha, risk) and factor-zoo positioning."""

import math

import numpy as np
import pandas as pd
import pytest

from priors.composite import run_factor_backtest
from priors.composite.report import composite_report, shapley
from priors.factors import FactorLibrary
from priors.spec import StrategySpec
from priors.zoo import factor_map, spanning, zoo_report

HS = "2015-01"


def zoo_lib(seed: int = 7) -> FactorLibrary:
    """Market factor, three 'families' of correlated predictors, and a redundant twin of A."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1960-01", "2020-12", freq="M", name="month")
    n = len(idx)
    mkt = 0.005 + 0.045 * rng.standard_normal(n)
    fam = {g: 0.03 * rng.standard_normal(n) for g in "XYZ"}
    cols = {"MKT_RF": mkt}
    cats = {"MKT_RF": None}
    for g, cat in zip("XYZ", ["momentum", "valuation", "profitability"]):
        for i in range(6):
            cols[f"{g}{i}"] = 0.004 + fam[g] + 0.012 * rng.standard_normal(n)
            cats[f"{g}{i}"] = cat
    cols["A"] = 0.006 + 0.02 * rng.standard_normal(n)
    cols["A_TWIN"] = cols["A"] + 0.002 * rng.standard_normal(n)        # nearly identical to A
    cols["B"] = 0.006 + 0.02 * rng.standard_normal(n)
    cats.update(A="other", A_TWIN="other", B="other")
    r = pd.DataFrame(cols, index=idx)
    info = pd.DataFrame(index=r.columns)
    info["source"] = ["french"] + ["cz"] * (len(r.columns) - 1)
    info["name"] = info.index
    info["year"] = 1990
    info["economic_category"] = [cats[c] for c in r.columns]
    return FactorLibrary(returns=r, rf=pd.Series(0.0, index=idx), info=info)


def spec(*ids, **fl):
    return StrategySpec.model_validate({"name": "t", "components": [{"factor": f} for f in ids], "factor_layer": fl})


def test_shapley_axioms_on_a_toy_game():
    players = ["a", "b", "c"]
    v = {}
    for mask in range(8):
        s = frozenset(p for i, p in enumerate(players) if mask >> i & 1)
        v[s] = 2.0 * len(s & {"a", "b"}) + (1.0 if "c" in s and "a" in s else 0.0)
    phi = shapley(v, players)
    assert sum(phi.values()) == pytest.approx(v[frozenset(players)])          # efficiency
    assert phi["a"] == pytest.approx(2.5) and phi["c"] == pytest.approx(0.5)   # interaction split evenly
    assert phi["b"] == pytest.approx(2.0)


def test_report_sections_are_consistent():
    lib = zoo_lib()
    res = run_factor_backtest(spec("A", "B", "X0"), lib, holdout_from=HS)
    rep = composite_report(res, lib)
    assert rep.shapley.sum() == pytest.approx(rep.composite_sharpe)
    assert rep.risk_contribution.sum() == pytest.approx(1.0)
    sr_without_a = (res.returns[["B", "X0"]].mean(axis=1))
    sr = sr_without_a.mean() / sr_without_a.std() * math.sqrt(12)
    assert rep.leave_one_out.loc["A", "sharpe_without"] == pytest.approx(sr)
    assert set(rep.shapley_by_decade.columns) == {"A", "B", "X0"}
    assert rep.flags[0].startswith("These contributions are measured in-sample")


def test_report_flags_redundant_and_correlated_factors():
    lib = zoo_lib()
    res = run_factor_backtest(spec("A", "A_TWIN", "B"), lib, holdout_from=HS)
    rep = composite_report(res, lib)
    text = " ".join(rep.flags)
    assert "A and A_TWIN are highly correlated" in text
    assert "adds little that the other components do not already capture" in text


def test_theory_vs_empirical_priority_flag():
    lib = zoo_lib()
    s = StrategySpec.model_validate({"name": "t", "components": [
        {"factor": "X0", "theory_priority": 1}, {"factor": "A", "theory_priority": 2}, {"factor": "B", "theory_priority": 3},
    ]})
    r = lib.returns.copy()
    r["X0"] -= 0.004                                      # strongest theory, no premium
    lib2 = FactorLibrary(returns=r, rf=lib.rf, info=lib.info)
    rep = composite_report(run_factor_backtest(s, lib2, holdout_from=HS), lib2)
    assert rep.priority.loc["X0", "theory_priority"] == 1
    assert any("X0 has the strongest theoretical support" in f for f in rep.flags)


def test_theory_weights_are_rescaled_for_subsets():
    lib = zoo_lib()
    s = StrategySpec.model_validate({"name": "t", "factor_layer": {"combination": "theory_weights"}, "components": [
        {"factor": "A", "weight": 0.6}, {"factor": "B", "weight": 0.3}, {"factor": "X0", "weight": 0.1}]})
    res = run_factor_backtest(s, lib, holdout_from=HS)
    rep = composite_report(res, lib)
    sub = (0.3 / 0.4) * res.returns["B"] + (0.1 / 0.4) * res.returns["X0"]
    assert rep.leave_one_out.loc["A", "sharpe_without"] == pytest.approx(sub.mean() / sub.std() * math.sqrt(12))


def test_spanning_recovers_alpha_and_handles_own_factors():
    lib = zoo_lib()
    r = lib.returns.copy()
    r["A"] = 0.003 + 0.5 * r["MKT_RF"] + 0.01 * np.random.default_rng(0).standard_normal(len(r))
    lib2 = FactorLibrary(returns=r, rf=lib.rf, info=lib.info)
    res = run_factor_backtest(spec("A", spanning_models=["CAPM"]), lib2, holdout_from=HS)
    (s,) = spanning(res, lib2)
    assert s.alpha_ann == pytest.approx(0.036, abs=0.01) and s.betas["MKT_RF"] == pytest.approx(0.5, abs=0.05)
    own = run_factor_backtest(spec("MKT_RF", spanning_models=["CAPM"]), lib2, holdout_from=HS)
    (o,) = spanning(own, lib2)
    assert math.isnan(o.alpha_t) and "zero by construction" in o.overlap_note


def test_factor_map_finds_the_right_family():
    lib = zoo_lib()
    res = run_factor_backtest(spec("X0", "X1", spanning_models=["CAPM"]), lib, holdout_from=HS)
    fm = factor_map(res, lib)
    assert fm.family == "momentum"
    assert set(fm.family_members) == {"X2", "X3", "X4", "X5"}          # own components excluded
    lone = factor_map(run_factor_backtest(spec("B", spanning_models=["CAPM"]), lib, holdout_from=HS), lib)
    assert lone.family == "none"


def test_zoo_report_runs_end_to_end():
    lib = zoo_lib()
    z = zoo_report(run_factor_backtest(spec("Y0", "A", spanning_models=["CAPM"]), lib, holdout_from=HS), lib)
    assert z.neighbors.index[0].startswith("Y") and "Descriptive only" in " ".join(z.flags)

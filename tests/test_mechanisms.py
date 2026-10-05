"""Mechanism test templates on synthetic data with planted effects."""

import numpy as np
import pandas as pd
import pytest

from priors.composite import run_factor_backtest
from priors.factors import FactorLibrary
from priors.mechanisms import (
    StateData, TEMPLATE_FOR_MECHANISM, bonferroni_cutoff, publication_decay_test, run_mechanism_tests,
)
from priors.spec import StrategySpec

HS = "2015-01"
IDX = pd.period_range("1965-01", "2020-12", freq="M", name="month")


def make(effect: str, size: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(IDX)
    sent = pd.Series(rng.standard_normal(n), index=IDX)
    rec = pd.Series((rng.random(n) < 0.15).astype(int), index=IDX)
    mkt = pd.Series(0.006 + 0.045 * rng.standard_normal(n), index=IDX)
    r = 0.004 + 0.02 * rng.standard_normal(n)
    if effect == "sentiment":
        r = r + size * sent.shift(1).fillna(0).to_numpy()
    elif effect == "recession":
        r = r - size * rec.to_numpy()
    elif effect == "decay":
        r = r - size * (IDX >= pd.Period("1995-01", "M"))
    lib = FactorLibrary(
        returns=pd.DataFrame({"F": r, "MKT_RF": mkt}, index=IDX), rf=pd.Series(0.0, index=IDX),
        info=pd.DataFrame({"source": ["cz", "french"], "name": ["F", "Market"], "year": [1995, 1993],
                           "sample_start": [1965, np.nan], "sample_end": [1990, np.nan]}, index=["F", "MKT_RF"]),
    )
    state = StateData(
        sentiment=lambda: pd.DataFrame({"SENT": sent, "SENT_ORTH": sent}),
        recessions=lambda: rec,
        vix=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    return lib, state


def run(lib, state, tests):
    s = StrategySpec.model_validate({"name": "t", "components": [{"factor": "F"}], "mechanism_tests": tests})
    return run_mechanism_tests(run_factor_backtest(s, lib, holdout_from=HS), lib, state)


def test_sentiment_uses_lagged_state():
    lib, state = make("sentiment", 0.01)
    (m,) = run(lib, state, ["sentiment"])
    assert m.verdict == "consistent with" and m.rows[0].t_stat > 5
    none, state0 = make("none", 0.0)
    assert run(none, state0, ["sentiment"])[0].verdict == "inconclusive"


def test_recession_effect_and_skipped_vix():
    lib, state = make("recession", 0.03)
    (m,) = run(lib, state, ["risk_regime"])
    assert m.rows[0].estimate < 0 and m.rows[0].t_stat < -3
    assert m.verdict == "consistent with"
    assert any("VIX data are unavailable" in n for n in m.notes)
    assert any("Bonferroni" in n for n in m.notes)


def test_bonferroni_cutoffs():
    assert bonferroni_cutoff(1) == pytest.approx(1.96, abs=0.01)
    assert bonferroni_cutoff(3) == pytest.approx(2.394, abs=0.01)


def test_publication_decay_and_out_of_sample_window():
    lib, state = make("decay", 0.008)
    s = StrategySpec.model_validate({"name": "t", "components": [{"factor": "F"}]})
    m = publication_decay_test(run_factor_backtest(s, lib, holdout_from=HS), lib)
    assert m.verdict == "consistent with" and m.rows[0].estimate < 0
    assert "1991-01 to 1994-12" in m.rows[0].detail
    assert "cannot separate the two" in m.explanation


def test_default_template_and_mechanism_map():
    lib, state = make("none", 0.0)
    out = run(lib, state, ["publication_decay"])
    assert [m.template for m in out] == ["publication_decay"]
    assert StrategySpec.model_validate({"name": "t", "components": [{"factor": "F"}]}).mechanism_tests == ["publication_decay"]
    assert "sentiment" in TEMPLATE_FOR_MECHANISM["behavioral_bias"]
    assert "risk_regime" in TEMPLATE_FOR_MECHANISM["risk_compensation"]

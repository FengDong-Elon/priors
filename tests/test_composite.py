"""Factor-layer spec and composite engine, on a synthetic factor library."""

import pandas as pd
import pytest
from pydantic import ValidationError

from priors.composite import run_factor_backtest
from priors.composite import weights as W
from priors.holdout import holdout_start
from priors.spec import StrategySpec

from conftest import make_lib

HS = pd.Period("2015-01", "M")


def spec(**kw) -> StrategySpec:
    base = {"name": "t", "components": [{"factor": "A"}, {"factor": "B"}]}
    base.update(kw)
    return StrategySpec.model_validate(base)


def test_holdout_rule():
    assert holdout_start("2026-10") == pd.Period("2021-10", "M")


def test_equal_weight_composite_is_average_and_stops_before_holdout():
    lib = make_lib()
    res = run_factor_backtest(spec(), lib, holdout_from=HS)
    assert res.sample["start"] == pd.Period("1990-01", "M")
    assert res.sample["end"] == HS - 1
    expected = lib.returns.loc[:HS - 1, ["A", "B"]].mean(axis=1)
    pd.testing.assert_series_equal(res.returns["composite"], expected, check_names=False)
    assert any("before trading costs" in w for w in res.warnings)
    assert any("Exploratory" in w for w in res.warnings)


def test_common_sample_and_limiting_factor():
    lib = make_lib()
    res = run_factor_backtest(spec(components=[{"factor": "A"}, {"factor": "C"}]), lib, holdout_from=HS)
    assert res.sample["start"] == pd.Period("2000-01", "M")
    assert res.sample["limited_by"] == "C"


def test_holdout_period_and_short_factor_warning():
    lib = make_lib()
    res = run_factor_backtest(spec(), lib, period="holdout", holdout_from=HS)
    assert res.sample["start"] == HS
    assert res.sample["end"] == pd.Period("2018-12", "M")   # B ends here
    assert any("Holdout used" in w for w in res.warnings)
    assert any("data for B end there" in w for w in res.warnings)


def test_inverse_vol_uses_only_past_data():
    lib = make_lib()
    w1 = W.inverse_vol(lib.returns[["A", "B"]].loc[:"2010-12"], 36)
    changed = lib.returns[["A", "B"]].copy()
    changed.loc["2011-01":] *= 10            # future shock
    w2 = W.inverse_vol(changed, 36).loc[:"2010-12"]
    pd.testing.assert_frame_equal(w1, w2)
    # Lower-vol factor A gets more weight
    assert (w1.dropna()["A"] > w1.dropna()["B"]).mean() > 0.95


def test_theory_weights_validation_and_use():
    with pytest.raises(ValidationError):
        spec(factor_layer={"combination": "theory_weights"})              # weights missing
    with pytest.raises(ValidationError):
        spec(components=[{"factor": "A", "weight": 0.5}, {"factor": "B", "weight": 0.2}],
             factor_layer={"combination": "theory_weights"})              # does not sum to 1
    with pytest.raises(ValidationError):
        spec(components=[{"factor": "A", "weight": 1.0}, {"factor": "B"}])  # weight without theory_weights
    s = spec(components=[{"factor": "A", "weight": 0.75}, {"factor": "B", "weight": 0.25}],
             factor_layer={"combination": "theory_weights"})
    res = run_factor_backtest(s, make_lib(), holdout_from=HS)
    assert (res.weights["A"] == 0.75).all()


def test_optimized_is_evaluated_only_after_training_window():
    s = spec(factor_layer={"combination": "optimized", "train_fraction": 0.5})
    res = run_factor_backtest(s, make_lib(), holdout_from=HS)
    train_start, train_end = res.train_window
    assert res.sample["start"] == train_end + 1
    assert res.sample["end"] == HS - 1
    w = res.weights.iloc[0]
    assert w.sum() == pytest.approx(1) and (w >= 0).all()


def test_mode_restrictions():
    s = spec(factor_layer={"combination": "inverse_vol"})
    s.check_mode("analyst")
    with pytest.raises(ValueError, match="not available in mentor"):
        s.check_mode("mentor")


def test_fingerprint_ignores_labels_but_content_hash_does_not():
    a = spec()
    b = spec(name="other", status="preregistered",
             components=[{"factor": "A", "source": "doi:x", "theory_priority": 1}, {"factor": "B"}])
    assert a.fingerprint() == b.fingerprint()
    assert a.content_hash() != b.content_hash()
    c = spec(stock_layer={"holdings": 30})
    assert c.fingerprint() != a.fingerprint()
    assert StrategySpec.from_yaml(b.to_yaml()) == b


def test_spec_rejects_bad_input():
    with pytest.raises(ValidationError):
        spec(components=[{"factor": "A"}, {"factor": "A"}])
    with pytest.raises(ValidationError):
        spec(stock_layer={"cost_bps_one_way": 80})
    with pytest.raises(ValidationError):
        spec(factor_layer={"start": "1990/01"})
    assert spec(factor_layer={"start": "1995-03-31"}).factor_layer.start == "1995-03"

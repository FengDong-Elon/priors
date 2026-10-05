import numpy as np
import pandas as pd
import pytest

from priors.data import SyntheticProvider
from priors.data.calendar import assign_periods, rebalance_dates
from priors.engine import run_backtest, summarize
from priors.engine.signals import compute_signal
from priors.spec import StockSpec

MOM_SPEC = """
name: synthetic momentum
signal: {name: momentum, params: {lookback: 12, skip: 1}}
portfolio: {quantiles: 10, side: long_short, weighting: equal}
universe: {min_price: null}
execution: {lag_days: 0, cost_bps_one_way: 0}
"""


def spec(**overrides) -> StockSpec:
    s = StockSpec.from_yaml(MOM_SPEC)
    return s.model_copy(update=overrides)


def test_finds_planted_effect():
    res = run_backtest(spec(), SyntheticProvider(momentum_strength=0.006, seed=1))
    m = res.metrics["gross"]
    assert m["t_stat_mean"] > 4
    assert m["sharpe_ci_low"] > 0


def test_no_false_positive_without_effect():
    t_stats = [
        run_backtest(spec(), SyntheticProvider(momentum_strength=0.0, seed=s)).metrics["gross"]["t_stat_mean"]
        for s in range(20)
    ]
    # Under the null, |t| > 1.96 should be rare (about 5%).
    assert sum(abs(t) > 1.96 for t in t_stats) <= 3
    assert abs(np.mean(t_stats)) < 0.8


def test_no_lookahead_in_weights():
    """Weights at date k must be identical whether or not later data exists."""
    panel = SyntheticProvider(seed=2).panel()
    s = spec()
    full = run_backtest(s, panel)
    cut = panel.signal_dates[150]
    truncated = run_backtest(s, panel.truncate(cut))
    # The truncated panel has no ret_next at its final date, so compare earlier dates.
    common = truncated.weights_long.index[:-1]
    pd.testing.assert_frame_equal(full.weights_long.loc[common], truncated.weights_long.loc[common])


def test_signal_cannot_see_forward_returns():
    from priors.engine.signals import register

    with pytest.raises(ValueError):
        @register("cheat", "uses the future", ("ret_next",))
        def cheat(panel):
            return panel["ret_next"]


def test_momentum_definition():
    panel = SyntheticProvider(n_periods=30, seed=3).panel()
    mom = compute_signal(panel, "momentum", {"lookback": 12, "skip": 1})
    r = panel["ret_past"]
    k = 20
    expected = (1 + r.iloc[k - 11:k]).prod() - 1  # periods k-11 .. k-1
    np.testing.assert_allclose(mom.iloc[k].to_numpy(), expected.to_numpy(), rtol=1e-10)


def test_costs_reduce_returns_by_turnover():
    provider = SyntheticProvider(seed=4)
    free = run_backtest(spec(), provider)
    costly_spec = spec(execution=spec().execution.model_copy(update={"cost_bps_one_way": 20}))
    costly = run_backtest(costly_spec, provider)
    pd.testing.assert_series_equal(free.returns["gross"], costly.returns["gross"])
    expected_cost = (free.turnover * 2 * 2) * 20 / 1e4  # one-way per leg -> traded across both legs
    np.testing.assert_allclose(costly.returns["cost"], expected_cost, rtol=1e-12)
    assert (costly.returns["net"] <= costly.returns["gross"] + 1e-15).all()


def test_legs_are_fully_invested():
    res = run_backtest(spec(), SyntheticProvider(seed=5))
    np.testing.assert_allclose(res.weights_long.sum(axis=1), 1.0)
    np.testing.assert_allclose(res.weights_short.sum(axis=1), 1.0)
    assert (res.n_long >= 39).all()  # 400 stocks / 10 deciles


def test_long_only_warns_about_risk_free():
    s = spec(portfolio=spec().portfolio.model_copy(update={"side": "long_only"}))
    res = run_backtest(s, SyntheticProvider(seed=6))
    assert res.weights_short is None
    assert any("Risk-free" in w for w in res.warnings)


def test_summarize_known_series():
    r = pd.Series([0.01, -0.02, 0.03, 0.0] * 30)
    m = summarize(r, 12)
    assert m["periods"] == 120
    assert m["mean_ann"] == pytest.approx(0.005 * 12)
    assert m["sharpe_ci_low"] < m["sharpe"] < m["sharpe_ci_high"]
    assert m["max_drawdown"] == pytest.approx(-0.02)


def test_spec_fingerprint_ignores_name():
    a = spec()
    b = a.model_copy(update={"name": "renamed"})
    c = a.model_copy(update={"rebalance": "weekly"})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert StockSpec.from_yaml(a.to_yaml()) == a


def test_spec_rejects_unknown_keys():
    with pytest.raises(Exception):
        StockSpec.from_yaml(MOM_SPEC + "\nleverage: 3\n")


def test_rebalance_calendar():
    days = pd.bdate_range("2024-01-01", "2024-04-15")
    sig, exe = rebalance_dates(days, "monthly", lag_days=1)
    assert list(sig.strftime("%Y-%m-%d")) == ["2024-01-31", "2024-02-29", "2024-03-29"]  # April incomplete
    assert list(exe.strftime("%Y-%m-%d")) == ["2024-02-01", "2024-03-01", "2024-04-01"]
    k = assign_periods(days, sig)
    assert k[pd.Timestamp("2024-01-31")] == 0 and k[pd.Timestamp("2024-02-01")] == 1
    assert k[pd.Timestamp("2024-04-10")] == -1


def test_nyse_breakpoints_split_nyse_stocks_evenly():
    from priors.engine.portfolio import nyse_mask, quantile_buckets

    panel = SyntheticProvider(seed=7).panel()
    sig = compute_signal(panel, "momentum", {})
    mask = sig.notna()
    nyse = nyse_mask(panel)
    b = quantile_buckets(sig, mask, 5, nyse)
    last = b.iloc[-1]
    counts = last[nyse.iloc[-1]].value_counts()
    assert set(counts.index) == {1, 2, 3, 4, 5}
    assert counts.max() - counts.min() <= 1          # NYSE stocks split evenly
    assert last.notna().sum() == mask.iloc[-1].sum()  # every eligible stock gets a bucket

"""Stock layer on synthetic data: universe cleaning, split handling, panel fields, construction,
backtest timing and costs, current holdings, and the gap report. No network."""

import numpy as np
import pandas as pd
import pytest

from priors.data.base import Capabilities
from priors.data.yfinance_provider import MARKET_PROXY, _raw_close, build_panel
from priors.spec import StrategySpec
from priors.stocklayer import FundamentalHistoryUnavailable, current_holdings, gap_report, run_stock_backtest
from priors.stocklayer.construct import composite_score, eligible, top_n_weights
from priors.universe import clean_listings, yahoo_symbol

CAPS = Capabilities(universe="us_equity", history_start=pd.Timestamp("2000-01-01").date(), includes_delisted=False,
                    delisting_returns=False, point_in_time_fundamentals=False, fundamentals=True,
                    institutional_holdings=False, historical_exchange=False)
HS = "2014-01"


def synthetic(n: int = 80, seed: int = 0, momentum: float = 0.0):
    """Daily prices for n stocks plus the market proxy. With momentum > 0, each stock's drift persists
    for about a year, so past winners keep winning."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2000-01-03", "2015-12-31")
    T = len(days)
    m = 0.0003 + 0.01 * rng.standard_normal(T)
    tickers = [f"S{i:02d}" for i in range(n)]
    drift_state = rng.standard_normal(n)
    rets = np.empty((T, n))
    for t in range(T):
        if t % 21 == 0:
            drift_state = 0.95 * drift_state + 0.3 * rng.standard_normal(n)
        rets[t] = m[t] + momentum * drift_state + 0.015 * rng.standard_normal(n)
    adj = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days, columns=tickers)
    adj[MARKET_PROXY] = 100 * np.cumprod(1 + m)
    vol = pd.DataFrame(1e6, index=days, columns=adj.columns)
    px = {"adj_close": adj, "close": adj.copy(), "raw_close": adj.copy(), "volume": vol}
    universe = pd.DataFrame({
        "ticker": tickers, "name": [f"Firm {t}" for t in tickers],
        "sector": ["Tech", "Energy", "Finance", "Health"] * (n // 4), "industry": "x",
        "mcap": np.linspace(5e10, 1e9, n), "nyse": True,
    })
    universe.attrs["label"] = "test universe"
    return px, universe


def panel_for(px, universe):
    return build_panel(px, universe, None, "monthly", 1, CAPS, "yfinance")


def spec(components, **sl):
    return StrategySpec.model_validate({"name": "t", "components": components, "stock_layer": sl})


MOM = [{"factor": "UMD", "stock_signal": "mom_12_1"}]
NO_FILTERS = {"filters": {"exclude_microcaps": False, "min_price": None, "min_adv_usd": None}}


# ------------------------------------------------------------- universe

def test_clean_listings():
    rows = pd.DataFrame([
        {"symbol": "BRK/B", "name": "Berkshire Hathaway Inc. Class B Common Stock", "marketCap": "1,000,000,000,000",
         "country": "United States", "sector": "Finance", "industry": "Ins"},
        {"symbol": "BAC^K", "name": "Bank of America Preferred", "marketCap": "5e9", "country": "United States",
         "sector": "Finance", "industry": "x"},
        {"symbol": "XYZW", "name": "XYZ Warrant", "marketCap": "1e9", "country": "United States", "sector": "", "industry": ""},
        {"symbol": "RLJ", "name": "RLJ Lodging Trust Common Shares of Beneficial Interest $0.01 par value",
         "marketCap": "3,000,000", "country": "United States", "sector": "Real Estate", "industry": "x"},
        {"symbol": "ACME", "name": "Acme Corp. Common Stock", "marketCap": "2,000,000", "country": "United States",
         "sector": "Industrials", "industry": "x"},
        {"symbol": "FORN", "name": "Foreign Co Common Stock", "marketCap": "9e9", "country": "Canada", "sector": "x", "industry": "x"},
    ])
    out = clean_listings(rows, {"BRK/B"})
    assert list(out["ticker"]) == ["BRK-B", "RLJ", "ACME"]
    assert out.iloc[1]["name"] == "RLJ Lodging Trust"
    assert out.iloc[0]["name"] == "Berkshire Hathaway Inc." and bool(out.iloc[0]["nyse"])
    assert yahoo_symbol("BF/B") == "BF-B"


def test_raw_close_undoes_splits():
    days = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame({"X": [10.0, 10, 10, 10, 10]}, index=days)        # split-adjusted
    split = pd.DataFrame({"X": [0.0, 0, 4.0, 0, 0]}, index=days)           # 4-for-1 on day 3
    raw = _raw_close(close, split)
    assert list(raw["X"]) == [40, 40, 10, 10, 10]


# ---------------------------------------------------------------- panel

def test_panel_fields_and_beta():
    px, u = synthetic(8)
    beta_true = 1.5
    m = px["adj_close"][MARKET_PROXY].pct_change().fillna(0)
    rng = np.random.default_rng(9)
    px["adj_close"]["S00"] = 50 * np.cumprod(1 + beta_true * m + 0.005 * rng.standard_normal(len(m)))
    px["close"]["S00"] = px["adj_close"]["S00"]
    p = panel_for(px, u)
    adj = px["adj_close"]["S01"]
    k = 30
    d0, d1 = p.signal_dates[k - 1], p.signal_dates[k]
    assert p["ret_past"].iloc[k]["S01"] == pytest.approx(adj.loc[d1] / adj.loc[d0] - 1)
    e0, e1 = p.exec_dates[k], p.exec_dates[k + 1]
    assert p["ret_next"].iloc[k]["S01"] == pytest.approx(adj.loc[e1] / adj.loc[e0] - 1)
    assert p["beta"].iloc[-1]["S00"] == pytest.approx(beta_true, abs=0.1)
    assert (p["high52"].dropna().to_numpy() <= 1 + 1e-12).all()


# -------------------------------------------------------- construction

def test_filters_and_composite():
    px, u = synthetic(20)
    p = panel_for(px, u)
    sl = spec(MOM).stock_layer
    mask = eligible(p, sl.filters.model_copy(update={"min_price": 1e9}), None)
    assert not mask.to_numpy().any()
    a = pd.DataFrame([[1.0, 2.0, 3.0]], columns=list("xyz"))
    b = pd.DataFrame([[3.0, np.nan, 1.0]], columns=list("xyz"))
    m = pd.DataFrame(True, index=a.index, columns=a.columns)
    score, count = composite_score({"a": a, "b": b}, m)
    assert list(count.iloc[0]) == [2, 1, 2]
    assert np.isnan(score.iloc[0]["y"])                          # with two signals, both are required
    assert score.iloc[0]["x"] == pytest.approx((1 / 3 + 1) / 2)
    single = composite_score({"a": a}, m)[0]
    assert single.iloc[0]["y"] == pytest.approx(2 / 3)           # one signal: that one is enough
    four = composite_score({"a": a, "b": b, "c": b, "d": a}, m)[0]
    assert four.iloc[0]["y"] == pytest.approx(2 / 3)             # two of four is enough
    w = top_n_weights(pd.DataFrame([[0.1, 0.9, 0.5, 0.7]], columns=list("abcd")), 2, "equal")
    assert list(w.iloc[0]) == [0, 0.5, 0, 0.5]


# ------------------------------------------------------------- backtest

def test_backtest_finds_planted_momentum_and_respects_holdout():
    px, u = synthetic(80, momentum=0.0012)
    p = panel_for(px, u)
    res = run_stock_backtest(spec(MOM, holdings=10, **NO_FILTERS), p, None, holdout_from=HS)
    assert res.sample["end"] == pd.Period(HS, "M") - 1
    assert res.metrics["excess"]["t_stat_mean"] > 2
    assert (res.n_holdings == 10).all()
    assert any("Survivorship" in w for w in res.warnings)
    assert not any("point-in-time" in w for w in res.warnings)


def test_costs_and_quarterly_rebalancing():
    px, u = synthetic(40, seed=2)
    p = panel_for(px, u)
    monthly = run_stock_backtest(spec(MOM, holdings=10, cost_bps_one_way=20, **NO_FILTERS), p, None, holdout_from=HS)
    r = monthly.returns
    assert (r["net"] == r["gross"] - r["cost"]).all()
    assert r["cost"].iloc[5] == pytest.approx(monthly.turnover.iloc[5] * 2 * 20 / 1e4)
    q = run_stock_backtest(spec(MOM, holdings=10, rebalance="quarterly", **NO_FILTERS), p, None, holdout_from=HS)
    off_quarter = ~q.turnover.index.month.isin([3, 6, 9, 12])
    assert (q.turnover[off_quarter].iloc[1:] == 0).all()
    assert q.turnover.mean() < monthly.turnover.mean()
    free = run_stock_backtest(spec(MOM, holdings=10, cost_bps_one_way=0, **NO_FILTERS), p, None, holdout_from=HS)
    assert (free.returns["cost"] == 0).all()


def test_fundamental_signals_cannot_be_backtested_without_pit_data():
    px, u = synthetic(20)
    with pytest.raises(FundamentalHistoryUnavailable, match="point-in-time"):
        run_stock_backtest(spec([{"factor": "HML", "stock_signal": "book_to_market"}]), panel_for(px, u), None)


# ------------------------------------------------------------- holdings

def test_current_holdings_mixes_price_and_fundamental_signals():
    px, u = synthetic(40, seed=4)
    p = panel_for(px, u)
    fund = pd.DataFrame({"book_value_ps": np.linspace(1, 80, 40)}, index=u["ticker"])
    for c in ("trailing_eps", "roe", "gross_profit", "total_assets", "total_assets_prev"):
        fund[c] = np.nan
    s = spec([{"factor": "HML", "stock_signal": "book_to_market"}, {"factor": "UMD", "stock_signal": "mom_12_1"}],
             holdings=10, **NO_FILTERS)
    h = current_holdings(s, p, u, fund, None)
    assert len(h.table) == 10 and h.table["weight"].sum() == pytest.approx(1)
    assert h.sector_weights.sum() == pytest.approx(1)
    assert {"rank_book_to_market", "rank_mom_12_1"} <= set(h.table.columns)
    assert "Not a recommendation" in h.label and h.universe_label == "test universe"
    assert h.as_of == str(p.signal_dates[-1].date())


# ------------------------------------------------------------------ gap

def test_gap_report_aligns_months():
    from priors.composite import run_factor_backtest
    from priors.factors import FactorLibrary

    px, u = synthetic(60, momentum=0.0012)
    p = panel_for(px, u)
    stock = run_stock_backtest(spec(MOM, holdings=10, **NO_FILTERS), p, None, holdout_from=HS)
    idx = pd.period_range("1995-01", "2015-12", freq="M", name="month")
    proxy = stock.returns["excess"].reindex(idx).fillna(0.0) + 0.001
    lib = FactorLibrary(returns=pd.DataFrame({"UMD": proxy}), rf=pd.Series(0.0, index=idx),
                        info=pd.DataFrame({"source": ["french"], "name": ["UMD"]}, index=["UMD"]))
    fac = run_factor_backtest(spec(MOM), lib, holdout_from=HS)
    g = gap_report(fac, stock)
    assert g.correlation == pytest.approx(1.0, abs=1e-9)
    assert g.start == str(stock.sample["start"]) and len(g.table) == 4

"""Integration checks against real data. Skipped unless the data paths are set.

Required environment variables: SHARADAR_SEP_PATH, SHARADAR_SF1_PATH,
SHARADAR_TICKERS_PATH, and FF_FACTORS_MONTHLY_PATH (a CSV with columns Date
[YYYY-MM], Mkt-RF and RF in percent, as distributed by Ken French). Optional:
FRENCH_MOM10_PATH, the unzipped "10_Portfolios_Prior_12_2.csv" from Ken French.
"""

import os

import pandas as pd
import pytest

REQUIRED = ("SHARADAR_SEP_PATH", "SHARADAR_SF1_PATH", "SHARADAR_TICKERS_PATH", "FF_FACTORS_MONTHLY_PATH")
pytestmark = pytest.mark.skipif(
    not all(os.environ.get(v) for v in REQUIRED), reason="real-data paths not configured"
)


def test_value_weighted_market_matches_fama_french():
    from priors.data import SharadarProvider
    from priors.engine.backtest import _market

    panel = SharadarProvider().panel("monthly", lag_days=0)
    active = pd.Series(panel.signal_dates >= "1999-01-01", index=panel.signal_dates)
    ours = _market(panel, active)
    ours.index = ours.index.to_period("M") + 1  # holdings formed at k earn month k+1

    ff = pd.read_csv(os.environ["FF_FACTORS_MONTHLY_PATH"])
    ff.index = pd.PeriodIndex(ff["Date"], freq="M")
    ff_mkt = (ff["Mkt-RF"] + ff["RF"]) / 100

    j = pd.concat([ours.rename("ours"), ff_mkt.rename("ff")], axis=1).dropna()
    assert len(j) > 300
    assert j.corr().iloc[0, 1] > 0.995
    assert abs(j["ours"].mean() - j["ff"].mean()) * 12 < 0.01  # within 1% a year


@pytest.mark.skipif(not os.environ.get("FRENCH_MOM10_PATH"), reason="FRENCH_MOM10_PATH not set")
def test_momentum_deciles_match_ken_french():
    """Replicates Ken French's 10 VW portfolios formed on prior (12-2) return, 10-1 spread."""
    import io

    from priors.data import SharadarProvider
    from priors.engine import run_backtest
    from priors.spec import StrategySpec

    lines = open(os.environ["FRENCH_MOM10_PATH"]).read().splitlines()
    start = next(i for i, l in enumerate(lines) if "Value Weight Returns -- Monthly" in l) + 1
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].strip())
    fr = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0).astype(float) / 100
    fr.index = pd.PeriodIndex(fr.index.astype(str), freq="M")
    french = fr["Hi PRIOR"] - fr["Lo PRIOR"]

    spec = StrategySpec.from_yaml("""
name: French momentum deciles
signal: {name: momentum, params: {lookback: 12, skip: 1}}
universe: {min_price: null, min_mcap_pctile: null}
portfolio: {quantiles: 10, breakpoints: nyse, side: long_short, weighting: value}
execution: {lag_days: 0, cost_bps_one_way: 0}
sample: {start: 1999-01-01}
""")
    ours = run_backtest(spec, SharadarProvider()).returns["gross"]
    ours.index = ours.index.to_period("M") + 1
    j = pd.concat([ours.rename("ours"), french.rename("french")], axis=1).dropna()
    assert len(j) > 300
    assert j.corr().iloc[0, 1] > 0.98
    assert abs(j["ours"].mean() - j["french"].mean()) * 12 < 0.01

"""State variables for mechanism tests: sentiment, recessions, volatility.

All series are monthly and indexed by ``Period('M')``. Each value is the one
observable at the end of that month. Mechanism tests must lag them by one month
before conditioning next-month returns on them.
"""

from __future__ import annotations

import io

import pandas as pd

from ._fetch import fetch

BW_URL = "https://pages.stern.nyu.edu/~jwurgler/data/SENTIMENT.xlsx"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _monthly_index(yyyymm: pd.Series) -> pd.PeriodIndex:
    s = yyyymm.astype(int).astype(str)
    return pd.PeriodIndex([pd.Period(f"{v[:4]}-{v[4:]}", "M") for v in s], name="month")


def parse_baker_wurgler(content: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(content), sheet_name="DATA")
    df.index = _monthly_index(df.pop("yearmo"))
    return df


def load_sentiment(refresh: bool = False) -> pd.DataFrame:
    """Baker-Wurgler sentiment: ``SENT`` and ``SENT_ORTH`` (orthogonalized to macro conditions)."""
    df = parse_baker_wurgler(fetch(BW_URL, key="bw_SENTIMENT.xlsx", max_age_days=90, refresh=refresh))
    return df[["SENT", "SENT_ORTH"]].dropna(how="all")


def parse_fred(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = df.columns[0]
    s = pd.Series(df.iloc[:, 1].astype(float).values, index=pd.to_datetime(df[date_col]))
    return s.dropna()


def load_fred(series_id: str, refresh: bool = False) -> pd.Series:
    text = fetch(FRED_CSV.format(sid=series_id), key=f"fred_{series_id}.csv", max_age_days=7, refresh=refresh)
    return parse_fred(text.decode("utf-8"))


def load_recessions(refresh: bool = False) -> pd.Series:
    """NBER recession indicator (1 = recession month), from FRED USREC."""
    s = load_fred("USREC", refresh=refresh)
    s.index = s.index.to_period("M")
    s.index.name = "month"
    return s.astype(int).rename("recession")


def load_vix(refresh: bool = False) -> pd.Series:
    """VIX at the last trading day of each month, from FRED VIXCLS (starts 1990)."""
    s = load_fred("VIXCLS", refresh=refresh)
    m = s.groupby(s.index.to_period("M")).last()
    m = m[m.index < pd.Timestamp.today().to_period("M")]  # drop the unfinished current month
    m.index.name = "month"
    return m.rename("vix")


def market_drawdown(mkt_total_return: pd.Series) -> pd.Series:
    """Drawdown of the market from its running peak (0 at a peak, negative below it)."""
    wealth = (1 + mkt_total_return.fillna(0)).cumprod()
    return (wealth / wealth.cummax() - 1).rename("market_drawdown")

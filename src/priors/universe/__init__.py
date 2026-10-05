"""The stock-layer universe (ARCHITECTURE §10.2).

The Russell 3000 is the 3,000 largest U.S. companies by market capitalization.
Its official constituent file (iShares IWV) is not reliably downloadable by a
script, so Priors builds the same kind of list from the Nasdaq stock screener,
which covers every stock listed on the NYSE, Nasdaq, and NYSE American:

1. U.S. companies only.
2. Common stock only: preferred shares, warrants, units, rights, notes,
   depositary shares, closed-end funds, and blank-check (SPAC) shells removed.
3. Ranked by market capitalization; the top 3,000 approximate the Russell 3000
   and the top 1,000 the Russell 1000.

The list is labeled as an approximation everywhere it is shown.
"""

from __future__ import annotations

import json
import re
from datetime import date

import pandas as pd

from ..factors._fetch import fetch

SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&offset=0&download=true"
SIZES = {"russell3000": 3000, "russell1000": 1000}
LABELS = {
    "russell3000": "Top 3,000 U.S. stocks by market cap (approximates the Russell 3000)",
    "russell1000": "Top 1,000 U.S. stocks by market cap (approximates the Russell 1000)",
}
EXCLUDE_NAME = re.compile(
    r"warrant|\bunits?\b|\brights?\b|preferred|depositary|\bnotes?\b|debenture|\bfund\b|acquisition corp",
    re.IGNORECASE,
)


def _screener(exchange: str | None, refresh: bool) -> pd.DataFrame:
    url = SCREENER + (f"&exchange={exchange}" if exchange else "")
    key = f"nasdaq_screener_{exchange or 'all'}_{date.today():%Y%m%d}.json"
    rows = json.loads(fetch(url, key=key, max_age_days=1, refresh=refresh))["data"]["rows"]
    return pd.DataFrame(rows)


def yahoo_symbol(symbol: str) -> str:
    """Nasdaq writes share classes as BRK/B; Yahoo writes BRK-B."""
    return symbol.strip().replace("/", "-")


def clean_listings(all_rows: pd.DataFrame, nyse_symbols: set[str]) -> pd.DataFrame:
    df = all_rows.copy()
    df["mcap"] = pd.to_numeric(df["marketCap"].astype(str).str.replace(",", ""), errors="coerce")
    df = df[(df["country"] == "United States") & (df["mcap"] > 0)]
    df = df[~df["symbol"].str.contains(r"\^", regex=True)]
    df = df[~df["name"].str.contains(EXCLUDE_NAME)]
    out = pd.DataFrame({
        "ticker": df["symbol"].map(yahoo_symbol),
        "name": df["name"].str.replace(
            r"\s+(Class [A-Z] )?(Common Stock|Common Shares|Ordinary Shares|Shares of Beneficial Interest).*$",
            "", regex=True).str.strip(),
        "sector": df["sector"].replace("", pd.NA),
        "industry": df["industry"].replace("", pd.NA),
        "mcap": df["mcap"],
        "nyse": df["symbol"].isin(nyse_symbols),
    })
    return out.drop_duplicates("ticker").sort_values("mcap", ascending=False).reset_index(drop=True)


def load_universe(name: str = "russell3000", refresh: bool = False) -> pd.DataFrame:
    """Today's universe: ticker, name, sector, industry, mcap (USD), nyse (bool), rank."""
    if name not in SIZES:
        raise ValueError(f"Unknown universe {name!r}. Available: {sorted(SIZES)}")
    listings = clean_listings(_screener(None, refresh), set(_screener("nyse", refresh)["symbol"]))
    u = listings.head(SIZES[name]).copy()
    u["rank"] = range(1, len(u) + 1)
    u.attrs["label"] = LABELS[name]
    u.attrs["as_of"] = str(date.today())
    return u

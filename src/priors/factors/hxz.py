"""Hou-Xue-Zhang q-factor loaders (global-q.org).

The file name carries the data vintage year (for example
``q5_factors_monthly_2025.csv``), so the current link is discovered from the
factors page instead of being hard-coded.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from ._fetch import fetch

PAGE = "https://global-q.org/factors.html"
SITE = "https://global-q.org"

COLUMNS = {"R_MKT": "Q_MKT", "R_ME": "Q_ME", "R_IA": "Q_IA", "R_ROE": "Q_ROE", "R_EG": "Q_EG"}


def monthly_url(page_html: str) -> str:
    links = re.findall(r'href="([^"]*q5_factors_monthly_(\d{4})\.csv)"', page_html)
    if not links:
        raise ValueError("q5 monthly factor file not found on the global-q factors page")
    path, _ = max(links, key=lambda x: int(x[1]))
    return path if path.startswith("http") else SITE + path


def parse_monthly(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.index = pd.PeriodIndex(
        [pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(df["year"], df["month"])],
        name="month",
    )
    return df[list(COLUMNS)].rename(columns=COLUMNS) / 100.0


def load_factors(refresh: bool = False) -> pd.DataFrame:
    """Monthly q5 factors in decimals: Q_MKT (excess market), Q_ME, Q_IA, Q_ROE, Q_EG."""
    page = fetch(PAGE, key="hxz_factors.html", max_age_days=30, refresh=refresh).decode("utf-8", "replace")
    url = monthly_url(page)
    key = "hxz_" + url.rsplit("/", 1)[-1]
    return parse_monthly(fetch(url, key=key, refresh=refresh).decode("utf-8"))

"""Sharadar adapter (bring your own data).

Reads local Parquet exports of three Sharadar tables:

* SEP: daily equity prices (ticker, date, close, closeadj, closeunadj, volume)
* SF1: fundamentals (ticker, dimension, date [= filing date], marketcap, price)
* TICKERS: metadata (ticker, category, exchange)

Nothing here downloads data or ships data. Paths come from the constructor or
from the SHARADAR_*_PATH environment variables.

Market cap is point-in-time: SF1's market cap from the latest as-reported (ARQ)
filing on or before the signal date, rolled forward with the split-adjusted
price. Share issuance after the filing is ignored until the next filing.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .base import Capabilities, DataProvider, Panel
from .calendar import assign_periods, rebalance_dates
from .cache import cache_dir

DEFAULT_CATEGORIES = ("Domestic Common Stock", "Domestic Common Stock Primary Class")
_BUILD_VERSION = "3"
# Filings older than this are treated as stale and yield no market cap.
_MAX_FILING_AGE_DAYS = 400


class SharadarProvider(DataProvider):
    name = "sharadar"

    def __init__(
        self,
        sep_path: str | Path | None = None,
        sf1_path: str | Path | None = None,
        tickers_path: str | Path | None = None,
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
        use_cache: bool = True,
    ):
        self.sep_path = Path(sep_path or _env("SHARADAR_SEP_PATH"))
        self.sf1_path = Path(sf1_path or _env("SHARADAR_SF1_PATH"))
        self.tickers_path = Path(tickers_path or _env("SHARADAR_TICKERS_PATH"))
        for p in (self.sep_path, self.sf1_path, self.tickers_path):
            if not p.exists():
                raise FileNotFoundError(f"Sharadar file not found: {p}")
        self.categories = tuple(categories)
        self.use_cache = use_cache

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            universe="us_equity",
            history_start=date(1998, 1, 1),
            includes_delisted=True,
            delisting_returns=False,
            point_in_time_fundamentals=True,
            fundamentals=True,
            institutional_holdings=True,
            historical_exchange=False,
        )

    # ------------------------------------------------------------------ panel

    def panel(self, freq: str = "monthly", lag_days: int = 1) -> Panel:
        long, signal_dates, exec_dates, nyse = self._load_or_build(freq, lag_days)
        fields = {}
        for col in ("ret_past", "ret_next", "price", "mcap", "dollar_volume", "volatility"):
            wide = long.pivot(index="k", columns="ticker", values=col)
            wide = wide.reindex(range(len(signal_dates)))
            wide.index = signal_dates
            fields[col] = wide.astype("float64")
        ids = fields["ret_next"].columns
        return Panel(
            freq=freq,
            lag_days=lag_days,
            signal_dates=signal_dates,
            exec_dates=exec_dates,
            fields=fields,
            nyse=nyse.reindex(ids, fill_value=False),
            source=self.name,
            capabilities=self.capabilities,
        )

    def _cache_key(self, freq: str, lag_days: int) -> str:
        h = hashlib.sha256()
        for p in (self.sep_path, self.sf1_path, self.tickers_path):
            st = p.stat()
            h.update(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
        h.update(f"{freq}|{lag_days}|{self.categories}|{_BUILD_VERSION}".encode())
        return h.hexdigest()[:16]

    def _load_or_build(self, freq: str, lag_days: int):
        key = self._cache_key(freq, lag_days)
        d = cache_dir() / "sharadar"
        files = {n: d / f"{key}_{n}.parquet" for n in ("long", "dates", "nyse")}
        if self.use_cache and all(f.exists() for f in files.values()):
            long = pd.read_parquet(files["long"])
            dates = pd.read_parquet(files["dates"])
            nyse = pd.read_parquet(files["nyse"])["nyse"]
            return (
                long,
                pd.DatetimeIndex(dates["signal_date"]),
                pd.DatetimeIndex(dates["exec_date"]),
                nyse,
            )
        long, signal_dates, exec_dates, nyse = self._build(freq, lag_days)
        if self.use_cache:
            d.mkdir(parents=True, exist_ok=True)
            long.to_parquet(files["long"], index=False)
            pd.DataFrame({"signal_date": signal_dates, "exec_date": exec_dates}).to_parquet(files["dates"], index=False)
            nyse.rename("nyse").to_frame().to_parquet(files["nyse"])
        return long, signal_dates, exec_dates, nyse

    def _build(self, freq: str, lag_days: int):
        con = duckdb.connect()
        meta = con.execute(
            "SELECT ticker, category, exchange FROM read_parquet(?)", [str(self.tickers_path)]
        ).df()
        meta = meta.drop_duplicates("ticker", keep="last")
        keep = meta[meta["category"].isin(self.categories)]
        nyse = keep.set_index("ticker")["exchange"].eq("NYSE")
        con.register("universe", keep[["ticker"]])

        con.execute(
            """
            CREATE TEMP TABLE daily AS
            SELECT s.ticker, CAST(s.date AS DATE) AS d, s.close, s.closeadj, s.closeunadj, s.volume
            FROM read_parquet(?) s JOIN universe u USING (ticker)
            WHERE s.closeadj > 0 AND s.close > 0 AND s.closeunadj > 0
            """,
            [str(self.sep_path)],
        )
        days = pd.DatetimeIndex(con.execute("SELECT DISTINCT d FROM daily ORDER BY d").df()["d"])
        signal_dates, exec_dates = rebalance_dates(days, freq, lag_days)
        cal = pd.DataFrame(
            {
                "d": days,
                "k_sig": assign_periods(days, signal_dates).values,
                "k_exec": assign_periods(days, exec_dates).values,
            }
        )
        con.register("cal", cal)
        con.execute(
            """
            CREATE TEMP TABLE x AS
            SELECT t.*, c.k_sig, c.k_exec,
                   t.closeadj / lag(t.closeadj) OVER (PARTITION BY t.ticker ORDER BY t.d) - 1 AS r
            FROM daily t JOIN cal c ON t.d = CAST(c.d AS DATE)
            """
        )
        sig = con.execute(
            """
            SELECT ticker, k_sig AS k,
                   arg_max(closeadj, d)   AS close_adj,
                   arg_max(closeunadj, d) AS price,
                   arg_max(close, d)      AS close_split,
                   avg(close * volume)    AS dollar_volume,  -- both split-adjusted
                   stddev_samp(r)         AS volatility
            FROM x WHERE k_sig >= 0 GROUP BY 1, 2
            """
        ).df()
        exe = con.execute(
            """
            SELECT ticker, k_exec AS k, arg_max(closeadj, d) AS exec_close
            FROM x WHERE k_exec >= 0 GROUP BY 1, 2
            """
        ).df()

        # Implied shares on today's split basis at each filing date: SF1 marketcap
        # divided by SF1 price (the split-adjusted close on the filing date). Do not
        # use sharesbas: it is not adjusted for ADR ratio changes (e.g. SCNI) and
        # can be off by 1000x.
        shares = con.execute(
            """
            SELECT ticker, CAST(date AS DATE) AS fd,
                   last(marketcap / price ORDER BY lastupdated) AS shares_adj
            FROM read_parquet(?)
            WHERE dimension = 'ARQ' AND marketcap > 0 AND price > 0
            GROUP BY 1, 2
            """,
            [str(self.sf1_path)],
        ).df()
        con.close()

        sig["signal_date"] = signal_dates[sig["k"].to_numpy()].astype("datetime64[ns]")
        sig = sig.sort_values("signal_date")
        shares["fd"] = pd.to_datetime(shares["fd"]).astype("datetime64[ns]")
        shares = shares.sort_values("fd")
        sig = pd.merge_asof(
            sig, shares, left_on="signal_date", right_on="fd", by="ticker", direction="backward"
        )
        stale = (sig["signal_date"] - sig["fd"]).dt.days > _MAX_FILING_AGE_DAYS
        sig["mcap"] = np.where(stale, np.nan, sig["shares_adj"] * sig["close_split"])

        sig = sig.sort_values(["ticker", "k"])
        prev = sig.groupby("ticker")["close_adj"].shift(1)
        prev_k = sig.groupby("ticker")["k"].shift(1)
        sig["ret_past"] = np.where(prev_k == sig["k"] - 1, sig["close_adj"] / prev - 1, np.nan)

        exe = exe.sort_values(["ticker", "k"])
        nxt = exe.groupby("ticker")["exec_close"].shift(-1)
        nxt_k = exe.groupby("ticker")["k"].shift(-1)
        exe["ret_next"] = np.where(nxt_k == exe["k"] + 1, nxt / exe["exec_close"] - 1, np.nan)

        long = sig[["ticker", "k", "ret_past", "price", "mcap", "dollar_volume", "volatility"]].merge(
            exe[["ticker", "k", "ret_next"]], on=["ticker", "k"], how="outer"
        )
        long = long[long["k"] < len(signal_dates)]
        for c in ("ret_past", "ret_next", "price", "mcap", "dollar_volume", "volatility"):
            long[c] = long[c].astype("float32")
        long["k"] = long["k"].astype("int32")
        return long.reset_index(drop=True), signal_dates, exec_dates, nyse


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set. See .env.example.")
    return v

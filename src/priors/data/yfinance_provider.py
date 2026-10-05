"""yfinance adapter: the free default stock-data source (ARCHITECTURE §14.2).

What it can and cannot do:

* **Current data is reliable.** Today's prices, share counts, and the latest
  statements are fine for building a current holdings list.
* **History has survivorship bias.** Yahoo only covers stocks that still trade,
  so a backtest on today's universe excludes every firm that was delisted.
* **Fundamentals are not point-in-time** and cover only the last four or five
  fiscal years, so fundamental signals cannot be backtested here.
* **Market cap history is approximate:** today's share count times the
  split-adjusted price. Share issuance and buybacks are ignored.

Prices are cached per day; fundamentals are cached per ticker for a week.
yfinance is an unofficial interface; cached Yahoo data is never redistributed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Capabilities, DataProvider, Panel
from .cache import cache_dir
from .calendar import assign_periods, rebalance_dates

MARKET_PROXY = "SPY"   # S&P 500 ETF (from 1993), used for beta; the Russell 3000 ETF only starts in 2000
MIN_STOCKS_PER_DAY = 50
CHUNK = 200
TRADING_DAYS_YEAR = 252


def _yf_dir() -> Path:
    d = cache_dir() / "yfinance"
    (d / "fund").mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ prices

def _download_chunk(tickers: list[str], start: str, retries: int = 4) -> pd.DataFrame:
    import yfinance as yf

    last = None
    for attempt in range(retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = yf.download(
                    tickers, start=start, auto_adjust=False, actions=True, progress=False,
                    threads=True, group_by="column",
                )
            if d is None or d.empty:
                raise RuntimeError("empty download")
            return d
        except Exception as e:  # network errors and rate limits are routine; back off and retry
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"yfinance download failed for {len(tickers)} tickers: {last}")


FIELDS = {"Adj Close": "adj_close", "Close": "close", "Volume": "volume", "Stock Splits": "split"}


def download_prices(tickers: list[str], start: str = "1995-01-01", refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Daily wide float32 frames: adj_close, close (split-adjusted), raw_close (as traded), volume.

    Downloaded in chunks straight into wide frames (no long-format reshaping), so peak memory stays
    near the size of the result. Cached for the day as one Parquet file per field. Tickers Yahoo does
    not know are dropped and show up as missing columns.
    """
    tickers = sorted(set(tickers) | {MARKET_PROXY})
    key = hashlib.sha256((",".join(tickers) + start).encode()).hexdigest()[:12]
    folder = _yf_dir() / f"prices_{date.today():%Y%m%d}_{key}"
    names = ("adj_close", "close", "volume", "split")
    if not refresh and all((folder / f"{n}.parquet").exists() for n in names):
        wide = {n: pd.read_parquet(folder / f"{n}.parquet") for n in names}
    else:
        parts: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
        for i in range(0, len(tickers), CHUNK):
            d = _download_chunk(tickers[i:i + CHUNK], start)
            level0 = set(d.columns.get_level_values(0))
            for src, dst in FIELDS.items():
                if src in level0:
                    parts[dst].append(d[src].astype("float32"))
            del d
        wide = {n: pd.concat(v, axis=1).sort_index() for n, v in parts.items() if v}
        keep = wide["adj_close"].columns[wide["adj_close"].notna().any()]
        wide = {n: f.reindex(columns=keep) for n, f in wide.items()}
        folder.mkdir(parents=True, exist_ok=True)
        for n, f in wide.items():
            f.to_parquet(folder / f"{n}.parquet")
    split = wide.pop("split", None)
    wide["raw_close"] = _raw_close(wide["close"], split).astype("float32")
    return wide


def _raw_close(close: pd.DataFrame, split: pd.DataFrame | None) -> pd.DataFrame:
    """Undo split adjustment: raw price = split-adjusted price x product of all later split ratios."""
    if split is None:
        return close
    s = split.reindex_like(close).fillna(0.0)
    ratio = s.where(s > 0, 1.0)
    # Splits on day t apply to prices before t: cumulative product of ratios strictly after each day.
    later = ratio.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return close * later


# ------------------------------------------------------------ fundamentals

FUND_FIELDS = (
    "shares_outstanding", "book_value_ps", "price_to_book", "trailing_eps", "roe", "price",
    "total_revenue", "cost_of_revenue", "gross_profit", "total_assets", "total_assets_prev",
    "fiscal_year_end", "sector", "short_name",
)


def _row(stmt: pd.DataFrame, name: str, col: int = 0) -> float:
    try:
        v = stmt.loc[name].iloc[col]
        return float(v) if pd.notna(v) else np.nan
    except (KeyError, IndexError):
        return np.nan


def _fetch_one(ticker: str, retries: int = 4) -> dict:
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # failures are retried and reported below
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            bs, inc = t.balance_sheet, t.income_stmt
            rev, cogs = _row(inc, "Total Revenue"), _row(inc, "Cost Of Revenue")
            gp = _row(inc, "Gross Profit")
            if np.isnan(gp) and not (np.isnan(rev) or np.isnan(cogs)):
                gp = rev - cogs
            return {
                "ticker": ticker,
                "shares_outstanding": info.get("sharesOutstanding"),
                "book_value_ps": info.get("bookValue"),
                "price_to_book": info.get("priceToBook"),
                "trailing_eps": info.get("trailingEps"),
                "roe": info.get("returnOnEquity"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "total_revenue": rev, "cost_of_revenue": cogs, "gross_profit": gp,
                "total_assets": _row(bs, "Total Assets", 0), "total_assets_prev": _row(bs, "Total Assets", 1),
                "fiscal_year_end": str(bs.columns[0].date()) if bs is not None and len(bs.columns) else None,
                "sector": info.get("sector"), "short_name": info.get("shortName"),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        except Exception:
            time.sleep(3 * (attempt + 1) + random.uniform(0, 2))  # Yahoo rate-limits bursts; back off with jitter
    return {"ticker": ticker, "error": "fetch failed", "fetched_at": datetime.now(timezone.utc).isoformat()}


def fetch_fundamentals(tickers: list[str], max_age_days: float = 7, workers: int = 3,
                       progress=None) -> pd.DataFrame:
    """Latest fundamentals per ticker, cached per ticker for ``max_age_days``."""
    d = _yf_dir() / "fund"
    rows, todo = {}, []
    now = time.time()
    for t in tickers:
        p = d / f"{t}.json"
        if p.exists() and (now - p.stat().st_mtime) / 86400 <= max_age_days:
            rows[t] = json.loads(p.read_text())
        else:
            todo.append(t)

    def work(t):
        r = _fetch_one(t)
        if "error" not in r:
            (d / f"{t}.json").write_text(json.dumps(r, default=str))
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(work, todo), start=1):
            rows[r["ticker"]] = r
            if progress:
                progress(i, len(todo))
    df = pd.DataFrame([rows[t] for t in tickers]).set_index("ticker")
    for c in FUND_FIELDS:
        if c not in df:
            df[c] = np.nan
    num = [c for c in FUND_FIELDS if c not in ("fiscal_year_end", "sector", "short_name")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df


# ------------------------------------------------------------------ panel

class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self, universe: pd.DataFrame, start: str = "1995-01-01", fundamentals: pd.DataFrame | None = None):
        self.universe = universe
        self.start = start
        self.fundamentals = fundamentals

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            universe="us_equity", history_start=date.fromisoformat(self.start),
            includes_delisted=False, delisting_returns=False, point_in_time_fundamentals=False,
            fundamentals=True, institutional_holdings=False, historical_exchange=False,
        )

    def panel(self, freq: str = "monthly", lag_days: int = 1, prices: dict | None = None) -> Panel:
        tickers = list(self.universe["ticker"])
        px = prices or download_prices(tickers, self.start)
        return build_panel(px, self.universe, self.fundamentals, freq, lag_days, self.capabilities, self.name)


PANEL_CHUNK = 500  # stocks processed at a time, to keep peak memory low on small hosts


def build_panel(px: dict[str, pd.DataFrame], universe: pd.DataFrame, fundamentals: pd.DataFrame | None,
                freq: str, lag_days: int, caps: Capabilities, source: str) -> Panel:
    """Monthly panel from daily frames. Stocks are processed in chunks so that peak memory stays well
    below the full daily data size (a free cloud host has about 2.7 GB)."""
    adj = px["adj_close"]
    stocks = [t for t in universe["ticker"] if t in adj.columns]
    # Trading calendar: days on which enough universe stocks have a price.
    days = adj.index[adj[stocks].notna().sum(axis=1) >= min(MIN_STOCKS_PER_DAY, len(stocks))]
    signal_dates, exec_dates = rebalance_dates(days, freq, lag_days)
    k_sig = assign_periods(days, signal_dates)
    k_exe = assign_periods(days, exec_dates)
    rm = adj[MARKET_PROXY].reindex(days).astype("float64").pct_change(fill_method=None) if MARKET_PROXY in adj else None

    def at(frame: pd.DataFrame, k: pd.Series, how: str) -> pd.DataFrame:
        g = frame[k.values >= 0].groupby(k[k >= 0].values)
        return getattr(g, how)().reindex(range(len(signal_dates)))

    parts: dict[str, list[pd.DataFrame]] = {}
    for i in range(0, len(stocks), PANEL_CHUNK):
        cols = stocks[i:i + PANEL_CHUNK]
        a, c, raw, vol = (px[f][cols].reindex(days).astype("float64")
                          for f in ("adj_close", "close", "raw_close", "volume"))
        r = a.pct_change(fill_method=None)
        r = r.where(r > -1)  # a return of -100% or worse is a data error in adjusted prices
        af = a.ffill(limit=5)
        close_sig, close_exe = at(af, k_sig, "last"), at(af, k_exe, "last")
        ret_past = close_sig / close_sig.shift(1) - 1
        ret_next = close_exe.shift(-1) / close_exe - 1
        out = {
            "ret_past": ret_past.where(ret_past > -1), "ret_next": ret_next.where(ret_next > -1),
            "price": at(raw, k_sig, "last"), "split_close": at(c, k_sig, "last"),
            "dollar_volume": at(c * vol, k_sig, "mean"), "volatility": at(r, k_sig, "std"),
            "high52": at(c / c.rolling(TRADING_DAYS_YEAR, min_periods=200).max(), k_sig, "last"),
        }
        if rm is not None:
            w, mp = TRADING_DAYS_YEAR, 126
            mean_r = r.rolling(w, min_periods=mp).mean()
            mean_m = rm.rolling(w, min_periods=mp).mean()
            cov = r.mul(rm, axis=0).rolling(w, min_periods=mp).mean() - mean_r.mul(mean_m, axis=0)
            var_m = (rm**2).rolling(w, min_periods=mp).mean() - mean_m**2
            out["beta"] = at(cov.div(var_m, axis=0), k_sig, "last")
        for k, v in out.items():
            parts.setdefault(k, []).append(v)
        del a, c, raw, vol, r, af
    fields = {k: pd.concat(v, axis=1) for k, v in parts.items()}

    split_close = fields.pop("split_close")
    shares = None
    if fundamentals is not None and "shares_outstanding" in fundamentals:
        shares = fundamentals["shares_outstanding"].reindex(stocks)
    if shares is None or shares.isna().all():
        shares = (universe.set_index("ticker")["mcap"].reindex(stocks) / split_close.ffill().iloc[-1])
    fields["mcap"] = split_close * shares
    for k, v in fields.items():
        v.index = signal_dates
        fields[k] = v.astype("float64")
    nyse = universe.set_index("ticker")["nyse"].reindex(stocks).fillna(False).astype(bool)
    return Panel(freq=freq, lag_days=lag_days, signal_dates=signal_dates, exec_dates=exec_dates,
                 fields=fields, nyse=nyse, source=source, capabilities=caps)

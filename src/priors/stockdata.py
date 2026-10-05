"""Stock-data snapshots: refreshed by a scheduled job, read by the app (ARCHITECTURE §13.3).

The app never downloads stock data while students use it. A daily job
(``priors refresh``) downloads the universe, prices, and fundamentals and writes
one snapshot per universe:

    <cache>/stock_snapshot/<universe>/
        meta.json            as_of, universe label, start date, counts
        universe.parquet
        adj_close.parquet, close.parquet, raw_close.parquet, volume.parquet
        fundamentals.parquet

``load_snapshot`` reads it and builds the monthly panel. If no snapshot exists,
the app runs without the stock layer and says so.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .data.cache import cache_dir
from .data.yfinance_provider import YFinanceProvider, download_prices, fetch_fundamentals
from .factors import french
from .flows.session import StockContext
from .universe import LABELS, load_universe

PRICE_FIELDS = ("adj_close", "close", "raw_close", "volume")


def snapshot_dir(universe: str = "russell3000") -> Path:
    return cache_dir() / "stock_snapshot" / universe


def refresh_snapshot(universe: str = "russell3000", start: str = "1995-01-01", fundamentals: bool = True,
                     log: Callable[[str], None] = print) -> dict:
    """Download everything and write a fresh snapshot. Takes minutes (prices) to half an hour (fundamentals).

    With ``fundamentals=False`` (prices only), fundamentals from the previous snapshot are kept if there
    are any, and their date is recorded separately.
    """
    old_fund = snapshot_dir(universe) / "fundamentals.parquet"
    old_meta_path = snapshot_dir(universe) / "meta.json"
    carried = None
    if not fundamentals and old_fund.exists():
        carried = (pd.read_parquet(old_fund),
                   json.loads(old_meta_path.read_text()).get("fundamentals_as_of") if old_meta_path.exists() else None)
    t0 = time.time()
    u = load_universe(universe, refresh=True)
    log(f"universe: {len(u)} stocks")
    px = download_prices(list(u["ticker"]), start, refresh=True)
    log(f"prices: {px['adj_close'].shape[0]} days x {px['adj_close'].shape[1]} series")
    fund = None
    if fundamentals:
        fund = fetch_fundamentals(list(u["ticker"]),
                                  progress=lambda i, n: (i % 250 == 0 or i == n) and log(f"fundamentals {i}/{n}"))
    d = snapshot_dir(universe)
    tmp = d.with_name(d.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    u.to_parquet(tmp / "universe.parquet", index=False)
    for f in PRICE_FIELDS:
        px[f].astype("float32").to_parquet(tmp / f"{f}.parquet")   # half the memory; ample precision
    fund_as_of = datetime.now(timezone.utc).isoformat(timespec="seconds") if fund is not None else None
    if fund is None and carried is not None:
        fund, fund_as_of = carried
    if fund is not None:
        fund.to_parquet(tmp / "fundamentals.parquet")
    meta = {
        "universe": universe, "label": LABELS[universe], "start": start,
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stocks": int(len(u)), "days": int(px["adj_close"].shape[0]),
        "fundamentals_coverage": None if fund is None else float(fund["book_value_ps"].notna().mean()),
        "fundamentals_as_of": fund_as_of,
        "seconds": round(time.time() - t0),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if d.exists():
        old = d.with_name(d.name + ".old")
        if old.exists():
            _rmtree(old)
        d.rename(old)
        tmp.rename(d)
        _rmtree(old)
    else:
        tmp.rename(d)
    log(f"snapshot written to {d} in {meta['seconds']} s")
    return meta


def _rmtree(p: Path) -> None:
    import shutil

    shutil.rmtree(p, ignore_errors=True)


@dataclass
class Snapshot:
    context: StockContext
    meta: dict


def load_snapshot(universe: str = "russell3000") -> Snapshot | None:
    """The latest snapshot as a StockContext, or None if the refresh job has never run."""
    d = snapshot_dir(universe)
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    u = pd.read_parquet(d / "universe.parquet")
    u.attrs["label"] = meta["label"]
    u.attrs["as_of"] = meta["as_of"][:10]
    px = {f: pd.read_parquet(d / f"{f}.parquet") for f in PRICE_FIELDS}
    fund = pd.read_parquet(d / "fundamentals.parquet") if (d / "fundamentals.parquet").exists() else None
    panel = YFinanceProvider(u, start=meta["start"], fundamentals=fund).panel("monthly", 1, prices=px)
    try:
        bp = french.load_nyse_me_breakpoints()
    except Exception:  # offline and never cached: the microcap filter is skipped and the report says so
        bp = None
    return Snapshot(StockContext(u, panel, fund, bp), meta)


# ------------------------------------------------------------ data sources

SOURCES = {
    "yfinance": "Free data (Yahoo Finance): current holdings are reliable; history excludes delisted firms",
    "sharadar": "Sharadar (your own licensed copy): includes delisted firms",
}


def available_sources() -> list[str]:
    """Data sources this installation can offer. Sharadar appears only when its files exist on this computer
    (paths in the environment, or from `priors sharadar-download` / the app's download form)."""
    import os

    from .data.sharadar_download import activate_configured_paths

    out = ["yfinance"]
    if os.environ.get("PRIORS_HOSTED") != "1" and activate_configured_paths():
        out.append("sharadar")
    return out


def sharadar_fundamentals(sf1_path: str | Path) -> pd.DataFrame:
    """Latest trailing-twelve-month fundamentals per ticker, in the same columns as the yfinance adapter."""
    import duckdb

    q = """
        WITH art AS (
            SELECT ticker, CAST(datekey AS DATE) AS dk, bvps, eps, roe, gp, revenue, cor, assets, sharesbas, price,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY CAST(datekey AS DATE) DESC, lastupdated DESC) AS rn
            FROM read_parquet(?) WHERE dimension = 'ART'),
        cur AS (SELECT * FROM art WHERE rn = 1),
        prev AS (SELECT a.ticker, a.assets AS assets_prev FROM art a JOIN cur c USING (ticker)
                 WHERE a.dk <= c.dk - INTERVAL 360 DAY
                 QUALIFY ROW_NUMBER() OVER (PARTITION BY a.ticker ORDER BY a.dk DESC) = 1)
        SELECT cur.ticker, sharesbas AS shares_outstanding, bvps AS book_value_ps, eps AS trailing_eps, roe,
               price, revenue AS total_revenue, cor AS cost_of_revenue, gp AS gross_profit, assets AS total_assets,
               assets_prev AS total_assets_prev, CAST(dk AS VARCHAR) AS fiscal_year_end
        FROM cur LEFT JOIN prev USING (ticker)
    """
    con = duckdb.connect()
    cols = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(sf1_path)]).df().columns
    date_col = "datekey" if "datekey" in cols else "date"
    df = con.execute(q.replace("datekey", date_col), [str(sf1_path)]).df()
    con.close()
    return df.set_index("ticker")


def load_sharadar(universe: str = "russell3000") -> Snapshot:
    """A StockContext from local Sharadar files (paths from the SHARADAR_*_PATH variables).

    The universe shown is the largest ``N`` stocks by the latest market cap; backtests use every U.S. common
    stock that passes the filters at each date, including firms that later delisted.
    """
    import os

    import duckdb

    from .data import SharadarProvider
    from .universe import SIZES

    prov = SharadarProvider()
    panel = prov.panel("monthly", 1)
    meta = duckdb.connect().execute(
        "SELECT ticker, name, sector, industry, exchange FROM read_parquet(?)",  # one row per ticker in any export
        [os.environ["SHARADAR_TICKERS_PATH"]],
    ).df().drop_duplicates("ticker", keep="last").set_index("ticker")
    n = SIZES[universe]
    # Point-in-time membership: the n largest stocks by market cap at each signal date, as Russell builds its indexes.
    members = panel["mcap"].rank(axis=1, ascending=False, method="first") <= n
    last = panel["mcap"].iloc[-1].where(members.iloc[-1]).dropna().sort_values(ascending=False)
    top = last
    u = pd.DataFrame({
        "ticker": top.index, "name": meta["name"].reindex(top.index).values,
        "sector": meta["sector"].reindex(top.index).values, "industry": meta["industry"].reindex(top.index).values,
        "mcap": top.values, "nyse": (meta["exchange"].reindex(top.index) == "NYSE").values,
    })
    u["rank"] = range(1, len(u) + 1)
    u.attrs["label"] = f"Sharadar: the {n:,} largest U.S. common stocks at each date, including firms that later delisted"
    as_of = str(panel.signal_dates[-1].date())
    u.attrs["as_of"] = as_of
    fund = sharadar_fundamentals(os.environ["SHARADAR_SF1_PATH"])
    try:
        bp = french.load_nyse_me_breakpoints()
    except Exception:
        bp = None
    meta_out = {"universe": universe, "label": u.attrs["label"], "as_of": as_of, "source": "sharadar",
                "stocks": int(panel["mcap"].iloc[-1].notna().sum())}
    return Snapshot(StockContext(u, panel, fund, bp, universe_mask=members), meta_out)


def load_source(source: str, universe: str = "russell3000") -> Snapshot | None:
    if source == "sharadar":
        return load_sharadar(universe)
    return load_snapshot(universe)


# ------------------------------------------------------ automatic daily refresh

def snapshot_age_hours(universe: str = "russell3000") -> float | None:
    meta = snapshot_dir(universe) / "meta.json"
    if not meta.exists():
        return None
    as_of = datetime.fromisoformat(json.loads(meta.read_text())["as_of"])
    return (datetime.now(timezone.utc) - as_of).total_seconds() / 3600


def ensure_snapshot(universe: str = "russell3000", max_age_hours: float = 24, start: str = "2000-01-01",
                    log: Callable[[str], None] = print, lock_minutes: float = 20) -> tuple[Snapshot | None, str]:
    """A snapshot that is at most ``max_age_hours`` old, building a prices-only one if needed.

    For hosts without a scheduler (Streamlit Community Cloud): the first visitor of the day triggers a
    download of a few minutes; a lock file keeps a second visitor from starting another. If the download
    fails (for example Yahoo refuses the host), the previous snapshot is used if there is one.
    Returns (snapshot or None, status message for the user).
    """
    age = snapshot_age_hours(universe)
    if age is not None and age <= max_age_hours:
        return load_snapshot(universe), "current"
    lock = snapshot_dir(universe).with_name(f"{universe}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and (time.time() - lock.stat().st_mtime) < lock_minutes * 60:
        snap = load_snapshot(universe)
        return snap, "Stock data is being refreshed by another session; showing the previous data." if snap else             "Stock data is being prepared by another session; try again in a few minutes."
    try:
        lock.write_text(datetime.now(timezone.utc).isoformat())
        refresh_snapshot(universe, start, fundamentals=False, log=log)
        return load_snapshot(universe), "refreshed"
    except Exception as e:
        snap = load_snapshot(universe)
        msg = f"Stock data could not be refreshed ({type(e).__name__})."
        return snap, msg + (" Showing the previous data." if snap else " The stock layer is unavailable.")
    finally:
        lock.unlink(missing_ok=True)

"""Download Sharadar tables with the user's own Nasdaq Data Link API key, to the user's own computer.

Uses the Nasdaq Data Link bulk-export API: request an export, wait until the file is ready, download the zip,
and convert the CSV inside to Parquet in a streaming way (low memory). Only the columns and history Priors needs
are requested. The API key is used for the download only and is never written to disk.

The files go to ``<PRIORS_HOME>/sharadar/`` and their paths are recorded in ``<PRIORS_HOME>/data_sources.json``,
which makes "Sharadar" appear as a data source in the app on that computer. This must not run on a hosted
deployment: licensed data belongs on the licensee's own machine.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests

API = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json"

TABLES = {
    "SEP": {
        "file": "sep.parquet",
        "params": {"date.gte": "1998-01-01",
                   "qopts.columns": "ticker,date,close,closeadj,closeunadj,volume,lastupdated"},
    },
    "SF1": {
        "file": "sf1.parquet",
        "params": {"dimension": "ARQ,ART",
                   "qopts.columns": "ticker,dimension,datekey,lastupdated,marketcap,price,bvps,eps,roe,gp,"
                                    "revenue,cor,assets,sharesbas"},
    },
    "TICKERS": {
        "file": "tickers.parquet",
        "params": {"table": "SEP",
                   "qopts.columns": "ticker,name,exchange,category,sector,industry,isdelisted"},
    },
}
ENV_FOR = {"SEP": "SHARADAR_SEP_PATH", "SF1": "SHARADAR_SF1_PATH", "TICKERS": "SHARADAR_TICKERS_PATH"}


class SharadarDownloadError(RuntimeError):
    pass


def priors_home() -> Path:
    return Path(os.environ.get("PRIORS_HOME", Path.home() / ".priors"))


def sources_config() -> Path:
    return priors_home() / "data_sources.json"


NETWORK_RETRIES = 6


def _get(session, url: str, api_key: str, what: str, **kw):
    """GET with retries on network errors; the key is removed from any error message."""
    last = None
    for attempt in range(NETWORK_RETRIES):
        try:
            return session.get(url, **kw)
        except Exception as e:  # DNS hiccups and dropped connections are common; retry them
            last = e
            time.sleep(min(5 * (attempt + 1), 30))
    raise SharadarDownloadError(f"{what}: request failed: {str(last).replace(api_key, '***')}") from None


def _export_link(table: str, api_key: str, params: dict, log: Callable[[str], None],
                 poll_seconds: float = 30, max_wait: float = 1800, session=requests) -> str:
    q = {**params, "qopts.export": "true", "api_key": api_key}
    waited = 0.0
    while True:
        try:
            r = _get(session, API.format(table=table), api_key, table, params=q, timeout=60)
            if r.status_code in (401, 403):
                raise SharadarDownloadError("Nasdaq Data Link rejected the API key, or the key has no Sharadar access.")
            if r.status_code == 429:  # never retry a rate limit: more requests can extend a suspension
                try:
                    msg = r.json().get("quandl_error", {}).get("message", "")
                except Exception:
                    msg = ""
                raise SharadarDownloadError(
                    "Nasdaq Data Link's rate limit was reached; the download stopped without retrying. "
                    + (f"Server message: {msg} " if msg else "") + "Try again later.")
            r.raise_for_status()
        except SharadarDownloadError:
            raise
        except Exception as e:  # error messages can contain the request URL, which includes the key
            raise SharadarDownloadError(f"{table}: request failed: {str(e).replace(api_key, '***')}") from None
        f = r.json()["datatable_bulk_download"]["file"]
        if f.get("status") == "fresh" and f.get("link"):
            return f["link"]
        if waited >= max_wait:
            raise SharadarDownloadError(f"The {table} export was not ready after {max_wait / 60:.0f} minutes; try again later.")
        log(f"{table}: export is being prepared ({f.get('status')}); waiting...")
        time.sleep(poll_seconds)
        waited += poll_seconds


def _zip_csv_to_parquet(content: bytes, out: Path, table: str) -> int:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as fh:
            reader = pacsv.open_csv(fh, read_options=pacsv.ReadOptions(block_size=1 << 24))
            writer, rows = None, 0
            tmp = out.with_suffix(".part")
            for batch in reader:
                t = pa.Table.from_batches([batch])
                if table == "SF1" and "datekey" in t.column_names and "date" not in t.column_names:
                    t = t.append_column("date", t.column("datekey"))   # the Sharadar provider reads 'date'
                if writer is None:
                    writer = pq.ParquetWriter(tmp, t.schema)
                writer.write_table(t.cast(writer.schema) if t.schema != writer.schema else t)
                rows += t.num_rows
            if writer is not None:
                writer.close()
                tmp.replace(out)
    return rows


def download_sharadar(api_key: str, log: Callable[[str], None] = print, session=requests) -> dict[str, str]:
    """Download SEP, SF1, and TICKERS to this computer and register them as a data source. Returns the paths."""
    if os.environ.get("PRIORS_HOSTED") == "1":
        raise SharadarDownloadError("Licensed data cannot be downloaded on a hosted deployment; run Priors on your own computer.")
    if not api_key or not api_key.strip():
        raise SharadarDownloadError("Enter your Nasdaq Data Link API key.")
    folder = priors_home() / "sharadar"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for table, spec in TABLES.items():
        link = _export_link(table, api_key.strip(), spec["params"], log, session=session)
        log(f"{table}: downloading...")
        r = _get(session, link, api_key, f"{table} file", timeout=1800)
        r.raise_for_status()
        out = folder / spec["file"]
        rows = _zip_csv_to_parquet(r.content, out, table)
        log(f"{table}: {rows:,} rows saved")
        paths[ENV_FOR[table]] = str(out)
    cfg = sources_config()
    cfg.write_text(json.dumps({"sharadar": paths}, indent=1), encoding="utf-8")
    return paths


def configured_paths() -> dict[str, str] | None:
    """Sharadar paths from the environment, or from a previous download on this computer."""
    env = {v: os.environ.get(v) for v in ENV_FOR.values()}
    if all(env.values()):
        return env
    cfg = sources_config()
    if cfg.exists():
        paths = json.loads(cfg.read_text(encoding="utf-8")).get("sharadar") or {}
        if all(paths.get(v) for v in ENV_FOR.values()):
            return paths
    return None


def activate_configured_paths() -> bool:
    """Put configured Sharadar paths into the environment, where the Sharadar provider reads them."""
    paths = configured_paths()
    if not paths or not all(Path(p).exists() for p in paths.values()):
        return False
    for k, v in paths.items():
        os.environ.setdefault(k, v)
    return True

"""Ken French Data Library loaders.

French's CSV files contain a text preamble, a monthly block (YYYYMM rows), and
then an annual block. Only the monthly block is read. Returns are converted from
percent to decimals; -99.99 and -999 mark missing values.
"""

from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd

from ._fetch import fetch

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# file stem -> {column in file: Priors factor id}
# MKT_RF, HML, and RF come from the FF3 file because it starts in 1926 (the FF5
# file starts in 1963). Over the overlap, HML and RF are identical in both files
# and Mkt-RF differs by at most 0.08 percentage points.
# SMB differs between the files (FF5 averages over three 2x3 sorts), so the FF3
# version is kept separately as SMB_FF3.
FACTOR_FILES: dict[str, dict[str, str]] = {
    "F-F_Research_Data_Factors": {"Mkt-RF": "MKT_RF", "SMB": "SMB_FF3", "HML": "HML", "RF": "RF"},
    "F-F_Research_Data_5_Factors_2x3": {"SMB": "SMB", "RMW": "RMW", "CMA": "CMA"},
    "F-F_Momentum_Factor": {"Mom": "UMD"},
}

_ROW = re.compile(r"^\s*(\d{6})\s*,")


def parse_monthly_block(text: str) -> pd.DataFrame:
    """Parse the first YYYYMM block of a French CSV into a DataFrame indexed by month."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _ROW.match(line)), None)
    if start is None or start == 0:
        raise ValueError("no monthly block found")
    header = [h.strip() for h in lines[start - 1].split(",")]
    rows = []
    for line in lines[start:]:
        if not _ROW.match(line):
            break
        rows.append([c.strip() for c in line.split(",")])
    width = len(header)
    rows = [r + [""] * (width - len(r)) for r in rows]
    df = pd.DataFrame([r[:width] for r in rows], columns=["yyyymm"] + header[1:])
    idx = pd.PeriodIndex([pd.Period(f"{s[:4]}-{s[4:]}", "M") for s in df.pop("yyyymm")], name="month")
    df.index = idx
    df = df.replace("", np.nan).astype(float)
    return df.where(~df.isin([-99.99, -999.0]))


def _read_zip_csv(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return z.read(name).decode("latin-1")


def load_file(stem: str, refresh: bool = False) -> pd.DataFrame:
    content = fetch(BASE + f"{stem}_CSV.zip", key=f"french_{stem}.zip", refresh=refresh)
    return parse_monthly_block(_read_zip_csv(content))


def load_factors(refresh: bool = False) -> pd.DataFrame:
    """Monthly French factors in decimals: MKT_RF, SMB_FF3, HML, RF (from 1926), SMB, RMW, CMA (from 1963), UMD."""
    frames = []
    for stem, cols in FACTOR_FILES.items():
        df = load_file(stem, refresh=refresh)
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{stem}: expected columns {sorted(missing)} not found")
        frames.append(df[list(cols)].rename(columns=cols))
    return pd.concat(frames, axis=1).sort_index() / 100.0


def load_nyse_me_breakpoints(refresh: bool = False) -> pd.DataFrame:
    """NYSE market-equity percentiles (5th, 10th, ..., 100th) in USD millions, by month."""
    text = _read_zip_csv(fetch(BASE + "ME_Breakpoints_CSV.zip", key="french_ME_Breakpoints.zip", refresh=refresh))
    rows = []
    for line in text.splitlines():
        if not _ROW.match(line):
            if rows:
                break
            continue
        rows.append([c.strip() for c in line.split(",")])
    months = pd.PeriodIndex([pd.Period(f"{r[0][:4]}-{r[0][4:]}", "M") for r in rows], name="month")
    values = np.array([[float(x) for x in r[2:22]] for r in rows])
    out = pd.DataFrame(values, index=months, columns=[f"p{p}" for p in range(5, 101, 5)])
    out.insert(0, "n_firms", [int(r[1]) for r in rows])
    return out

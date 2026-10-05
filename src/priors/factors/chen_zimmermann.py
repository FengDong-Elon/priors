"""Chen & Zimmermann Open Source Asset Pricing loaders.

Source: https://www.openassetpricing.com (Chen and Zimmermann, 2022, "Open Source
Cross-Sectional Asset Pricing," Critical Finance Review). The data are hosted in
a public Google Drive folder per release. Two files are used:

* ``PredictorLSretWide.csv``: monthly long-short returns (percent) for every
  predictor, built as in the original paper and signed so that the long side is
  the side the original paper predicts to outperform.
* ``PredictorAltPorts_DecilesVW.zip``: the same predictors rebuilt as
  value-weighted decile sorts (decile 10 minus decile 1, signed the same way).
  These are comparable to the value-weighted Ken French factors and are loaded
  with the suffix ``_VW`` (for example ``Mom12m_VW``).
* ``SignalDoc.csv``: one row per signal with the original paper's authors, year,
  journal, sample, reported return and t-stat, and economic category.

File ids are looked up from the release folder listing. If the listing page
changes format, the known ids for the pinned release are used instead.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from ._fetch import fetch

RELEASE = "2025.10"
RELEASE_FOLDER = "1qQDuTsnyvWfEJR6nPBQZ8xxlq6bkLG_y"
# Known file ids for the pinned release (fallback if folder listing fails).
KNOWN_IDS = {
    "SignalDoc.csv": "1rdi0jTPSA6xtn6TpQAMT5WyEczQ1gK59",
    "PredictorLSretWide.csv": "10sOryk_ddjkXagaajTKUk1nwJs2ZLRiI",
    "PredictorAltPorts_DecilesVW.zip": "1_1WWZqilrt1gleeyAFwjv5aobd0QRbS3",
}
_PATHS = {
    "SignalDoc.csv": [],
    "PredictorLSretWide.csv": ["Portfolios", "Full Sets OP"],
    "PredictorAltPorts_DecilesVW.zip": ["Portfolios", "Full Sets Alt"],
}
VW_SUFFIX = "_VW"

_ENTRY = re.compile(
    r'<a href="https://drive\.google\.com/(?:drive/folders|file/d)/([-\w]+)[^"]*"[^>]*>.*?'
    r'<div class="flip-entry-title">(.*?)</div>',
    re.S,
)


def list_folder(folder_id: str) -> dict[str, str]:
    """Map entry name -> Drive id for a public folder."""
    html = fetch(
        f"https://drive.google.com/embeddedfolderview?id={folder_id}",
        key=f"cz_folder_{folder_id}.html",
        max_age_days=7,
    ).decode("utf-8", "replace")
    return {name: fid for fid, name in _ENTRY.findall(html)}


def file_id(name: str) -> str:
    try:
        folder = RELEASE_FOLDER
        for sub in _PATHS[name]:
            folder = list_folder(folder)[sub]
        return list_folder(folder)[name]
    except Exception:
        return KNOWN_IDS[name]


def _download(name: str, refresh: bool) -> bytes:
    fid = file_id(name)
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    content = fetch(url, key=f"cz_{RELEASE}_{name}", max_age_days=180, refresh=refresh)
    if content[:15].lstrip().lower().startswith(b"<!doctype html") or content[:5].lower() == b"<html":
        raise ValueError(f"Google Drive returned a web page instead of {name}; try again later.")
    return content


def parse_returns(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.index = pd.PeriodIndex(pd.to_datetime(df.pop("date")), freq="M", name="month")
    return df.astype(float) / 100.0


def load_returns(refresh: bool = False) -> pd.DataFrame:
    """Monthly long-short returns (decimals), one column per predictor acronym."""
    return parse_returns(_download("PredictorLSretWide.csv", refresh).decode("utf-8"))


def parse_vw_deciles(content: bytes) -> pd.DataFrame:
    import zipfile

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(z.open(name), usecols=["signalname", "port", "date", "ret"], dtype={"port": str})
    ls = df[df["port"] == "LS"].pivot(index="date", columns="signalname", values="ret")
    ls.index = pd.PeriodIndex(pd.to_datetime(ls.index), freq="M", name="month")
    ls.columns = [f"{c}{VW_SUFFIX}" for c in ls.columns]
    ls.columns.name = None
    return ls.sort_index().astype(float) / 100.0


def load_vw_returns(refresh: bool = False) -> pd.DataFrame:
    """Value-weighted decile long-short returns (decimals), columns '<acronym>_VW'."""
    return parse_vw_deciles(_download("PredictorAltPorts_DecilesVW.zip", refresh))


DOC_COLUMNS = {
    "Acronym": "id",
    "LongDescription": "name",
    "Authors": "authors",
    "Year": "year",
    "Journal": "journal",
    "Cat.Economic": "economic_category",
    "Cat.Data": "data_category",
    "SampleStartYear": "sample_start",
    "SampleEndYear": "sample_end",
    "Sign": "sign",
    "Return": "op_return_monthly_pct",
    "T-Stat": "op_tstat",
    "Predictability in OP": "op_predictability",
    "Signal Rep Quality": "replication_quality",
    "Test in OP": "op_test",
    "Stock Weight": "op_weighting",
    "Detailed Definition": "definition",
}


NUMERIC_COLUMNS = (
    "year", "sample_start", "sample_end", "sign", "op_return_monthly_pct", "op_tstat", "gscholar_cites",
)


def parse_signal_doc(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    raw = raw[raw["Cat.Signal"] == "Predictor"]
    cites = next((c for c in raw.columns if c.startswith("GScholarCites")), None)
    doc = raw[list(DOC_COLUMNS)].rename(columns=DOC_COLUMNS)
    doc["gscholar_cites"] = raw[cites] if cites else pd.NA
    doc["gscholar_cites_asof"] = cites.removeprefix("GScholarCites") if cites else None
    for col in NUMERIC_COLUMNS:
        doc[col] = pd.to_numeric(doc[col], errors="coerce")
    return doc.set_index("id")


def load_signal_doc(refresh: bool = False) -> pd.DataFrame:
    """Documentation for every predictor, indexed by acronym."""
    return parse_signal_doc(_download("SignalDoc.csv", refresh).decode("utf-8-sig"))

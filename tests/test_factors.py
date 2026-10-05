"""Factor-library parsers, tested offline on small fixtures in each source's format.

A network test at the bottom checks the live sources and is skipped when offline
or when PRIORS_SKIP_NETWORK is set.
"""

import os

import numpy as np
import pandas as pd
import pytest

from priors.factors import chen_zimmermann, french, hxz, regimes

FRENCH_TEXT = """This file was created using the 202608 CRSP database.
Some description.

,Mkt-RF,SMB,HML,RF
196307,   -0.39,   -0.48,   -0.84,    0.27
196308,    5.07,   -0.80,    1.65,  -99.99
196309,   -1.57,   -0.43,    0.19,    0.27

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1964,   12.0,   1.0,   2.0,   3.5
"""


def test_french_reads_monthly_block_only():
    df = french.parse_monthly_block(FRENCH_TEXT)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3
    assert df.index[0] == pd.Period("1963-07", "M")
    assert df.loc[pd.Period("1963-07", "M"), "Mkt-RF"] == pytest.approx(-0.39)
    assert np.isnan(df.loc[pd.Period("1963-08", "M"), "RF"])  # -99.99 is missing


def test_hxz_picks_latest_vintage_and_converts_percent():
    page = (
        '<a href="/uploads/x/q5_factors_monthly_2024.csv">old</a>'
        '<a href="/uploads/x/q5_factors_monthly_2025.csv">new</a>'
    )
    assert hxz.monthly_url(page) == "https://global-q.org/uploads/x/q5_factors_monthly_2025.csv"
    text = "year,month,R_F,R_MKT,R_ME,R_IA,R_ROE,R_EG\n1967,1,0.39,8.19,6.81,-2.92,1.89,-2.55\n"
    df = hxz.parse_monthly(text)
    assert list(df.columns) == ["Q_MKT", "Q_ME", "Q_IA", "Q_ROE", "Q_EG"]
    assert df.iloc[0]["Q_MKT"] == pytest.approx(0.0819)


def test_cz_returns_and_signal_doc():
    text = "date,Mom12m,BM\n1990-01-31,1.5,\n1990-02-28,-2.0,0.5\n"
    r = chen_zimmermann.parse_returns(text)
    assert r.index[1] == pd.Period("1990-02", "M")
    assert r.loc[pd.Period("1990-01", "M"), "Mom12m"] == pytest.approx(0.015)
    assert np.isnan(r.loc[pd.Period("1990-01", "M"), "BM"])

    cols = list(chen_zimmermann.DOC_COLUMNS) + ["Cat.Signal", "GScholarCites202509"]
    row = {c: "x" for c in cols}
    keep = {**row, "Acronym": "Mom12m", "Cat.Signal": "Predictor", "Year": 1993, "GScholarCites202509": 9000}
    drop = {**row, "Acronym": "Junk", "Cat.Signal": "Placebo"}
    doc = chen_zimmermann.parse_signal_doc(pd.DataFrame([keep, drop]).to_csv(index=False))
    assert list(doc.index) == ["Mom12m"]
    assert doc.loc["Mom12m", "gscholar_cites"] == 9000
    assert doc.loc["Mom12m", "gscholar_cites_asof"] == "202509"


def test_drive_folder_listing_regex():
    html = (
        '<div class="flip-entry"><a href="https://drive.google.com/file/d/ABCdef_123-xyz/view?usp=drive_web" '
        'target="_blank"><div class="flip-entry-title">SignalDoc.csv</div></a></div>'
        '<a href="https://drive.google.com/drive/folders/FOLDERid_9"><div class="x">'
        '<div class="flip-entry-title">Portfolios</div></div></a>'
    )
    found = {n: i for i, n in chen_zimmermann._ENTRY.findall(html)}
    assert found == {"SignalDoc.csv": "ABCdef_123-xyz", "Portfolios": "FOLDERid_9"}


def test_fred_parser_and_drawdown():
    s = regimes.parse_fred("observation_date,VIXCLS\n2020-03-31,53.5\n2020-04-01,.\n")
    assert len(s) == 1 and s.iloc[0] == 53.5
    dd = regimes.market_drawdown(pd.Series([0.10, -0.20, 0.05]))
    assert dd.iloc[0] == 0
    assert dd.iloc[1] == pytest.approx(-0.20)
    assert dd.iloc[2] == pytest.approx(0.8 * 1.05 - 1)


@pytest.mark.skipif(bool(os.environ.get("PRIORS_SKIP_NETWORK")), reason="network tests disabled")
def test_live_library_sanity():
    from priors.factors import FactorLibrary

    try:
        lib = FactorLibrary.load()
    except Exception as e:  # sources are occasionally unreachable
        pytest.skip(f"data sources unreachable: {e}")
    r = lib.get(["MKT_RF", "HML", "UMD", "Mom12m"], "1967-01", "2024-12")
    # Published magnitudes: equity premium ~6-8%/yr, momentum ~6-8%/yr.
    assert 0.04 < r["MKT_RF"].mean() * 12 < 0.10
    assert 0.04 < r["UMD"].mean() * 12 < 0.10
    assert r["UMD"].corr(r["Mom12m"]) > 0.6
    cz = lib.info.index[lib.info["source"] == "cz"]
    assert len(cz) > 200
    assert (lib.returns[cz].mean() > 0).mean() > 0.9  # signed so the predicted side is long

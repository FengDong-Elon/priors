"""The factor library: every factor return series Priors knows, in one table.

``FactorLibrary.returns`` holds monthly returns in decimals, indexed by
``Period('M')``, one column per factor id:

* Ken French: ``MKT_RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``, ``UMD``, ``SMB_FF3``
* Hou-Xue-Zhang: ``Q_MKT``, ``Q_ME``, ``Q_IA``, ``Q_ROE``, ``Q_EG``
* Chen-Zimmermann: the predictor acronym, for example ``Mom12m`` or ``BM``, built
  as in the original paper (most are equal-weighted)
* Chen-Zimmermann value-weighted deciles: the acronym plus ``_VW``, for example
  ``Mom12m_VW``; comparable in construction to the Ken French factors

All are zero-investment (long-short or excess) returns. The risk-free rate is
kept separately in ``FactorLibrary.rf``.

``FactorLibrary.info`` describes each factor: source, name, original paper,
publication year, and (for Chen-Zimmermann) the original paper's reported
return and t-stat.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from . import chen_zimmermann, french, hxz

FRENCH_INFO = {
    "MKT_RF": ("Market excess return", "Fama & French", 1993, "JFE"),
    "SMB": ("Size (small minus big), FF5 version", "Fama & French", 2015, "JFE"),
    "SMB_FF3": ("Size (small minus big), FF3 version", "Fama & French", 1993, "JFE"),
    "HML": ("Value (high minus low book-to-market)", "Fama & French", 1993, "JFE"),
    "RMW": ("Profitability (robust minus weak)", "Fama & French", 2015, "JFE"),
    "CMA": ("Investment (conservative minus aggressive)", "Fama & French", 2015, "JFE"),
    "UMD": ("Momentum (up minus down)", "Carhart", 1997, "JF"),
}
HXZ_INFO = {
    "Q_MKT": ("Market excess return (q-factor model)", "Hou, Xue & Zhang", 2015, "RFS"),
    "Q_ME": ("Size (q-factor model)", "Hou, Xue & Zhang", 2015, "RFS"),
    "Q_IA": ("Investment (q-factor model)", "Hou, Xue & Zhang", 2015, "RFS"),
    "Q_ROE": ("Profitability, ROE (q-factor model)", "Hou, Xue & Zhang", 2015, "RFS"),
    "Q_EG": ("Expected growth (q5 model)", "Hou, Mo, Xue & Zhang", 2021, "RF"),
}

# Pre-specified factor models available for spanning regressions.
MODELS: dict[str, list[str]] = {
    "CAPM": ["MKT_RF"],
    "FF3": ["MKT_RF", "SMB_FF3", "HML"],
    "FF5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
    "FF5_UMD": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "UMD"],
    "Q5": ["Q_MKT", "Q_ME", "Q_IA", "Q_ROE", "Q_EG"],
}


def _info_frame(table: dict, source: str) -> pd.DataFrame:
    rows = [
        {"id": k, "source": source, "name": n, "authors": a, "year": y, "journal": j}
        for k, (n, a, y, j) in table.items()
    ]
    return pd.DataFrame(rows).set_index("id")


@dataclass(frozen=True)
class FactorLibrary:
    returns: pd.DataFrame
    rf: pd.Series
    info: pd.DataFrame
    sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(
        cls, refresh: bool = False, include: tuple[str, ...] = ("french", "hxz", "cz", "cz_vw"),
    ) -> "FactorLibrary":
        frames, infos, sources = [], [], {}
        rf = pd.Series(dtype=float, name="RF")
        if "french" in include:
            ff = french.load_factors(refresh=refresh)
            rf = ff.pop("RF").rename("RF")
            frames.append(ff)
            infos.append(_info_frame(FRENCH_INFO, "french"))
            sources["french"] = "Ken French Data Library"
        if "hxz" in include:
            frames.append(hxz.load_factors(refresh=refresh))
            infos.append(_info_frame(HXZ_INFO, "hxz"))
            sources["hxz"] = "Hou-Xue-Zhang, global-q.org"
        if "cz" in include:
            cz = chen_zimmermann.load_returns(refresh=refresh)
            doc = chen_zimmermann.load_signal_doc(refresh=refresh)
            clash = set(cz.columns) & set().union(*(f.columns for f in frames)) if frames else set()
            if clash:
                raise ValueError(f"Chen-Zimmermann ids clash with other sources: {sorted(clash)}")
            frames.append(cz)
            doc = doc.reindex(cz.columns)
            doc.insert(0, "source", "cz")
            infos.append(doc)
            sources["cz"] = f"Chen & Zimmermann Open Source Asset Pricing, release {chen_zimmermann.RELEASE}"
            if "cz_vw" in include:
                vw = chen_zimmermann.load_vw_returns(refresh=refresh)
                base = [c.removesuffix(chen_zimmermann.VW_SUFFIX) for c in vw.columns]
                vdoc = doc.drop(columns="source").reindex(base)
                vdoc.index = vw.columns
                vdoc["op_weighting"] = "VW"
                vdoc["name"] = vdoc["name"].astype(str) + " (value-weighted deciles)"
                vdoc.insert(0, "source", "cz_vw")
                frames.append(vw)
                infos.append(vdoc)
                sources["cz_vw"] = "Chen & Zimmermann value-weighted decile portfolios"
        returns = pd.concat(frames, axis=1).sort_index()
        info = pd.concat(infos)
        info = info.loc[returns.columns]
        return cls(returns=returns, rf=rf, info=info, sources=sources)

    def get(self, ids: list[str], start=None, end=None, how: str = "common") -> pd.DataFrame:
        """Returns for ``ids`` over [start, end]. ``how='common'`` keeps months where all are present."""
        unknown = [i for i in ids if i not in self.returns.columns]
        if unknown:
            raise KeyError(f"Unknown factor ids: {unknown}")
        df = self.returns[ids]
        if start is not None:
            df = df.loc[pd.Period(start, "M"):]
        if end is not None:
            df = df.loc[: pd.Period(end, "M")]
        return df.dropna() if how == "common" else df

    def coverage(self) -> pd.DataFrame:
        """First and last month with data, and number of months, for every factor."""
        r = self.returns
        return pd.DataFrame(
            {"first": r.apply(lambda s: s.first_valid_index()),
             "last": r.apply(lambda s: s.last_valid_index()),
             "months": r.notna().sum()}
        )

    def is_value_weighted(self, factor: str) -> bool:
        """True for Ken French, Hou-Xue-Zhang, and value-weighted Chen-Zimmermann series."""
        info = self.info.loc[factor]
        if info.get("source") in ("french", "hxz", "cz_vw"):
            return True
        return str(info.get("op_weighting", "")).upper() == "VW"

    def model(self, name: str) -> list[str]:
        return list(MODELS[name])

    @property
    def snapshot_id(self) -> str:
        """Content hash of the return data, recorded with every result for reproducibility."""
        h = hashlib.sha256()
        h.update(pd.util.hash_pandas_object(self.returns, index=True).values.tobytes())
        h.update(pd.util.hash_pandas_object(self.rf, index=True).values.tobytes())
        return h.hexdigest()[:16]

"""Expected-return priors, computed by code (ARCHITECTURE §5.2, §7.1).

For each factor:

1. **Base:** the factor's mean annual return over the original paper's sample
   (Chen-Zimmermann documents it). When no sample is documented, the
   pre-publication period is used: all data before the publication year.
2. **Decay:** McLean & Pontiff (2016) find anomaly returns about 26% lower out
   of sample and about 58% lower after publication. The expected range is
   ``[base x (1 - 0.58), base x (1 - 0.26)]``.

The market factor is not an anomaly, so only the out-of-sample haircut applies:
``[base x (1 - 0.26), base]``.

A composite's range is the weighted sum of its components' ranges (equal
weights unless the spec fixes theory weights).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..factors import FactorLibrary
from ..spec.strategy import StrategySpec

OOS_DECAY = 0.26
POST_PUB_DECAY = 0.58
MARKET_FACTORS = frozenset({"MKT_RF", "Q_MKT"})
SOURCE = "McLean & Pontiff (2016), doi:10.1111/jofi.12365"


@dataclass(frozen=True)
class FactorPrior:
    factor: str
    base_mean_ann: float
    base_start: str
    base_end: str
    base_window: str            # "original sample" or "pre-publication period"
    low: float
    high: float
    op_tstat: float | None = None
    replication_quality: str | None = None
    note: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def factor_prior(lib: FactorLibrary, factor: str) -> FactorPrior:
    info = lib.info.loc[factor]
    r = lib.returns[factor].dropna()
    s0, s1 = info.get("sample_start"), info.get("sample_end")
    if info.get("source") == "cz" and pd.notna(s0) and pd.notna(s1):
        start, end, kind = pd.Period(f"{int(s0)}-01", "M"), pd.Period(f"{int(s1)}-12", "M"), "original sample"
    else:
        start, end, kind = r.index[0], pd.Period(f"{int(info['year']) - 1}-12", "M"), "pre-publication period"
    window = r.loc[start:end]
    if len(window) < 24:
        raise ValueError(f"{factor}: only {len(window)} months in the {kind}; no prior can be computed.")
    base = float(window.mean() * 12)

    note = None
    if factor in MARKET_FACTORS:
        low, high = base * (1 - OOS_DECAY), base
        note = "Market premium: only the out-of-sample haircut applies."
    else:
        low, high = base * (1 - POST_PUB_DECAY), base * (1 - OOS_DECAY)
    if base <= 0:
        low, high = 0.0, 0.0
        note = "The factor earned no positive premium in its reference window."
    elif str(info.get("op_weighting", "")).upper() == "EW":
        note = (
            "Equal-weighted portfolios, as in the original paper: small stocks carry as much weight as large "
            "ones, and returns are before trading costs. A value-weighted or investable version usually earns less."
        )

    tstat = info.get("op_tstat")
    rq = info.get("replication_quality")
    return FactorPrior(
        factor=factor, base_mean_ann=base, base_start=str(window.index[0]), base_end=str(window.index[-1]),
        base_window=kind, low=low, high=high,
        op_tstat=float(tstat) if pd.notna(tstat) else None,
        replication_quality=str(rq) if pd.notna(rq) else None,
        note=note,
    )


def composite_prior(spec: StrategySpec, lib: FactorLibrary) -> tuple[float, float, list[FactorPrior]]:
    """Expected annual-return range for the spec's composite, plus each component's prior."""
    priors = [factor_prior(lib, f) for f in spec.factor_ids]
    if spec.factor_layer.combination == "theory_weights":
        w = [c.weight for c in spec.components]
    else:  # equal weights; inverse-vol and optimized weights are not known before testing
        w = [1 / len(priors)] * len(priors)
    low = sum(wi * p.low for wi, p in zip(w, priors))
    high = sum(wi * p.high for wi, p in zip(w, priors))
    return low, high, priors

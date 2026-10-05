"""Mechanism tests (ARCHITECTURE §11): why might the premium exist?

Each test is a fixed template with a prediction stated before the data are seen.
The pre-registration record fixes which templates run (``spec.mechanism_tests``);
the LLM never designs a test after seeing results.

* ``sentiment`` (behavioral bias): the premium is larger after periods of high
  investor sentiment (Stambaugh, Yu & Yuan 2012). Uses the Baker-Wurgler index,
  lagged one month.
* ``risk_regime`` (risk compensation): the strategy does worse in bad times:
  NBER recessions, deep market drawdowns, and high-VIX months (state lagged one
  month where it is a market variable).
* ``publication_decay`` (data mining or arbitrage): the premium is smaller
  after the original sample and after publication (McLean & Pontiff 2016).

Verdicts are "consistent with", "inconsistent with", or "inconclusive": a
coefficient counts only when its Newey-West |t| clears the cutoff, which is 1.96
for a single test and Bonferroni-adjusted when one template tests several
regimes (2.39 for three). Nothing here proves a mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..composite.engine import FactorResult
from ..composite.report import ols_nw
from ..factors import FactorLibrary
from ..factors import regimes as R
from scipy.stats import norm

T_CUTOFF = 1.96
DRAWDOWN_CUT = -0.20
TEMPLATE_FOR_MECHANISM = {
    "behavioral_bias": ["sentiment", "publication_decay"],
    "risk_compensation": ["risk_regime", "publication_decay"],
    "limits_to_arbitrage": ["publication_decay"],   # size and illiquidity tests need stock data (v2)
    "market_friction": ["publication_decay"],        # net-of-cost comparison needs stock data (v2)
}


@dataclass
class TestRow:
    label: str
    estimate: float          # difference or slope, annualized where it is a return
    t_stat: float
    months: int
    detail: str


@dataclass
class MechanismResult:
    template: str
    mechanism: str
    prediction: str
    rows: list[TestRow]
    verdict: str             # "consistent with", "inconsistent with", "inconclusive"
    explanation: str
    notes: list[str] = field(default_factory=list)


def bonferroni_cutoff(n_tests: int, alpha: float = 0.05) -> float:
    return float(norm.ppf(1 - alpha / 2 / max(n_tests, 1)))


def _verdict(est: float, t: float, predicted_sign: int, cutoff: float = T_CUTOFF) -> str:
    if not np.isfinite(t) or abs(t) < cutoff:
        return "inconclusive"
    return "consistent with" if np.sign(est) == predicted_sign else "inconsistent with"


def _dummy_test(r: pd.Series, flag: pd.Series, label: str, on: str, off: str) -> TestRow:
    """Regress returns on a 0/1 state dummy: the slope is (mean in state) - (mean otherwise)."""
    data = pd.concat([r.rename("r"), flag.astype(float).rename("d")], axis=1).dropna()
    n_on = int(data["d"].sum())
    if n_on < 12 or len(data) - n_on < 12:
        return TestRow(label, np.nan, np.nan, len(data), f"Too few {on} months ({n_on}) to test.")
    fit = ols_nw(data["r"], data[["d"]])
    m_on = data.loc[data["d"] == 1, "r"].mean() * 12
    m_off = data.loc[data["d"] == 0, "r"].mean() * 12
    return TestRow(label, fit["betas"]["d"] * 12, fit["beta_t"]["d"], len(data),
                   f"{on}: {m_on:.1%} per year ({n_on} months); {off}: {m_off:.1%} per year ({len(data) - n_on} months).")


# ---------------------------------------------------------------- templates

def sentiment_test(result: FactorResult, sentiment: pd.DataFrame) -> MechanismResult:
    r = result.returns["composite"]
    col = "SENT_ORTH" if sentiment["SENT_ORTH"].notna().sum() > 120 else "SENT"
    s = sentiment[col].shift(1).reindex(r.index)         # sentiment known at the end of the previous month
    z = (s - s.mean()) / s.std()
    data = pd.concat([r.rename("r"), z.rename("z")], axis=1).dropna()
    rows = []
    if len(data) >= 60:
        fit = ols_nw(data["r"], data[["z"]])
        rows.append(TestRow(
            "Slope on lagged sentiment (per standard deviation)", fit["betas"]["z"] * 12, fit["beta_t"]["z"],
            fit["months"], f"Baker-Wurgler {col}, lagged one month; {data.index[0]} to {data.index[-1]}."))
        high = s.reindex(data.index) > s.reindex(data.index).median()
        rows.append(_dummy_test(data["r"], high, "High minus low sentiment", "after high sentiment",
                                "after low sentiment"))
    main = rows[0] if rows else TestRow("", np.nan, np.nan, 0, "")
    verdict = _verdict(main.estimate, main.t_stat, +1)
    expl = {
        "consistent with": "Returns are higher after high-sentiment months, as the behavioral explanation predicts.",
        "inconsistent with": "Returns are lower after high-sentiment months, the opposite of the behavioral prediction.",
        "inconclusive": "Returns do not differ reliably with lagged sentiment.",
    }[verdict]
    return MechanismResult("sentiment", "behavioral bias",
                           "Higher returns after high investor sentiment (Stambaugh, Yu & Yuan 2012).",
                           rows, verdict, expl)


def risk_regime_test(result: FactorResult, recessions: pd.Series, market_total: pd.Series,
                     vix: pd.Series | None) -> MechanismResult:
    r = result.returns["composite"]
    rows = [_dummy_test(r, recessions.reindex(r.index) == 1, "Recession minus expansion",
                        "NBER recessions", "expansions")]
    dd = R.market_drawdown(market_total).shift(1).reindex(r.index)     # drawdown known at the start of the month
    rows.append(_dummy_test(r, dd <= DRAWDOWN_CUT, "Deep market drawdown minus other months",
                            f"market more than {abs(DRAWDOWN_CUT):.0%} below its peak", "other months"))
    notes = []
    if vix is not None and vix.notna().sum() > 60:
        v = vix.shift(1).reindex(r.index)
        cut = v.quantile(2 / 3)
        rows.append(_dummy_test(r, (v >= cut).where(v.notna()), "High-VIX minus other months",
                                f"lagged VIX in its top third (>= {cut:.0f})", "other months"))
    else:
        notes.append("VIX data are unavailable for this sample; that regime was skipped.")

    usable = [x for x in rows if np.isfinite(x.t_stat)]
    cut = bonferroni_cutoff(len(usable))
    sig = [x for x in usable if abs(x.t_stat) >= cut]
    near = [x for x in usable if T_CUTOFF <= abs(x.t_stat) < cut]
    lower = [x for x in sig if x.estimate < 0]
    higher = [x for x in sig if x.estimate > 0]
    if lower and not higher:
        verdict = "consistent with"
        expl = (f"The strategy earns significantly less in {len(lower)} of {len(usable)} bad-times regimes, as a "
                "risk premium predicts: it tends to lose when investors can least afford it.")
    elif higher and not lower:
        verdict = "inconsistent with"
        expl = (f"The strategy earns significantly more in {len(higher)} of {len(usable)} bad-times regimes: it "
                "behaves more like a hedge than like compensation for bad-times risk.")
    else:
        verdict = "inconclusive"
        expl = "Returns in bad times do not differ reliably from returns in other months."
    if len(usable) > 1:
        notes.append(f"{len(usable)} regimes were tested, so each needs |t| >= {cut:.2f} (Bonferroni) to count.")
    for x in near:
        notes.append(f"{x.label} has t = {x.t_stat:.2f}: significant on its own, but not after adjusting for "
                     f"{len(usable)} tests.")
    return MechanismResult("risk_regime", "risk compensation",
                           "Lower returns in bad times: recessions, market drawdowns, high volatility.",
                           rows, verdict, expl, notes)


def publication_decay_test(result: FactorResult, lib: FactorLibrary) -> MechanismResult:
    """Per component and for the composite: the change in mean return after each paper's
    original sample and after publication (McLean & Pontiff 2016)."""
    rows, notes = [], []
    post_flags = []
    for f in result.spec.factor_ids:
        info = lib.info.loc[f]
        r = result.returns[f]
        pub = pd.Period(f"{int(info['year'])}-01", "M")
        post = pd.Series(r.index >= pub, index=r.index)
        rows.append(_dummy_test(r, post, f"{f}: after minus before publication ({pub.year})",
                                "after publication", "before"))
        s1 = info.get("sample_end")
        if pd.notna(s1):
            end = pd.Period(f"{int(s1)}-12", "M")
            between = pd.Series((r.index > end) & (r.index < pub), index=r.index)
            if between.sum() >= 12:
                rows[-1].detail += f" Out-of-sample but before publication ({end + 1} to {pub - 1}): {between.sum()} months."
        post_flags.append(post)
    if len(result.spec.factor_ids) > 1:
        latest = max(pd.Period(f"{int(lib.info.loc[f, 'year'])}-01", "M") for f in result.spec.factor_ids)
        r = result.returns["composite"]
        rows.append(_dummy_test(r, pd.Series(r.index >= latest, index=r.index),
                                f"Composite: after minus before the latest publication ({latest.year})",
                                "after", "before"))
    main = rows[-1]
    verdict = _verdict(main.estimate, main.t_stat, -1)
    expl = {
        "consistent with": "Returns are significantly lower after publication. This is consistent with data "
                           "mining in the original studies, and also with investors trading the anomaly away "
                           "once it became known; the test cannot separate the two.",
        "inconsistent with": "Returns are significantly higher after publication, the opposite of the usual decay.",
        "inconclusive": "Returns after publication do not differ reliably from returns before it.",
    }[verdict]
    notes.append("McLean & Pontiff (2016) find average anomaly returns about 26% lower out of sample and 58% "
                 "lower after publication.")
    return MechanismResult("publication_decay", "data mining or arbitrage",
                           "Lower returns after the original sample and after publication.",
                           rows, verdict, expl, notes)


# ------------------------------------------------------------------ runner

@dataclass
class StateData:
    sentiment: Callable[[], pd.DataFrame] = R.load_sentiment
    recessions: Callable[[], pd.Series] = R.load_recessions
    vix: Callable[[], pd.Series] = R.load_vix


def run_mechanism_tests(result: FactorResult, lib: FactorLibrary, data: StateData | None = None) -> list[MechanismResult]:
    data = data or StateData()
    out = []
    for name in result.spec.mechanism_tests:
        if name == "sentiment":
            out.append(sentiment_test(result, data.sentiment()))
        elif name == "risk_regime":
            market = lib.returns["MKT_RF"] + lib.rf.reindex(lib.returns.index).fillna(0.0)
            try:
                vix = data.vix()
            except Exception:
                vix = None
            out.append(risk_regime_test(result, data.recessions(), market, vix))
        elif name == "publication_decay":
            out.append(publication_decay_test(result, lib))
    return out


__all__ = [
    "MechanismResult", "StateData", "TEMPLATE_FOR_MECHANISM", "TestRow", "bonferroni_cutoff", "publication_decay_test",
    "risk_regime_test", "run_mechanism_tests", "sentiment_test",
]

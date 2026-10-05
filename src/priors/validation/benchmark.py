"""Literature benchmark check (ARCHITECTURE §7.1).

The question is whether the student's result looks like what the literature
leads us to expect. Each component is compared in two windows:

* **Reference window** (the original sample, or the pre-publication period):
  the result should be close to the factor's base mean. A large gap points to a
  construction difference or a data problem.
* **After the reference window** (out of sample relative to the paper): the
  result should fall in the decayed range [low, high] from ``expected.py``.

The composite is compared with its expected range after the latest reference
window among its components. Comparisons are statistical: a result counts as
"far above" or "far below" only when the 95% confidence interval of its mean
lies entirely outside the expected range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..composite import FactorResult
from ..engine.metrics import newey_west_t
from ..factors import FactorLibrary
from ..literature.expected import FactorPrior, composite_prior

MIN_MONTHS = 36


@dataclass
class WindowCheck:
    label: str
    start: str
    end: str
    months: int
    mean_ann: float
    ci_low: float
    ci_high: float
    expected_low: float
    expected_high: float
    verdict: str          # "far above", "above", "within", "below", "far below", "too short"


@dataclass
class BenchmarkCheck:
    composite: WindowCheck | None
    components: list[WindowCheck]
    priors: list[FactorPrior]
    flags: list[str] = field(default_factory=list)


def _check(label: str, r: pd.Series, low: float, high: float) -> WindowCheck:
    r = r.dropna()
    n = len(r)
    if n < MIN_MONTHS:
        return WindowCheck(label, str(r.index[0]) if n else "", str(r.index[-1]) if n else "", n,
                           float("nan"), float("nan"), float("nan"), low, high, "too short")
    mean = float(r.mean() * 12)
    t = newey_west_t(r)
    se = abs(mean / t) if t and math.isfinite(t) and t != 0 else float("nan")
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    if lo > high:
        verdict = "far above"
    elif hi < low:
        verdict = "far below"
    elif mean > high:
        verdict = "above"
    elif mean < low:
        verdict = "below"
    else:
        verdict = "within"
    return WindowCheck(label, str(r.index[0]), str(r.index[-1]), n, mean, lo, hi, low, high, verdict)


def benchmark_check(result: FactorResult, lib: FactorLibrary) -> BenchmarkCheck:
    low, high, priors = composite_prior(result.spec, lib)
    rets = result.returns
    comps, flags = [], []
    for p in priors:
        r = rets[p.factor]
        ref_end = pd.Period(p.base_end, "M")
        ref = _check(f"{p.factor}: reference window", r.loc[pd.Period(p.base_start, "M"): ref_end],
                     p.base_mean_ann, p.base_mean_ann)
        if ref.verdict != "too short":
            # In the reference window the target is the base mean itself; judge by the CI only.
            inside = ref.ci_low <= p.base_mean_ann <= ref.ci_high
            ref.verdict = "consistent" if inside else ("far above" if ref.ci_low > p.base_mean_ann else "far below")
            if not inside:
                flags.append(
                    f"{p.factor} earns {ref.mean_ann:.1%} per year in its reference window ({ref.start} to "
                    f"{ref.end}), different from the {p.base_mean_ann:.1%} reference. Check the construction."
                )
        after = _check(f"{p.factor}: after the reference window", r.loc[ref_end + 1:], p.low, p.high)
        comps += [ref, after]

    latest_end = max(pd.Period(p.base_end, "M") for p in priors)
    comp = _check("Composite: after all reference windows", rets["composite"].loc[latest_end + 1:], low, high)
    if comp.verdict == "far above":
        flags.append(
            f"The composite earns {comp.mean_ann:.1%} per year after the papers' samples, well above the "
            f"expected {low:.1%} to {high:.1%}. Results this far above the literature call for a check for "
            "look-ahead bias, data errors, or overfitting."
        )
    elif comp.verdict == "far below":
        flags.append(
            f"The composite earns {comp.mean_ann:.1%} per year after the papers' samples, well below the "
            f"expected {low:.1%} to {high:.1%}. This is consistent with stronger-than-average decay, or with "
            "a construction that differs from the original papers."
        )
    elif comp.verdict in ("above", "below"):
        flags.append(
            f"The composite earns {comp.mean_ann:.1%} per year after the papers' samples, {comp.verdict} the "
            f"expected {low:.1%} to {high:.1%}, but the 95% confidence interval ({comp.ci_low:.1%} to "
            f"{comp.ci_high:.1%}) overlaps that range, so the difference is not statistically clear."
        )
    elif comp.verdict == "too short":
        flags.append("Too few months after the papers' samples to compare with the literature.")
    return BenchmarkCheck(composite=comp, components=comps, priors=priors, flags=flags)

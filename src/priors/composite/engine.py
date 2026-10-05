"""Factor-layer backtest: combine published factor returns into one composite.

The sample is the set of months in which every component has a return (the
common sample). It is split by the calendar holdout rule (``priors.holdout``):

* ``period="in_sample"``: months before the holdout start. The default.
* ``period="holdout"``: the sealed months. Opening the holdout is a logged,
  once-per-hypothesis event; the registry enforces that, not this function.

Weights may use history from before the sample start (for example, trailing
volatility), but never from the month being evaluated or later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .. import __version__
from ..engine.metrics import summarize
from ..factors import FactorLibrary
from ..holdout import holdout_start as default_holdout_start
from ..spec.strategy import StrategySpec
from . import weights as W

MIN_MONTHS = 36

BEFORE_COSTS = (
    "Factor returns are before trading costs. Net-of-cost results come from the stock layer."
)
EXPLORATORY = "Exploratory, not theory-backed: this result did not pass the theory gate."


@dataclass
class FactorResult:
    spec: StrategySpec
    period: Literal["in_sample", "holdout"]
    returns: pd.DataFrame            # 'composite' plus one column per component
    weights: pd.DataFrame            # weight of each component, per month
    sample: dict                     # start, end, months, and which factor limits the sample
    holdout_start: pd.Period
    warnings: list[str] = field(default_factory=list)
    train_window: tuple[pd.Period, pd.Period] | None = None   # 'optimized' only
    data_snapshot: str = ""
    engine_version: str = __version__

    @property
    def metrics(self) -> dict[str, dict[str, float]]:
        return {c: summarize(self.returns[c], 12) for c in self.returns.columns}

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics)

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "spec_fingerprint": self.spec.fingerprint(),
            "data_snapshot": self.data_snapshot,
            "engine_version": self.engine_version,
            "holdout_start": str(self.holdout_start),
            "period": self.period,
        }


def run_factor_backtest(
    spec: StrategySpec,
    lib: FactorLibrary,
    period: Literal["in_sample", "holdout"] = "in_sample",
    holdout_from: pd.Period | str | None = None,
) -> FactorResult:
    ids = spec.factor_ids
    hs = pd.Period(holdout_from, "M") if holdout_from is not None else default_holdout_start()
    full = lib.get(ids, how="common")
    if full.empty:
        raise ValueError(f"The factors {ids} have no months in common.")

    fl = spec.factor_layer
    start = max(full.index[0], pd.Period(fl.start, "M")) if fl.start else full.index[0]
    in_sample = full.loc[start: hs - 1]
    holdout = full.loc[hs:]
    warnings: list[str] = [BEFORE_COSTS]
    if spec.status == "exploratory":
        warnings.append(EXPLORATORY)

    # ---------------------------------------------------------- weights
    train_window = None
    if fl.combination == "equal_weight":
        w = W.equal_weight(full)
    elif fl.combination == "theory_weights":
        w = W.fixed_weights(full, {c.factor: c.weight for c in spec.components})
    elif fl.combination == "inverse_vol":
        w = W.inverse_vol(full, fl.vol_lookback_months)
    else:  # optimized
        n_train = int(len(in_sample) * fl.train_fraction)
        if n_train < MIN_MONTHS:
            raise ValueError(f"Training window has {n_train} months; at least {MIN_MONTHS} are needed.")
        train = in_sample.iloc[:n_train] if period == "in_sample" else in_sample
        w_star = W.max_sharpe_long_only(train)
        w = W.fixed_weights(full, w_star.to_dict())
        train_window = (train.index[0], train.index[-1])
        if period == "in_sample":
            in_sample = in_sample.iloc[n_train:]
        warnings.append(
            f"Optimized weights were estimated on {train_window[0]} to {train_window[1]} and are evaluated "
            "only on later months. Optimized weights often look better in the training window than they "
            "turn out to be; compare them with equal weights."
        )

    # ---------------------------------------------------------- sample
    window = in_sample if period == "in_sample" else holdout
    window = window.loc[w.loc[window.index].notna().all(axis=1)]
    if fl.combination == "inverse_vol" and len(window) < len(in_sample if period == "in_sample" else holdout):
        warnings.append(
            f"Inverse-volatility weights need {W.MIN_VOL_MONTHS} months of history; "
            f"the evaluated sample starts at {window.index[0] if len(window) else 'n/a'}."
        )
    if period == "in_sample" and len(window) < MIN_MONTHS:
        raise ValueError(f"Only {len(window)} in-sample months; at least {MIN_MONTHS} are needed.")
    if period == "holdout":
        if window.empty:
            raise ValueError("No holdout months with data for these factors.")
        warnings.append(f"Holdout used: these results cover the sealed period from {hs}.")
        lib_end = lib.returns.index[-1]
        if window.index[-1] < lib_end:
            short = [i for i in ids if lib.returns[i].last_valid_index() == window.index[-1]]
            warnings.append(
                f"The holdout ends at {window.index[-1]} because data for {', '.join(short)} end there."
            )

    wt = w.loc[window.index]
    composite = (wt * window).sum(axis=1)
    returns = pd.concat([composite.rename("composite"), window], axis=1)

    firsts = {i: lib.returns[i].first_valid_index() for i in ids}
    limited_by = max(firsts, key=lambda i: firsts[i])
    sample = {
        "start": window.index[0],
        "end": window.index[-1],
        "months": len(window),
        "limited_by": (
            limited_by
            if len(ids) > 1 and start == full.index[0] and any(firsts[i] < full.index[0] for i in ids)
            else None
        ),
    }
    if sample["limited_by"]:
        warnings.append(
            f"The common sample starts at {full.index[0]} because {limited_by} has no earlier data."
        )

    return FactorResult(
        spec=spec,
        period=period,
        returns=returns,
        weights=wt,
        sample=sample,
        holdout_start=hs,
        warnings=warnings,
        train_window=train_window,
        data_snapshot=lib.snapshot_id,
    )

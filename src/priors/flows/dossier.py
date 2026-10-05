"""The facts dossier: every number an agent may use, computed by code.

Agents that discuss results (Performance Analyst, Advocate, Reviewer 2,
Dr. Dong, Explainer) receive the dossier as text and are told to use no other
numbers. ``unverified_numbers`` then checks their output: any number that does
not appear in the dossier (allowing for rounding) is reported, so a figure the
model made up is caught before a student sees it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from ..composite import FactorResult
from ..composite.report import CompositeReport, ols_nw
from ..mechanisms import MechanismResult
from ..stocklayer import GapReport, Holdings, StockResult
from ..validation import ValidationReport
from ..zoo import ZooReport


def _pct(x: float) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.1%}"


def _num(x: float, d: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{d}f}"


@dataclass
class Dossier:
    sections: dict[str, list[str]] = field(default_factory=dict)
    risk_notes: list[str] = field(default_factory=list)     # must reach the student in every mode

    def add(self, section: str, line: str) -> None:
        self.sections.setdefault(section, []).append(line)

    def text(self) -> str:
        parts = []
        for name, lines in self.sections.items():
            parts.append(f"## {name}\n" + "\n".join(f"- {x}" for x in lines))
        if self.risk_notes:
            parts.append("## Required risk notes\n" + "\n".join(f"- {x}" for x in self.risk_notes))
        return "\n\n".join(parts)


def build_dossier(
    factor: FactorResult,
    validation: ValidationReport | None = None,
    composite: CompositeReport | None = None,
    zoo: ZooReport | None = None,
    mechanisms: list[MechanismResult] | None = None,
    stock: StockResult | None = None,
    gap: GapReport | None = None,
    holdings: Holdings | None = None,
    ledger_summary: dict | None = None,
    status: str = "exploratory",
) -> Dossier:
    d = Dossier()
    spec = factor.spec
    m = factor.metrics["composite"]
    d.add("Strategy", f"Components: {', '.join(spec.factor_ids)}; combination: {spec.factor_layer.combination}.")
    d.add("Strategy", f"Status: {status}.")
    d.add("Factor layer", f"Sample {factor.sample['start']} to {factor.sample['end']} ({factor.sample['months']} months); "
                          f"the holdout from {factor.holdout_start} is sealed." if factor.period == "in_sample"
          else f"HOLDOUT results, {factor.sample['start']} to {factor.sample['end']}.")
    d.add("Factor layer", f"Composite: mean {_pct(m['mean_ann'])} per year, volatility {_pct(m['vol_ann'])}, "
                          f"Sharpe {_num(m['sharpe'])} (95% CI {_num(m['sharpe_ci_low'])} to {_num(m['sharpe_ci_high'])}), "
                          f"Newey-West t {_num(m['t_stat_mean_nw'])}, max drawdown {_pct(m['max_drawdown'])}.")
    for f in spec.factor_ids:
        fm = factor.metrics[f]
        d.add("Factor layer", f"{f}: mean {_pct(fm['mean_ann'])}, Sharpe {_num(fm['sharpe'])}.")
    d.risk_notes += [w for w in factor.warnings]

    if validation:
        b = validation.benchmark
        if b and b.composite and b.composite.verdict != "too short":
            c = b.composite
            d.add("Literature benchmark", f"After the papers' samples ({c.start} to {c.end}): mean {_pct(c.mean_ann)} "
                                          f"(95% CI {_pct(c.ci_low)} to {_pct(c.ci_high)}) versus expected {_pct(c.expected_low)} "
                                          f"to {_pct(c.expected_high)}: {c.verdict}.")
        ds = validation.deflated
        d.add("Multiple testing", f"Trials in this project: {ds.n_trials}. Deflated Sharpe probability {_num(ds.dsr)}; "
                                  f"luck benchmark Sharpe {_num(ds.sr0_ann)}; passes t > 3: {'yes' if ds.passes_hlz else 'no'}.")
        p = validation.placebo
        if p:
            d.add("Placebo", f"Beats {_pct(p.percentile)} of {len(p.placebo_sharpes)} random {p.k}-factor combinations "
                             f"({p.pool} pool, {p.window_start} to {p.window_end}); best of 10 random: Sharpe "
                             f"{_num(p.best_of_n.get(10, float('nan')))}.")
        st = validation.stability
        if len(st.halves) == 2:
            d.add("Stability", f"Sharpe first half {_num(st.halves['sharpe'].iloc[0])}, second half "
                               f"{_num(st.halves['sharpe'].iloc[1])}; rolling 36-month Sharpe positive "
                               f"{_pct(st.share_positive_rolling)} of the time.")
        for dec, row in st.by_decade.iterrows():
            d.add("Stability", f"{dec}: mean {_pct(row['mean_ann'])}, Sharpe {_num(row['sharpe'])}.")
        d.risk_notes += [f for f in validation.flags if f not in d.risk_notes and ("not" in f or "below" in f or "Exploratory" in f)]

    if composite:
        for f in composite.shapley.index:
            d.add("Composite analysis",
                  f"{f}: Shapley share {_pct(composite.shapley_share[f])}, risk share {_pct(composite.risk_contribution[f])}, "
                  f"Sharpe change if removed {_num(composite.leave_one_out.loc[f, 'change_if_removed'])}, incremental alpha "
                  f"{_pct(composite.incremental_alpha.loc[f, 'alpha_ann'])} (t {_num(composite.incremental_alpha.loc[f, 'alpha_t'])}), "
                  f"theory priority {composite.priority.loc[f, 'theory_priority']}.")
        d.risk_notes.append(composite.guard_rail)

    if zoo:
        for s in zoo.spanning:
            if math.isfinite(s.alpha_t):
                d.add("Factor models", f"{s.model}: alpha {_pct(s.alpha_ann)} per year (t {_num(s.alpha_t)}), R-squared {_num(s.r2)}.")
            else:
                d.add("Factor models", f"{s.model}: not informative (every component is a model factor).")
        if len(zoo.neighbors):
            top = zoo.neighbors.head(3)
            d.add("Factor models", "Most similar published factors (descriptive): " + "; ".join(
                f"{i} ({_num(r.correlation)})" for i, r in top.iterrows()))
        d.add("Factor models", f"Factor-map family: {zoo.map.family} ({len(zoo.map.family_members)} published factors).")

    for mr in mechanisms or []:
        d.add("Mechanism tests", f"{mr.template} ({mr.mechanism}): {mr.verdict}. " + " ".join(
            f"{r.label}: {_pct(r.estimate)} (t {_num(r.t_stat)})." for r in mr.rows if math.isfinite(r.t_stat)))

    if stock:
        sm = stock.metrics
        d.add("Stock layer", f"{stock.spec.stock_layer.holdings} stocks, {stock.spec.stock_layer.rebalance} rebalancing, "
                             f"{stock.spec.stock_layer.cost_bps_one_way:.0f} bp one-way; {stock.sample['start']} to {stock.sample['end']}.")
        d.add("Stock layer", f"Net {_pct(sm['net']['mean_ann'])} per year vs eligible universe {_pct(sm['benchmark']['mean_ann'])}; "
                             f"excess {_pct(sm['excess']['mean_ann'])} (t {_num(sm['excess']['t_stat_mean_nw'])}); "
                             f"turnover {_pct(stock.turnover.mean() * 12)} per year; costs {_pct(stock.returns['cost'].mean() * 12)} per year.")
        fit = ols_nw(stock.returns["net"], stock.returns[["benchmark"]])
        d.add("Stock layer", f"Regressed on the eligible universe: beta {_num(fit['betas']['benchmark'])}, alpha "
                             f"{_pct(fit['alpha_ann'])} per year (t {_num(fit['alpha_t'])}). Use this, not the raw excess, "
                             "for strategies whose beta is far from 1.")
        d.risk_notes += [w for w in stock.warnings if w not in d.risk_notes]
    if gap:
        d.add("Factor-to-stock gap", gap.flags[0])
    if holdings:
        s = holdings.summary
        d.add("Current holdings", f"As of {holdings.as_of}: {s['holdings']} stocks from {s['eligible_stocks']} eligible; "
                                  f"largest sector {s['top_sector']} ({_pct(s['top_sector_weight'])}); median market cap "
                                  f"${s['median_mcap'] / 1e9:.1f}bn.")
        d.risk_notes.append(holdings.label)
    if ledger_summary:
        d.add("Process", f"Trials: {ledger_summary['trials']} ({ledger_summary['preregistered_trials']} pre-registered, "
                         f"{ledger_summary['exploratory_trials']} exploratory); holdouts opened: {ledger_summary['holdouts_opened']}.")
    return d


_NUM = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?%?")
_ALLOWED = {str(y) for y in range(1900, 2100)} | {str(i) for i in range(0, 13)}


def _tolerance(token: str) -> float:
    """Half a unit in the last decimal place the token shows: '0.7' -> 0.05, '12%' -> 0.5."""
    body = token.lstrip("-").rstrip("%")
    places = len(body.split(".")[1]) if "." in body else 0
    return 0.5 * 10 ** (-places) + 1e-9


def unverified_numbers(text: str, dossier_text: str) -> list[str]:
    """Numbers in ``text`` that match no dossier number, allowing for rounding.

    A number matches when its absolute value equals a dossier number of the same
    kind (percent or not), rounded to the decimals it shows; signs are often put
    into words ("a shortfall of 3.7%"). Integers that appear inside a dossier name
    (52 in High52) are accepted. Years and small integers (0-12, counts and list
    markers) are ignored.
    """
    pool = [(tok.endswith("%"), abs(float(tok.rstrip("%")))) for tok in _NUM.findall(dossier_text)]
    missing = []
    for tok in _NUM.findall(text):
        bare = tok.lstrip("-")
        if bare in _ALLOWED:
            continue
        if "." not in bare and not bare.endswith("%") and bare in dossier_text:
            continue
        is_pct, v, tol = tok.endswith("%"), abs(float(tok.rstrip("%"))), _tolerance(tok)
        if not any(p == is_pct and abs(v - w) <= tol for p, w in pool):
            missing.append(tok)
    return sorted(set(missing))

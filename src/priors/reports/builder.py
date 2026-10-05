"""The research report (ARCHITECTURE §12.2): one HTML document for the in-app view, the same
document rendered to PDF for submission.

Rules:
* Exhibits follow the house style: numbered, bold title above, three-line tables, black-and-white
  figures, and a "Note" below that gives only definitions, data sources, and design.
* Exploratory results never appear in the main body. If the current result is exploratory, its
  exhibits move to an appendix and every exhibit carries the exploratory label (§6.2).
* The required risk notes, the trial ledger summary, and a summary of the conversation are always
  included, so the process behind a result is visible.
"""

from __future__ import annotations

import base64
import html
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import jinja2
import markdown as md
import pandas as pd

from ..composite.report import ols_nw
from ..flows.agents import AnalystReview, RefereeReport
from ..flows.session import DrDongReview, Session
from ..registry.registry import SEEN_BEFORE
from . import charts

EXPLORATORY_LABEL = "Exploratory, not theory-backed"
SIGNAL_LABEL = {
    "mom_12_1": "Momentum", "st_reversal": "Reversal", "low_vol": "Low vol", "low_beta": "Low beta",
    "high52": "52-wk high", "size_small": "Small size", "book_to_market": "B/M", "earnings_yield": "E/P",
    "roe": "ROE", "gross_profitability": "Gross prof.", "asset_growth_low": "Low asset gr.",
}
DISCLAIMER = ("Priors is a research and education tool. Nothing in this report is investment advice or a "
              "recommendation to buy or sell any security.")
SOURCES = ("Factor returns: Ken French Data Library; Chen and Zimmermann Open Source Asset Pricing; Hou, Xue and "
           "Zhang q-factors. Stock data: Yahoo Finance via yfinance.")


def pct(x, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.{d}f}%"


def num(x, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def text_html(s: str) -> str:
    """Markdown from an agent, with any raw HTML escaped first."""
    return md.markdown(html.escape(s), extensions=["sane_lists"])


@dataclass
class Exhibit:
    kind: str                          # "table" or "figure"
    title: str
    note: str
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    src: str = ""
    label: str | None = None
    number: str = ""
    left: int = 1                      # number of leading text columns aligned left
    small: bool = False                # smaller type for wide tables
    widths: list[int] | None = None    # column widths in percent


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)      # trusted HTML built here
    bullets: list[str] = field(default_factory=list)          # plain text, escaped by the template
    exhibits: list[Exhibit] = field(default_factory=list)
    html_blocks: list[str] = field(default_factory=list)      # agent text converted with text_html
    compact: bool = False                                     # smaller type for long lists


def _short(s, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "."


def _img(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


@dataclass
class Report:
    title: str
    html: str

    def save_html(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.html, encoding="utf-8")
        return p

    def save_pdf(self, path: str | Path) -> Path:
        from xhtml2pdf import pisa

        p = Path(path)
        with p.open("wb") as f:
            res = pisa.CreatePDF(self.html, dest=f, encoding="utf-8")
        if res.err:
            raise RuntimeError(f"PDF rendering failed with {res.err} error(s).")
        return p


# ----------------------------------------------------------------- sections

def _results_sections(s: Session, label: str | None) -> list[Section]:
    r = s.results
    res = r.outcome.result
    out: list[Section] = []
    period = "holdout" if res.period == "holdout" else "in-sample"
    sample_note = (f"Sample {res.sample['start']} to {res.sample['end']} ({res.sample['months']} months, {period}). "
                   f"Monthly long-short factor returns before trading costs. Sharpe ratios are annualized; the 95% "
                   f"confidence interval uses the Lo (2002) standard error; t-statistics use Newey-West standard errors. "
                   + SOURCES.split(". ")[0] + ".")

    # Factor layer
    sec = Section("Factor-layer results")
    rows = []
    for c in res.returns.columns:
        m = res.metrics[c]
        rows.append([c if c != "composite" else "Composite", pct(m["mean_ann"]), pct(m["vol_ann"]), num(m["sharpe"]),
                     f"[{num(m['sharpe_ci_low'])}, {num(m['sharpe_ci_high'])}]", num(m["t_stat_mean_nw"]),
                     pct(m["max_drawdown"])])
    sec.exhibits.append(Exhibit("table", "Performance of the composite and its components",
                                f"Combination: {res.spec.factor_layer.combination.replace('_', ' ')}. " + sample_note,
                                ["Series", "Mean", "Volatility", "Sharpe", "95% CI", "t (NW)", "Max drawdown"], rows,
                                label=label))
    sec.exhibits.append(Exhibit("figure", "Growth of $1 invested in the composite and its components",
                                "Cumulative product of one plus the monthly return, log scale. " + sample_note,
                                src=_img(charts.cumulative_growth(res.returns)), label=label))
    if len(r.validation.stability.rolling_sharpe):
        sec.exhibits.append(Exhibit("figure", "Rolling 36-month Sharpe ratio of the composite",
                                    "Annualized Sharpe ratio over the trailing 36 months.",
                                    src=_img(charts.rolling_sharpe(r.validation.stability.rolling_sharpe)), label=label))
    out.append(sec)

    # Validation
    v = r.validation
    sec = Section("Validation")
    if v.benchmark:
        rows = []
        for c in [v.benchmark.composite] + v.benchmark.components:
            if c is None or c.verdict == "too short":
                continue
            rows.append([c.label, f"{c.start} to {c.end}", pct(c.mean_ann), f"[{pct(c.ci_low)}, {pct(c.ci_high)}]",
                         f"{pct(c.expected_low)} to {pct(c.expected_high)}", c.verdict])
        sec.exhibits.append(Exhibit("table", "Comparison with the literature",
                                    "Reference window: the original paper's sample (Chen and Zimmermann documentation) or, "
                                    "when not documented, the period before publication. Expected range after the reference "
                                    "window: reference mean times one minus the McLean and Pontiff (2016) decay estimates "
                                    "(58% after publication, 26% out of sample). 95% confidence intervals use Newey-West "
                                    "standard errors.",
                                    ["Window", "Months", "Mean", "95% CI", "Expected", "Verdict"], rows, left=2, label=label))
    d = v.deflated
    sec.paragraphs.append(
        f"<b>Multiple testing.</b> Distinct specifications tested in this project: {d.n_trials}. Deflated Sharpe "
        f"probability (Bailey and López de Prado, 2014): {num(d.dsr)}. Expected best Sharpe ratio among {d.n_trials} "
        f"strategies with no edge: {num(d.sr0_ann)}. Newey-West t-statistic {num(d.t_stat_nw)} "
        f"({'meets' if d.passes_hlz else 'does not meet'} the Harvey, Liu and Zhu (2016) hurdle of 3.0).")
    if v.placebo:
        p = v.placebo
        note = (f"Each draw combines {p.k} predictor{'s' if p.k > 1 else ''} chosen at random from {p.pool_size} published "
                f"predictors ({p.pool} versions) with the same method as the strategy, over {p.window_start} to "
                f"{p.window_end}; {len(p.placebo_sharpes):,} draws, random seed {p.seed}.")
        sec.exhibits.append(Exhibit("figure", "The strategy against random combinations of published predictors",
                                    note, src=_img(charts.placebo_histogram(p.placebo_sharpes, p.student_sharpe)), label=label))
        sec.exhibits.append(Exhibit("figure", "Expected best Sharpe ratio when trying N random combinations",
                                    note + " The curve shows the average of the maximum over N draws.",
                                    src=_img(charts.best_of_n(p.best_of_n, p.student_sharpe)), label=label))
    rows = [[str(i), str(int(x["months"])), pct(x["mean_ann"]), num(x["sharpe"]), num(x["t_stat_mean_nw"]),
             pct(x["max_drawdown"])] for i, x in v.stability.by_decade.iterrows()]
    if rows:
        sec.exhibits.append(Exhibit("table", "Composite performance by decade",
                                    "Decades with at least 24 months of data. " + sample_note,
                                    ["Decade", "Months", "Mean", "Sharpe", "t (NW)", "Max drawdown"], rows, label=label))
    out.append(sec)

    # Multi-factor
    if r.composite is not None:
        c = r.composite
        sec = Section("Multi-factor analysis")
        rows = []
        for f in c.shapley.index:
            rows.append([f, str(c.priority.loc[f, "theory_priority"]) if pd.notna(c.priority.loc[f, "theory_priority"]) else "-",
                         str(c.priority.loc[f, "empirical_rank"]), pct(c.shapley_share[f]), pct(c.risk_contribution[f]),
                         num(c.leave_one_out.loc[f, "change_if_removed"]),
                         f"{pct(c.incremental_alpha.loc[f, 'alpha_ann'])} ({num(c.incremental_alpha.loc[f, 'alpha_t'])})"])
        sec.exhibits.append(Exhibit("table", "Contribution of each component",
                                    "Theory priority: set before testing (1 = strongest evidence). Empirical rank: by Shapley "
                                    "contribution to the composite Sharpe ratio (exact over all subsets). Risk share: "
                                    "component's share of composite variance. Removal effect: change in composite Sharpe "
                                    "when the component is dropped. Incremental alpha: intercept from regressing the "
                                    "component on the other components, annualized, Newey-West t in parentheses.",
                                    ["Factor", "Theory priority", "Empirical rank", "Sharpe share", "Risk share",
                                     "Removal effect", "Incremental alpha (t)"], rows, label=label))
        sec.exhibits.append(Exhibit("figure", "Sharpe contribution and risk share by component",
                                    "Open bars: share of the composite Sharpe ratio (Shapley). Hatched bars: share of "
                                    "composite variance.", src=_img(charts.shapley_bars(c.shapley_share, c.risk_contribution)),
                                    label=label))
        sec.bullets.append(c.guard_rail)
        out.append(sec)

    # Zoo
    if r.zoo is not None:
        z = r.zoo
        sec = Section("Position in the factor zoo")
        rows = []
        for sp in z.spanning:
            if math.isfinite(sp.alpha_t):
                rows.append([sp.model, f"{sp.start} to {sp.end}", pct(sp.alpha_ann), num(sp.alpha_t), num(sp.r2)])
            else:
                rows.append([sp.model, f"{sp.start} to {sp.end}", "-", "-", "not informative"])
        sec.exhibits.append(Exhibit("table", "Alpha against standard factor models",
                                    "Time-series regression of the composite's monthly return on each model's factors, chosen "
                                    "before testing. Alpha annualized; Newey-West t. A model containing every component "
                                    "explains the composite by construction and is marked not informative.",
                                    ["Model", "Sample", "Alpha", "t (NW)", "R-squared"], rows, left=2, label=label))
        rows = [[i, str(x["name"])[:60], str(int(x["year"])) if pd.notna(x["year"]) else "", num(x["correlation"])]
                for i, x in z.neighbors.head(8).iterrows()]
        sec.exhibits.append(Exhibit("table", "Most similar published factors (descriptive)",
                                    "Correlation of monthly returns with the composite over overlapping months (at least 120). "
                                    "Selected after seeing the results; descriptive only.",
                                    ["Factor", "Description", "Year", "Correlation"], rows, left=2, label=label))
        sec.exhibits.append(Exhibit("figure", "Map of published factors (descriptive)",
                                    "Classical multidimensional scaling of the correlation distance sqrt(2(1 - rho)) among "
                                    "published factors from 1963-07. Filled markers: factors in the same average-linkage "
                                    "cluster as the strategy (average correlation about 0.4 or more). Star: the strategy.",
                                    src=_img(charts.factor_map(z.map.coords, family=z.map.family_members)), label=label))
        out.append(sec)

    # Mechanisms
    if r.mechanisms:
        sec = Section("Mechanism tests")
        rows = []
        for m in r.mechanisms:
            for row in m.rows:
                rows.append([m.template.replace("_", " "), row.label, pct(row.estimate), num(row.t_stat), m.verdict])
        sec.exhibits.append(Exhibit("table", "Tests of the proposed mechanisms",
                                    "Templates fixed before testing. Sentiment: Baker and Wurgler index lagged one month. "
                                    "Risk regimes: NBER recessions; market more than 20% below its peak; lagged VIX in its top "
                                    "third (Bonferroni-adjusted cutoff when several regimes are tested). Publication decay: "
                                    "returns before and after the original paper's publication year. Differences annualized; "
                                    "Newey-West t.",
                                    ["Template", "Test", "Estimate", "t (NW)", "Verdict"], rows, left=2, label=label))
        sec.bullets += [f"{m.template.replace('_', ' ').capitalize()}: {m.verdict} {m.mechanism}. {m.explanation}"
                        for m in r.mechanisms]
        out.append(sec)

    # Stock layer
    if r.stock is not None or r.holdings is not None:
        sec = Section("Implementation on real stocks")
        st = r.stock
        if st is not None:
            sl = st.spec.stock_layer
            rows = []
            for col, name in (("gross", "Portfolio, gross"), ("net", "Portfolio, net of costs"),
                              ("benchmark", "Eligible universe (equal weight)"), ("excess", "Net minus universe")):
                m = st.metrics[col]
                rows.append([name, pct(m["mean_ann"]), pct(m["vol_ann"]), num(m["sharpe"]), num(m["t_stat_mean_nw"]),
                             pct(m["max_drawdown"])])
            sec.exhibits.append(Exhibit("table", "Stock portfolio built from the strategy's signals",
                                        f"Top {sl.holdings} stocks by average signal rank, {sl.weighting.replace('_', ' ')} weights, "
                                        f"{sl.rebalance} rebalancing, {sl.cost_bps_one_way:.0f} bp one-way cost on turnover. "
                                        f"Universe: {sl.universe} (largest U.S. common stocks by market cap). Sample "
                                        f"{st.sample['start']} to {st.sample['end']}. Stock data: Yahoo Finance via yfinance.",
                                        ["Series", "Mean", "Volatility", "Sharpe", "t (NW)", "Max drawdown"], rows,
                                        label=label))
        if r.gap is not None:
            rows = [[k, pct(x["mean_ann"]), num(x["sharpe"]), pct(x["max_drawdown"])] for k, x in r.gap.table.iterrows()]
            sec.exhibits.append(Exhibit("table", "From factor to stock portfolio",
                                        f"Common months {r.gap.start} to {r.gap.end}. Correlation between the factor composite "
                                        f"and the stock portfolio's net excess return: {num(r.gap.correlation)}.",
                                        ["Series", "Mean", "Sharpe", "Max drawdown"], rows, label=label))
            sec.bullets += r.gap.reasons
        h = r.holdings
        if h is not None:
            sig_cols = [c for c in h.table.columns if c.startswith("rank_")]
            rows = [[t, _short(x["name"], 26), _short(x["sector"], 18) if pd.notna(x["sector"]) else "", pct(x["weight"]),
                     num(x["score"])] + [num(x[c]) for c in sig_cols]
                    for t, x in h.table.iterrows()]
            sec.exhibits.append(Exhibit("table", f"Current holdings under the strategy's rules (as of {h.as_of})",
                                        f"{h.label} Universe: {h.universe_label}. Score: average percentile rank across "
                                        "signals among eligible stocks. Signal columns: percentile rank of each signal.",
                                        ["Ticker", "Name", "Sector", "Weight", "Score"]
                                        + [SIGNAL_LABEL.get(c[5:], c[5:]) for c in sig_cols], rows,
                                        left=3, small=True, label=label,
                                        widths=[9, 27, 18, 9, 9] + [round(28 / max(len(sig_cols), 1))] * len(sig_cols)))
            rows = [[k, pct(v)] for k, v in h.sector_weights.items()]
            sec.exhibits.append(Exhibit("table", "Sector weights of the current holdings",
                                        "Sector classification from the Nasdaq stock screener.", ["Sector", "Weight"], rows,
                                        label=label))
        out.append(sec)
    return out


# ------------------------------------------------------------- main report

PLAIN_CONTRIBUTION = (
    "<b>How to read this table.</b> \"Sharpe share\" is how much of the strategy's reward for risk each factor "
    "provided; the shares add up to 100%. A negative share means the factor pulled the combination down. "
    "\"Risk share\" is how much of the ups and downs came from that factor."
)
WARNING_PRIORITY = ("Exploratory", "after the papers' samples", "Harvey-Liu-Zhu", "Survivorship", "before trading costs",
                    "Long-only")


def _by_title(sections: list[Section]) -> dict[str, Exhibit]:
    return {e.title.split(" (as of")[0]: e for sec in sections for e in sec.exhibits}


def _top_warnings(notes: list[str], k: int = 3) -> list[str]:
    picked = []
    for key in WARNING_PRIORITY:
        for n in notes:
            if key in n and n not in picked:
                picked.append(n)
                break
        if len(picked) == k:
            break
    return picked


def _glance(s: Session, review: DrDongReview | None, status: str) -> Section:
    r = s.results
    res = r.outcome.result
    reg = s.registration
    sec = Section("At a glance")
    rows = [["Hypothesis", f"{reg.id}: {reg.hypothesis.statement}" if reg else "None registered"],
            ["Status", "Pre-registered" if status == "preregistered" else EXPLORATORY_LABEL],
            ["Strategy", f"{', '.join(res.spec.factor_ids)}; {res.spec.factor_layer.combination.replace('_', ' ')}"]]
    if status == "preregistered":
        m = res.metrics["composite"]
        rows += [["Sample", f"{res.sample['start']} to {res.sample['end']} ({res.sample['months']} months); "
                            f"holdout sealed from {res.holdout_start}"],
                 ["Result", f"Mean {pct(m['mean_ann'])} per year; Sharpe {num(m['sharpe'])} "
                            f"(95% CI {num(m['sharpe_ci_low'])} to {num(m['sharpe_ci_high'])}); t (NW) {num(m['t_stat_mean_nw'])}"]]
        b = r.validation.benchmark
        if b and b.composite and b.composite.verdict != "too short":
            c = b.composite
            rows.append(["Versus the literature", f"{c.verdict} the expected range after the papers' samples "
                                                  f"({pct(c.mean_ann)} vs {pct(c.expected_low)} to {pct(c.expected_high)})"])
    else:
        rows.append(["Results", "Shown only in the technical appendix, because they are exploratory."])
    if reg is not None:
        h = reg.hypothesis
        rows.insert(2, ["Mechanism", f"{h.mechanism.replace('_', ' ') if h.mechanism else 'n/a'}. In the student's words: "
                                     f"{h.mechanism_explanation}"])
        rows.insert(3, ["Expected return", f"{pct(h.expected_low)} to {pct(h.expected_high)} per year (set before testing)"])
    if review is not None:
        rows.append(["Dr. Dong", review.report.recommendation.replace("_", " ").capitalize()
                     + ("" if s.mode == "dr_dong" else " (full referee report in the technical appendix)")])
    sec.exhibits.append(Exhibit("table", "Summary", "Status is decided by the trial registry, not by the student. "
                                "Registration record and hash: technical appendix.",
                                ["Item", "Value"], rows, left=2, widths=[22, 78]))
    sec.bullets += [f"Warning: {w}" for w in _top_warnings(r.dossier.risk_notes)]
    return sec


def _credibility(s: Session) -> Exhibit:
    v = s.results.validation
    rows = []
    b = v.benchmark
    if b and b.composite and b.composite.verdict != "too short":
        c = b.composite
        rows.append(["Literature benchmark", c.verdict, f"{pct(c.mean_ann)} per year after the papers' samples (95% CI "
                     f"{pct(c.ci_low)} to {pct(c.ci_high)}) vs expected {pct(c.expected_low)} to {pct(c.expected_high)}"])
    d = v.deflated
    rows.append(["Multiple testing", "passes t > 3" if d.passes_hlz else "below t > 3",
                 f"t (NW) {num(d.t_stat_nw)}; deflated Sharpe probability {num(d.dsr)} with {d.n_trials} trial(s)"])
    if v.placebo:
        p = v.placebo
        rows.append(["Random-factor placebo", f"beats {pct(p.percentile, 0)}",
                     f"best of 10 random combinations: Sharpe {num(p.best_of_n.get(10, float('nan')))}; "
                     f"this strategy {num(p.student_sharpe)}"])
    st = v.stability
    if len(st.halves) == 2:
        rows.append(["Stability", f"positive in {pct(st.share_positive_rolling, 0)} of rolling windows",
                     f"Sharpe {num(st.halves['sharpe'].iloc[0])} in the first half, {num(st.halves['sharpe'].iloc[1])} in the second"])
    for m in s.results.mechanisms:
        if m.template == "publication_decay" and m.rows:
            x = m.rows[-1]
            rows.append(["Publication decay", m.verdict, f"{pct(x.estimate)} per year after publication (t {num(x.t_stat)})"])
    return Exhibit("table", "Credibility checks",
                   "Literature benchmark: expected range from the original papers with McLean and Pontiff (2016) decay; "
                   "verdict from the Newey-West 95% confidence interval. Multiple testing: Harvey, Liu and Zhu (2016) "
                   "t > 3 hurdle and the Bailey and López de Prado (2014) deflated Sharpe ratio. Placebo: share of random "
                   "combinations of published predictors, built the same way, with a lower Sharpe ratio. Stability: "
                   "rolling 36-month Sharpe ratio and sample halves. Publication decay: change in mean return after "
                   "publication.", ["Check", "Verdict", "Key statistics"], rows, left=3, widths=[22, 22, 56])


def _main_sections(s: Session, explanation, analyst, review, status) -> list[Section]:
    mode = s.mode
    out = [_glance(s, review, status)]
    if s.registration is not None and s.registration.in_sample_seen:
        out[0].bullets.append(SEEN_BEFORE)
    if status != "preregistered":
        return out

    ex = _by_title(_results_sections(s, None))
    core = Section("Results")
    core.exhibits += [ex["Performance of the composite and its components"],
                      ex["Growth of $1 invested in the composite and its components"], _credibility(s)]
    out.append(core)

    if "Contribution of each component" in ex:
        comp = Section("What each factor contributes")
        if mode == "mentor":
            comp.paragraphs.append(PLAIN_CONTRIBUTION)
        comp.exhibits.append(ex["Contribution of each component"])
        out.append(comp)

    if mode in ("analyst", "dr_dong"):
        deep = Section("Factor models and mechanisms")
        deep.exhibits += [ex[t] for t in ("Alpha against standard factor models", "Tests of the proposed mechanisms") if t in ex]
        out.append(deep)

    r = s.results
    if r.stock is not None or r.holdings is not None:
        stock = Section("On real stocks")
        if r.stock is not None:
            st = r.stock
            fit = ols_nw(st.returns["net"], st.returns[["benchmark"]])
            m = st.metrics
            rows = [["Portfolio, net of costs", pct(m["net"]["mean_ann"]), num(m["net"]["sharpe"])],
                    ["Eligible universe", pct(m["benchmark"]["mean_ann"]), num(m["benchmark"]["sharpe"])],
                    ["Alpha vs universe (beta-adjusted)", f"{pct(fit['alpha_ann'])} (t {num(fit['alpha_t'])})",
                     f"beta {num(fit['betas']['benchmark'])}"]]
            sl = st.spec.stock_layer
            stock.exhibits.append(Exhibit("table", "Stock portfolio summary",
                                          f"Top {sl.holdings} stocks, {sl.rebalance} rebalancing, {sl.cost_bps_one_way:.0f} bp "
                                          f"one-way cost; {st.sample['start']} to {st.sample['end']}. Alpha: intercept from "
                                          "regressing the portfolio's net return on the eligible universe, annualized, "
                                          "Newey-West t.", ["Series", "Mean", "Sharpe / beta"], rows))
        stock.bullets += [n for n in r.notes if "fundamental signal" in n]
        full = ex.get("Current holdings under the strategy's rules")
        if r.holdings is not None and full is not None:
            stock.exhibits.append(Exhibit("table", f"Top 10 current holdings (as of {r.holdings.as_of})",
                                          full.note + " The full list is in the technical appendix.", full.columns,
                                          full.rows[:10], left=3, small=True, widths=full.widths, label=full.label))
        out.append(stock)

    if explanation and mode in ("mentor", "analyst"):
        sec = Section("Explanation")
        sec.html_blocks.append(text_html(explanation))
        out.append(sec)
    if analyst is not None and mode == "analyst":
        sec = Section("Performance Analyst")
        sec.bullets += analyst.diagnosis[:4]
        sec.bullets += [f"Suggestion: {x.suggestion}{' (a new trial; only the holdout can judge it)' if x.new_trial else ''}"
                        for x in analyst.suggestions[:3]]
        out.append(sec)
    if review is not None and mode == "dr_dong":
        rep = review.report
        sec = Section("Dr. Dong's verdict")
        sec.paragraphs.append(f"<b>Recommendation: {rep.recommendation.replace('_', ' ').upper()}.</b>")
        if mode == "dr_dong":
            sec.html_blocks.append("<p><b>Major comments</b></p>"
                                   + text_html("\n".join(f"{i}. {x}" for i, x in enumerate(rep.major_comments, 1))))
        if rep.what_would_change_my_mind:
            sec.html_blocks.append("<p><b>What would change my mind</b></p>" + text_html(
                "\n".join(f"{i}. {x}" for i, x in enumerate(rep.what_would_change_my_mind, 1))))
        sec.html_blocks.append(f"<p><i>{html.escape(review.footer)} The full report and the debate are in the "
                               "technical appendix.</i></p>")
        out.append(sec)
    return out


def _appendix_sections(s: Session, explanation, analyst, review, status) -> list[Section]:
    label = None if status == "preregistered" else EXPLORATORY_LABEL
    out: list[Section] = []
    expl = [r for r in s.registry.ledger.runs(layer="factor") if r["status"] == "exploratory"]
    if expl:
        sec = Section("Exploratory runs in this project")
        rows = [[r["ts"][:10], ", ".join(r.get("factors", [])) or r["spec_fingerprint"],
                 r.get("combination", "").replace("_", " "), f"{r['sample_start']} to {r['sample_end']}",
                 num(r["sharpe"]), r.get("deviation_of") or ""] for r in expl]
        sec.exhibits.append(Exhibit("table", "Exploratory runs", "Every backtest run that was not pre-registered, from the "
                                    "trial ledger. These runs count toward the deflated Sharpe ratio.",
                                    ["Date", "Factors", "Combination", "Sample", "Sharpe", "Deviation of"], rows,
                                    left=4, label=EXPLORATORY_LABEL))
        out.append(sec)

    reg = s.registration
    if reg is not None:
        h = reg.hypothesis
        pre = Section("Pre-registration record")
        pre.bullets += [
            f"Registered {reg.created_at}; record hash {reg.record_hash}.",
            f"Hypothesis: {h.statement}",
            f"Components: {', '.join(f'{c.factor} (priority {c.theory_priority})' for c in reg.spec.components)}; "
            f"combination: {reg.spec.factor_layer.combination.replace('_', ' ')}.",
            f"Mechanism tests: {', '.join(reg.spec.mechanism_tests)}. "
            f"Factor models: {', '.join(reg.spec.factor_layer.spanning_models)}.",
            f"Holdout sealed from {reg.holdout_start}.",
        ]
        out.append(pre)

    if s.card is not None:
        c = s.card
        ev = Section("What the literature says")
        ev.paragraphs.append(f"<b>Claim.</b> {html.escape(c.claim)} <b>Evidence grade:</b> {c.grade} "
                             f"({html.escape(c.grade_justification)})")
        ev.bullets += [f"Supporting: {f.citation.short()}, {f.citation.title}. {f.finding}" for f in c.key_papers]
        ev.bullets += [f"Contrary: {f.citation.short()}, {f.citation.title}. {f.finding}" for f in c.contrary_evidence]
        ev.bullets += [f"Boundary condition: {b}" for b in c.boundary_conditions]
        out.append(ev)

    results = _results_sections(s, label)
    if status != "preregistered":
        for sec in results:
            sec.title = f"Exploratory results: {sec.title.lower()}"
    out += results

    if explanation and (s.mode == "dr_dong" or status != "preregistered"):
        sec = Section("Explanation")
        sec.html_blocks.append(text_html(explanation))
        out.append(sec)
    if analyst is not None:
        sec = Section("Performance Analyst review")
        sec.bullets += analyst.diagnosis
        sec.bullets += [f"Suggestion: {x.suggestion} ({x.rationale})"
                        f"{' New trial: only the holdout can judge it.' if x.new_trial else ''}" for x in analyst.suggestions]
        sec.bullets += [f"Risk: {x}" for x in analyst.key_risks]
        out.append(sec)
    if review is not None:
        rep: RefereeReport = review.report
        sec = Section("Referee report by Dr. Dong (full)")
        sec.paragraphs.append(f"<b>Recommendation: {rep.recommendation.replace('_', ' ').upper()}.</b>")
        sec.html_blocks.append(text_html(rep.summary))
        for title, items in (("Major comments", rep.major_comments), ("Minor comments", rep.minor_comments),
                             ("Tested results", rep.tested_results), ("Descriptive results", rep.descriptive_results),
                             ("What would change my mind", rep.what_would_change_my_mind)):
            if items:
                sec.html_blocks.append(f"<p><b>{title}</b></p>"
                                       + text_html("\n".join(f"{i}. {x}" for i, x in enumerate(items, 1))))
        sec.html_blocks.append(f"<p><i>{html.escape(review.footer)}</i></p>")
        out.append(sec)
        deb = Section("The debate")
        for who, turn in review.debate:
            body = "\n".join(f"- {a.point}" for a in turn.arguments)
            if turn.concessions:
                body += "\n\nConcedes: " + "; ".join(turn.concessions)
            deb.html_blocks.append(f"<p><b>{html.escape(who)}</b></p>" + text_html(body))
        out.append(deb)

    proc = Section("Process")
    led = s.registry.ledger.summary()
    proc.bullets.append(f"Distinct specifications tested: {led['trials']} ({led['preregistered_trials']} pre-registered, "
                        f"{led['exploratory_trials']} exploratory); backtest runs: {led['runs']}; registrations: "
                        f"{led['registrations']}; holdouts opened: {led['holdouts_opened']}.")
    proc.bullets += [f"Student: {t['text']}" for t in s.transcript if t["role"] == "student"]
    if s.card is not None:
        proc.bullets.append(f"Evidence card {s.card.id} built from {len(s.papers)} retrieved papers.")
    out.append(proc)
    return out


def _render(sections: list[Section], title: str, s: Session, student: str | None, status: str,
            prefix: str = "") -> Report:
    t = f = 0
    for sec in sections:
        for e in sec.exhibits:
            if e.kind == "table":
                t += 1
                e.number = f"Table {prefix}{t}"
            else:
                f += 1
                e.number = f"Figure {prefix}{f}"
    tpl = jinja2.Environment(autoescape=jinja2.select_autoescape(default=True)).from_string(
        resources.files("priors.reports").joinpath("report.html.j2").read_text(encoding="utf-8"))
    doc = tpl.render(
        title=title, strategy=s.spec.name if s.spec else "Strategy", student=student, status=status,
        exploratory_label=EXPLORATORY_LABEL, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        mode=s.mode.replace("_", " ").title(), main=sections, appendix=[], disclaimer=DISCLAIMER,
    )
    return Report(title=title, html=doc)


def _unsupported_sections(s: Session) -> tuple[list[Section], list[Section]]:
    """A literature assessment for an idea that ends without a backtest because it lacks support."""
    u = s.unsupported
    c = s.card
    main = [Section("At a glance")]
    rows = [["Idea", s.idea.summary if s.idea else ""],
            ["Outcome", "Not tested: the literature does not support the idea well enough to test it without data mining"],
            ["Evidence grade", f"{c.grade}: {c.grade_justification}" if c else "n/a"]]
    if s.gate is not None:
        rows.append(["Theory gate", "passed" if s.gate.passed else "not passed"])
    main[0].exhibits.append(Exhibit("table", "Summary", "No backtest was run, so no performance numbers are reported.",
                                    ["Item", "Value"], rows, left=2, widths=[22, 78]))
    sec = Section("Why the idea was not tested")
    sec.html_blocks.append(text_html(u.summary))
    sec.bullets += u.why_not_supported
    main.append(sec)
    sec = Section("What would change this")
    sec.bullets += u.evidence_needed
    main.append(sec)
    sec = Section("Better-supported directions to explore")
    sec.bullets += u.alternative_directions
    main.append(sec)
    if c is not None:
        ev = Section("What the literature says")
        ev.paragraphs.append(f"<b>Claim.</b> {html.escape(c.claim)}")
        ev.bullets += [f"Supporting: {f.citation.short()}, {f.citation.title}. {f.finding}" for f in c.key_papers]
        ev.bullets += [f"Contrary: {f.citation.short()}, {f.citation.title}. {f.finding}" for f in c.contrary_evidence]
        main.append(ev)
    proc = Section("Process")
    proc.paragraphs.append("Concluding that an idea lacks support, and not testing it, is a legitimate research result: "
                           "testing it anyway would be a search for a pattern without a reason to expect one.")
    proc.bullets += [f"Student: {t['text']}" for t in s.transcript + s.coaching if t["role"] == "student"]
    appendix = [Section("Retrieved papers")]
    appendix[0].bullets += [f"{p.citation.short()}, {p.citation.title}. {p.citation.venue or ''}" for p in s.papers]
    if c is not None:
        appendix[0].bullets += [f"Boundary condition: {b}" for b in c.boundary_conditions]
    main.append(proc)
    return main, appendix


def build_reports(session: Session, explanation: str | None = None, analyst: AnalystReview | None = None,
                  review: DrDongReview | None = None, student: str | None = None) -> tuple[Report, Report]:
    """(main report, technical appendix). Defaults to what the session has saved.

    The main report's length depends on the mode (about 4 pages in Mentor, 6 in Analyst, 8 in Dr. Dong);
    every exhibit, the full evidence card, the full referee report, and the debate are in the appendix.
    """
    s = session
    if s.results is None:
        if s.unsupported is None:
            raise ValueError("Run the tests, or conclude that the idea is not supported, before building a report.")
        main, appendix = _unsupported_sections(s)
        return (_render(main, "Priors research report: literature assessment", s, student, "not tested"),
                _render(appendix, "Priors research report: technical appendix", s, student, "not tested", prefix="A"))
    explanation = explanation if explanation is not None else s.explanation
    analyst = analyst if analyst is not None else s.analyst_review
    review = review if review is not None else s.dr_dong
    status = s.results.outcome.status

    main = _main_sections(s, explanation, analyst, review, status)
    risks = Section("Other required risk notes", compact=True)
    shown = set(_top_warnings(s.results.dossier.risk_notes))      # already listed under "At a glance"
    risks.bullets += [n for n in s.results.dossier.risk_notes if n not in shown]
    main.append(risks)
    led = s.registry.ledger.summary()
    tail = Section("Process")
    tail.paragraphs.append(f"{led['trials']} specification(s) tested ({led['exploratory_trials']} exploratory); holdouts "
                           f"opened: {led['holdouts_opened']}. Every other exhibit, the full evidence card, the full "
                           "referee report, and the conversation are in the technical appendix.")
    main.append(tail)
    return (_render(main, "Priors research report", s, student, status),
            _render(_appendix_sections(s, explanation, analyst, review, status),
                    "Priors research report: technical appendix", s, student, status, prefix="A"))


def build_report(session: Session, **kw) -> Report:
    """The main report only. See ``build_reports`` for the technical appendix."""
    return build_reports(session, **kw)[0]

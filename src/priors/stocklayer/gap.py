"""Factor-to-stock gap report (ARCHITECTURE §10.7).

The factor layer is a long-short portfolio of published factors (decile sorts,
before costs). The stock layer is a long-only top-N portfolio of today's
universe, after costs. The two should move together, but their returns differ
for known reasons; this report shows both on the same months and names the
reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..composite import FactorResult
from ..engine.metrics import summarize
from .backtest import StockResult

REASONS = [
    "Long-only versus long-short: the stock portfolio holds only the attractive end and carries market risk; "
    "the factor earns the spread between the two ends.",
    "Concentration: the stock portfolio holds a fixed number of stocks; the factor holds a whole decile or more.",
    "Costs: stock-layer returns are after trading costs; factor returns are before costs.",
    "Universe: today's large and mid-size U.S. stocks versus all listed stocks at the time.",
]


@dataclass
class GapReport:
    table: pd.DataFrame        # rows: factor composite, stock gross, stock net, stock excess
    correlation: float         # factor composite vs stock excess return
    months: int
    start: str
    end: str
    reasons: list[str] = field(default_factory=lambda: list(REASONS))
    flags: list[str] = field(default_factory=list)


def gap_report(factor: FactorResult, stock: StockResult) -> GapReport:
    f = factor.returns["composite"]
    s = stock.returns
    common = f.index.intersection(s.index)
    if len(common) < 24:
        raise ValueError(f"Only {len(common)} months in common between the two layers.")
    series = {
        "Factor composite (long-short, before costs)": f.loc[common],
        "Stock portfolio, gross": s.loc[common, "gross"],
        "Stock portfolio, net of costs": s.loc[common, "net"],
        "Stock portfolio minus eligible universe (net)": s.loc[common, "excess"],
    }
    rows = {}
    for k, r in series.items():
        m = summarize(r, 12)
        rows[k] = {"mean_ann": m["mean_ann"], "vol_ann": m["vol_ann"], "sharpe": m["sharpe"],
                   "max_drawdown": m["max_drawdown"]}
    table = pd.DataFrame(rows).T
    corr = float(f.loc[common].corr(s.loc[common, "excess"]))

    flags = [
        f"Over {common[0]} to {common[-1]}, the factor composite earned {table.iloc[0]['mean_ann']:.1%} per year "
        f"before costs; the stock portfolio earned {table.iloc[3]['mean_ann']:.1%} per year above the eligible "
        f"universe after costs. Correlation between the two: {corr:.2f}."
    ]
    if corr < 0.3:
        flags.append(
            "The stock portfolio tracks the factor weakly. Check that the stock signals match the factors, and "
            "remember the stock layer holds only a few dozen stocks."
        )
    if any("Survivorship" in w for w in stock.warnings):
        flags.append("The stock-layer history excludes delisted firms, so its returns are likely overstated.")
    return GapReport(table=table, correlation=corr, months=len(common), start=str(common[0]),
                     end=str(common[-1]), flags=flags)

"""The validation suite: runs automatically after every factor-layer test (ARCHITECTURE §7)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..composite import FactorResult
from ..factors import FactorLibrary
from ..registry.ledger import Ledger
from .benchmark import BenchmarkCheck, benchmark_check
from .deflated import DeflatedSharpe, deflated_sharpe
from .placebo import Placebo, random_factor_placebo
from .stability import Stability, stability


@dataclass
class ValidationReport:
    benchmark: BenchmarkCheck | None
    deflated: DeflatedSharpe
    placebo: Placebo | None
    stability: Stability
    flags: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def validate(result: FactorResult, lib: FactorLibrary, ledger: Ledger | None = None,
             placebo_draws: int = 1000) -> ValidationReport:
    """Run every check. A check that cannot run is reported in ``errors``; the others still run."""
    errors: dict[str, str] = {}
    trial_sharpes = [t["sharpe"] for t in ledger.trials("factor")] if ledger else []
    if not trial_sharpes:
        trial_sharpes = [result.metrics["composite"]["sharpe"]]

    try:
        bench = benchmark_check(result, lib)
    except Exception as e:  # e.g. a factor without enough history for a prior
        bench, errors["benchmark"] = None, str(e)
    dsr = deflated_sharpe(result.returns["composite"], trial_sharpes)
    try:
        plc = random_factor_placebo(result, lib, n_draws=placebo_draws)
    except Exception as e:
        plc, errors["placebo"] = None, str(e)
    stab = stability(result)

    flags = list(result.warnings)
    for part in (bench, dsr, plc, stab):
        if part is not None:
            flags += part.flags
    return ValidationReport(benchmark=bench, deflated=dsr, placebo=plc, stability=stab, flags=flags, errors=errors)


__all__ = [
    "BenchmarkCheck", "DeflatedSharpe", "Placebo", "Stability", "ValidationReport",
    "benchmark_check", "deflated_sharpe", "random_factor_placebo", "stability", "validate",
]

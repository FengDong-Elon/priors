"""The theory gate (ARCHITECTURE §6.1).

A hypothesis may be pre-registered only if code confirms three things:

1. Mechanism: a mechanism class is chosen and explained in the student's own words.
2. Literature support: at least one *retrieved* citation is attached.
3. Expected range: an expected range for the composite's annualized mean return.

An LLM may add an advisory assessment (``GateResult.advisory``), but the pass /
fail decision is made here, in code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..literature.citations import Citation

Mechanism = Literal["risk_compensation", "behavioral_bias", "limits_to_arbitrage", "market_friction"]

MECHANISM_LABELS = {
    "risk_compensation": "Risk compensation: investors are paid for bearing a risk that hurts in bad times.",
    "behavioral_bias": "Behavioral bias: investors make systematic mistakes that prices reflect.",
    "limits_to_arbitrage": "Limits to arbitrage: mispricing persists because correcting it is costly or risky.",
    "market_friction": "Market friction: costs, constraints, or illiquidity that investors are compensated for.",
}

MIN_STATEMENT_CHARS = 20
MIN_EXPLANATION_CHARS = 60


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(description="The hypothesis in one or two sentences.")
    mechanism: Mechanism | None = None
    mechanism_explanation: str = Field("", description="Why the premium should exist, in the student's own words.")
    citations: list[Citation] = Field(default_factory=list)
    expected_low: float | None = Field(None, description="Expected annualized mean return of the composite, low end.")
    expected_high: float | None = Field(None, description="Expected annualized mean return of the composite, high end.")
    evidence_card_ids: list[str] = Field(default_factory=list, description="Evidence cards supporting the idea.")

    @model_validator(mode="after")
    def _range(self) -> "Hypothesis":
        lo, hi = self.expected_low, self.expected_high
        if (lo is None) != (hi is None):
            raise ValueError("give both expected_low and expected_high, or neither")
        return self


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    advisory: str | None = None

    def explain(self) -> str:
        if self.passed:
            return "The hypothesis passes the theory gate."
        return "The hypothesis does not pass the theory gate yet:\n" + "\n".join(f"- {f}" for f in self.failures)


def in_core_library(c: Citation) -> bool:
    from ..literature.citations import core_library, normalize_doi

    lib = core_library()
    return c.retrieved_via == "core_library" and (
        (c.doi is not None and normalize_doi(c.doi) in lib) or c.title.lower() in lib
    )


def check_gate(
    h: Hypothesis,
    advisory: str | None = None,
    is_retrieved: Callable[[Citation], bool] = in_core_library,
    card_exists: Callable[[str], bool] | None = None,
) -> GateResult:
    """Run the gate. ``is_retrieved`` confirms that a citation really came from a retrieval.

    The default accepts only core-library papers; the registry supplies a check
    against the project's search log so that live search results count too.
    When ``card_exists`` is given, at least one attached evidence card must exist.
    """
    failures = []
    unverified = [c.short() for c in h.citations if not is_retrieved(c)]
    if unverified:
        failures.append(
            "These citations were not found by a literature search in this session and cannot be used: "
            + ", ".join(unverified)
        )
    if len(h.statement.strip()) < MIN_STATEMENT_CHARS:
        failures.append("State the hypothesis in at least a full sentence.")
    if h.mechanism is None:
        failures.append(
            "Choose a mechanism: risk compensation, behavioral bias, limits to arbitrage, or market friction."
        )
    if len(h.mechanism_explanation.strip()) < MIN_EXPLANATION_CHARS:
        failures.append(
            "Explain in your own words (at least a few sentences) why this premium should exist."
        )
    if not any(is_retrieved(c) for c in h.citations):
        failures.append("Attach at least one paper found by the literature search that supports the idea.")
    if card_exists is not None and not any(card_exists(i) for i in h.evidence_card_ids):
        failures.append("Attach an evidence card built from the literature search for this idea.")
    if h.expected_low is None:
        failures.append("Give an expected range for the annualized return, based on the literature.")
    else:
        lo, hi = h.expected_low, h.expected_high
        if not (0 <= lo < hi <= 0.5):
            failures.append(
                "The expected range must satisfy 0 <= low < high <= 50% per year. Factor portfolios are "
                "built long on the side the theory predicts will outperform, so the expected return is positive."
            )
    return GateResult(passed=not failures, failures=failures, advisory=advisory)

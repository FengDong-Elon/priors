"""Evidence cards (ARCHITECTURE §5.2) and the Evidence Synthesizer.

A card has two kinds of content, and they come from different places:

* **From code:** the papers (only those the search retrieved), the original
  paper's statistics and replication grade (Chen-Zimmermann), and the expected
  return range (``expected.py``).
* **From the LLM:** the claim, mechanisms, each paper's finding in words,
  boundary conditions, contrary evidence, and the evidence grade.

The LLM refers to papers only by label (P1, P2, ...). Labels that do not match a
retrieved paper are dropped and reported, so a card can never cite a paper the
search did not return.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..factors import FactorLibrary
from ..llm import LLM
from .citations import Citation
from .expected import SOURCE as DECAY_SOURCE
from .expected import FactorPrior, factor_prior
from .search import Paper

_LABEL = re.compile(r"\bP\d+\b")

MechanismTag = Literal["risk_compensation", "behavioral_bias", "limits_to_arbitrage", "market_friction", "data_mining"]
Grade = Literal["strong", "moderate", "weak", "contested"]


# ------------------------------------------------------------ LLM draft

class PaperFinding(BaseModel):
    paper: str = Field(description="Paper label, e.g. 'P2'.")
    finding: str = Field(description="What this paper finds, in one or two sentences, based only on its abstract.")


class CardDraft(BaseModel):
    claim: str = Field(description="The research line's central claim in one sentence.")
    mechanisms: list[MechanismTag] = Field(description="Mechanisms the papers propose.")
    mechanism_notes: str = Field(description="How the papers explain the effect, two to four sentences.")
    key_papers: list[PaperFinding] = Field(description="Papers that support the claim.")
    contrary_evidence: list[PaperFinding] = Field(description="Papers that fail to find the effect or explain it away.")
    boundary_conditions: list[str] = Field(description="Where the effect is weaker or absent.")
    grade: Grade
    grade_justification: str = Field(description="One sentence explaining the grade.")


# ------------------------------------------------------------- the card

class CitedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation: Citation
    finding: str


class EvidenceCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    topic: str
    factor_ids: list[str]
    claim: str
    mechanisms: list[MechanismTag]
    mechanism_notes: str
    key_papers: list[CitedFinding]
    contrary_evidence: list[CitedFinding]
    boundary_conditions: list[str]
    grade: Grade
    grade_justification: str
    priors: list[dict]
    decay_source: str = DECAY_SOURCE
    warnings: list[str] = Field(default_factory=list)
    created_at: str
    model_tier: str = "deep"

    @property
    def citations(self) -> list[Citation]:
        return [f.citation for f in self.key_papers + self.contrary_evidence]

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceCard":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


SYNTH_SYSTEM = """You are the Evidence Synthesizer in Priors, a teaching tool for evidence-based factor investing.
You write the qualitative part of an evidence card about one line of research, for students.

Rules:
- Use only the numbered papers provided. Refer to them by label (P1, P2, ...). Never mention any other paper.
- Base each paper's finding on its abstract. If a paper has no abstract, describe only what its title makes
  clear, and keep it general.
- Do not state any number (return, t-statistic, Sharpe ratio, sample size) unless it appears in the
  abstract you are summarizing. The card's statistics are filled in by code, not by you.
- List contrary evidence when the provided papers contain it (failed replications, post-publication decline,
  cost-based explanations). If none of the papers contradict the claim, return an empty list; do not invent one.
- Grade the evidence: "strong" (repeatedly replicated, out of sample, economically meaningful), "moderate"
  (well documented but with known weaknesses), "weak" (thin or fragile), "contested" (credible papers disagree).
  Take the code-provided replication facts into account.
- Write plainly and precisely. Say "consistent with," not "proves."""


def _facts_block(priors: list[FactorPrior]) -> str:
    lines = []
    for p in priors:
        line = (
            f"- {p.factor}: mean {p.base_mean_ann:.1%}/yr over the {p.base_window} "
            f"({p.base_start} to {p.base_end})"
        )
        if p.op_tstat is not None:
            line += f"; original paper t-stat {p.op_tstat:.2f}"
        if p.replication_quality:
            line += f"; Chen-Zimmermann replication quality: {p.replication_quality}"
        lines.append(line)
    return "\n".join(lines)


def _paper_block(papers: list[Paper]) -> str:
    out = []
    for i, p in enumerate(papers, start=1):
        abstract = p.abstract or "(no abstract available)"
        out.append(f"P{i}. {p.brief()}\nAbstract: {abstract}")
    return "\n\n".join(out)


def synthesize(
    llm: LLM, topic: str, papers: list[Paper], factor_ids: list[str], lib: FactorLibrary,
    clock=lambda: datetime.now(timezone.utc),
) -> EvidenceCard:
    if not papers:
        raise ValueError("No papers to synthesize: run a literature search first.")
    priors = [factor_prior(lib, f) for f in factor_ids]
    user = (
        f"Topic: {topic}\nFactors: {', '.join(factor_ids)}\n\n"
        f"Facts computed by code (do not restate them as your own findings):\n{_facts_block(priors)}\n\n"
        f"Papers:\n{_paper_block(papers)}"
    )
    draft = llm.structured("deep", SYNTH_SYSTEM, user, CardDraft)

    labels = {f"P{i}": p for i, p in enumerate(papers, start=1)}
    warnings: list[str] = []

    def resolve(items: list[PaperFinding]) -> list[CitedFinding]:
        out = []
        for it in items:
            p = labels.get(it.paper.strip().upper())
            if p is None:
                warnings.append(f"Dropped a reference to {it.paper!r}, which is not one of the retrieved papers.")
                continue
            out.append(CitedFinding(citation=p.citation, finding=it.finding))
        return out

    def relabel(text: str) -> str:
        """Replace P-labels in free text with author-year references students can read."""
        def sub(m: re.Match) -> str:
            p = labels.get(m.group(0).upper())
            if p is None:
                warnings.append(f"Removed a reference to {m.group(0)!r} from the text: not a retrieved paper.")
                return "[unverified reference removed]"
            return p.citation.short()
        return _LABEL.sub(sub, text)

    key, contra = resolve(draft.key_papers), resolve(draft.contrary_evidence)
    for f in key + contra:
        f.finding = relabel(f.finding)
    grade = draft.grade
    if not key:
        warnings.append("No retrieved paper supports the claim, so the grade is set to 'weak'.")
        grade = "weak"

    created = clock().isoformat(timespec="seconds")
    body = json.dumps({"topic": topic, "factors": factor_ids, "draft": draft.model_dump(), "at": created},
                      sort_keys=True)
    return EvidenceCard(
        id="E-" + hashlib.sha256(body.encode()).hexdigest()[:10],
        topic=topic, factor_ids=factor_ids, claim=draft.claim, mechanisms=draft.mechanisms,
        mechanism_notes=relabel(draft.mechanism_notes), key_papers=key, contrary_evidence=contra,
        boundary_conditions=[relabel(b) for b in draft.boundary_conditions], grade=grade,
        grade_justification=relabel(draft.grade_justification),
        priors=[p.as_dict() for p in priors], warnings=warnings, created_at=created,
    )

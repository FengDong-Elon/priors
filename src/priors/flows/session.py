"""One student's working session across the three modes (ARCHITECTURE §3).

The session is UI-agnostic: the app turns chat messages into these calls.

Mentor:   interview -> research -> propose -> set_explanation -> check_gate -> register -> run -> explain
Analyst:  load_spec (or continue from Mentor) -> run -> analyze -> explain
Dr. Dong: load_spec (or continue) -> run -> review -> explain

``upgrade`` moves between modes and keeps everything already done. Every
backtest goes through the registry, so status and the trial ledger are always
decided by code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from ..composite.report import CompositeReport, composite_report
from ..data.base import Panel
from ..factors import FactorLibrary
from ..literature.evidence import EvidenceCard, synthesize
from ..literature.expected import composite_prior
from ..literature.search import Paper, gather_papers
from ..llm import LLM
from ..mechanisms import MechanismResult, StateData, run_mechanism_tests
from ..registry import GateResult, Hypothesis, Registration, Registry, RunOutcome, check_gate
from ..spec.strategy import MODE_COMBINATIONS, StrategySpec
from ..stocklayer import (
    FundamentalHistoryUnavailable, GapReport, Holdings, StockResult, current_holdings, default_signal,
    gap_report, run_stock_backtest,
)
from ..validation import ValidationReport, validate
from ..zoo import ZooReport, zoo_report
from . import agents as A
from .dossier import Dossier, build_dossier, unverified_numbers

Mode = Literal["mentor", "analyst", "dr_dong"]
MODES: tuple[str, ...] = ("mentor", "analyst", "dr_dong")


class FlowError(RuntimeError):
    """A step was called before the steps it depends on."""


@dataclass
class StockContext:
    universe: pd.DataFrame
    panel: Panel
    fundamentals: pd.DataFrame | None = None
    breakpoints: pd.DataFrame | None = None
    universe_mask: pd.DataFrame | None = None     # point-in-time membership (signal dates x tickers), if known


@dataclass
class Results:
    outcome: RunOutcome
    validation: ValidationReport
    composite: CompositeReport | None
    zoo: ZooReport | None
    mechanisms: list[MechanismResult]
    stock: StockResult | None
    gap: GapReport | None
    holdings: Holdings | None
    dossier: Dossier
    notes: list[str] = field(default_factory=list)


@dataclass
class Checked:
    """An agent's output plus any numbers it used that are not in the dossier."""
    value: object
    unverified: list[str]


@dataclass
class DrDongReview:
    debate: list[tuple[str, A.DebateTurn]]
    report: A.RefereeReport
    footer: str
    unverified: list[str]


def card_text(card: EvidenceCard | None) -> str:
    if card is None:
        return "No evidence card."
    lines = [f"Claim: {card.claim}", f"Evidence grade: {card.grade} ({card.grade_justification})",
             f"Mechanisms: {', '.join(card.mechanisms)}. {card.mechanism_notes}", "Supporting papers:"]
    lines += [f"- {f.citation.short()}: {f.finding}" for f in card.key_papers]
    lines += ["Contrary evidence:"] + ([f"- {f.citation.short()}: {f.finding}" for f in card.contrary_evidence] or ["- none found"])
    lines += ["Boundary conditions:"] + [f"- {b}" for b in card.boundary_conditions]
    return "\n".join(lines)


_WORD = re.compile(r"[a-z0-9]+")


class Session:
    def __init__(
        self, project_dir: str | Path, lib: FactorLibrary, llm: LLM, mode: Mode = "mentor",
        stock: StockContext | None = None, state_data: StateData | None = None,
        clock: Callable[[], datetime] | None = None, require_evidence: bool = True,
        allow_exploratory: bool = True, required_stage: str = "dr_dong",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.registry = Registry(project_dir, clock=clock, require_evidence=require_evidence)
        self.lib, self.llm, self.mode, self.stock = lib, llm, mode, stock
        self.state_data = state_data
        self.transcript: list[dict] = []
        self.idea: A.ClarifiedIdea | None = None
        self.papers: list[Paper] = []
        self.card: EvidenceCard | None = None
        self.proposal: A.ArchitectProposal | None = None
        self.spec: StrategySpec | None = None
        self.hypothesis: Hypothesis | None = None
        self.gate: GateResult | None = None
        self.registration: Registration | None = None
        self.results: Results | None = None
        self.explanation: str | None = None
        self.analyst_review: A.AnalystReview | None = None
        self.dr_dong: DrDongReview | None = None
        self.allow_exploratory = allow_exploratory      # set by the class; public users may explore
        self.required_stage = required_stage            # "tested" or "dr_dong"
        self.coaching: list[dict] = []                  # revision conversation before testing
        self.unsupported: A.UnsupportedAssessment | None = None
        self.revisions = 0
        self._load_state()

    # ---------------------------------------------------------------- modes

    def upgrade(self, mode: Mode) -> None:
        """Switch mode, keeping the idea, evidence, spec, registration, ledger, and results."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    # --------------------------------------------------------------- Mentor

    def interview(self, message: str) -> A.InterviewTurn:
        self.transcript.append({"role": "student", "text": message})
        convo = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in self.transcript)
        turn = self.llm.structured("deep", A.INTERVIEWER_SYSTEM, f"Conversation so far:\n{convo}", A.InterviewTurn,
                                   max_tokens=2000)
        self.transcript.append({"role": "mentor", "text": turn.reply})
        if turn.ready and turn.idea is not None:
            self.idea = turn.idea
        self.save_state()
        return turn

    def set_idea(self, idea: A.ClarifiedIdea) -> None:
        self.idea = idea

    def research(self) -> EvidenceCard:
        if self.idea is None:
            raise FlowError("Clarify the idea first (interview).")
        text = f"{self.idea.summary} {self.idea.signal} {self.idea.direction}"
        self.papers = gather_papers(self.llm, text, log=self.registry.retrievals)
        linked = [f for p in self.papers for f in p.factor_ids if f in self.lib.returns.columns]
        factors = list(dict.fromkeys(linked))[:4]
        self.card = synthesize(self.llm, self.idea.summary, self.papers, factors, self.lib)
        self.card.save(self.registry.cards_dir)
        self.save_state()
        return self.card

    def candidates(self, k: int = 25) -> pd.DataFrame:
        """Library factors that match the idea and the evidence card, for the Architect to choose from."""
        words = set(_WORD.findall(" ".join(filter(None, [
            self.idea.summary if self.idea else "", self.idea.signal if self.idea else "",
            self.card.claim if self.card else "",
        ])).lower())) - {"the", "and", "stocks", "stock", "returns", "return", "with", "that", "high", "low"}
        linked = {f for p in self.papers for f in p.factor_ids}
        info = self.lib.info
        cat = info.get("economic_category", pd.Series("", index=info.index))
        text = (info["name"].fillna("").astype(str) + " " + cat.fillna("").astype(str)).str.lower()
        score = text.map(lambda t: len(words & set(_WORD.findall(t)))).astype(float)
        score[score.index.isin(linked)] += 5
        score[score.index.isin([f + "_VW" for f in linked])] += 4
        top = score[score > 0].sort_values(ascending=False).index[:k]
        out = info.loc[top, ["source", "name", "year"]].copy()
        out["stock_signal"] = [default_signal(f) for f in top]
        return out

    def propose(self) -> tuple[StrategySpec, Hypothesis]:
        if self.idea is None or self.card is None:
            raise FlowError("Research the idea first.")
        cands = self.candidates()
        table = "\n".join(f"- {i}: {r['name']} ({r['source']}, {r['year']}); stock signal: {r['stock_signal'] or 'none'}"
                          for i, r in cands.iterrows())
        user = (f"Idea: {self.idea.model_dump_json()}\n\nEvidence card:\n{card_text(self.card)}\n\n"
                f"Candidate factors:\n{table}\n\nAllowed combination methods in {self.mode} mode: "
                f"{', '.join(MODE_COMBINATIONS[self.mode])}.")
        prop = self.llm.structured("deep", A.ARCHITECT_SYSTEM, user, A.ArchitectProposal, max_tokens=4000)
        valid = [c for c in prop.components if c.factor in cands.index][:4]
        if not valid:
            raise FlowError("The Architect proposed no factors from the candidate list; refine the idea.")
        self.proposal = prop
        cites = {f: p.citation for p in self.papers for f in p.factor_ids}
        components = []
        for rank, c in enumerate(sorted(valid, key=lambda c: c.theory_priority), start=1):
            cite = cites.get(c.factor) or cites.get(c.factor.removesuffix("_VW"))
            components.append({
                "factor": c.factor, "stock_signal": default_signal(c.factor), "theory_priority": rank,
                "source": f"{cite.short()}, doi:{cite.doi}" if cite and cite.doi else None,
            })
        tests = list(dict.fromkeys(["publication_decay", *prop.mechanism_tests]))
        self.spec = StrategySpec.model_validate({
            "name": prop.name.strip()[:60] or "Strategy", "components": components, "mechanism_tests": tests,
        })
        low, high, _ = composite_prior(self.spec, self.lib)
        self.hypothesis = Hypothesis(
            statement=prop.hypothesis, mechanism=prop.mechanism,
            mechanism_explanation=self.idea.student_reason if self.idea else "",
            citations=[f.citation for f in self.card.key_papers],
            expected_low=round(max(low, 0.0), 4), expected_high=round(max(high, low + 0.005, 0.005), 4),
            evidence_card_ids=[self.card.id],
        )
        self.save_state()
        return self.spec, self.hypothesis

    def set_explanation(self, text: str) -> None:
        """The student's own explanation of why the premium should exist."""
        if self.hypothesis is None:
            raise FlowError("Propose a hypothesis first.")
        self.hypothesis = self.hypothesis.model_copy(update={"mechanism_explanation": text})

    def check_gate(self, advice: bool = True) -> GateResult:
        if self.hypothesis is None:
            raise FlowError("Propose a hypothesis first.")
        advisory = None
        if advice:
            user = (f"Hypothesis: {self.hypothesis.statement}\nMechanism: {self.hypothesis.mechanism}\n"
                    f"Student's explanation: {self.hypothesis.mechanism_explanation}\n\n{card_text(self.card)}")
            adv = self.llm.structured("fast", A.GATEKEEPER_SYSTEM, user, A.GateAdvice, max_tokens=1500)
            advisory = adv.assessment + ("\n" + "\n".join(f"- {s}" for s in adv.suggestions) if adv.suggestions else "")
        self.gate = check_gate(
            self.hypothesis, advisory=advisory, is_retrieved=self.registry.is_retrieved,
            card_exists=self.registry.card_exists if self.registry.require_evidence else None,
        )
        return self.gate

    def register(self) -> Registration:
        if self.hypothesis is None or self.spec is None:
            raise FlowError("Propose a hypothesis first.")
        self.registration = self.registry.register(self.hypothesis, self.spec, advisory=self.gate.advisory if self.gate else None)
        self.spec = self.registration.spec
        self.save_state()
        return self.registration

    # ----------------------------------------------------------------- runs

    def load_spec(self, spec: StrategySpec) -> None:
        """Analyst and Dr. Dong modes start from a student's own spec."""
        self.spec = spec

    def run(self, period: Literal["in_sample", "holdout"] = "in_sample", stock: bool = True,
            placebo_draws: int = 1000) -> Results:
        if self.spec is None:
            raise FlowError("No strategy yet.")
        self.spec.check_mode(self.mode)
        status, _, _ = self.registry._classify(self.spec)
        if status == "exploratory" and not self.allow_exploratory:
            raise FlowError("In this class, a strategy must pass the theory gate and be pre-registered before it "
                            "can be tested.")
        outcome = self.registry.run_factor(self.spec, self.lib, period=period)
        res = outcome.result
        notes = list(outcome.notes)
        val = validate(res, self.lib, self.registry.ledger, placebo_draws=placebo_draws)
        comp = composite_report(res, self.lib) if len(self.spec.components) > 1 else None
        zoo = zoo_report(res, self.lib)
        mech = run_mechanism_tests(res, self.lib, self.state_data)

        st = gap = hold = None
        if stock and self.stock is not None and any(c.stock_signal for c in self.spec.components):
            ctx = self.stock
            try:
                st = run_stock_backtest(res.spec, ctx.panel, ctx.breakpoints, holdout_from=res.holdout_start,
                                        universe_mask=ctx.universe_mask)
                gap = gap_report(res, st)
            except FundamentalHistoryUnavailable as e:
                notes.append(str(e))
            except ValueError as e:
                notes.append(f"Stock-layer backtest not run: {e}")
            try:
                hold = current_holdings(res.spec, ctx.panel, ctx.universe, ctx.fundamentals, ctx.breakpoints,
                                        universe_mask=ctx.universe_mask)
            except ValueError as e:
                notes.append(f"Current holdings not built: {e}")

        dossier = build_dossier(res, val, comp, zoo, mech, st, gap, hold,
                                self.registry.ledger.summary(), status=outcome.status)
        dossier.risk_notes += [n for n in notes if n not in dossier.risk_notes]
        self.results = Results(outcome, val, comp, zoo, mech, st, gap, hold, dossier, notes)
        return self.results

    # ------------------------------------------------------- talking about results

    def _checked(self, tier: str, system: str, user: str, schema, render: Callable[[object], str],
                 max_tokens: int = 8000) -> Checked:
        facts = self.results.dossier.text()
        value = self.llm.structured(tier, system, user, schema, max_tokens=max_tokens)
        bad = unverified_numbers(render(value), facts)
        if bad:
            retry = (user + "\n\nYour previous answer used numbers that are not in the FACTS dossier: "
                     + ", ".join(bad) + ". Rewrite it using only numbers from the dossier.")
            value = self.llm.structured(tier, system, retry, schema, max_tokens=max_tokens)
            bad = unverified_numbers(render(value), facts)
        return Checked(value, bad)

    def _need_results(self) -> None:
        if self.results is None:
            raise FlowError("Run the tests first.")

    def explain(self) -> tuple[str, list[str], list[str]]:
        """(explanation text at this mode's depth, required risk notes, unverified numbers)."""
        self._need_results()
        user = f"FACTS:\n{self.results.dossier.text()}\n\nEvidence card:\n{card_text(self.card)}"
        # Deep tier: this is the text students read, and a fast model confused t-statistics with Sharpe ratios.
        out = self._checked("deep", A.explainer_system(self.mode), user, A.Explanation, lambda e: e.text, 8000)
        self.explanation = out.value.text
        self.save_state()
        return out.value.text, list(self.results.dossier.risk_notes), out.unverified

    def analyze(self) -> Checked:
        self._need_results()
        user = f"FACTS:\n{self.results.dossier.text()}\n\nEvidence card:\n{card_text(self.card)}"
        out = self._checked("deep", A.ANALYST_SYSTEM, user, A.AnalystReview, lambda r: json.dumps(r.model_dump()))
        self.analyst_review = out.value
        self.save_state()
        return out

    def review(self, rounds: int = 2) -> DrDongReview:
        """The Dr. Dong debate: Advocate and Reviewer 2 alternate, then Dr. Dong writes the report."""
        self._need_results()
        facts = self.results.dossier.text()
        base = f"FACTS:\n{facts}\n\nEvidence card:\n{card_text(self.card)}"
        debate: list[tuple[str, A.DebateTurn]] = []
        unverified: set[str] = set()

        def history() -> str:
            return "\n\n".join(f"{who}:\n" + "\n".join(f"- {a.point} [{a.evidence}]" for a in t.arguments)
                               + ("\nConcedes: " + "; ".join(t.concessions) if t.concessions else "")
                               for who, t in debate)

        for _ in range(rounds):
            for who, system in (("Advocate", A.ADVOCATE_SYSTEM), ("Reviewer 2", A.REVIEWER2_SYSTEM)):
                user = base + (f"\n\nDebate so far:\n{history()}" if debate else "")
                out = self._checked("deep", system, user, A.DebateTurn, lambda t: json.dumps(t.model_dump()))
                debate.append((who, out.value))
                unverified |= set(out.unverified)
        user = base + f"\n\nDebate:\n{history()}"
        rep = self._checked("deep", A.DR_DONG_SYSTEM, user, A.RefereeReport, lambda r: json.dumps(r.model_dump()))
        unverified |= set(rep.unverified)
        self.dr_dong = DrDongReview(debate=debate, report=rep.value, footer=A.DR_DONG_FOOTER, unverified=sorted(unverified))
        self.save_state()
        return self.dr_dong

    # ------------------------------------------------------------ revising

    def coach(self, message: str) -> A.CoachTurn:
        """Revision conversation before any test: help the student strengthen (or abandon) the hypothesis."""
        if self.hypothesis is None:
            raise FlowError("Propose a hypothesis first.")
        self.coaching.append({"role": "student", "text": message})
        gate = self.gate.explain() if self.gate else "The gate has not been checked yet."
        convo = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in self.coaching)
        user = (f"Hypothesis: {self.hypothesis.statement}\nMechanism: {self.hypothesis.mechanism}\n"
                f"Student's explanation: {self.hypothesis.mechanism_explanation}\n\nTheory gate: {gate}\n"
                f"Gate advice: {self.gate.advisory if self.gate and self.gate.advisory else 'none'}\n\n"
                f"Evidence card:\n{card_text(self.card)}\n\nConversation:\n{convo}")
        turn = self.llm.structured("deep", A.COACH_SYSTEM, user, A.CoachTurn, max_tokens=2000)
        self.coaching.append({"role": "mentor", "text": turn.reply})
        self.save_state()
        return turn

    def research_again(self, refined_idea: str) -> EvidenceCard:
        """Search the literature again for a refined idea, before any test; the strategy is proposed anew."""
        if self.results is not None:
            raise FlowError("Results already exist; start a revision instead.")
        if self.idea is None:
            raise FlowError("Clarify the idea first.")
        self.idea = self.idea.model_copy(update={"summary": refined_idea.strip()})
        self.proposal = self.spec = self.hypothesis = self.gate = None
        return self.research()

    REVISION_NOTICE = (
        "You have seen results, so a revised strategy is a new trial. It is recorded in the trial ledger, it raises "
        "the bar the deflated Sharpe ratio applies to every result in this project, and it needs its own "
        "pre-registration. Because the in-sample data has already been seen, only the sealed holdout can judge it."
    )

    def start_revision(self) -> str:
        """After results: clear the current strategy so a revised one can be proposed and registered."""
        if self.results is None:
            raise FlowError("Nothing to revise yet: before testing, edit the hypothesis or talk to the Mentor.")
        self.revisions += 1
        self.proposal = self.spec = self.hypothesis = self.gate = self.registration = None
        self.results = None
        self.explanation, self.analyst_review, self.dr_dong = None, None, None
        self.coaching = []
        self.save_state()
        return self.REVISION_NOTICE

    def reopen(self) -> None:
        """Withdraw a 'not supported' conclusion and keep working on the same idea and evidence card."""
        self.unsupported = None
        self.save_state()

    def start_new_idea(self) -> None:
        """Start over with a new idea in the same project. The trial ledger and registrations are kept."""
        self.transcript, self.papers, self.coaching = [], [], []
        self.idea = self.card = self.proposal = self.spec = self.hypothesis = self.gate = None
        self.registration = self.results = self.unsupported = None
        self.explanation, self.analyst_review, self.dr_dong = None, None, None
        self.save_state()

    def conclude_unsupported(self) -> A.UnsupportedAssessment:
        """End the project without a backtest because the literature does not support the idea."""
        if self.card is None:
            raise FlowError("Search the literature first.")
        gate = self.gate.explain() if self.gate else "The gate was not checked."
        user = (f"Idea: {self.idea.summary if self.idea else ''}\n"
                f"Hypothesis: {self.hypothesis.statement if self.hypothesis else 'none proposed'}\n"
                f"Student's explanation: {self.hypothesis.mechanism_explanation if self.hypothesis else ''}\n"
                f"Theory gate: {gate}\n\nEvidence card:\n{card_text(self.card)}")
        self.unsupported = self.llm.structured("deep", A.UNSUPPORTED_SYSTEM, user, A.UnsupportedAssessment,
                                               max_tokens=4000)
        self.save_state()
        return self.unsupported

    def progress(self) -> dict:
        """Checklist of steps and whether the project is finished under the class's rules."""
        reg = self.registration
        tested = reg is not None and any(  # from the ledger, so it survives reopening the project
            r["hypothesis_id"] == reg.id and r["status"] == "preregistered" for r in self.registry.ledger.runs())
        steps = {
            "Idea": self.idea is not None,
            "Literature": self.card is not None,
            "Theory gate": bool(self.gate and self.gate.passed) or reg is not None,   # registration requires passing
            "Pre-registered": reg is not None,
            "Tested": tested,
            "Dr. Dong review": self.dr_dong is not None,
        }
        if self.unsupported is not None:
            status = "complete: not supported by the literature"
        elif steps["Tested"] and (self.required_stage == "tested" or steps["Dr. Dong review"]):
            status = "complete"
        else:
            status = "in progress"
        if self.required_stage == "tested":
            steps.pop("Dr. Dong review")
        return {"steps": steps, "status": status, "revisions": self.revisions}

    # ----------------------------------------------------------- persistence

    @property
    def state_path(self) -> Path:
        return self.registry.root / "session.json"

    def save_state(self) -> None:
        """Save everything the LLM produced, so a session can resume (or a report be rebuilt) without new calls.

        Test results are not saved: ``run`` recomputes them deterministically from the spec and the data.
        """
        dump = lambda m: m.model_dump(mode="json") if m is not None else None  # noqa: E731
        state = {
            "mode": self.mode, "transcript": self.transcript, "idea": dump(self.idea),
            "papers": [{"citation": p.citation.model_dump(mode="json"), "abstract": p.abstract, "cited_by": p.cited_by,
                        "factor_ids": p.factor_ids, "score": p.score} for p in self.papers],
            "card_id": self.card.id if self.card else None, "proposal": dump(self.proposal), "spec": dump(self.spec),
            "hypothesis": dump(self.hypothesis), "registration_id": self.registration.id if self.registration else None,
            "explanation": self.explanation, "analyst_review": dump(self.analyst_review),
            "coaching": self.coaching, "unsupported": dump(self.unsupported), "revisions": self.revisions,
            "gate": None if self.gate is None else {"passed": self.gate.passed, "failures": self.gate.failures,
                                                    "advisory": self.gate.advisory},
            "dr_dong": None if self.dr_dong is None else {
                "debate": [[who, t.model_dump(mode="json")] for who, t in self.dr_dong.debate],
                "report": self.dr_dong.report.model_dump(mode="json"), "footer": self.dr_dong.footer,
                "unverified": self.dr_dong.unverified},
        }
        tmp = self.state_path.with_suffix(".part")
        tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        from ..literature.citations import Citation

        st = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.mode = st.get("mode", self.mode)
        self.transcript = st.get("transcript", [])
        self.idea = A.ClarifiedIdea.model_validate(st["idea"]) if st.get("idea") else None
        self.papers = [Paper(citation=Citation.model_validate(p["citation"]), abstract=p["abstract"], cited_by=p["cited_by"],
                             factor_ids=p["factor_ids"], score=p["score"]) for p in st.get("papers", [])]
        if st.get("card_id"):
            path = self.registry.cards_dir / f"{st['card_id']}.json"
            self.card = EvidenceCard.load(path) if path.exists() else None
        self.proposal = A.ArchitectProposal.model_validate(st["proposal"]) if st.get("proposal") else None
        self.spec = StrategySpec.model_validate(st["spec"]) if st.get("spec") else None
        self.hypothesis = Hypothesis.model_validate(st["hypothesis"]) if st.get("hypothesis") else None
        if st.get("registration_id"):
            self.registration = self.registry.get(st["registration_id"])
        self.explanation = st.get("explanation")
        self.coaching = st.get("coaching", [])
        self.revisions = st.get("revisions", 0)
        self.unsupported = A.UnsupportedAssessment.model_validate(st["unsupported"]) if st.get("unsupported") else None
        if st.get("gate"):
            g = st["gate"]
            self.gate = GateResult(passed=g["passed"], failures=g["failures"], advisory=g["advisory"])
        self.analyst_review = A.AnalystReview.model_validate(st["analyst_review"]) if st.get("analyst_review") else None
        if st.get("dr_dong"):
            d = st["dr_dong"]
            self.dr_dong = DrDongReview(debate=[(w, A.DebateTurn.model_validate(t)) for w, t in d["debate"]],
                                        report=A.RefereeReport.model_validate(d["report"]), footer=d["footer"],
                                        unverified=d["unverified"])

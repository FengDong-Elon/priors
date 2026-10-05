"""The three mode flows end to end, with a scripted fake LLM and synthetic data (no network)."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from priors.factors import FactorLibrary
from priors.flows import FlowError, Session, StockContext, build_dossier, unverified_numbers
from priors.flows import session as S
from priors.literature.search import search_core
from priors.llm import FakeLLM
from priors.mechanisms import StateData
from priors.spec import StrategySpec

CLOCK = lambda: datetime(2016, 1, 15, tzinfo=timezone.utc)  # noqa: E731  (holdout from 2011-01)


def flow_lib(seed: int = 5) -> FactorLibrary:
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1960-01", "2015-12", freq="M", name="month")
    n = len(idx)
    cols = {"MKT_RF": 0.005 + 0.045 * rng.standard_normal(n),
            "Mom12m": 0.008 + 0.04 * rng.standard_normal(n),
            "Mom12m_VW": 0.005 + 0.04 * rng.standard_normal(n),
            "BM": 0.004 + 0.03 * rng.standard_normal(n)}
    r = pd.DataFrame(cols, index=idx)
    info = pd.DataFrame({
        "source": ["french", "cz", "cz_vw", "cz"], "name": ["Market", "Momentum (12 month)",
                                                            "Momentum (12 month) (value-weighted deciles)", "Book to market"],
        "year": [1993, 1993, 1993, 1980], "sample_start": [np.nan, 1965, 1965, 1962],
        "sample_end": [np.nan, 1989, 1989, 1976], "economic_category": [None, "momentum", "momentum", "valuation"],
        "op_weighting": [None, "EW", "VW", "EW"],
    }, index=r.columns)
    return FactorLibrary(returns=r, rf=pd.Series(0.0, index=idx), info=info)


IDEA = {"summary": "Buy stocks with strong past-year returns.", "signal": "past 12-month return",
        "direction": "high past returns", "horizon": "one month",
        "student_reason": "People react slowly to good news, so prices keep drifting up for months after it."}


def responder(tier, system, user, schema):
    name = schema.__name__
    if name == "InterviewTurn":
        ready = "winners" in user.lower()
        return {"reply": "Why do you think it works?" if not ready else "Great, let me check the research.",
                "ready": ready, "idea": IDEA if ready else None}
    if name == "QueryPlan":
        return {"queries": ["momentum"]}
    if name == "CardDraft":
        return {"claim": "Past winners outperform.", "mechanisms": ["behavioral_bias"], "mechanism_notes": "Underreaction.",
                "key_papers": [{"paper": "P1", "finding": "Winners keep winning."}], "contrary_evidence": [],
                "boundary_conditions": [], "grade": "strong", "grade_justification": "Replicated."}
    if name == "ArchitectProposal":
        return {"name": "Investable momentum", "hypothesis": "Past-year winners outperform past-year losers over the next month.",
                "mechanism": "behavioral_bias",
                "components": [{"factor": "Mom12m_VW", "theory_priority": 1, "reason": "Investable momentum."},
                               {"factor": "NOT_A_FACTOR", "theory_priority": 2, "reason": "x"}],
                "mechanism_tests": ["sentiment"], "notes": "Value-weighted version chosen."}
    if name == "GateAdvice":
        return {"assessment": "Coherent.", "suggestions": []}
    if name == "Explanation":
        return {"text": "Your strategy was tested. It earned a Sharpe of 9.99."} if "previous answer" not in user \
            else {"text": "Your strategy was tested; see the facts above."}
    if name == "AnalystReview":
        return {"diagnosis": ["Momentum works in this sample."], "suggestions": [], "key_risks": ["Crashes."]}
    if name == "DebateTurn":
        return {"arguments": [{"point": "The premium is documented.", "evidence": "Factor layer"}], "concessions": []}
    if name == "CoachTurn":
        return {"reply": "Which paper supports your mechanism?", "next_step": "edit_explanation"}
    if name == "UnsupportedAssessment":
        return {"summary": "The idea lacks support.", "why_not_supported": ["No paper documents it."],
                "evidence_needed": ["A mechanism paper."], "alternative_directions": ["Momentum."]}
    if name == "RefereeReport":
        return {"summary": "A momentum strategy.", "recommendation": "major_revision", "major_comments": ["Decay."],
                "minor_comments": [], "tested_results": [], "descriptive_results": [], "what_would_change_my_mind": ["Holdout."]}
    raise AssertionError(name)


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "gather_papers", lambda llm, text, log=None: _papers(log))
    state = StateData(sentiment=lambda: pd.DataFrame({"SENT": [0.0], "SENT_ORTH": [0.0]},
                                                     index=pd.PeriodIndex(["1990-01"], freq="M")),
                      recessions=lambda: pd.Series(dtype=float), vix=lambda: pd.Series(dtype=float))
    return Session(tmp_path / "proj", flow_lib(), FakeLLM(responder), mode="mentor", state_data=state, clock=CLOCK)


def _papers(log):
    papers = search_core("momentum winners losers", k=2)
    for p in papers:
        log.record(p.citation, "momentum")
        p.factor_ids = ["Mom12m"]
    return papers


def test_mentor_flow_end_to_end(session):
    with pytest.raises(FlowError):
        session.research()                                          # no clear idea yet
    assert not session.interview("I like stocks that went up").ready
    with pytest.raises(FlowError):
        session.propose()                                           # no evidence yet
    assert session.interview("buy recent winners").ready
    card = session.research()
    assert card.key_papers and (session.registry.cards_dir / f"{card.id}.json").exists()

    spec, hyp = session.propose()
    assert spec.factor_ids == ["Mom12m_VW"]                       # invalid id dropped
    assert spec.components[0].stock_signal == "mom_12_1"
    assert spec.mechanism_tests == ["publication_decay", "sentiment"]
    assert hyp.evidence_card_ids == [card.id] and 0 < hyp.expected_low < hyp.expected_high

    gate = session.check_gate()
    assert gate.passed and gate.advisory.startswith("Coherent")
    reg = session.register()
    assert reg.id == "H-0001"

    res = session.run(placebo_draws=50)
    assert res.outcome.status == "preregistered"
    assert "Status: preregistered." in res.dossier.text()

    text, risks, bad = session.explain()
    assert "9.99" not in text and bad == []                       # made-up number caught and rewritten
    assert any("before trading costs" in r for r in risks)


def test_upgrade_keeps_state_and_dr_dong_reviews(session):
    session.interview("buy recent winners")
    session.research()
    session.propose()
    session.register()
    session.run(placebo_draws=50)
    session.upgrade("dr_dong")
    review = session.review(rounds=1)
    assert [who for who, _ in review.debate] == ["Advocate", "Reviewer 2"]
    assert review.report.recommendation == "major_revision"
    assert "not personally reviewed by Dr. Feng Dong" in review.footer
    assert session.registration.id == "H-0001"
    assert session.analyze().value.key_risks == ["Crashes."]


def test_mode_limits_and_exploratory_runs(session):
    s = StrategySpec.model_validate({"name": "x", "components": [{"factor": "Mom12m"}, {"factor": "BM"}],
                                     "factor_layer": {"combination": "inverse_vol"}})
    session.load_spec(s)
    with pytest.raises(ValueError, match="not available in mentor"):
        session.run(placebo_draws=50)
    session.upgrade("analyst")
    res = session.run(placebo_draws=50)
    assert res.outcome.status == "exploratory" and res.composite is not None


def test_dossier_numbers_guard():
    d = "Sharpe 0.74; mean 5.8% per year"
    assert unverified_numbers("Sharpe about 0.7 and 5.8%", d) == []
    assert unverified_numbers("Sharpe 0.95", d) == ["0.95"]
    assert unverified_numbers("a shortfall of 3.7% and the 52-week high", "excess -3.7%; High52_VW") == []


def test_session_resumes_without_new_llm_calls(session, tmp_path):
    session.interview("buy recent winners")
    session.research()
    session.propose()
    session.register()
    session.run(placebo_draws=50)
    session.upgrade("dr_dong")
    session.review(rounds=1)
    session.explain()
    calls = len(session.llm.calls)

    fresh_llm = FakeLLM(responder)
    again = Session(session.registry.root, session.lib, fresh_llm, state_data=session.state_data, clock=CLOCK)
    assert again.mode == "dr_dong" and again.registration.id == "H-0001"
    assert again.card.id == session.card.id and again.spec == session.spec
    assert again.dr_dong.report.recommendation == "major_revision" and again.explanation
    again.run(placebo_draws=50)
    assert fresh_llm.calls == [] and calls > 0

"""Report builder: main report by mode, technical appendix, exploratory placement, escaping, PDF output."""

import pandas as pd
import pypdf
import pytest

from priors.flows import Session
from priors.flows import session as S
from priors.mechanisms import StateData
from priors.llm import FakeLLM
from priors.reports import EXPLORATORY_LABEL, build_report, build_reports
from priors.spec import StrategySpec

from test_flows import CLOCK, _papers, flow_lib, responder

STATE = StateData(sentiment=lambda: pd.DataFrame({"SENT": [0.0], "SENT_ORTH": [0.0]}, index=pd.PeriodIndex(["1990-01"], freq="M")),
                  recessions=lambda: pd.Series(dtype=float), vix=lambda: pd.Series(dtype=float))


@pytest.fixture
def registered(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "gather_papers", lambda llm, text, log=None: _papers(log))
    s = Session(tmp_path / "p", flow_lib(), FakeLLM(responder), state_data=STATE, clock=CLOCK)
    s.interview("buy recent winners")
    s.research()
    s.propose()
    s.register()
    s.run(placebo_draws=50)
    return s


def test_mentor_main_report_is_short_and_plain(registered):
    s = registered
    two = s.spec.model_copy(update={"components": [*s.spec.components,
                                                   s.spec.components[0].model_copy(update={"factor": "BM", "theory_priority": 2})]})
    s.spec = two
    s.register()                       # H-0002: a two-factor composite, so the contribution table appears
    s.run(placebo_draws=50)
    s.explain()
    main, appendix = build_reports(s, student="Test Student")
    h = main.html
    for part in ("At a glance", "Credibility checks", "How to read this table", "Explanation",
                 "Other required risk notes", "technical appendix", "Test Student", "Status: pre-registered"):
        assert part in h, part
    for absent in ("Alpha against standard factor models", "Map of published factors", "Composite performance by decade",
                   "The debate"):
        assert absent not in h, absent
    a = appendix.html
    for part in ("Pre-registration record", "What the literature says", "Map of published factors",
                 "Composite performance by decade", "Table A1."):
        assert part in a, part


def test_dr_dong_main_report_has_verdict_and_models(registered, tmp_path):
    s = registered
    s.upgrade("dr_dong")
    s.review(rounds=1)
    s.explain()
    main, appendix = build_reports(s)
    h = main.html
    assert "Dr. Dong&#39;s verdict" in h or "Dr. Dong's verdict" in h
    assert "Alpha against standard factor models" in h and "Tests of the proposed mechanisms" in h
    assert "not personally reviewed by Dr. Feng Dong" in h
    assert "Explanation" not in h.split("At a glance")[1].split("Other required risk notes")[0]   # moved to the appendix
    assert "Referee report by Dr. Dong (full)" in appendix.html and "The debate" in appendix.html
    pdf = main.save_pdf(tmp_path / "main.pdf")
    reader = pypdf.PdfReader(pdf)
    assert 2 <= len(reader.pages) <= 10
    assert "At a glance" in "".join(p.extract_text() for p in reader.pages)
    assert appendix.save_pdf(tmp_path / "appendix.pdf").exists()


def test_exploratory_results_stay_out_of_the_main_report(tmp_path):
    s = Session(tmp_path / "p", flow_lib(), FakeLLM(responder), mode="analyst", state_data=STATE, clock=CLOCK)
    s.load_spec(StrategySpec.model_validate({"name": "x", "components": [{"factor": "Mom12m"}, {"factor": "BM"}]}))
    s.run(placebo_draws=50)
    main, appendix = build_reports(s)
    assert "Performance of the composite" not in main.html and "Credibility checks" not in main.html
    assert "Shown only in the technical appendix" in main.html
    assert "Exploratory results: factor-layer results" in appendix.html
    assert appendix.html.count(EXPLORATORY_LABEL) >= 3 and "Exploratory runs" in appendix.html


def test_agent_text_is_escaped(registered):
    rep = build_report(registered, explanation="**Bold** <script>alert(1)</script>")
    assert "<script>" not in rep.html and "&lt;script&gt;" in rep.html and "<strong>Bold</strong>" in rep.html

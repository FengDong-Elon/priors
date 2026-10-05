"""Literature module: search parsing, retrieval log, priors, evidence cards, and the LLM layer.

No network and no API calls: OpenAlex responses are fixtures and the LLM is a fake.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from priors.factors import FactorLibrary
from priors.literature import Citation
from priors.literature.evidence import CardDraft, EvidenceCard, synthesize
from priors.literature.expected import composite_prior, factor_prior
from priors.literature.log import RetrievalLog
from priors.literature.search import abstract_from_index, parse_openalex, plan_queries, search, search_core
from priors.llm import FakeLLM, LLMRefusal
from priors.llm.anthropic_llm import AnthropicLLM, strict_schema
from priors.registry import Hypothesis, Registry
from priors.spec import StrategySpec

OPENALEX_FIXTURE = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1000/mom.2001",
            "title": "Momentum revisited",
            "publication_year": 2001,
            "authorships": [{"author": {"display_name": "Ann Smith"}}, {"author": {"display_name": "Bo Lee"}}],
            "primary_location": {"source": {"display_name": "Journal of Finance"}},
            "cited_by_count": 500,
            "abstract_inverted_index": {"Winners": [0], "keep": [1], "winning.": [2]},
        },
        {"id": "https://openalex.org/W2", "doi": None, "title": None, "publication_year": 2001, "authorships": []},
    ]
}


def lib_with_doc() -> FactorLibrary:
    rng = np.random.default_rng(1)
    idx = pd.period_range("1960-01", "2020-12", freq="M", name="month")
    r = pd.DataFrame({"MOM": 0.01 + 0.04 * rng.standard_normal(len(idx)),
                      "VAL": 0.004 + 0.03 * rng.standard_normal(len(idx))}, index=idx)
    info = pd.DataFrame({
        "source": ["cz", "french"], "name": ["Momentum", "Value"], "year": [1993, 1993],
        "sample_start": [1965, np.nan], "sample_end": [1989, np.nan],
        "op_tstat": [4.0, np.nan], "replication_quality": ["1_good", np.nan],
    }, index=["MOM", "VAL"])
    return FactorLibrary(returns=r, rf=pd.Series(0.0, index=idx), info=info)


def draft(**kw) -> dict:
    base = dict(
        claim="Past winners outperform past losers over the next 3 to 12 months.",
        mechanisms=["behavioral_bias"], mechanism_notes="Underreaction to news.",
        key_papers=[{"paper": "P1", "finding": "Winners keep winning."}],
        contrary_evidence=[], boundary_conditions=["Weaker in large caps."],
        grade="strong", grade_justification="Replicated widely.",
    )
    base.update(kw)
    return base


# ------------------------------------------------------------- search

def test_openalex_parsing_and_abstract():
    assert abstract_from_index({"b": [1], "a": [0]}) == "a b"
    papers = parse_openalex(OPENALEX_FIXTURE)
    assert len(papers) == 1                      # the record without a title is skipped
    p = papers[0]
    assert p.citation.doi == "10.1000/mom.2001" and p.citation.retrieved_via == "openalex"
    assert p.abstract == "Winners keep winning." and p.cited_by == 500


def test_core_search_finds_original_papers():
    top = search_core("momentum winners losers", k=3)
    assert "Jegadeesh" in top[0].citation.authors and top[0].citation.year == 1993
    assert "Mom12m" in top[0].factor_ids


def test_retrieval_log_guards_citations(tmp_path):
    log = RetrievalLog(tmp_path / "r.jsonl")
    papers = search("momentum", log=log, use_web=False)
    assert papers and all(log.is_retrieved(p.citation) for p in papers)
    forged = papers[0].citation.model_copy(update={"title": "A different title"})
    assert not log.is_retrieved(forged)
    remembered = Citation(doi="10.1/x", title="From memory", authors="X", year=2000, retrieved_via="openalex")
    assert not log.is_retrieved(remembered)


def test_plan_queries_uses_fast_tier():
    llm = FakeLLM(lambda tier, s, u, schema: {"queries": [" momentum ", "momentum replication", ""]})
    assert plan_queries(llm, "buy recent winners") == ["momentum", "momentum replication"]
    assert llm.calls[0]["tier"] == "fast"


# -------------------------------------------------------------- priors

def test_factor_prior_uses_original_sample_then_decay():
    lib = lib_with_doc()
    p = factor_prior(lib, "MOM")
    assert p.base_window == "original sample" and p.base_start == "1965-01" and p.base_end == "1989-12"
    base = lib.returns["MOM"].loc["1965-01":"1989-12"].mean() * 12
    assert p.base_mean_ann == pytest.approx(base)
    assert p.low == pytest.approx(base * 0.42) and p.high == pytest.approx(base * 0.74)
    v = factor_prior(lib, "VAL")
    assert v.base_window == "pre-publication period" and v.base_end == "1992-12"


def test_composite_prior_is_weighted_sum():
    lib = lib_with_doc()
    s = StrategySpec.model_validate({"name": "t", "components": [{"factor": "MOM"}, {"factor": "VAL"}]})
    low, high, priors = composite_prior(s, lib)
    assert low == pytest.approx((priors[0].low + priors[1].low) / 2)
    assert high == pytest.approx((priors[0].high + priors[1].high) / 2)


# -------------------------------------------------------- evidence card

def test_card_drops_unretrieved_labels_and_keeps_code_numbers():
    lib = lib_with_doc()
    papers = search_core("momentum", k=2)
    llm = FakeLLM(lambda *a: draft(key_papers=[{"paper": "P1", "finding": "ok"}, {"paper": "P9", "finding": "fake"}]))
    card = synthesize(llm, "Momentum", papers, ["MOM"], lib)
    assert [f.citation for f in card.key_papers] == [papers[0].citation]
    assert any("'P9'" in w for w in card.warnings)
    assert card.priors[0]["factor"] == "MOM"                   # filled by code
    assert llm.calls[0]["tier"] == "deep"
    assert "do not state any number" in llm.calls[0]["system"].lower()


def test_card_without_support_is_graded_weak(tmp_path):
    lib = lib_with_doc()
    card = synthesize(FakeLLM(lambda *a: draft(key_papers=[{"paper": "P7", "finding": "x"}])),
                      "Momentum", search_core("momentum", k=1), ["MOM"], lib)
    assert card.grade == "weak"
    path = card.save(tmp_path)
    assert EvidenceCard.load(path) == card


def test_registry_requires_card_and_accepts_logged_web_citation(tmp_path):
    lib = lib_with_doc()
    reg = Registry(tmp_path / "proj")
    web_paper = parse_openalex(OPENALEX_FIXTURE)[0]
    h = Hypothesis(
        statement="Stocks with high past returns keep outperforming for several months.",
        mechanism="behavioral_bias",
        mechanism_explanation="Investors underreact to firm news, so prices drift in the direction of recent returns for months.",
        citations=[web_paper.citation], expected_low=0.02, expected_high=0.06,
    )
    s = StrategySpec.model_validate({"name": "t", "components": [{"factor": "MOM"}]})
    res = pytest.raises(Exception, reg.register, h, s)
    assert "not found by a literature search" in str(res.value) and "evidence card" in str(res.value)

    reg.retrievals.record(web_paper.citation, "momentum")       # the search returned it
    card = synthesize(FakeLLM(lambda *a: draft()), "Momentum", [web_paper], ["MOM"], lib)
    card.save(reg.cards_dir)
    r = reg.register(h.model_copy(update={"evidence_card_ids": [card.id]}), s)
    assert r.id == "H-0001"


# ------------------------------------------------------------ LLM layer

class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_client(text: str, stop_reason: str = "end_turn"):
    resp = SimpleNamespace(
        stop_reason=stop_reason, stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )
    return SimpleNamespace(messages=FakeMessages(resp), beta=SimpleNamespace(messages=FakeMessages(resp)))


def test_strict_schema_closes_every_object():
    schema = strict_schema(CardDraft)
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    pf = schema["$defs"]["PaperFinding"]
    assert pf["additionalProperties"] is False and set(pf["required"]) == {"paper", "finding"}


def test_anthropic_deep_tier_uses_fallbacks_and_effort():
    client = fake_client(json.dumps(draft()))
    llm = AnthropicLLM(client=client)
    out = llm.structured("deep", "sys", "user", CardDraft)
    assert out.grade == "strong"
    kw = client.beta.messages.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["effort"] == "high" and kw["output_config"]["format"]["type"] == "json_schema"
    assert llm.usage.calls == 1 and llm.usage.input_tokens == 100


def test_anthropic_fast_tier_plain_call_and_refusal():
    client = fake_client(json.dumps({"queries": ["a"]}))
    from priors.literature.search import QueryPlan
    AnthropicLLM(client=client).structured("fast", "s", "u", QueryPlan)
    kw = client.messages.kwargs
    assert kw["model"] == "claude-haiku-4-5" and "fallbacks" not in kw and "effort" not in kw["output_config"]

    refused = fake_client("", stop_reason="refusal")
    with pytest.raises(LLMRefusal, match="cyber"):
        AnthropicLLM(client=refused).structured("deep", "s", "u", CardDraft)


def test_labels_in_free_text_become_author_year():
    lib = lib_with_doc()
    papers = search_core("momentum winners losers", k=2)
    llm = FakeLLM(lambda *a: draft(boundary_conditions=["Weaker in large caps (P1).", "See P5."],
                                   mechanism_notes="P2 argues underreaction."))
    card = synthesize(llm, "Momentum", papers, ["MOM"], lib)
    assert papers[0].citation.short() in card.boundary_conditions[0]
    assert "[unverified reference removed]" in card.boundary_conditions[1]
    assert card.mechanism_notes.startswith(papers[1].citation.short())
    assert "PE ratio" not in card.mechanism_notes  # words like 'P/E' or 'PE' are untouched


def test_gather_puts_core_library_first(monkeypatch):
    from priors.literature import search as S
    monkeypatch.setattr(S, "search_openalex", lambda q, k=4: parse_openalex(OPENALEX_FIXTURE))
    monkeypatch.setattr(S, "fill_abstracts", lambda papers: None)
    llm = FakeLLM(lambda *a: {"queries": ["value momentum everywhere", "momentum replication"]})
    papers = S.gather_papers(llm, "combine value and momentum")
    assert papers[0].citation.retrieved_via == "core_library"
    assert any("Asness" in p.citation.authors for p in papers)
    assert sum(p.citation.doi == "10.1000/mom.2001" for p in papers) == 1   # web duplicate dropped

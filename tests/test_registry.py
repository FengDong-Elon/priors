"""Theory gate, pre-registration, the trial ledger, and holdout control."""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from priors.literature import Citation, from_core_library
from priors.registry import (
    GateFailed, HoldoutError, Hypothesis, LedgerCorrupted, RecordCorrupted, Registry, check_gate,
)
from priors.spec import StrategySpec

# A fixed clock in 2016 puts the holdout at 2011-01, inside the synthetic data (1990-2020).
CLOCK = lambda: datetime(2016, 1, 15, tzinfo=timezone.utc)  # noqa: E731
JT = "10.1111/j.1540-6261.1993.tb04702.x"


def good_hypothesis(**kw) -> Hypothesis:
    base = dict(
        statement="Stocks that did well over the past year keep outperforming for a few months.",
        mechanism="behavioral_bias",
        mechanism_explanation=(
            "Investors underreact to news, so prices adjust slowly and past winners keep drifting up "
            "while past losers keep drifting down."
        ),
        citations=[from_core_library(JT)],
        expected_low=0.02,
        expected_high=0.08,
    )
    base.update(kw)
    return Hypothesis(**base)


def spec(**kw) -> StrategySpec:
    base = {"name": "t", "components": [{"factor": "A"}, {"factor": "B"}]}
    base.update(kw)
    return StrategySpec.model_validate(base)


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "project", clock=CLOCK, require_evidence=False)


# ------------------------------------------------------------------ gate

def test_gate_lists_every_missing_piece():
    h = Hypothesis(statement="momentum")
    res = check_gate(h)
    assert not res.passed
    text = res.explain()
    for phrase in ("full sentence", "mechanism", "own words", "Attach at least one paper", "expected range"):
        assert phrase in text


def test_gate_rejects_unretrieved_citation():
    fake = Citation(title="A paper I remember", authors="Some One", year=2001, retrieved_via="openalex")
    res = check_gate(good_hypothesis(citations=[fake]))
    assert not res.passed
    assert any("not found by a literature search" in f for f in res.failures)


def test_gate_rejects_bad_range_and_accepts_good_hypothesis():
    assert not check_gate(good_hypothesis(expected_low=0.05, expected_high=0.02)).passed
    assert check_gate(good_hypothesis()).passed


def test_core_library_lookup():
    c = from_core_library("https://doi.org/" + JT.upper())
    assert c.year == 1993 and "Jegadeesh" in c.authors
    with pytest.raises(KeyError):
        from_core_library("10.9999/not-a-paper")


# --------------------------------------------------------- registration

def test_register_locks_spec_and_detects_tampering(reg):
    with pytest.raises(GateFailed):
        reg.register(Hypothesis(statement="too short"), spec())
    r = reg.register(good_hypothesis(), spec())
    assert r.id == "H-0001"
    assert r.spec.status == "preregistered" and r.spec.hypothesis_id == "H-0001"
    assert r.holdout_start == pd.Period("2011-01", "M")
    path = reg.reg_dir / "H-0001.json"
    body = json.loads(path.read_text())
    body["hypothesis"]["expected_high"] = 0.5
    path.write_text(json.dumps(body))
    with pytest.raises(RecordCorrupted):
        reg.get("H-0001")


def test_status_is_decided_by_registry_not_by_spec(reg, lib):
    # A spec claiming to be pre-registered, with no registration, runs as exploratory.
    out = reg.run_factor(spec(status="preregistered"), lib)
    assert out.status == "exploratory"
    assert any("Exploratory" in w for w in out.result.warnings)

    r = reg.register(good_hypothesis(), spec(components=[{"factor": "A"}, {"factor": "C"}]))
    out = reg.run_factor(r.spec, lib)
    assert out.status == "preregistered" and out.hypothesis_id == r.id
    assert not any("Exploratory" in w for w in out.result.warnings)
    assert out.result.sample["end"] == pd.Period("2010-12", "M")


def test_changed_spec_is_a_logged_deviation(reg, lib):
    r = reg.register(good_hypothesis(), spec())
    changed = r.spec.model_copy(update={"factor_layer": r.spec.factor_layer.model_copy(update={"start": "2000-01"})})
    out = reg.run_factor(changed, lib)
    assert out.status == "exploratory" and out.deviation_of == r.id
    assert out.ledger_row["deviation_of"] == r.id


def test_holdout_once_and_only_for_registered(reg, lib):
    with pytest.raises(HoldoutError):
        reg.run_factor(spec(), lib, period="holdout")
    r = reg.register(good_hypothesis(), spec())
    out = reg.run_factor(r.spec, lib, period="holdout")
    assert out.result.sample["start"] == pd.Period("2011-01", "M")
    assert any("Holdout used" in w for w in out.result.warnings)
    with pytest.raises(HoldoutError, match="already been opened"):
        reg.run_factor(r.spec, lib, period="holdout")


def test_promotion_after_exploration_marks_in_sample_as_seen(reg, lib):
    s = spec()
    reg.run_factor(s, lib)                       # explored first
    r = reg.register(good_hypothesis(), s)
    assert r.in_sample_seen
    out = reg.run_factor(r.spec, lib)
    assert any("tested before it was registered" in n for n in out.notes)


# ---------------------------------------------------------------- ledger

def test_trial_count_counts_distinct_factor_specs(reg, lib):
    s = spec()
    reg.run_factor(s, lib)
    reg.run_factor(s, lib)                                            # same spec again
    reg.run_factor(s.model_copy(update={"name": "renamed"}), lib)     # label change only
    reg.run_factor(spec(stock_layer={"holdings": 25}), lib)           # stock layer only
    assert reg.ledger.trial_count("factor") == 1
    reg.run_factor(spec(components=[{"factor": "A"}, {"factor": "C"}]), lib)
    summary = reg.ledger.summary()
    assert summary["trials"] == 2 and summary["exploratory_trials"] == 2 and summary["runs"] == 5


def test_ledger_chain_detects_edits_and_deletions(reg, lib):
    for f in ("B", "C"):
        reg.run_factor(spec(components=[{"factor": "A"}, {"factor": f}]), lib)
    reg.ledger.verify()
    lines = reg.ledger.path.read_text().splitlines()

    edited = json.loads(lines[0])
    edited["sharpe"] = 9.9
    reg.ledger.path.write_text("\n".join([json.dumps(edited)] + lines[1:]) + "\n")
    with pytest.raises(LedgerCorrupted, match="modified"):
        reg.ledger.verify()

    reg.ledger.path.write_text(lines[1] + "\n")
    with pytest.raises(LedgerCorrupted, match="out of sequence"):
        reg.ledger.verify()

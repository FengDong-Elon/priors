"""In-app classroom access: enrollment, weekly budgets, and cost recording (no network)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from priors.classroom import BudgetExceeded, BudgetedLLM, NotEnrolled, classroom_from_env, enroll, parse_classes
from priors.literature.search import QueryPlan
from priors.llm import LLMRefusal
from priors.llm.anthropic_llm import AnthropicLLM
from priors.proxy import Ledger

from test_literature import fake_client

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)
CONFIG = {"classes": {"FIN3500": {"weekly_budget_usd": 0.01, "students": ["AB1234"]},
                      "OPEN": {"weekly_budget_usd": 1.0, "allowed_models": ["claude-haiku-4-5"]}}}


def budgeted(tmp_path, code="FIN3500", sid="ab1234", stop="end_turn"):
    classes = parse_classes(CONFIG)
    inner = AnthropicLLM(client=fake_client(json.dumps({"queries": ["a"]}), stop_reason=stop))
    return BudgetedLLM(inner, enroll(classes, code, sid), Ledger(tmp_path / "u.sqlite"), clock=lambda: NOW)


def test_enrollment_rules():
    classes = parse_classes(CONFIG)
    assert enroll(classes, " FIN3500 ", " AB1234 ").student_id == "ab1234"
    with pytest.raises(NotEnrolled, match="Unknown class"):
        enroll(classes, "NOPE", "ab1234")
    with pytest.raises(NotEnrolled, match="roster"):
        enroll(classes, "FIN3500", "zz9999")
    assert enroll(classes, "OPEN", "anyone").config.students is None


def test_budget_is_charged_and_enforced(tmp_path):
    llm = budgeted(tmp_path)
    llm.structured("fast", "s", "u", QueryPlan)           # 100 in / 50 out tokens on Haiku: $0.00035
    assert llm.spent() == pytest.approx((100 * 1 + 50 * 5) / 1e6)
    for _ in range(40):
        try:
            llm.structured("fast", "s", "u", QueryPlan)
        except BudgetExceeded as e:
            assert "Weekly budget" in str(e) or "weekly budget" in str(e)
            break
    else:
        pytest.fail("budget never enforced")
    assert llm.spent() >= 0.01


def test_refused_calls_are_still_charged(tmp_path):
    llm = budgeted(tmp_path, stop="refusal")
    with pytest.raises(LLMRefusal):
        llm.structured("deep", "s", "u", QueryPlan)
    assert llm.spent() > 0


def test_model_allowlist(tmp_path):
    llm = budgeted(tmp_path, code="OPEN", sid="x")
    llm.structured("fast", "s", "u", QueryPlan)
    with pytest.raises(NotEnrolled, match="not enabled"):
        llm.structured("deep", "s", "u", QueryPlan)       # Opus is not on this class's list


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PRIORS_CLASSROOM", raising=False)
    monkeypatch.delenv("PRIORS_CLASSROOM_CONFIG", raising=False)
    assert classroom_from_env() is None
    monkeypatch.setenv("PRIORS_CLASSROOM", "classes:\n  FIN3500:\n    weekly_budget_usd: 2\n")
    assert classroom_from_env()["FIN3500"].weekly_budget_usd == 2.0


def test_app_connects_with_a_class_code(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PRIORS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PRIORS_AUTO_REFRESH", "0")
    monkeypatch.setenv("PRIORS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PRIORS_CLASSROOM", "classes:\n  FIN3500:\n    weekly_budget_usd: 2\n    students: [ab1234]\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-instructor")
    monkeypatch.setenv("PRIORS_ALLOW_LOCAL_KEY", "0")
    app = Path(__file__).resolve().parents[1] / "src" / "priors" / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    assert at.radio[0].options[0] == "Class code"
    at.text_input[0].set_value("FIN3500")
    at.text_input[1].set_value("zz9999")
    [b for b in at.button if b.label == "Connect"][0].click().run()
    assert any("roster" in e.value for e in at.error)
    at.text_input[1].set_value("AB1234")
    [b for b in at.button if b.label == "Connect"][0].click().run()
    assert isinstance(at.session_state["llm"], BudgetedLLM)
    assert any("of $2.00 used" in c.value for c in at.caption)


def test_dated_model_ids_use_the_right_price():
    from priors.proxy import CONSERVATIVE, PRICES, price_for

    assert price_for("claude-haiku-4-5-20251001") == PRICES["claude-haiku-4-5"]
    assert price_for("claude-opus-5-5") == PRICES["claude-opus-5-5"]
    assert price_for("claude-opus-5-20260101") == PRICES["claude-opus-5"]      # not confused with opus-5-5
    assert price_for("some-new-model") == CONSERVATIVE


def test_classroom_mode_does_not_need_the_proxy_packages():
    """A hosted app installs only the app extras, without FastAPI."""
    import subprocess
    import sys

    code = ("import sys; import priors.classroom, priors.budget; "
            "assert 'fastapi' not in sys.modules and 'priors.proxy' not in sys.modules")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

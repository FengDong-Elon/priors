"""Classroom access without a separate server (ARCHITECTURE §13.2).

When the app is deployed (for example on Streamlit Community Cloud), the
instructor stores two secrets:

* ``ANTHROPIC_API_KEY``: a key from an Anthropic workspace created for the
  course, with a monthly spend limit set in the Anthropic Console. That limit
  is the hard cap; nothing here can exceed it.
* ``PRIORS_CLASSROOM``: the class configuration, in the same YAML shape as the
  proxy's ``classroom.yaml`` (class codes, weekly budget per student, optional
  roster, allowed models, whether exploratory backtests are allowed, and
  which stage counts as a finished project).

Students sign in with the class code and their student id. ``BudgetedLLM``
checks the student's spending this week before each call and records the cost
after it. The usage log is a SQLite file; on hosts whose disk is reset on
restart it is a soft limit, which is why the workspace spend limit matters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
import yaml

from .llm.anthropic_llm import AnthropicLLM
from .llm.base import T, Tier, Usage
from .proxy import ClassConfig, Ledger, cost_usd, iso_week, parse_classes  # noqa: F401  (parse_classes re-exported)

CONFIG_ENV = "PRIORS_CLASSROOM"            # YAML text (for example from Streamlit secrets)
CONFIG_PATH_ENV = "PRIORS_CLASSROOM_CONFIG"  # or a path to classroom.yaml


class BudgetExceeded(RuntimeError):
    pass


class NotEnrolled(PermissionError):
    pass


def classroom_from_env() -> dict[str, ClassConfig] | None:
    """Class configuration from PRIORS_CLASSROOM (YAML text) or PRIORS_CLASSROOM_CONFIG (a file), if either is set."""
    text = os.environ.get(CONFIG_ENV)
    if not text and os.environ.get(CONFIG_PATH_ENV) and Path(os.environ[CONFIG_PATH_ENV]).exists():
        text = Path(os.environ[CONFIG_PATH_ENV]).read_text(encoding="utf-8")
    if not text:
        return None
    classes = parse_classes(yaml.safe_load(text) or {})
    return classes or None


@dataclass
class Student:
    class_code: str
    student_id: str
    config: ClassConfig


def enroll(classes: dict[str, ClassConfig], class_code: str, student_id: str) -> Student:
    code, sid = class_code.strip(), student_id.strip().lower()
    conf = classes.get(code)
    if conf is None:
        raise NotEnrolled("Unknown class code.")
    if not sid:
        raise NotEnrolled("Enter your student id.")
    if conf.students is not None and sid not in conf.students:
        raise NotEnrolled("This student id is not on the class roster.")
    return Student(code, sid, conf)


class BudgetedLLM:
    """An AnthropicLLM that enforces one student's weekly budget and records every call."""

    def __init__(self, inner: AnthropicLLM, student: Student, ledger: Ledger, clock=None):
        self.inner, self.student, self.ledger = inner, student, ledger
        self._clock = clock

    @property
    def usage(self) -> Usage:
        return self.inner.usage

    def _week(self) -> str:
        return iso_week(self._clock() if self._clock else None)

    def spent(self) -> float:
        return self.ledger.spent(self.student.class_code, self.student.student_id, self._week())

    def remaining(self) -> float:
        return max(self.student.config.weekly_budget_usd - self.spent(), 0.0)

    def structured(self, tier: Tier, system: str, user: str, schema: type[T], max_tokens: int = 16000) -> T:
        model = self.inner.models[tier]
        if model not in self.student.config.allowed_models:
            raise NotEnrolled(f"Model {model} is not enabled for this class.")
        if self.remaining() <= 0:
            raise BudgetExceeded(
                f"Your weekly budget of ${self.student.config.weekly_budget_usd:.2f} is used up. It resets next Monday (UTC)."
            )
        self.inner.last_call = None
        try:
            return self.inner.structured(tier, system, user, schema, max_tokens)
        finally:
            call = self.inner.last_call
            if call is not None:   # a call that reached the API is charged even if its output was unusable
                self.ledger.record(self.student.class_code, self.student.student_id, self._week(), call["model"],
                                   call, cost_usd(call["model"], call))


def classroom_llm(classes: dict[str, ClassConfig], class_code: str, student_id: str, ledger_path: str | Path,
                  api_key: str | None = None) -> BudgetedLLM:
    student = enroll(classes, class_code, student_id)
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("The instructor's ANTHROPIC_API_KEY is not configured on this deployment.")
    return BudgetedLLM(AnthropicLLM(client=anthropic.Anthropic(api_key=key)), student, Ledger(ledger_path))

"""Class settings, model prices, and the usage ledger.

Shared by the in-app classroom mode (``priors.classroom``) and the optional proxy service
(``priors.proxy``). It needs no web-server packages, so the app can use it on any host.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

# USD per million tokens: (input, output, cache read)
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
}
CONSERVATIVE = (5.0, 25.0, 0.50)


@dataclass
class ClassConfig:
    weekly_budget_usd: float
    students: set[str] | None = None
    allowed_models: set[str] = field(default_factory=lambda: {"claude-opus-5-5", "claude-haiku-4-5"})
    allow_exploratory: bool = False      # may students backtest before passing the theory gate?
    required_stage: str = "dr_dong"      # what counts as a finished project: "tested" or "dr_dong"


STAGES = ("tested", "dr_dong")


def parse_classes(raw: dict) -> dict[str, ClassConfig]:
    """Class settings from the YAML structure shared by the proxy and the in-app classroom mode."""
    out = {}
    for code, c in (raw.get("classes") or {}).items():
        stage = str(c.get("required_stage", "dr_dong"))
        if stage not in STAGES:
            raise ValueError(f"{code}: required_stage must be one of {STAGES}")
        out[str(code)] = ClassConfig(
            weekly_budget_usd=float(c["weekly_budget_usd"]),
            students={str(s).strip().lower() for s in c["students"]} if c.get("students") else None,
            allowed_models=set(c.get("allowed_models") or ClassConfig(0).allowed_models),
            allow_exploratory=bool(c.get("allow_exploratory", False)),
            required_stage=stage,
        )
    return out


def load_config(path: str | Path) -> dict[str, ClassConfig]:
    return parse_classes(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def price_for(model: str) -> tuple[float, float, float]:
    """Prices for a model id, including dated ids such as 'claude-haiku-4-5-20251001'. Unknown models are
    charged conservatively."""
    if model in PRICES:
        return PRICES[model]
    matches = [k for k in PRICES if model.startswith(k + "-")]
    return PRICES[max(matches, key=len)] if matches else CONSERVATIVE


def cost_usd(model: str, usage: dict) -> float:
    pin, pout, pcache = price_for(model)
    return (usage.get("input_tokens", 0) * pin + usage.get("output_tokens", 0) * pout
            + usage.get("cache_creation_input_tokens", 0) * pin * 1.25
            + usage.get("cache_read_input_tokens", 0) * pcache) / 1e6


def iso_week(now: datetime | None = None) -> str:
    y, w, _ = (now or datetime.now(timezone.utc)).isocalendar()
    return f"{y}-W{w:02d}"


class Ledger:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)   # a fresh deployment has no data folder yet
        self.path = str(path)
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS usage (
                ts TEXT, class TEXT, student TEXT, week TEXT, model TEXT,
                input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL)""")

    def spent(self, cls: str, student: str, week: str) -> float:
        with sqlite3.connect(self.path) as con:
            row = con.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE class=? AND student=? AND week=?",
                              (cls, student, week)).fetchone()
        return float(row[0])

    def record(self, cls: str, student: str, week: str, model: str, usage: dict, cost: float) -> None:
        with self._lock, sqlite3.connect(self.path) as con:
            con.execute("INSERT INTO usage VALUES (?,?,?,?,?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(timespec="seconds"), cls, student, week, model,
                         usage.get("input_tokens", 0), usage.get("output_tokens", 0), cost))

    def report(self, cls: str, week: str) -> list[dict]:
        with sqlite3.connect(self.path) as con:
            rows = con.execute("""SELECT student, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd)
                                  FROM usage WHERE class=? AND week=? GROUP BY student ORDER BY 5 DESC""",
                               (cls, week)).fetchall()
        return [{"student": r[0], "calls": r[1], "input_tokens": r[2], "output_tokens": r[3], "cost_usd": round(r[4], 4)}
                for r in rows]

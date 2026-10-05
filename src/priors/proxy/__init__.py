"""Classroom proxy (ARCHITECTURE §12.2, §13.2).

A small service that sits between students and the Anthropic API:

* Students authenticate with ``<class code>:<student id>`` as their API key.
* The proxy checks the class, the roster (if one is configured), the model, and
  the student's spending this ISO week, then forwards the request with the
  instructor's key (``ANTHROPIC_API_KEY`` on the server).
* After each response it records the cost from the response's token usage.

Configuration (YAML, path in ``PRIORS_CLASSROOM_CONFIG``)::

    classes:
      FIN3500-F26:
        weekly_budget_usd: 2.00
        students: [ab1234, cd5678]          # optional roster; omit to accept any id
        allowed_models: [claude-opus-5-5, claude-haiku-4-5]

Run with ``priors proxy`` (uvicorn). Usage is stored in SQLite.
Streaming requests are refused, because Priors does not use them and their
cost could not be recorded reliably.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

UPSTREAM = "https://api.anthropic.com"
# USD per million tokens: (input, output, cache read)
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
}
CONSERVATIVE = (5.0, 25.0, 0.50)
FORWARD_HEADERS = ("anthropic-version", "anthropic-beta", "content-type")


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


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": "priors_proxy_error", "message": message}}, status_code=status)


def create_app(classes: dict[str, ClassConfig], api_key: str, db_path: str | Path,
               upstream: str = UPSTREAM, transport: httpx.BaseTransport | None = None,
               clock=lambda: datetime.now(timezone.utc)) -> FastAPI:
    app = FastAPI(title="Priors classroom proxy", docs_url=None, redoc_url=None)
    ledger = Ledger(db_path)
    client = httpx.Client(base_url=upstream, timeout=600, transport=transport)

    def identify(request: Request) -> tuple[str, str, ClassConfig] | JSONResponse:
        token = request.headers.get("x-api-key", "")
        if ":" not in token:
            return _error(401, "Use '<class code>:<student id>' as the key.")
        cls, student = token.split(":", 1)
        student = student.strip().lower()
        conf = classes.get(cls)
        if conf is None:
            return _error(401, "Unknown class code.")
        if conf.students is not None and student not in conf.students:
            return _error(401, "This student id is not on the class roster.")
        return cls, student, conf

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/usage")
    def usage(request: Request):
        who = identify(request)
        if isinstance(who, JSONResponse):
            return who
        cls, student, conf = who
        week = iso_week(clock())
        spent = ledger.spent(cls, student, week)
        return {"week": week, "spent_usd": round(spent, 4), "budget_usd": conf.weekly_budget_usd,
                "remaining_usd": round(max(conf.weekly_budget_usd - spent, 0), 4)}

    @app.post("/v1/messages")
    async def messages(request: Request):
        who = identify(request)
        if isinstance(who, JSONResponse):
            return who
        cls, student, conf = who
        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return _error(400, "Request body is not JSON.")
        if payload.get("stream"):
            return _error(400, "Streaming is not supported by the classroom proxy.")
        model = payload.get("model", "")
        if model not in conf.allowed_models:
            return _error(403, f"Model {model!r} is not enabled for this class.")
        week = iso_week(clock())
        if ledger.spent(cls, student, week) >= conf.weekly_budget_usd:
            return _error(429, f"Weekly budget of ${conf.weekly_budget_usd:.2f} used up. It resets next Monday (UTC).")

        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_HEADERS}
        headers["x-api-key"] = api_key
        up = await run_in_threadpool(client.post, "/v1/messages", params=dict(request.query_params), content=body,
                                     headers=headers)
        if up.status_code == 200:
            data = up.json()
            served = data.get("model", model)
            u = data.get("usage") or {}
            ledger.record(cls, student, week, served, u, cost_usd(served, u))
        return Response(content=up.content, status_code=up.status_code,
                        media_type=up.headers.get("content-type", "application/json"))

    app.state.ledger = ledger
    return app


def app_from_env() -> FastAPI:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY (the instructor's key) on the proxy server.")
    cfg = os.environ.get("PRIORS_CLASSROOM_CONFIG", "classroom.yaml")
    db = os.environ.get("PRIORS_PROXY_DB", "priors_proxy_usage.sqlite")
    return create_app(load_config(cfg), key, db)

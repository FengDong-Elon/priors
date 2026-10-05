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
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from ..budget import (  # noqa: F401  (re-exported for callers of the proxy module)
    CONSERVATIVE, PRICES, STAGES, ClassConfig, Ledger, cost_usd, iso_week, load_config, parse_classes, price_for,
)

UPSTREAM = "https://api.anthropic.com"
FORWARD_HEADERS = ("anthropic-version", "anthropic-beta", "content-type")


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

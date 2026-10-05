"""Classroom proxy, LLM access, stock snapshots, and a click-through of the Streamlit app (no network)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from priors.llm.access import classroom_token, make_llm
from priors.proxy import ClassConfig, cost_usd, create_app, iso_week, load_config

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


# ------------------------------------------------------------------- proxy

def upstream(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = json.loads(request.content)
        return httpx.Response(200, json={"model": body["model"], "type": "message", "content": [],
                                         "usage": {"input_tokens": 100_000, "output_tokens": 20_000}})
    return httpx.MockTransport(handler)


@pytest.fixture
def proxy(tmp_path):
    calls = []
    classes = {"FIN3500": ClassConfig(weekly_budget_usd=1.0, students={"ab1234"}),
               "OPEN": ClassConfig(weekly_budget_usd=5.0)}
    app = create_app(classes, "sk-instructor", tmp_path / "u.sqlite", transport=upstream(calls), clock=lambda: NOW)
    return TestClient(app), calls


def post(client, key, model="claude-opus-5-5", **extra):
    return client.post("/v1/messages?beta=true", headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                       json={"model": model, "max_tokens": 10, "messages": [], **extra})


def test_proxy_forwards_with_instructor_key_and_records_cost(proxy):
    client, calls = proxy
    r = post(client, "FIN3500:AB1234")
    assert r.status_code == 200
    assert calls[0].headers["x-api-key"] == "sk-instructor"
    assert calls[0].url.params["beta"] == "true"
    u = client.get("/usage", headers={"x-api-key": "FIN3500:ab1234"}).json()
    assert u["week"] == iso_week(NOW) and u["spent_usd"] == pytest.approx(cost_usd("claude-opus-5-5", {"input_tokens": 100_000, "output_tokens": 20_000}))
    assert u["spent_usd"] == pytest.approx(0.8)


def test_proxy_rejects_bad_credentials_models_streams_and_overspend(proxy):
    client, calls = proxy
    assert post(client, "nonsense").status_code == 401
    assert post(client, "NOPE:ab1234").status_code == 401
    assert post(client, "FIN3500:zz9999").status_code == 401          # not on the roster
    assert post(client, "OPEN:anyone", model="claude-fable-5-1").status_code == 403
    assert post(client, "OPEN:anyone", stream=True).status_code == 400
    assert post(client, "FIN3500:ab1234").status_code == 200           # $0.80 of $1.00
    assert post(client, "FIN3500:ab1234").status_code == 200           # crosses the budget
    r = post(client, "FIN3500:ab1234")
    assert r.status_code == 429 and "Weekly budget" in r.json()["error"]["message"]
    assert len(calls) == 2


def test_config_loading(tmp_path):
    p = tmp_path / "classroom.yaml"
    p.write_text("classes:\n  FIN3500:\n    weekly_budget_usd: 2.5\n    students: [AB1234]\n")
    c = load_config(p)["FIN3500"]
    assert c.weekly_budget_usd == 2.5 and c.students == {"ab1234"} and "claude-opus-5-5" in c.allowed_models


def test_llm_access_modes():
    assert classroom_token("FIN3500", " AB1234 ") == "FIN3500:ab1234"
    with pytest.raises(ValueError):
        classroom_token("FIN:3500", "x")
    via_proxy = make_llm(proxy_url="http://localhost:8787/", class_code="FIN3500", student_id="ab1234")
    assert str(via_proxy.client.base_url).startswith("http://localhost:8787")
    assert via_proxy.client.api_key == "FIN3500:ab1234"
    own = make_llm(api_key="sk-test")
    assert own.client.api_key == "sk-test"
    with pytest.raises(ValueError):
        make_llm()


# ---------------------------------------------------------------- snapshot

def test_snapshot_round_trip(tmp_path, monkeypatch):
    import priors.stockdata as SD
    from test_stocklayer import synthetic

    monkeypatch.setenv("PRIORS_CACHE_DIR", str(tmp_path / "cache"))
    px, u = synthetic(32)
    px["raw_close"] = px["close"]
    monkeypatch.setattr(SD, "load_universe", lambda name, refresh=False: u)
    monkeypatch.setattr(SD, "download_prices", lambda tickers, start, refresh=False: px)
    monkeypatch.setattr(SD, "fetch_fundamentals", lambda tickers, progress=None: pd.DataFrame(
        {"book_value_ps": 1.0, "shares_outstanding": 1e6}, index=u["ticker"]))
    monkeypatch.setattr(SD.french, "load_nyse_me_breakpoints", lambda: None)
    assert SD.load_snapshot("russell3000") is None
    meta = SD.refresh_snapshot("russell3000", "2000-01-01", log=lambda m: None)
    assert meta["stocks"] == 32 and meta["fundamentals_coverage"] == 1.0
    snap = SD.load_snapshot("russell3000")
    assert len(snap.context.panel.ids) == 32 and snap.context.fundamentals is not None
    assert snap.context.universe.attrs["label"].startswith("Top 3,000")


# --------------------------------------------------------------------- app

APP = Path(__file__).resolve().parents[1] / "src" / "priors" / "app" / "streamlit_app.py"


def test_app_click_through_mentor(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import priors.factors as F
    import priors.stockdata as SD
    from priors.flows import session as S
    from priors.llm import FakeLLM
    from priors.mechanisms import StateData
    from test_flows import _papers, flow_lib, responder

    monkeypatch.setenv("PRIORS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PRIORS_AUTO_REFRESH", "0")
    monkeypatch.setenv("PRIORS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(F.FactorLibrary, "load", classmethod(lambda cls, **kw: flow_lib()))
    monkeypatch.setattr(SD, "load_source", lambda source, universe: None)
    monkeypatch.setattr(SD, "available_sources", lambda: ["yfinance"])
    monkeypatch.setattr(S, "gather_papers", lambda llm, text, log=None: _papers(log))
    state = StateData(sentiment=lambda: pd.DataFrame({"SENT": [0.0], "SENT_ORTH": [0.0]},
                                                     index=pd.PeriodIndex(["1990-01"], freq="M")),
                      recessions=lambda: pd.Series(dtype=float), vix=lambda: pd.Series(dtype=float))
    monkeypatch.setattr(S, "StateData", lambda: state)
    monkeypatch.setattr(S, "run_mechanism_tests", lambda res, lib, data=None: [])

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert any("Connect in the sidebar" in i.value for i in at.info)

    at.session_state["llm"] = FakeLLM(responder)
    at.session_state["who"] = ("own key", "")
    at.run()

    def click(label):
        b = [b for b in at.button if b.label == label]
        assert b, f"no button {label!r}; have {[x.label for x in at.button]}"
        b[0].click().run()

    click("Start in Mentor")
    assert not at.exception
    at.chat_input[0].set_value("buy recent winners").run()
    click("Search the research")
    click("Yes, conclude: not supported")          # inside a confirmation popover
    assert any("not supported by the literature" in h.value for h in at.subheader)
    click("Withdraw this conclusion and keep working")
    assert at.session_state["session"].unsupported is None

    def broken():
        raise RuntimeError("upstream overloaded")
    at.session_state["session"].propose = broken
    click("Propose a strategy")                    # a failed step keeps its error message on screen
    assert any("upstream overloaded" in e.value for e in at.error)
    del at.session_state["session"].propose
    click("Propose a strategy")
    click("Check the theory gate")
    assert any("passes the theory gate" in s.value for s in at.success)
    click("Pre-register (this locks the plan)")
    click("Run the tests")
    assert not at.exception, at.exception
    assert any(m.label == "Sharpe ratio" for m in at.metric)
    assert at.session_state["session"].results.outcome.status == "preregistered"


    # explore first, then register the same plan and open the sealed holdout (Mentor mode)
    at.session_state["session"].start_new_idea()
    at.run()
    at.chat_input[0].set_value("buy recent winners").run()
    click("Search the research")
    click("Propose a strategy")
    click("Check the theory gate")
    click("Explore without registering")
    assert at.session_state["session"].results.outcome.status == "exploratory"
    click("Pre-register (this locks the plan)")
    assert at.session_state["session"].registration.in_sample_seen
    at.checkbox[0].check().run()
    click("Open the holdout")                      # the synthetic data end before the holdout window
    assert not at.exception, at.exception
    assert any("No holdout months" in e.value for e in at.error)
    reg = at.session_state["session"].registration
    assert not at.session_state["session"].registry.ledger.holdout_opened(reg.id)   # a failed run keeps the holdout


def test_sharadar_source_only_when_configured(tmp_path, monkeypatch):
    import priors.stockdata as SD

    for v in ("SHARADAR_SEP_PATH", "SHARADAR_SF1_PATH", "SHARADAR_TICKERS_PATH"):
        monkeypatch.delenv(v, raising=False)
    assert SD.available_sources() == ["yfinance"]
    for v in ("SHARADAR_SEP_PATH", "SHARADAR_SF1_PATH", "SHARADAR_TICKERS_PATH"):
        f = tmp_path / f"{v}.parquet"
        f.write_bytes(b"x")
        monkeypatch.setenv(v, str(f))
    assert SD.available_sources() == ["yfinance", "sharadar"]


def test_fundamental_signal_on_a_point_in_time_source_gives_a_clear_message():
    from dataclasses import replace

    from priors.spec import StrategySpec
    from priors.stocklayer import FundamentalHistoryUnavailable, run_stock_backtest
    from test_stocklayer import panel_for, synthetic

    px, u = synthetic(20)
    p = panel_for(px, u)
    pit = replace(p, capabilities=replace(p.capabilities, point_in_time_fundamentals=True, includes_delisted=True))
    spec = StrategySpec.model_validate({"name": "t", "components": [{"factor": "HML", "stock_signal": "book_to_market"}]})
    with pytest.raises(FundamentalHistoryUnavailable, match="not built for any data source"):
        run_stock_backtest(spec, pit, None)


def test_ensure_snapshot_builds_when_stale_and_falls_back(tmp_path, monkeypatch):
    import priors.stockdata as SD
    from test_stocklayer import synthetic

    monkeypatch.setenv("PRIORS_CACHE_DIR", str(tmp_path / "cache"))
    px, u = synthetic(32)
    calls = []
    monkeypatch.setattr(SD, "load_universe", lambda name, refresh=False: u)

    def download(tickers, start, refresh=False):
        calls.append(start)
        if len(calls) > 1:
            raise ConnectionError("Yahoo refused")
        return px
    monkeypatch.setattr(SD, "download_prices", download)
    monkeypatch.setattr(SD.french, "load_nyse_me_breakpoints", lambda: None)

    snap, status = SD.ensure_snapshot("russell3000", log=lambda m: None)
    assert status == "refreshed" and snap is not None and calls == ["2000-01-01"]
    assert SD.ensure_snapshot("russell3000", log=lambda m: None)[1] == "current"      # fresh: no download
    monkeypatch.setattr(SD, "snapshot_age_hours", lambda universe="russell3000": 48.0)
    snap2, status2 = SD.ensure_snapshot("russell3000", log=lambda m: None)          # stale, download fails
    assert snap2 is not None and "Showing the previous data" in status2
    lock = SD.snapshot_dir("russell3000").with_name("russell3000.lock")
    assert not lock.exists()
    lock.write_text("x")                                                            # another session is refreshing
    snap3, status3 = SD.ensure_snapshot("russell3000", log=lambda m: None)
    assert snap3 is not None and "another session" in status3


def test_theme_config_matches_the_cli_flags():
    import tomllib

    from priors.app.theme import CLIENT, SIDEBAR, THEME, cli_flags

    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert {k: v for k, v in cfg["theme"].items() if k != "sidebar"} == THEME
    assert cfg["theme"]["sidebar"] == SIDEBAR and cfg["client"] == CLIENT
    assert "--theme.primaryColor=" + THEME["primaryColor"] in cli_flags()
    assert "--theme.showWidgetBorder=true" in cli_flags()

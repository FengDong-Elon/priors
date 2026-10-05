"""Class rules, revision loop, literature-only conclusions, progress, project files, and Sharadar downloads."""

import io
import json
import zipfile

import pandas as pd
import pytest

from priors.data import sharadar_download as SDL
from priors.flows import FlowError, Session
from priors.flows import session as S
from priors.flows.projectfile import ProjectFileError, export_project, import_project, read_manifest
from priors.llm import FakeLLM
from priors.mechanisms import StateData
from priors.proxy import parse_classes
from priors.reports import build_reports

from test_flows import CLOCK, _papers, flow_lib, responder

STATE = StateData(sentiment=lambda: pd.DataFrame({"SENT": [0.0], "SENT_ORTH": [0.0]}, index=pd.PeriodIndex(["1990-01"], freq="M")),
                  recessions=lambda: pd.Series(dtype=float), vix=lambda: pd.Series(dtype=float))


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "gather_papers", lambda llm, text, log=None: _papers(log))

    def make(**kw):
        return Session(tmp_path / "p", flow_lib(), FakeLLM(responder), state_data=STATE, clock=CLOCK, **kw)
    return make


def to_proposal(s):
    s.interview("buy recent winners")
    s.research()
    s.propose()
    return s


# ------------------------------------------------------------- class rules

def test_class_settings_parse():
    c = parse_classes({"classes": {"A": {"weekly_budget_usd": 1, "allow_exploratory": True, "required_stage": "tested"},
                                   "B": {"weekly_budget_usd": 1}}})
    assert c["A"].allow_exploratory and c["A"].required_stage == "tested"
    assert not c["B"].allow_exploratory and c["B"].required_stage == "dr_dong"
    with pytest.raises(ValueError):
        parse_classes({"classes": {"C": {"weekly_budget_usd": 1, "required_stage": "whenever"}}})


def test_strict_class_blocks_unregistered_tests(make_session):
    s = to_proposal(make_session(allow_exploratory=False))
    with pytest.raises(FlowError, match="must pass the theory gate"):
        s.run(placebo_draws=50)
    s.check_gate()
    s.register()
    assert s.run(placebo_draws=50).outcome.status == "preregistered"


# ------------------------------------------------------------ revising

def test_coaching_before_testing_is_free(make_session):
    s = to_proposal(make_session())
    turn = s.coach("Is my explanation good enough?")
    assert turn.next_step == "edit_explanation" and len(s.coaching) == 2
    card = s.research_again("Buy stocks with strong 12-month returns, excluding the last month")
    assert card is not None and s.spec is None and s.idea.summary.startswith("Buy stocks")
    assert s.registry.ledger.trial_count() == 0


def test_revision_after_results_is_a_new_trial(make_session):
    s = to_proposal(make_session())
    s.check_gate()
    s.register()
    s.run(placebo_draws=50)
    with pytest.raises(FlowError):
        s.research_again("anything")
    notice = s.start_revision()
    assert "new trial" in notice and s.spec is None and s.results is None and s.revisions == 1
    s.propose()
    s.check_gate()
    reg2 = s.register()
    assert reg2.id == "H-0002" and reg2.in_sample_seen      # the same spec was already tested
    assert s.progress()["revisions"] == 1


def test_unsupported_conclusion_and_report(make_session):
    s = to_proposal(make_session())
    a = s.conclude_unsupported()
    assert a.alternative_directions == ["Momentum."]
    assert s.progress()["status"] == "complete: not supported by the literature"
    main, appendix = build_reports(s)
    assert "literature assessment" in main.html and "Why the idea was not tested" in main.html
    assert "Retrieved papers" in appendix.html
    again = Session(s.registry.root, s.lib, FakeLLM(responder), state_data=STATE, clock=CLOCK)
    assert again.unsupported == a                           # saved with the project


def test_progress_follows_the_class_stage(make_session):
    s = to_proposal(make_session(required_stage="tested"))
    assert s.progress()["status"] == "in progress" and "Dr. Dong review" not in s.progress()["steps"]
    s.check_gate()
    s.register()
    s.run(placebo_draws=50)
    assert s.progress()["status"] == "complete"
    d = to_proposal(make_session(required_stage="dr_dong"))   # same folder: loads the tested project
    d.required_stage = "dr_dong"
    assert d.progress()["status"] == "in progress"


# ---------------------------------------------------------- project files

def test_project_file_round_trip_and_tamper_checks(make_session, tmp_path):
    s = to_proposal(make_session())
    s.check_gate()
    s.register()
    s.run(placebo_draws=50)
    data = export_project(s.registry.root, "Stu", "My project")
    assert read_manifest(data)["project"] == "My project"
    dest = import_project(data, tmp_path / "restored")
    again = Session(dest, s.lib, FakeLLM(responder), state_data=STATE, clock=CLOCK)
    assert again.registration.id == "H-0001" and again.spec == s.spec

    with pytest.raises(ProjectFileError, match="already exists"):
        import_project(data, dest)

    # edit the ledger inside the file: the hash chain no longer verifies
    zin = zipfile.ZipFile(io.BytesIO(data))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zout:
        for i in zin.infolist():
            content = zin.read(i)
            if i.filename == "ledger.jsonl":
                rows = [json.loads(x) for x in content.decode().splitlines()]
                rows[-1]["sharpe"] = 9.99
                content = ("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n").encode()
            zout.writestr(i, content)
    with pytest.raises(ProjectFileError, match="do not verify"):
        import_project(buf.getvalue(), tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_project_file_rejects_unsafe_entries(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("priors_project.json", json.dumps({"format": 1}))
        z.writestr("../evil.json", "{}")
    with pytest.raises(ProjectFileError, match="not allowed"):
        import_project(buf.getvalue(), tmp_path / "x")
    with pytest.raises(ProjectFileError, match="not a Priors project"):
        import_project(b"not a zip", tmp_path / "y")


# --------------------------------------------------------- Sharadar download

class FakeResponse:
    def __init__(self, status=200, json_data=None, content=b""):
        self.status_code, self._json, self.content = status, json_data, content

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def zipped_csv(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("export.csv", text)
    return buf.getvalue()


class FakeNDL:
    def __init__(self):
        self.calls = []
        self.polls = 0

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if url.startswith("https://files/"):
            table = url.rsplit("/", 1)[-1]
            return FakeResponse(content=zipped_csv({
                "SEP": "ticker,date,close,closeadj,closeunadj,volume,lastupdated\nAAA,2020-01-02,10,10,10,100,2020-01-03\n",
                "SF1": "ticker,dimension,datekey,lastupdated,marketcap,price,bvps,eps,roe,gp,revenue,cor,assets,sharesbas\n"
                       "AAA,ART,2020-02-01,2020-02-02,1e9,10,5,1,0.1,50,100,50,500,1e8\n",
                "TICKERS": "ticker,name,exchange,category,sector,industry,isdelisted\nAAA,Alpha,NYSE,Domestic Common Stock,Tech,Software,N\n",
            }[table]))
        if params.get("api_key") == "bad":
            return FakeResponse(status=401)
        table = url.split("/SHARADAR/")[1].split(".json")[0]
        self.polls += 1
        status = "creating" if self.polls == 1 else "fresh"
        return FakeResponse(json_data={"datatable_bulk_download": {"file": {"status": status, "link": f"https://files/{table}"}}})


def test_sharadar_download_writes_parquet_and_registers_source(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIORS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PRIORS_HOSTED", raising=False)
    for v in SDL.ENV_FOR.values():
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(SDL.time, "sleep", lambda s: None)
    fake = FakeNDL()
    paths = SDL.download_sharadar("goodkey", log=lambda m: None, session=fake)
    sf1 = pd.read_parquet(paths["SHARADAR_SF1_PATH"])
    assert "date" in sf1.columns and sf1.loc[0, "bvps"] == 5          # 'date' added for the Sharadar provider
    assert pd.read_parquet(paths["SHARADAR_SEP_PATH"]).shape[0] == 1
    assert not any("goodkey" in json.dumps(json.loads(SDL.sources_config().read_text())) for _ in [0])  # key not saved
    sep_params = next(p for u, p in fake.calls if "SEP" in u and p)
    assert sep_params["date.gte"] == "1998-01-01" and sep_params["qopts.export"] == "true"
    from priors.stockdata import available_sources
    assert "sharadar" in available_sources()


def test_sharadar_download_refuses_bad_key_and_hosted(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIORS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(SDL.time, "sleep", lambda s: None)
    with pytest.raises(SDL.SharadarDownloadError, match="rejected"):
        SDL.download_sharadar("bad", log=lambda m: None, session=FakeNDL())
    monkeypatch.setenv("PRIORS_HOSTED", "1")
    with pytest.raises(SDL.SharadarDownloadError, match="hosted"):
        SDL.download_sharadar("goodkey", log=lambda m: None, session=FakeNDL())


def test_progress_survives_reopening(make_session):
    s = to_proposal(make_session(required_stage="tested"))
    s.check_gate()
    s.register()
    s.run(placebo_draws=50)
    again = Session(s.registry.root, s.lib, FakeLLM(responder), state_data=STATE, clock=CLOCK, required_stage="tested")
    assert again.results is None                       # results are recomputed, not stored
    steps = again.progress()["steps"]
    assert steps["Theory gate"] and steps["Tested"] and again.progress()["status"] == "complete"


def test_sharadar_rate_limit_stops_without_retrying(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIORS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PRIORS_HOSTED", raising=False)

    class Limited:
        calls = 0

        def get(self, url, params=None, timeout=None):
            Limited.calls += 1
            return FakeResponse(status=429, json_data={"quandl_error": {"message": "speed limit"}})
    with pytest.raises(SDL.SharadarDownloadError, match="rate limit"):
        SDL.download_sharadar("goodkey", log=lambda m: None, session=Limited())
    assert Limited.calls == 1

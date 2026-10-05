"""Priors: the Streamlit app (ARCHITECTURE §3, §13).

Start screen: connect (class code or own API key), name the project, pick a mode.
Then three tabs: Work (the mode's flow), Results, and Report.
Run with ``priors app``.
"""

from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from priors import __version__
from priors.app.theme import CSS
from priors.classroom import BudgetedLLM, BudgetExceeded, NotEnrolled, classroom_from_env, classroom_llm
from priors.factors import FactorLibrary
from priors.data.sharadar_download import SharadarDownloadError, download_sharadar
from priors.flows import FlowError, Session
from priors.flows.projectfile import ProjectFileError, export_project, import_project, read_manifest
from priors.flows.agents import ClarifiedIdea
from priors.literature.expected import composite_prior
from priors.llm import LLMRefusal
from priors.llm.access import classroom_token, make_llm
from priors.registry import GateFailed, HoldoutError, Hypothesis
from priors.reports import EXPLORATORY_LABEL, build_reports
from priors.reports import charts
from priors.spec import StrategySpec
from priors.spec.strategy import MODE_COMBINATIONS
from priors.stockdata import SOURCES, available_sources, ensure_snapshot, load_source
from priors.stocklayer import default_signal

MODES = {
    "mentor": ("Mentor", "For a first idea. Mentor asks questions, finds the research, and builds a testable "
                         "strategy with you. Plain language."),
    "analyst": ("Analyst", "For a strategy you can already describe. Pick factors and settings, run the tests, "
                           "and get a diagnosis with suggestions."),
    "dr_dong": ("Dr. Dong", "For a finished strategy. Two AI reviewers debate it, and Dr. Dong writes a strict, "
                            "journal-style referee report."),
}
MODE_LEVEL = {"mentor": "Guided · first idea", "analyst": "Hands-on · your own design", "dr_dong": "Referee review · strict"}
WORKFLOW = [("Literature", "An evidence card from published research"),
            ("Theory gate", "A mechanism in your own words"),
            ("Pre-registration", "The plan is locked before any test"),
            ("Tests", "Benchmarks, placebo, sealed holdout"),
            ("Report", "Main report, appendix, referee review")]
PRIVACY = ("Your messages are sent to an AI provider (Anthropic). Do not enter personal information. "
           "Priors is for education; nothing here is investment advice.")
REPO = "https://github.com/FengDong-Elon/priors"
MANUAL = f"{REPO}/blob/main/USER_MANUAL.md"
DOI = "https://doi.org/10.5281/zenodo.23149420"
load_dotenv()


def _secrets_to_env() -> None:
    """On a hosted deployment, secrets (instructor key, class configuration) arrive via st.secrets."""
    try:
        for k in ("ANTHROPIC_API_KEY", "PRIORS_CLASSROOM", "PRIORS_PROXY_URL"):
            if k in st.secrets and not os.environ.get(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:  # no secrets file: local use
        pass


_secrets_to_env()
CLASSES = classroom_from_env()
PROXY_URL = os.environ.get("PRIORS_PROXY_URL", "")
# On a personal computer only: lets the owner use the key in .env without pasting it.
LOCAL_KEY = os.environ.get("ANTHROPIC_API_KEY") if os.environ.get("PRIORS_ALLOW_LOCAL_KEY") == "1" else None
HOSTED = os.environ.get("PRIORS_HOSTED") == "1"   # set on a hosted deployment: no local-only features

st.set_page_config(page_title="Priors · Evidence-based factor investing", page_icon=":material/menu_book:",
                   layout="wide", menu_items={"Get help": MANUAL, "Report a bug": f"{REPO}/issues",
                                              "About": f"**Priors {__version__}**: a literature-first teaching tool "
                                                       f"for evidence-based factor investing. [Source]({REPO}) · "
                                                       f"[Cite]({DOI})"})
st.html(CSS)


def html(markup: str, sidebar: bool = False) -> None:
    (st.sidebar if sidebar else st).markdown(markup, unsafe_allow_html=True)


def masthead(kicker: str, title: str, sub: str = "") -> None:
    html(f'<div class="pr-mast"><div class="pr-kicker">{escape(kicker)}</div><div class="pr-title">{escape(title)}'
         f'</div>{f"<div class=pr-sub>{escape(sub)}</div>" if sub else ""}</div>')


def section(title: str) -> None:
    html(f'<div class="pr-section">{escape(title)}</div>')


def steps(items: list[tuple[str, str]], done: list[bool] | None = None) -> None:
    """A row of numbered steps; with ``done``, finished steps are shaded and the next one is underlined."""
    now = next((i for i, d in enumerate(done or []) if not d), None) if done is not None else None
    cells = []
    for i, (name, desc) in enumerate(items):
        cls = "pr-step" + (" done" if done and done[i] else "") + (" now" if i == now else "")
        mark = "Done" if done and done[i] else ("Next" if i == now else f"Step {i + 1}")
        cells.append(f'<div class="{cls}"><span class="n">{mark}</span><b>{escape(name)}</b>{escape(desc)}</div>')
    html(f'<div class="pr-steps">{"".join(cells)}</div>')


# ----------------------------------------------------------------- resources

@st.cache_resource(show_spinner="Loading published factor data...")
def library() -> FactorLibrary:
    return FactorLibrary.load()


AUTO_REFRESH = os.environ.get("PRIORS_AUTO_REFRESH", "1") == "1"   # build a prices-only snapshot when stale


@st.cache_resource(show_spinner="Preparing stock data (the first visit of the day can take a few minutes)...",
                   ttl=6 * 3600)
def snapshot(source: str, universe: str):
    if source == "yfinance" and AUTO_REFRESH:
        snap, status = ensure_snapshot(universe, log=lambda m: None)
        if snap is not None:
            snap.meta["status"] = status
        return snap
    return load_source(source, universe)


def projects_root() -> Path:  # noqa: E302
    return Path(os.environ.get("PRIORS_HOME", Path.home() / ".priors")) / "projects"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "default"


def state():
    return st.session_state


def session() -> Session | None:
    return state().get("session")


def guarded(label: str, fn, *args, **kwargs):
    """Run a step with a spinner and turn expected errors into messages for the student."""
    try:
        with st.spinner(label):
            return fn(*args, **kwargs)
    except GateFailed as e:
        st.error(e.result.explain())
    except (FlowError, HoldoutError, ValueError) as e:
        st.error(str(e))
    except LLMRefusal as e:
        st.error(f"The AI declined this request: {e}")
    except (BudgetExceeded, NotEnrolled, SharadarDownloadError, ProjectFileError) as e:
        st.error(str(e))
    except Exception as e:  # network and API errors
        st.error(f"Something went wrong: {type(e).__name__}: {e}")
    return None


# ------------------------------------------------------------------- sidebar

def sidebar() -> None:
    with st.sidebar:
        html('<div class="pr-brand">Priors</div>'
             '<div class="pr-brand-sub">Start from the literature, not from the chart.</div>')
        st.divider()
        html('<div class="pr-label">Connection</div>')
        if "llm" not in state():
            options = (["Class code"] if (CLASSES or PROXY_URL) else []) + ["My own API key"] + (["This computer's key"] if LOCAL_KEY else [])
            how = st.radio("Connect", options, label_visibility="collapsed")
            if how == "Class code":
                code = st.text_input("Class code")
                sid = st.text_input("Student id")
                if st.button("Connect", type="primary", width="stretch"):
                    try:
                        if CLASSES:
                            state()["llm"] = classroom_llm(CLASSES, code, sid, projects_root().parent / "classroom_usage.sqlite")
                        else:
                            classroom_token(code, sid)
                            state()["llm"] = make_llm(proxy_url=PROXY_URL, class_code=code, student_id=sid)
                        state()["who"] = (code, sid.strip().lower())
                        st.rerun()
                    except (ValueError, NotEnrolled, RuntimeError) as e:
                        st.error(str(e))
            elif how == "This computer's key":
                st.caption("Uses the API key saved in this computer's .env file.")
                if st.button("Connect", type="primary", width="stretch"):
                    state()["llm"] = make_llm(api_key=LOCAL_KEY)
                    state()["who"] = ("local key", "")
                    st.rerun()
            else:
                key = st.text_input("Anthropic API key", type="password", help="Kept in this browser session only.")
                if st.button("Connect", type="primary", width="stretch") and key:
                    state()["llm"] = make_llm(api_key=key)
                    state()["who"] = ("own key", "")
                    st.rerun()
        else:
            who = state().get("who", ("", ""))
            html(f'<div class="pr-status"><span class="pr-dot"></span>Connected · {escape(who[0])}'
                 f'{" / " + escape(who[1]) if who[1] else ""}</div>')
            llm = state()["llm"]
            if isinstance(llm, BudgetedLLM):
                st.caption(f"This week: ${llm.spent():.2f} of ${llm.student.config.weekly_budget_usd:.2f} used")
            elif who[0] not in ("own key", "local key") and PROXY_URL:
                try:
                    import httpx  # only needed with a separate proxy service

                    u = httpx.get(PROXY_URL.rstrip("/") + "/usage", timeout=10,
                                  headers={"x-api-key": classroom_token(*who)}).json()
                    st.caption(f"This week: ${u['spent_usd']:.2f} of ${u['budget_usd']:.2f} used")
                except Exception:
                    st.caption("Budget: unavailable")
        s = session()
        if s is not None:
            st.divider()
            html('<div class="pr-label">Project</div>')
            led = s.registry.ledger.summary()
            snap = state().get("snapshot_meta")
            rows = {"Name": state().get("project_name") or "", "Mode": MODES[s.mode][0],
                    "Status": s.progress()["status"].capitalize(),
                    "Specifications tested": f"{led['trials']} ({led['exploratory_trials']} exploratory)",
                    "Holdouts opened": str(led["holdouts_opened"]),
                    "Stock data": (f"{snap.get('source', 'yfinance')}, {snap['as_of'][:10]}" if snap else "not available")}
            html('<table class="pr-kv">' + "".join(f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>"
                                                    for k, v in rows.items()) + "</table>")
            st.write("")
            st.download_button("Download project file", export_project(s.registry.root, state().get("student", ""),
                                                                       state().get("project_name", "")),
                               f"{slug(state().get('project_name', 'project'))}.priors.zip", "application/zip",
                               help="Save your work. Upload it later on the start screen to continue.",
                               icon=":material/download:", width="stretch")
            others = [m for m in MODES if m != s.mode]
            c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
            target = c1.selectbox("Switch mode", others, format_func=lambda m: MODES[m][0])
            if c2.button("Switch", width="stretch"):
                s.upgrade(target)
                s.save_state()
                st.rerun()
            if st.button("Close project", icon=":material/close:", width="stretch"):
                state().pop("session", None)
                st.rerun()
        st.divider()
        html(f'<div class="pr-foot"><a href="{MANUAL}" target="_blank">User manual</a> · '
             f'<a href="{REPO}" target="_blank">Source code</a> · <a href="{DOI}" target="_blank">Cite</a><br>'
             f'Priors {__version__} · MIT license<br><br>{escape(PRIVACY)}</div>')


# --------------------------------------------------------------- start screen

def start_screen() -> None:
    masthead("Evidence-based factor investing · a teaching tool", "Priors",
             "Every idea starts from what the research says, is tested without data mining, and is reviewed the way "
             "a journal would review it.")
    steps(WORKFLOW)
    if "llm" not in state():
        st.info("Connect in the sidebar first: use your class code, or your own Anthropic API key.",
                icon=":material/login:")
        section("Three ways to work")
        mode_cards(clickable=False)
        return
    section("Your project")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        student = c1.text_input("Your name or id", value=state().get("who", ("", ""))[1])
        project = c2.text_input("Project name", value="my first strategy")
        sources = available_sources()
        c3, c4 = st.columns(2)
        universe = c3.selectbox("Stock universe", ["russell3000", "russell1000"],
                                format_func=lambda u: "Top 3,000 U.S. stocks" if u == "russell3000" else "Top 1,000 U.S. stocks")
        source = c4.selectbox("Stock data", sources, format_func=lambda x: SOURCES[x], disabled=len(sources) == 1)
    if not HOSTED and "sharadar" not in sources:
        with st.expander("Use your own Sharadar data (downloads to this computer)"):
            st.caption("Enter your own Nasdaq Data Link API key with a Sharadar subscription. The data is saved on "
                       "this computer only; the key is used for the download and is not stored. The download can take "
                       "a while.")
            key = st.text_input("Nasdaq Data Link API key", type="password", key="ndl_key")
            if st.button("Download Sharadar") and key:
                box = st.empty()
                done = guarded("Downloading Sharadar...", download_sharadar, key, log=lambda m: box.caption(m))
                if done:
                    st.success("Sharadar is ready. Choose it under Stock data.")
                    st.rerun()
    with st.expander("Continue a saved project"):
        up = st.file_uploader("Project file (.priors.zip)", type=["zip"])
        if up is not None and st.button("Open the saved project"):
            data = up.getvalue()
            try:
                man = read_manifest(data)
                dest = projects_root() / slug(student or man.get("student") or "student") / slug(man.get("project") or project)
                import_project(data, dest, overwrite=True)
                open_project(student or man.get("student", ""), man.get("project") or project, "mentor", universe, source,
                             keep_mode=True)
            except ProjectFileError as e:
                st.error(str(e))
    section("Choose a mode")
    chosen = mode_cards(clickable=True)
    if chosen:
        open_project(student, project, chosen, universe, source)


def mode_cards(clickable: bool) -> str | None:
    """The three modes side by side; returns the mode whose Start button was pressed."""
    chosen = None
    for col, (mode, (name, desc)) in zip(st.columns(3), MODES.items()):
        with col.container(border=True):
            html(f'<div class="pr-kicker">{escape(MODE_LEVEL[mode])}</div>')
            st.markdown(f"### {name}")
            st.write(desc)
            if clickable and st.button(f"Start in {name}", key=f"start_{mode}", width="stretch",
                                       type="primary" if mode == "mentor" else "secondary"):
                chosen = mode
    return chosen


def open_project(student: str, project: str, mode: str, universe: str, source: str = "yfinance",
                 keep_mode: bool = False) -> None:
    lib = library()
    snap = snapshot(source, universe)
    root = projects_root() / slug(student or "student") / slug(project)
    llm = state()["llm"]
    cfg = llm.student.config if isinstance(llm, BudgetedLLM) else None
    s = Session(root, lib, llm, mode=mode, stock=snap.context if snap else None,
                allow_exploratory=cfg.allow_exploratory if cfg else True,
                required_stage=cfg.required_stage if cfg else "dr_dong")
    if not keep_mode:
        s.mode = mode                   # the mode chosen now wins over the one saved last time
    state().update(session=s, project_name=project, student=student,
                   snapshot_meta=snap.meta if snap else None)
    if snap is None:
        state()["data_notice"] = "Stock data is not available right now, so the stock layer is skipped. Everything else works."
    elif snap.meta.get("status") not in (None, "current", "refreshed"):
        state()["data_notice"] = snap.meta["status"]
    st.rerun()


# ----------------------------------------------------------------- work tab

def show_card(s: Session) -> None:
    c = s.card
    with st.expander(f"Evidence card: {c.claim}", expanded=s.spec is None):
        st.markdown(f"**Evidence grade:** {c.grade}. {c.grade_justification}")
        for f in c.key_papers:
            st.markdown(f"- **{f.citation.short()}**, *{f.citation.title}*. {f.finding}")
        if c.contrary_evidence:
            st.markdown("**Contrary evidence**")
            for f in c.contrary_evidence:
                st.markdown(f"- **{f.citation.short()}**, *{f.citation.title}*. {f.finding}")
        for b in c.boundary_conditions:
            st.markdown(f"- Boundary: {b}")


def show_spec(s: Session) -> None:
    sp = s.spec
    section(sp.name)
    st.dataframe(pd.DataFrame([{"factor": c.factor, "name": s.lib.info.loc[c.factor, "name"] if c.factor in s.lib.info.index else "",
                                "theory priority": c.theory_priority, "stock signal": c.stock_signal or "factor layer only"}
                               for c in sp.components]), hide_index=True, width="stretch")
    st.caption(f"Combination: {sp.factor_layer.combination.replace('_', ' ')} · mechanism tests: "
               f"{', '.join(sp.mechanism_tests)} · stock layer: top {sp.stock_layer.holdings}, "
               f"{sp.stock_layer.rebalance}, {sp.stock_layer.cost_bps_one_way:.0f} bp")


def gate_and_register(s: Session) -> None:
    h = s.hypothesis
    st.markdown(f"**Hypothesis:** {h.statement}")
    st.markdown(f"**Expected return (from the literature):** {h.expected_low:.1%} to {h.expected_high:.1%} per year")
    text = st.text_area("In your own words: why should this premium exist?", value=h.mechanism_explanation, height=110)
    if text != h.mechanism_explanation:
        s.set_explanation(text)
        s.save_state()
    c1, c2 = st.columns(2)
    if c1.button("Check the theory gate", type="primary"):
        guarded("Checking...", s.check_gate)
    if s.gate is not None:
        if s.gate.passed:
            st.success("The hypothesis passes the theory gate.")
        else:
            st.warning(s.gate.explain())
        if s.gate.advisory:
            st.info(s.gate.advisory)
        if s.gate.passed and c2.button("Pre-register (this locks the plan)"):
            guarded("Registering...", s.register)
            st.rerun()
    revise_panel(s)
    if s.allow_exploratory:
        st.caption("You can also test without registering, but results will be labeled exploratory and kept out of "
                   "the main report.")
        if st.button("Explore without registering"):
            run_tests(s)
    else:
        st.caption("In this class, the theory gate and pre-registration come before any test.")


def revise_panel(s: Session) -> None:
    with st.expander("Revise with the Mentor (free before any test)", expanded=bool(s.gate and not s.gate.passed)):
        for t in s.coaching:
            st.chat_message("user" if t["role"] == "student" else "assistant").write(t["text"])
        q = st.text_area("Ask the Mentor, or describe the change you are considering", key="coach_q")
        if st.button("Ask the Mentor") and q.strip():
            guarded("Thinking...", s.coach, q)
            st.rerun()
        refined = st.text_input("A refined idea to search the literature again", key="refined_idea")
        c1, c2, c3 = st.columns(3)
        if c1.button("Search the literature again") and refined.strip():
            guarded("Searching the literature again...", s.research_again, refined)
            st.rerun()
        if c2.button("Propose a different strategy"):
            guarded("Designing a strategy...", s.propose)
            st.rerun()
        if c3.button("Conclude: not supported"):
            guarded("Writing the literature assessment...", s.conclude_unsupported)
            st.rerun()


def revision_after_results(s: Session) -> None:
    with st.expander("Revise the strategy (counts as a new trial)"):
        st.write(Session.REVISION_NOTICE)
        ok = st.checkbox("I understand that a revision is a new trial.", key="rev_ok")
        if ok and st.button("Start a revision"):
            guarded("Starting a revision...", s.start_revision)
            st.rerun()


def show_unsupported(s: Session) -> None:
    u = s.unsupported
    st.subheader("Conclusion: not supported by the literature")
    st.markdown(u.summary)
    for title, items in (("Why it was not tested", u.why_not_supported), ("What would change this", u.evidence_needed),
                         ("Better-supported directions", u.alternative_directions)):
        st.markdown(f"**{title}**\n" + "\n".join(f"- {x}" for x in items))
    st.caption("This is a complete result. Download the report on the Report tab.")


def run_tests(s: Session, period: str = "in_sample") -> None:
    res = guarded("Running the tests (about half a minute)...", s.run, period=period)
    if res is not None:
        guarded("Writing the explanation...", s.explain)
        st.rerun()


def mentor_work(s: Session) -> None:
    for t in s.transcript:
        st.chat_message("user" if t["role"] == "student" else "assistant").write(t["text"])
    if s.unsupported is not None:
        show_unsupported(s)
        return
    if s.idea is None:
        if not s.transcript:
            st.chat_message("assistant").write("Hi! What investment idea would you like to explore? "
                                               "Describe it in a sentence or two.")
        msg = st.chat_input("Your idea or answer")
        if msg:
            guarded("Thinking...", s.interview, msg)
            st.rerun()
        return
    st.success(f"Idea: {s.idea.summary}")
    if s.card is None:
        if st.button("Search the research", type="primary"):
            guarded("Searching the literature and reading abstracts (about half a minute)...", s.research)
            st.rerun()
        return
    show_card(s)
    if s.spec is None:
        c1, c2 = st.columns(2)
        if c1.button("Propose a strategy", type="primary"):
            guarded("Designing a strategy from published factors...", s.propose)
            st.rerun()
        if c2.button("The research does not support it: conclude"):
            guarded("Writing the literature assessment...", s.conclude_unsupported)
            st.rerun()
        return
    show_spec(s)
    if s.registration is None and s.results is None:
        gate_and_register(s)
        return
    if s.registration is not None:
        st.success(f"Registered as {s.registration.id}. Holdout sealed from {s.registration.holdout_start}.")
    if s.results is None:
        if st.button("Run the tests", type="primary"):
            run_tests(s)
        return
    show_explanation(s)
    revision_after_results(s)


def show_explanation(s: Session) -> None:
    if s.results.outcome.status != "preregistered":
        st.warning(EXPLORATORY_LABEL)
    if s.explanation:
        st.markdown(s.explanation)
    with st.expander("Required risk notes"):
        for n in s.results.dossier.risk_notes:
            st.markdown(f"- {n}")


def spec_builder(s: Session) -> None:
    lib = s.lib
    info = lib.info
    ids = list(info.index)
    current = s.spec.factor_ids if s.spec else []
    with st.form("spec"):
        st.markdown("**Factors** (published factor portfolios)")
        chosen = st.multiselect("Factors", ids, default=[f for f in current if f in ids],
                                format_func=lambda f: f"{f}: {info.loc[f, 'name']}", max_selections=8)
        combos = MODE_COMBINATIONS[s.mode]
        combo = st.selectbox("Combination", combos, format_func=lambda c: c.replace("_", " "))
        c1, c2, c3 = st.columns(3)
        holdings = c1.slider("Stocks to hold", 10, 100, s.spec.stock_layer.holdings if s.spec else 40)
        cost = c2.slider("Trading cost (bp, one-way)", 0, 50, int(s.spec.stock_layer.cost_bps_one_way) if s.spec else 20)
        rebalance = c3.selectbox("Rebalance", ["monthly", "quarterly"])
        f1, f2, f3 = st.columns(3)
        micro = f1.checkbox("Exclude microcaps", True)
        price = f2.checkbox("Price at least $5", True)
        adv = f3.checkbox("Daily volume at least $1M", True)
        tests = st.multiselect("Mechanism tests", ["publication_decay", "sentiment", "risk_regime"],
                               default=s.spec.mechanism_tests if s.spec else ["publication_decay"])
        name = st.text_input("Strategy name", value=s.spec.name if s.spec else "My strategy")
        submitted = st.form_submit_button("Save strategy", type="primary")
    if submitted:
        if not chosen:
            st.error("Pick at least one factor.")
            return
        try:
            spec = StrategySpec.model_validate({
                "name": name,
                "components": [{"factor": f, "stock_signal": default_signal(f), "theory_priority": i + 1,
                                "weight": (round(1 / len(chosen), 6) if combo == "theory_weights" else None)}
                               for i, f in enumerate(chosen)],
                "factor_layer": {"combination": combo},
                "stock_layer": {"holdings": holdings, "cost_bps_one_way": cost, "rebalance": rebalance,
                                "filters": {"exclude_microcaps": micro, "min_price": 5.0 if price else None,
                                            "min_adv_usd": 1_000_000 if adv else None}},
                "mechanism_tests": list(dict.fromkeys(["publication_decay", *tests])),
            })
            if combo == "theory_weights":
                total = sum(c.weight for c in spec.components)
                spec.components[-1].weight = round(spec.components[-1].weight + 1 - total, 6)
            s.load_spec(spec)
            s.results = None
            s.save_state()
            st.rerun()
        except Exception as e:
            st.error(str(e))


def register_from_analyst(s: Session) -> None:
    with st.expander("Pre-register this strategy (recommended)"):
        stmt = st.text_area("Hypothesis (what should happen, and where)", key="h_stmt")
        mech = st.selectbox("Mechanism", ["behavioral_bias", "risk_compensation", "limits_to_arbitrage", "market_friction"],
                            format_func=lambda m: m.replace("_", " "))
        why = st.text_area("Why should the premium exist? (your own words)", key="h_why")
        if st.button("Build the evidence card and check the gate"):
            s.set_idea(ClarifiedIdea(summary=stmt, signal=", ".join(s.spec.factor_ids), direction="as the factors are signed",
                                     horizon="monthly", student_reason=why))
            card = guarded("Searching the literature...", s.research)
            if card is not None:
                low, high, _ = composite_prior(s.spec, s.lib)
                s.hypothesis = Hypothesis(statement=stmt, mechanism=mech, mechanism_explanation=why,
                                          citations=[f.citation for f in card.key_papers],
                                          expected_low=round(max(low, 0), 4), expected_high=round(max(high, low + 0.005), 4),
                                          evidence_card_ids=[card.id])
                s.save_state()
                guarded("Checking...", s.check_gate)
        if s.gate is not None and s.hypothesis is not None:
            if s.gate.passed:
                st.success("Passes the theory gate.")
            else:
                st.warning(s.gate.explain())
            if s.gate.passed and s.registration is None and st.button("Pre-register"):
                guarded("Registering...", s.register)
                st.rerun()


def analyst_or_dr_dong_work(s: Session) -> None:
    if s.card is not None:
        show_card(s)
    if s.spec is None or st.toggle("Edit the strategy", value=s.spec is None):
        spec_builder(s)
    if s.spec is None:
        return
    show_spec(s)
    if s.registration is None:
        register_from_analyst(s)
    else:
        st.success(f"Registered as {s.registration.id}. Holdout sealed from {s.registration.holdout_start}.")
    if s.registration is None and not s.allow_exploratory:
        st.info("In this class, pre-register the strategy (above) before testing it.")
    elif st.button("Run the tests", type="primary"):
        run_tests(s)
    if s.results is None:
        return
    show_explanation(s)
    revision_after_results(s)
    if s.mode == "analyst":
        if st.button("Ask the Performance Analyst"):
            guarded("Analyzing...", s.analyze)
        if s.analyst_review:
            r = s.analyst_review
            st.markdown("**Diagnosis**\n" + "\n".join(f"- {x}" for x in r.diagnosis))
            for x in r.suggestions:
                st.markdown(f"- **{x.suggestion}**: {x.rationale}{' *(a new trial: only the holdout can judge it)*' if x.new_trial else ''}")
    else:
        if st.button("Request Dr. Dong's review (about four minutes)", type="primary"):
            guarded("Advocate and Reviewer 2 are debating; Dr. Dong is writing the report...", s.review)
            st.rerun()
        if s.dr_dong:
            rep = s.dr_dong.report
            st.subheader(f"Dr. Dong: {rep.recommendation.replace('_', ' ').upper()}")
            st.markdown(rep.summary)
            for title, items in (("Major comments", rep.major_comments), ("Minor comments", rep.minor_comments),
                                 ("What would change my mind", rep.what_would_change_my_mind)):
                if items:
                    st.markdown(f"**{title}**\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)))
            st.caption(s.dr_dong.footer)
    if s.registration is not None:
        with st.expander("Open the sealed holdout (only once per hypothesis)"):
            ok = st.checkbox("I understand this can be done only once and the result is final.")
            if ok and st.button("Open the holdout"):
                run_tests(s, period="holdout")


# -------------------------------------------------------------- results tab

def results_tab(s: Session) -> None:
    r = s.results
    if r is None:
        st.info("No test results: this idea was concluded without a backtest." if s.unsupported
                else "Run the tests to see results.")
        return
    res = r.outcome.result
    if r.outcome.status != "preregistered":
        st.warning(EXPLORATORY_LABEL)
    m = res.metrics["composite"]
    c = st.columns(4)
    c[0].metric("Mean return per year", f"{m['mean_ann']:.1%}", border=True)
    c[1].metric("Sharpe ratio", f"{m['sharpe']:.2f}", help=f"95% CI {m['sharpe_ci_low']:.2f} to {m['sharpe_ci_high']:.2f}", border=True)
    c[2].metric("t-statistic (NW)", f"{m['t_stat_mean_nw']:.2f}", help="Harvey, Liu and Zhu (2016) suggest t > 3.", border=True)
    c[3].metric("Max drawdown", f"{m['max_drawdown']:.0%}", border=True)
    st.caption(f"{res.sample['start']} to {res.sample['end']} · before trading costs · {res.period.replace('_', ' ')}")
    st.image(charts.cumulative_growth(res.returns), width="stretch")
    for name, lines in r.dossier.sections.items():
        if name in ("Strategy", "Factor layer"):
            continue
        with st.expander(name):
            for line in lines:
                st.markdown(f"- {line}")
    if r.composite is not None:
        section("What each factor contributes")
        st.dataframe(pd.DataFrame({"Sharpe share": r.composite.shapley_share, "Risk share": r.composite.risk_contribution,
                                   "Removal effect": r.composite.leave_one_out["change_if_removed"]}).style.format("{:.2f}"))
    if r.holdings is not None:
        section(f"Current holdings, as of {r.holdings.as_of}")
        st.caption(r.holdings.label)
        st.dataframe(r.holdings.table.drop(columns=["mcap"]), width="stretch")
    for n in r.notes:
        st.info(n)


def report_tab(s: Session) -> None:
    if s.results is None and s.unsupported is None:
        st.info("Run the tests first, or conclude that the idea is not supported.")
        return
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("### Main report")
        st.write("A short report to hand in: the hypothesis, the results, the credibility checks, and the verdict.")
    with c2.container(border=True):
        st.markdown("### Technical appendix")
        st.write("Every exhibit, the full evidence card, the full referee report, and the conversation.")
    if st.button("Build the PDF reports", type="primary", icon=":material/picture_as_pdf:"):
        built = guarded("Building...", build_reports, s, student=state().get("student"))
        if built is None:
            return
        main, app = built
        tmp = s.registry.root / "reports"
        tmp.mkdir(exist_ok=True)
        state()["pdf_main"] = main.save_pdf(tmp / "main.pdf").read_bytes()
        state()["pdf_app"] = app.save_pdf(tmp / "appendix.pdf").read_bytes()
    if "pdf_main" in state():
        c1, c2 = st.columns(2)
        c1.download_button("Download the main report (PDF)", state()["pdf_main"], "priors_report.pdf", "application/pdf",
                           icon=":material/download:", width="stretch")
        c2.download_button("Download the technical appendix (PDF)", state()["pdf_app"], "priors_appendix.pdf",
                           "application/pdf", icon=":material/download:", width="stretch")


# ---------------------------------------------------------------------- main

def main() -> None:
    sidebar()
    s = session()
    if s is None:
        start_screen()
        return
    if "llm" in state():
        s.llm = state()["llm"]
    masthead(f"{MODES[s.mode][0]} mode · {MODE_LEVEL[s.mode]}", state().get("project_name") or "Project")
    prog = s.progress()
    steps([(name, "") for name in prog["steps"]], done=list(prog["steps"].values()))
    if state().get("data_notice"):
        st.info(state()["data_notice"], icon=":material/info:")
    work, results, report = st.tabs([":material/edit_note: Work", ":material/monitoring: Results",
                                     ":material/description: Report"])
    with work:
        if s.mode == "mentor":
            mentor_work(s)
        else:
            analyst_or_dr_dong_work(s)
    with results:
        results_tab(s)
    with report:
        report_tab(s)


main()

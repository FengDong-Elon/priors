# Priors — Architecture (v0.1 draft)

> *Start from the literature, not from the chart.*
>
> Most AI investing tools try to convince you a strategy works. Priors starts from what the research already says, and then tries to prove your strategy wrong.

Status: design draft, 2026-09-25. No code yet.

---

## 1. Positioning

Priors is an **AI research assistant for systematic investing**. It helps investors with limited programming skills turn an investment idea into a factor or strategy, then backtest it and audit it using the methods of academic finance.

It is an **AI researcher, not an AI trader**.

| | Typical AI-trading projects (TradingAgents, ai-hedge-fund) | Priors |
|---|---|---|
| Output | A buy/sell/hold decision on a ticker for a given day | A transparent, reusable strategy spec |
| Where the LLM reasons | Every trading day, for every ticker | Once, at design time and at audit time |
| Where performance numbers come from | An LLM-in-the-loop simulation | A deterministic backtest engine |
| Evidence base | News, sentiment, the model's own knowledge | Retrieved academic literature, with replication priors |
| Horizon tested | Months, a handful of tickers | 15+ years, the full cross-section |
| Stance | Advocate | Skeptic (falsification-first) |

**Out of scope:** single-stock recommendations. Priors can show a strategy's current holdings as the mechanical output of its rules, and explain why a stock was selected (its factor exposures). It never answers "should I buy X?" Instead, it redirects the user: "what characteristic of X do you like? Let's turn that into a factor and test it."

---

## 2. Design principles

1. **The LLM never produces a performance number.** Every return, Sharpe ratio, drawdown, t-stat, and p-value comes from deterministic code. LLMs design, explain, and critique.
2. **Citations only come from retrieved sources.** A paper can be cited only if the literature module actually retrieved it in the current session (DOI or stable URL attached). Citing from model memory is blocked.
3. **Every trial is recorded.** Every spec version and every backtest run is written to an append-only trial ledger. Reported significance is adjusted for the number of trials.
4. **Pre-register, then test.** The hypothesis, the expected sign and magnitude, and the chosen mechanism tests are locked before the first backtest.
5. **One analysis, three explanation depths.** The underlying analysis is identical in every mode. Only the explanation layer changes. Simplifying must never drop a risk warning.
6. **Bring your own data.** Code is open; data is not. Every data adapter declares its quality limitations, and the Referee reads them.
7. **Reproducible by default.** Every result can be regenerated from its spec, the data snapshot identifier, the engine version, and the random seed.
8. **English only.** All code, UI, prompts, and documentation are in English.

---

## 3. The three modes

The three modes are three entry points into one pipeline. They share the same engine and artifacts.

```
 Idea ─► Literature ─► Evidence ─► Hypothesis ─► Spec ─► Backtest ─► Validation ─► Referee report
 └──────────── Mentor ──────────────────────────┘
                                     └─────── Analyst ─────────┘
                                                      └────────── Referee ─────────────┘
```

| Mode | User | Starts from | Ends with | Explanation depth |
|---|---|---|---|---|
| **Mentor** | Beginner, or anyone with a rough idea | A sentence ("cheap stocks that are going up") | A pre-registered hypothesis, a spec, and a first backtest | Plain language, analogies, no unexplained jargon |
| **Analyst** | Investor with a working strategy | A spec, rules in plain text, or uploaded code | Backtest, performance analysis, robustness checks, concrete suggestions | Intermediate: standard terms, briefly explained |
| **Referee** | Professional | A spec plus results (from Priors or uploaded) | A journal-style referee report | Technical and precise, with statistical detail |

Users can switch mode at any time. Each reply also offers "Explain simpler" and "More technical" buttons, because a user's skill level and their chosen mode do not always match.

---

## 4. Agents

Models run in two tiers. Both are configurable, and other providers can be swapped in.

- **Fast tier**: high-volume, mechanical work. Default: Claude Haiku 4.5.
- **Deep tier**: judgment-heavy work, with extended thinking. Default: Claude Opus 5.5.

| Agent | Tier | Modes | Input | Output |
|---|---|---|---|---|
| **Interviewer** | Deep | Mentor | The user's idea and answers | A clarified idea: which assets, which signal, what horizon, and why it should work |
| **Librarian** | Fast | All | Search queries | Candidate papers with metadata and abstracts, from retrieved sources only |
| **Evidence Synthesizer** | Deep | All | Candidate papers | Evidence cards (§5.2), including conflicts between papers and replication status |
| **Architect** | Deep | Mentor, Analyst | Evidence cards and the clarified idea | A strategy spec (§5.1), with a citation for each design choice |
| **Registrar** | — (code) | All | Hypothesis and spec | A locked pre-registration record (§5.3) |
| **Engineer** | Fast | Analyst | Rules in plain text or uploaded code | A strategy spec (translated), with the ambiguities flagged |
| **Engine** | — (code) | All | Spec and data | Backtest results (§6) |
| **Performance Analyst** | Deep | Analyst | Results, validation suite output | Diagnostics and ranked improvement suggestions |
| **Advocate** | Deep | Referee | Everything so far | The strongest honest case that the alpha is real |
| **Reviewer 2** | Deep | Referee | Everything so far, plus the Advocate's case | The strongest case that it is not |
| **Risk Officer** | Deep | Referee | Results, capacity, cost analysis | Implementation risks: costs, capacity, drawdowns, concentration |
| **Editor** | Deep | Referee | The debate transcript and all evidence | The final referee report (§5.5) and a recommendation |
| **Explainer** | Fast | All | Structured output of any agent, current mode | User-facing text at the right depth (§11) |

The **Referee debate** borrows TradingAgents' bull/bear structure, but the question is different. Instead of "buy this stock?", it asks "is this alpha real?" The Advocate and Reviewer 2 go back and forth for a configurable number of rounds (default 2). The Editor decides, weighing the arguments by the evidence each side cites, not by how confidently they are stated.

**Orchestration.** A plain Python state machine that calls the Anthropic SDK with tool use. The pipeline is mostly linear and deterministic, so a framework adds little. LangGraph remains an option if branching grows.

---

## 5. Core artifacts

Every artifact is a versioned, human-readable file (YAML/JSON plus a rendered view). Artifacts are what the user sees, edits, and exports.

### 5.1 Strategy spec

```yaml
spec_version: 1
name: "12-1 momentum, large caps"
hypothesis_id: H-0007            # links to the pre-registration record
universe:
  asset_class: us_equity
  filter: { min_price: 5, min_mcap_pctile_nyse: 20 }   # exclude microcaps
signal:
  type: formula
  expr: "ret(t-12, t-1)"         # cumulative return months t-12..t-1, skipping month t
  source: "Jegadeesh & Titman (1993), doi:10.1111/j.1540-6261.1993.tb04702.x"
portfolio:
  construction: quantile_sort
  quantiles: 10
  side: long_short               # long_only | long_short
  weighting: value               # equal | value | rank | inverse_vol
rebalance: monthly
execution:
  signal_lag_days: 1             # signal at close t, trade at close t+1
  cost_bps_one_way: 15
benchmark: us_market
sample: { start: 2000-01, end: holdout }   # the holdout period is sealed, see §7.6
```

### 5.2 Evidence card

One card per research line (for example, "time-series momentum" or "accruals anomaly").

- **Claim.** One sentence.
- **Mechanism(s) proposed.** Risk compensation, behavioral bias, limits to arbitrage, market friction, or data mining.
- **Key papers.** Each has a DOI, sample, market, effect size, and the specific table or figure reference.
- **Replication status.** From Chen & Zimmermann, Hou-Xue-Zhang (2020), and Jensen-Kelly-Pedersen (2023) where applicable.
- **Post-publication decay prior.** Default from McLean & Pontiff (2016): about −26% out of sample and about −58% after publication. This is overridden when anomaly-specific evidence exists.
- **Boundary conditions.** Microcaps only? Pre-2000 only? Gross of costs only?
- **Contrary evidence.** Papers that fail to find the effect, or that explain it away.
- **Evidence grade.** Strong / moderate / weak / contested, with a one-line justification.
- **Expected effect under realistic conditions.** A prior range that the backtest will be compared against.

### 5.3 Pre-registration record

Written before the first backtest. It is immutable, timestamped, and content-hashed.

- The hypothesis in words
- Expected sign and magnitude range, for example "long-short 3–6% per year"
- Primary metric, for example "net Sharpe over the in-sample period"
- The mechanism tests chosen (§9) and the factor models used for spanning (§8)
- The planned robustness checks

Deviations after registration are allowed, but they are logged and shown in every report as "exploratory, not pre-registered."

### 5.4 Trial ledger

An append-only log. Each row records: timestamp, spec hash, data snapshot identifier, engine version, headline metrics, and whether the run was pre-registered or exploratory. The number of trials feeds into the deflated Sharpe ratio (§7.2).

### 5.5 Referee report

Formatted like a journal referee report:

1. **Summary** of the strategy and its claimed edge
2. **Recommendation**: Reject / Major revision / Minor revision / Accept (for paper trading)
3. **Major comments**: problems that could invalidate the result
4. **Minor comments**: improvements and clarifications
5. **Statistical summary**, clearly separating results that were tested from results that are only descriptive
6. **What would change my mind**: the specific evidence that would reverse the recommendation

---

## 6. Backtest engine

A **vectorized, cross-sectional engine** written for factor strategies. It is not an event-driven trading simulator.

- **Operations.** Signal → rank or sort → weights → holdings → returns → costs.
- **Timing conventions.** Signal known at close t, traded at close t+1 (configurable, never zero). Fundamentals use the filing date when the data provider supports point-in-time data. When it doesn't, the engine applies a conservative lag and flags it.
- **Portfolio types.** Long-only, long-short, quantile sorts, rank weights, value or equal weights, inverse-volatility weights. For time-series strategies, the signal times exposure to a single asset or ETF basket.
- **Costs.** A proportional one-way cost in basis points, plus an optional spread proxy based on size. Turnover is always reported.
- **Outputs.** Gross and net returns; CAGR; volatility; Sharpe with a 95% CI; max drawdown and its duration; turnover; hit rate; exposures; returns by year and by subperiod.
- **Determinism.** Same spec, same data snapshot, same engine version, same seed → identical numbers.

---

## 7. Validation suite

Runs automatically after every backtest. The results feed Analyst and Referee.

### 7.1 Literature benchmark check
The backtest is compared with the evidence card's expected range.
- **Far above** → alarm. Suspect look-ahead bias, survivorship bias, data errors, or overfitting.
- **Far below** → check whether the implementation departs from the original design (breakpoints, weighting, lags, universe).
- **Within range** → proceed.

### 7.2 Multiple-testing adjustment
Deflated Sharpe ratio (Bailey & López de Prado, 2014), computed from the number of trials in the ledger. The Harvey-Liu-Zhu (2016) t > 3 threshold is shown as a reference.

### 7.3 Placebo test (random-strategy null)
1,000 placebo portfolios are built by **shuffling the mapping between stock identities and signal values**, holding the shuffle fixed over time. This keeps the signal's persistence, and therefore its turnover and number of holdings, while destroying any link to the actual stocks. The output is a histogram of placebo Sharpe ratios with the real strategy marked, plus an empirical p-value.

### 7.4 Cost sensitivity
Net Sharpe at 0, 5, 10, 15, 25, and 50 bp one-way, plus the breakeven cost.

### 7.5 Stability
Subperiods (by decade and by market regime), rolling 36-month Sharpe, and sensitivity to parameters within the pre-registered grid only.

### 7.6 Sealed holdout
The most recent N years, and optionally all data after the LLM's training cutoff, are sealed by default. The user can open the holdout **once per hypothesis**. Opening it is logged. Results after opening are labeled "holdout used."

### 7.7 Data-quality warnings
Derived from the data adapter's capability flags (§12). Example: "This data source excludes delisted firms. Returns are likely overstated."

---

## 8. Factor zoo positioning

The question here is "what is this strategy, really?"

- **Factor library.** Ken French (market, SMB, HML, RMW, CMA, UMD); Hou-Xue-Zhang q-factors; and the Chen & Zimmermann open-source long-short portfolio returns (200+ published anomalies).
- **Spanning regressions** against **pre-specified models only**: CAPM, FF3, FF5, FF5+UMD, and q5. These models are chosen at pre-registration. The output is the alpha that remains, the factor loadings, and R².
- **Nearest neighbors.** The strategy's return correlation with every factor in the library, and the top-k closest published anomalies. Because it is selected after the fact, this part is **labeled descriptive only**.
- **Factor map.** A 2-D embedding (hierarchical clustering or MDS on the correlation matrix) of the whole library, with the user's strategy plotted, so the user can see which "family" it belongs to.
- **Alignment rules.** Long-only strategies are compared as excess over the market, or as a long-short version built for the purpose. Weekly strategies are aggregated to monthly. Library coverage gaps at the end of the sample are shown explicitly.

## 9. Mechanism tests

The question here is "why does it work?"

The mechanism tests come from a **fixed template library**. The Evidence Synthesizer proposes which templates apply, the Registrar locks them in the pre-registration record, and the engine runs them. The LLM never designs a test on its own after the results are in.

| Mechanism | Testable prediction | Test | Data | v1? |
|---|---|---|---|---|
| Behavioral / limits to arbitrage | Stronger among small, illiquid, high-idiosyncratic-volatility stocks | Double sorts, interaction regression | Market cap, Amihud illiquidity, IVOL (all price-based) | Yes |
| Behavioral / sentiment | Stronger after periods of high investor sentiment (Stambaugh, Yu & Yuan 2012) | Conditional returns on the lagged Baker-Wurgler index | Baker-Wurgler (public) | Yes |
| Risk compensation | Loses in bad times; returns covary with the state of the economy | Returns in NBER recessions, market drawdowns, and VIX regimes; conditional beta | NBER, FRED, VIX (public) | Yes |
| Frictions | Disappears net of costs; concentrated in microcaps | Microcap exclusion, net-of-cost comparison | Market cap, cost model | Yes |
| Data mining | Decays after publication | Returns before vs. after the publication date | Publication date from the literature module | Yes |
| Behavioral / limits to arbitrage | Stronger where institutional ownership is low | Double sort on institutional ownership | Institutional holdings (for example, Sharadar SF3) | v2 |

Results are always reported as **"consistent with," "inconsistent with," or "inconclusive."** They are never reported as proof of a mechanism.

Order of operations: spanning (§8) comes first. Mechanism tests then run on the strategy **and** on the part of its return that the pre-registered factor model does not explain.

---

## 10. Failed Strategy Museum

A public, curated gallery of strategies that were falsified: the idea, the spec, the evidence card, the results, and the referee report explaining why they failed. Examples:
- A post-earnings drift strategy whose net Sharpe falls to about 0 at 15 bp
- A conditional reversal strategy with annual turnover in the thousands of percent

Users can submit their own falsified strategies. Negative results are treated as first-class knowledge.

---

## 11. Explanation layer

Agents return structured results (numbers, verdicts, flags). The Explainer renders them at the depth of the current mode, following the style guide for that mode.

| | Mentor | Analyst | Referee |
|---|---|---|---|
| Vocabulary | Everyday words; any unavoidable term gets a one-line gloss | Standard industry terms | Full technical vocabulary |
| Numbers | Rounded, concrete ("$100 would have fallen to about $65") | Metrics with brief context | Point estimates with CIs, test statistics, adjustments |
| Examples | Everyday analogies | Market episodes (2008, 2020) | Citations and specific tables |
| Statistics | "How sure we are," in words | CIs mentioned | Tested vs. descriptive clearly separated, multiple-testing adjusted |
| Required risk content | **Same in all modes**: large drawdowns, overfitting risk, data-quality warnings, decay expectations |||

---

## 12. Data layer

```
Engine / agents
      │  depend only on the DataProvider interface
      ▼
DataProvider (abstract)
  ├── FreeProvider      (default: Stooq/Tiingo prices, FRED, Ken French, Chen-Zimmermann)
  ├── SharadarProvider  (bring your own Nasdaq Data Link key)
  └── ... community adapters (FMP, EODHD, ...)
```

Each provider declares its capabilities:

```python
Capabilities(
    includes_delisted: bool,
    point_in_time_fundamentals: bool,
    fundamentals: bool,
    institutional_holdings: bool,
    history_start: date,
    universe: str,              # e.g. "us_equity", "us_etf"
)
```

**Licensing rules:**
- Adapter code is open source.
- Raw data, caches, and derived firm-level datasets are **never committed**. Data directories are listed in `.gitignore`, and keys live in `.env`.
- CI tests use synthetic data only.
- The free default must be able to run the whole pipeline end to end on ETFs and price-based signals without any key.

---

## 13. Tech stack and repository layout

- **Language:** Python 3.11+
- **LLM:** Anthropic SDK (a provider interface allows others)
- **Data and computation:** pandas, numpy, statsmodels, scipy; DuckDB or Parquet for local caches
- **UI:** Streamlit (v1). The core stays UI-agnostic so a web front end can be added later.
- **Packaging:** `pyproject.toml`; the PyPI distribution name is `priors-quant`, and the repo name is `priors`

```
priors/
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
├── .env.example
├── src/priors/
│   ├── agents/          # interviewer, librarian, synthesizer, architect, referee debate, explainer
│   ├── prompts/         # per-agent prompts + per-mode style guides
│   ├── literature/      # search clients, citation guard, core library, replication priors
│   ├── spec/            # spec schema, validation, rendering
│   ├── registry/        # pre-registration records, trial ledger
│   ├── engine/          # vectorized backtest, costs, metrics
│   ├── validation/      # benchmark check, deflated Sharpe, placebo, costs, stability, holdout
│   ├── zoo/             # factor library, spanning, factor map
│   ├── mechanisms/      # mechanism test templates
│   ├── data/            # DataProvider interface + adapters
│   ├── reports/         # referee report, performance report, export package
│   └── app/             # Streamlit UI
├── museum/              # Failed Strategy Museum entries
├── core_library/        # curated structured summaries of ~100 key papers (no full texts)
└── tests/               # synthetic data only
```

---

## 14. Roadmap

### v1 (target: working demo)
1. Data layer: interface, free provider, Sharadar provider (local use)
2. Spec schema and backtest engine with metrics
3. Literature module: Librarian, citation guard, evidence cards, and a first 30 papers in the core library
4. Pre-registration record and trial ledger
5. Validation suite: benchmark check, deflated Sharpe, placebo, cost sensitivity, stability, holdout
6. Factor zoo positioning
7. Mechanism tests: the five templates that run on free data
8. Mentor, Analyst, and Referee flows, including the Advocate / Reviewer 2 debate and the referee report
9. Explanation layer with three depths
10. Streamlit UI, and a Failed Strategy Museum seeded with 3–5 entries

**End-to-end test:** reproduce known results (12-1 momentum and at least one of the author's own strategies) through the full pipeline using Sharadar, and check the numbers against the published or previously verified figures.

### v2
- Institutional-ownership mechanism test; more templates
- One-click paper replication with a replication-fidelity score
- Live strategy health monitor (actual performance versus the expected range from the backtest)
- Export of a replication package

### Later
- Additional markets (A-shares via free adapters)
- Integration with the 2027 Factor Calendar as Mentor-mode learning content

---

## 15. Risks and open questions

- **LLM knowledge of history.** Even at the strategy level, the model "knows" which anomalies worked. Mitigations: the sealed holdout, post-cutoff data, and the literature-benchmark check.
- **Literature coverage.** Top journals are paywalled. v1 relies on abstracts, open-access working papers, and the curated core library.
- **Free-data quality.** Survivorship bias and no point-in-time fundamentals. This is disclosed through capability flags; fundamentals-based strategies need a user-supplied provider.
- **Cost of the deep tier.** The Referee debate is the most expensive step. Token use should be capped, and the debate result cached for each spec hash.
- **Compliance.** Priors is a research and education tool. It gives no personalized investment advice and no recommendations on single securities. A disclaimer appears in the UI and in every report.

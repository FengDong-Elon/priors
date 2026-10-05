# Priors — Architecture (v0.3 draft)

> *Start from the literature, not from the chart.*
>
> Most AI investing tools try to convince you a strategy works. Priors starts from what the research already says, and then tries to prove your strategy wrong.

Status: v1 built (2026-10-04). This document describes the design; see the change log (§18) for decisions made while building, USER_MANUAL.md for how to use Priors, and DEPLOY.md for classroom setup.

**What changed in v0.3:** a stock layer is added on top of the factor layer. Factor-level data supplies the historical evidence. Stock-level data (yfinance by default, or the user's own source) turns the strategy into an implementable portfolio and a current holdings list on the Russell 3000 (§10). Defaults are set for the number of holdings, transaction costs, rebalancing, and data refresh, and the rule for building the core literature library is written down (§5.6). See §18 for the full change log.

---

## 1. Purpose and positioning

Priors is an **open-source teaching tool for evidence-based factor investing**. Students describe an investment idea in plain language. Priors helps them:
- ground the idea in the academic literature,
- test it without data mining,
- combine factors into a strategy,
- see what that strategy holds today,
- and read a referee-style critique of the result.

It is an **AI researcher, not an AI trader**. Its main job is to teach the research process: theory first, pre-registration, honest testing, and skepticism toward one's own results.

**Who it is for**

| Primary | Secondary |
|---|---|
| Undergraduate and master's students in investments, asset pricing, and quant courses | Other instructors who want a classroom tool; self-learners |

**What it is not**
- Not a research platform for new academic results
- Not a trading system, and not a source of investment advice
- It never recommends individual securities. Current holdings are shown only as the mechanical output of a student's rules (§10.6).

**License:** MIT. Code is open; data is never shipped (§14).

| | Typical AI-trading projects (TradingAgents, ai-hedge-fund) | Priors |
|---|---|---|
| Output | A buy/sell/hold decision on a ticker for a given day | A transparent strategy spec and a report |
| Where the LLM reasons | Every trading day, for every ticker | At design time and at review time |
| Where performance numbers come from | An LLM-in-the-loop simulation | Deterministic code |
| Evidence base | News, sentiment, the model's own knowledge | Retrieved academic literature |
| Stance | Advocate | Skeptic (falsification first) |

---

## 2. Design principles

1. **Theory before data.** A hypothesis needs an economic mechanism and retrieved literature support before it can be tested (§6). Ideas without that support can still be explored, but every exploratory result is labeled.
2. **Evidence from factors, implementation from stocks.** Historical evidence comes from published factor-portfolio data. Stock-level data turns the strategy into holdings and costs. Every result states which layer it comes from (§10).
3. **The LLM never produces a performance number.** Every return, Sharpe ratio, drawdown, t-stat, and p-value comes from deterministic code. LLMs guide, explain, and critique.
4. **Citations only come from retrieved sources.** A paper can be cited only if the literature module retrieved it in the current session, with a DOI or stable URL attached. Citing from model memory is blocked.
5. **Every trial is recorded.** Every spec version and every run is written to an append-only trial ledger. Reported significance is adjusted for the number of trials.
6. **One analysis, three explanation depths.** The analysis is identical in every mode. Only the explanation changes, and simplifying must never drop a risk warning.
7. **Process is visible.** Reports include the trial ledger and a summary of the conversation, so an instructor can see how a student reached a conclusion.
8. **Data limits are always disclosed.** Every data source declares what it lacks, such as delisted firms or point-in-time fundamentals, and the warnings follow every result built on it.
9. **Reproducible by default.** Every result can be regenerated from its spec, data snapshot identifier, engine version, and random seed.
10. **English only.** All code, UI, prompts, and documentation are in English.

---

## 3. User experience

### 3.1 Flow

```
 Start ─► Pick a mode ─► Chat ─► Theory gate ─► Pre-register ─► Factor-layer tests ─► Stock-layer implementation ─► Report
          (Mentor /               (or: exploratory path, labeled) ─┘                     (holdings, costs, current list)  (web + PDF)
           Analyst /
           Dr. Dong)
```

1. **Start screen.** The student enters a class access code or their own API key (§13), then picks one of three modes. Each mode card says in one or two sentences what the mode does and does not do.
2. **Chat.** A conversational interface specific to the chosen mode.
3. **Artifacts panel.** Next to the chat. It shows the current hypothesis, evidence cards, the spec, the trial ledger, and results as they are produced.
4. **Report.** Viewable in the app and downloadable as PDF (§12).

### 3.2 The three modes

| Mode | Typical user | Starts from | Ends with | Explanation depth |
|---|---|---|---|---|
| **Mentor** | Beginner, or anyone with a rough idea | A sentence ("cheap stocks that are going up") | A theory-backed, pre-registered hypothesis, a first test, and a simple stock portfolio | Plain language, analogies, no unexplained jargon |
| **Analyst** | Student with a defined strategy | A set of factors and rules | Tests, a multi-factor composite analysis, an implementation analysis, and concrete suggestions | Intermediate: standard terms, briefly explained |
| **Dr. Dong** | Advanced student | A strategy plus results | A journal-style referee report, signed by Dr. Dong | Technical and precise, with statistical detail |

### 3.3 Mode upgrade

A session can move **up** the chain: Mentor → Analyst → Dr. Dong. The hypothesis, evidence cards, spec, pre-registration, and trial ledger carry over. This mirrors the learning path from idea, to mature strategy, to critical review. Moving down is also allowed; it only changes the explanation depth.

Each reply also offers **"Explain simpler"** and **"More technical"** buttons, because a student's level and chosen mode do not always match.

---

## 4. Agents

Models run in two tiers. The provider (Anthropic or OpenAI) and the model for each tier are configurable.

- **Fast tier**: high-volume, mechanical work (for example Claude Haiku 4.5).
- **Deep tier**: judgment-heavy work (for example Claude Opus 5.5).

| Agent | Tier | Modes | Output |
|---|---|---|---|
| **Interviewer** | Deep | Mentor | A clarified idea: which signal, what horizon, and why it should work |
| **Librarian** | Fast | All | Candidate papers with metadata and abstracts, from retrieved sources only |
| **Evidence Synthesizer** | Deep | All | Evidence cards (§5.2) |
| **Theory Gatekeeper** | Deep + code | All | A pass / fail decision on the theory gate, with reasons (§6) |
| **Architect** | Deep | Mentor, Analyst | A strategy spec, with a citation for each design choice |
| **Registrar** | Code | All | A locked pre-registration record (§5.3) |
| **Engine** | Code | All | Factor-layer and stock-layer results (§7–§10) |
| **Performance Analyst** | Deep | Analyst | Diagnostics and ranked suggestions |
| **Advocate** | Deep | Dr. Dong | The strongest honest case that the premium is real |
| **Reviewer 2** | Deep | Dr. Dong | The strongest case that it is not |
| **Dr. Dong** (editor) | Deep | Dr. Dong | Weighs the debate and writes the final referee report and recommendation (§5.7) |
| **Explainer** | Fast | All | Student-facing text at the depth of the current mode (§12) |

**The Dr. Dong debate.** The Advocate and Reviewer 2 argue for a configurable number of rounds (default 2). Dr. Dong decides, weighing arguments by the evidence each side cites, not by how confidently they are stated.

**Orchestration.** A plain Python state machine with tool use. The pipeline is mostly linear, so a framework adds little.

---

## 5. Core artifacts

Every artifact is a versioned, human-readable file (YAML/JSON plus a rendered view).

### 5.1 Strategy spec

```yaml
spec_version: 3
name: "Value + momentum"
hypothesis_id: H-0007
status: preregistered            # preregistered | exploratory

components:
  - factor: HML                  # Ken French
    stock_signal: book_to_market # stock-layer equivalent (§10.3); null = factor layer only
    source: "Fama & French (1993), doi:10.1016/0304-405X(93)90023-5"
    theory_priority: 2           # set before testing, from evidence strength (§9.4)
  - factor: Mom12m               # Chen-Zimmermann signal name
    stock_signal: mom_12_1
    source: "Jegadeesh & Titman (1993), doi:10.1111/j.1540-6261.1993.tb04702.x"
    theory_priority: 1

factor_layer:                    # historical evidence (§7–§9)
  combination: equal_weight      # equal_weight | inverse_vol | theory_weights | optimized
  start: null                    # first month; null = earliest common month. The holdout (§7.6) is applied automatically.

stock_layer:                     # implementation (§10)
  universe: russell3000
  filters:
    min_price: 5
    exclude_microcaps: true      # below the NYSE 20th market-cap percentile
    min_adv_usd: 1_000_000       # average daily dollar volume
  signal_combination: rank_average
  holdings: 40                   # allowed 10–100
  weighting: equal               # equal | inverse_vol (Analyst, Dr. Dong)
  rebalance: monthly             # monthly | quarterly
  cost_bps_one_way: 20           # allowed 0–50
```

### 5.2 Evidence card

One card per line of research, for example "value premium."

- **Claim:** one sentence.
- **Mechanism(s):** risk compensation, behavioral bias, limits to arbitrage, market friction, or data mining.
- **Key papers:** DOI, sample, market, effect size, and the table or figure reference.
- **Replication status:** from Chen & Zimmermann, Hou-Xue-Zhang (2020), and Jensen-Kelly-Pedersen (2023).
- **Post-publication decay prior:** McLean & Pontiff (2016) by default, overridden when factor-specific evidence exists.
- **Boundary conditions and contrary evidence.**
- **Evidence grade:** strong, moderate, weak, or contested, with a one-line justification.
- **Expected effect range:** the prior that results are compared against.

### 5.3 Pre-registration record

Written before the first test. It is immutable, timestamped, and content-hashed. It contains:
- the hypothesis and its mechanism, in words;
- the expected sign and magnitude range;
- the components, their theory priorities, and both the factor-layer and stock-layer settings;
- the primary metric, the factor models used for spanning, and the mechanism tests chosen.

### 5.4 Trial ledger

Append-only. Each row records the timestamp, spec hash, data snapshot, engine version, layer, headline metrics, and **status (pre-registered or exploratory)**. The number of trials feeds into the deflated Sharpe ratio (§7.2).

Changing the stock-layer settings (holdings, weighting, filters, rebalancing) after seeing results also counts as a new trial.

### 5.5 Referee report

1. **Summary** of the strategy and its claimed edge
2. **Recommendation:** Reject, Major revision, Minor revision, or Accept
3. **Major comments:** problems that could invalidate the result
4. **Minor comments:** improvements and clarifications
5. **Statistical summary**, separating tested results from descriptive ones
6. **What would change my mind**

### 5.6 Core literature library

A curated set of structured paper summaries (no full texts) that the Librarian searches first. v1 targets about 30 papers.

**Selection rule:**
1. **Coverage.** The original paper for each factor in the library, plus the key methods papers: Fama-French, Carhart, Hou-Xue-Zhang, McLean-Pontiff, Harvey-Liu-Zhu, DeMiguel-Garlappi-Uppal, Bailey-López de Prado, Jensen-Kelly-Pedersen, Chen-Zimmermann, Novy-Marx & Velikov (costs), and similar.
2. **Ranking.** Citation count first, measured from OpenAlex at build time and never from memory. Among papers of similar citation weight, newer papers come first.

Each summary records the citation count and the date it was retrieved, so the ranking can be reproduced and refreshed.

### 5.7 The Dr. Dong persona

The top mode is named after the author, as a small in-joke for students. Dr. Dong is the strictest and most objective reviewer in Priors. The persona's standards are written into its prompt:

- **Tested vs. descriptive.** Only formally tested results may support a claim. Descriptive patterns are labeled as such.
- **Ex-ante justification.** Every specification choice must have been justified before the results were seen. Choices made after seeing results are treated as exploratory.
- **Uncertainty up front.** Point estimates are always reported with confidence intervals in the main text.
- **Hedged causal language.** "Consistent with," never "proves."
- **Falsify, don't rescue.** A weakening result is run through further checks and closed if it fails. It is not reframed to survive.
- **No flattery.** A recommendation of Reject is given whenever the evidence warrants it, regardless of how much work the student put in.

Every Dr. Dong report carries a footer: *"Dr. Dong is an AI reviewer persona modeled on the author's research standards. This report was generated automatically and was not personally reviewed by Dr. Feng Dong."* The footer keeps students from presenting an AI report as the instructor's personal judgment.

---

## 6. Theory gate and exploratory path

This is the core of Priors' teaching value.

### 6.1 The gate

A hypothesis passes the gate only if all three conditions hold:

1. **Mechanism.** The student states, in their own words, why the factor should earn a premium. The mechanism must be classified as risk compensation, behavioral bias, limits to arbitrage, or market friction.
2. **Literature support.** At least one retrieved paper supports the mechanism or the effect, with an evidence card attached.
3. **Expected range.** An expected sign and approximate magnitude, taken from the evidence card.

In Mentor mode, the Interviewer and Librarian actively help the student find a mechanism and supporting papers. If none can be found, Priors says so plainly: "There is no theoretical support for this idea yet."

The LLM writes an advisory assessment. The final pass / fail decision is made in code, which checks that a mechanism class is set, at least one retrieved citation is attached, and an expected range is set.

### 6.2 Exploratory path

Students may skip the gate and test anyway, because exploration is part of learning. The rules:

- Every exploratory run is stored in the ledger with `status: exploratory`.
- Every exploratory chart, table, number, and holdings list carries an **"Exploratory, not theory-backed"** label, in the app and in exports.
- Exploratory results never appear in the main body of a report. They are listed in a separate appendix that states how many exploratory runs were made.
- Exploratory runs count toward the deflated Sharpe ratio.
- An exploratory idea can later be **promoted** by passing the gate. Promotion requires a new pre-registration and is tested on the sealed holdout (§7.6), because the in-sample data has already been seen.

### 6.3 Teaching the cost of data mining

- **Ledger counter.** For example: "You have run 14 tests. After adjusting for that, the deflated Sharpe ratio is 0.21."
- **Random-factor demonstration (§7.3).** Shows students that picking the best of many random combinations can produce an impressive result.

---

## 7. Validation suite (factor layer)

Runs automatically after every factor-layer test. Stock-layer checks are in §10.

### 7.1 Literature benchmark check
Results are compared with the evidence card's expected range.
- **Far above:** warn about overfitting or data problems.
- **Far below:** check post-publication decay and differences from the original design.

### 7.2 Multiple-testing adjustment
The deflated Sharpe ratio (Bailey & López de Prado, 2014) is computed from the trial count in the ledger. The Harvey-Liu-Zhu (2016) threshold of t > 3 is shown for reference.

### 7.3 Random-factor placebo
The placebo draws **random sets of k factors** from the Chen-Zimmermann library, where k matches the number of factors in the student's strategy. Each set is combined with the same method as the student's strategy, and this is repeated 1,000 times. The output is a histogram of placebo Sharpe ratios with the student's strategy marked, plus an empirical percentile. It answers the question: "Is your combination better than combining published factors at random?"

A second view shows **the best of N random combinations** as N grows, to illustrate data mining directly.

The pool matches the student's construction. When every component is value-weighted (Ken French, Hou-Xue-Zhang, or a Chen-Zimmermann `_VW` series), draws come from the value-weighted decile versions of the published predictors; otherwise from the original-paper constructions, most of which are equal-weighted and earn higher gross Sharpe ratios. The placebo window starts when at least 100 predictors have complete data.

### 7.4 Costs
Factor-portfolio returns are gross of costs and are always labeled "before costs." Net-of-cost results come from the stock layer (§10.5).

### 7.5 Stability
- Subperiods, by decade and by market regime
- Rolling 36-month Sharpe ratio
- Results before vs. after each factor's publication date

### 7.6 Sealed holdout
The most recent **5 years** are sealed for both layers. Opening the holdout is allowed **once per hypothesis**, and the opening is logged. Results after opening are labeled "holdout used."

---

## 8. Factor zoo positioning

The question here is "what is this factor, really?"

- **Factor library:** Ken French (market, SMB, HML, RMW, CMA, UMD), Hou-Xue-Zhang q-factors, and the Chen & Zimmermann long-short portfolios.
- **Spanning regressions** against **pre-registered models only** (CAPM, FF3, FF5, FF5+UMD, q5). The output is the remaining alpha, the factor loadings, and R².
- **Nearest neighbors:** the factors most correlated with the student's strategy. Labeled **descriptive only**, because the comparison is chosen after the fact.
- **Factor map:** a 2-D map of the library (clustering or MDS on correlations) with the student's strategy plotted, showing which "family" it belongs to.

---

## 9. Multi-factor composite analysis (factor layer)

### 9.1 Two kinds of composite
- **Factor layer:** a **portfolio of factor portfolios**, meaning weights applied to the long-short return series. This is the source of historical evidence.
- **Stock layer:** a **signal composite**: each stock's signals are ranked within the universe, the ranks are averaged, and the top N stocks are held (§10.4). This is the implementable portfolio.

The two constructions differ, so their results differ. The report explains the gap (§10.7).

### 9.2 Factor-layer combination methods (part of the pre-registration)

| Method | Available in | Notes |
|---|---|---|
| **Equal weight** | All modes (the only option in Mentor) | The default. Justified by DeMiguel, Garlappi & Uppal (2009): estimated weights often lose to 1/N out of sample. |
| **Inverse volatility** | Analyst, Dr. Dong | An equal risk budget. Volatility is estimated from past data only. |
| **Theory weights** | Analyst, Dr. Dong | Fixed weights set by the student before testing, with a written reason. |
| **Optimized** | Analyst, Dr. Dong | Mean-variance or IC-based. Always estimated on a training window and evaluated on a later window, with an overfitting warning. |

### 9.3 The composite report

| Section | Content | Question answered |
|---|---|---|
| Composite performance | Return, volatility, Sharpe with 95% CI, drawdown, by subperiod | How does the whole strategy do? |
| Correlation | Correlation matrix of the component factors | Are these factors repeating the same bet? |
| Leave-one-out | Change in the composite Sharpe when each factor is removed | Whose absence hurts most? |
| Incremental alpha | Each factor regressed on the other components and on the pre-registered factor model | Does it add information the others don't have? |
| Shapley decomposition | The composite Sharpe divided fairly among the components (exact for k ≤ 8) | How much does each factor contribute? |
| Risk contribution | Each component's share of composite variance | Where does the risk come from? |
| Stability | Rolling and subperiod contributions | Is the contribution consistent over time? |

### 9.4 Priority: theory vs. evidence
The report shows two rankings side by side:
- **Theory priority**, set before testing. It is based on the evidence grade, replication status, and expected decay.
- **Empirical contribution**, measured after testing. It is based on the Shapley share and incremental alpha.

When the two disagree, the report says so and explains possible reasons. Examples: "strong theory but redundant with momentum," or "large in-sample contribution but weak theoretical support; treat with caution."

**Guard rail:** empirical contributions are in-sample. Re-weighting or dropping factors based on them creates a new spec. That spec is counted in the ledger and can only be judged on the sealed holdout. Priors states this explicitly whenever a student tries it.

---

## 10. Stock layer: implementation and current holdings

### 10.1 Role
The factor layer answers: "Has this worked historically, and how strong is the evidence?" The stock layer answers: "If I implement this on real stocks, what would I hold today, how much would I trade, and what would it cost?" Every stock-layer output is labeled with its data source and that source's limitations.

### 10.2 Universe and default filters
- **Universe:** an approximation of the Russell 3000: the 3,000 largest U.S. common stocks by market cap, built daily from the Nasdaq stock screener (all NYSE, Nasdaq, and NYSE American listings). Preferred shares, warrants, units, rights, notes, depositary shares, closed-end funds, and SPAC shells are removed, and share-class tickers are converted to Yahoo format (BRK/B becomes BRK-B). The official iShares IWV holdings file blocks scripted downloads, so it is not used. The top 1,000 approximate the Russell 1000. The label "approximates the Russell 3000" is shown wherever the universe appears.
- **Default filters**, which students can switch off:
  - Price ≥ $5
  - Exclude microcaps: market cap below the NYSE 20th percentile, using the Ken French ME breakpoints
  - Average daily dollar volume ≥ $1M
- If filters are switched off, the report warns: *"Portfolio includes microcaps; the cost assumption may substantially understate real trading costs."*
- **Coverage report:** how many universe members had usable data for each signal.

### 10.3 Stock-level signals
Only factors that can be computed from the stock data source have a `stock_signal`. With yfinance, v1 targets:

| Family | Signals | yfinance history |
|---|---|---|
| Price-based | 12-1 momentum, short-term reversal, low volatility, beta, 52-week high | Full price history |
| Size | Market cap | Current; history approximated from price × shares |
| Value | Book-to-market, earnings yield | Current; ~4–5 years of statements |
| Profitability / quality | ROE, gross profitability | Current; ~4–5 years of statements |
| Investment | Asset growth | Current; ~4–5 years of statements |

Factors without a stock-level equivalent are analyzed at the factor layer only and marked **"factor layer only"** in the UI.

### 10.4 Portfolio construction
- **Signal composite:** each signal is ranked within the filtered universe; ranks are averaged (equal weight in Mentor mode); missing signals are handled by averaging the available ranks, and the coverage is reported.
- **Holdings:** top **40** by default (allowed range 10–100).
- **Weighting:** equal by default; inverse volatility in Analyst and Dr. Dong.
- **Rebalancing:** monthly by default; quarterly optional.
- Signals are computed at close t and traded at close t+1.

### 10.5 Costs and turnover
- Default **20 bp one-way**; students can set 0–50 bp. The UI always says "one-way."
- Costs are charged on actual turnover at each rebalance.
- Every report includes a cost sensitivity table at 0, 10, 20, 30, and 50 bp, plus the breakeven cost.
- The report always shows the mean and median market cap of the holdings, so students can see where the portfolio really sits.

### 10.6 Current holdings list
- The latest rebalance under the student's rules: ticker, name, sector, each signal's rank, the composite rank, and the weight.
- Sector concentration and summary exposures.
- Always labeled: *"Mechanical output of your rules, for education. Not a recommendation to buy or sell any security."*
- Survivorship bias does not affect a list built from today's data, so this view is reliable even with yfinance.

### 10.7 Stock-level backtest and the factor-to-stock gap
- **Allowed, with explicit limits.** With yfinance:
  - **Price-based signals:** backtests run, but always with a prominent warning: *"No delisted firms; results are biased upward (survivorship bias)."*
  - **Fundamental signals:** statement history covers only about 4–5 years, which falls inside the sealed holdout, so pre-holdout backtests are **not available**. Only the current holdings list is produced.
- With a source that includes delisted firms and point-in-time fundamentals (Sharadar, EODHD, or similar, supplied by the user), both kinds run without these warnings.
- **Gap report:** the factor layer is a value-weighted decile long-short portfolio, while the stock layer is a long-only top-N portfolio. The report shows both side by side and explains the difference: long-only captures only part of a long-short premium, and the universe, weighting, and costs differ.
- **Universe comparison (optional view):** the same strategy on the Russell 1000, the filtered Russell 3000, and the unfiltered Russell 3000. Students see the premium grow as the universe extends into small caps, together with costs and risk. This is descriptive and counts as a trial if it changes the spec.

---

## 11. Mechanism tests

Tests come from a **fixed template library**. The Evidence Synthesizer proposes templates, the Registrar locks them, and the engine runs them.

| Mechanism | Testable prediction | Test | v1? |
|---|---|---|---|
| Behavioral / sentiment | Stronger after high investor sentiment (Stambaugh, Yu & Yuan 2012) | Returns conditional on the lagged Baker-Wurgler index | Yes |
| Risk compensation | Loses in bad times | Returns in NBER recessions, market drawdowns, and VIX regimes | Yes |
| Data mining | Decays after publication | Returns before vs. after the publication date (from Chen-Zimmermann) | Yes |
| Limits to arbitrage | Stronger in small, illiquid stocks | Double sorts on size and illiquidity | v2 (needs survivorship-free stock data) |
| Frictions | Disappears net of costs | Net-of-cost comparison across size groups | v2 (same) |

Results are reported as **"consistent with," "inconsistent with," or "inconclusive."** They are never reported as proof of a mechanism.

A coefficient counts only when its Newey-West |t| clears 1.96, or the Bonferroni-adjusted cutoff when one template tests several regimes (2.39 for the three risk regimes). The templates to run are part of the spec (`mechanism_tests`) and are locked at pre-registration; `publication_decay` always runs. Defaults by mechanism: behavioral bias → sentiment; risk compensation → risk regimes.

---

## 12. Explanation layer and reports

### 12.1 Explanation depth

| | Mentor | Analyst | Dr. Dong |
|---|---|---|---|
| Vocabulary | Everyday words; any unavoidable term gets a one-line gloss | Standard terms | Full technical vocabulary |
| Numbers | Rounded, concrete ("$100 would have fallen to about $65") | Metrics with brief context | Estimates with CIs, test statistics, adjustments |
| Statistics | "How sure we are," in words | CIs mentioned | Tested vs. descriptive separated, multiple-testing adjusted |
| Required risk content | **Same in all modes**: drawdowns, overfitting risk, data limitations, survivorship warnings, decay expectations |||

### 12.2 Report output
- An **in-app view**, plus a **PDF download** for assignments.
- Every report includes:
  - the pre-registration record;
  - factor-layer results and validation;
  - the composite report, if the strategy has more than one factor;
  - the stock-layer implementation: costs, turnover, the factor-to-stock gap, and the current holdings;
  - a summary of the trial ledger, with pre-registered and exploratory counts;
  - a short summary of the conversation.
- Exploratory results appear only in an appendix (§6.2).
- Every report carries a disclaimer: research and education only, not investment advice.

---

## 13. Interface, LLM access, data refresh, and deployment

### 13.1 Interface
**Streamlit** for v1. It is pure Python, so students and instructors can read and modify it. The core library stays UI-agnostic, so a web front end can replace Streamlit later.

### 13.2 LLM access

| | Classroom | Public |
|---|---|---|
| Who pays | The instructor, or a center or department budget | The user |
| How | Students enter a **class access code**. A small proxy holds the instructor's API key and enforces a **per-student budget** (tokens or dollars per week). | The user enters **their own API key**. It is kept in the session only and never stored. |
| Provider | Configured by the instructor | Anthropic or OpenAI, chosen by the user |

Rough cost: a few cents per conversation turn on the deep tier. Debate results are cached per spec hash, and fast-tier models handle routine work.

### 13.3 Stock-data refresh
- **Classroom deployment:** a scheduled job on the server pulls the universe file and stock data **once per day** and writes a cache. Students read only from the cache. This avoids many students hitting Yahoo from one IP address.
- **Local use:** data is fetched on the student's own machine and cached locally, with incremental updates.
- Fundamentals for about 3,000 tickers take tens of minutes to refresh, so the job runs off-hours, with retries and rate limiting.

### 13.4 Deployment
Streamlit Community Cloud (free) or a university server. The classroom proxy and the refresh job are small, separate services, so the API key never reaches the browser.

### 13.5 Privacy and academic integrity
- The start screen tells students that conversations are sent to the LLM provider, and that they should not enter personal information.
- By default, the server logs nothing beyond budget counters. An instructor who wants usage data for teaching evaluation or research should check FERPA and IRB requirements first. Priors makes this opt-in and visible to students.
- Reports include the trial ledger and a conversation summary (§12.2), so the process behind a result is visible.

---

## 14. Data layer

### 14.1 Factor layer (free, no key)

| Source | Content |
|---|---|
| Ken French Data Library | FF3/FF5 factors, UMD, sorted portfolios, NYSE ME breakpoints |
| Chen & Zimmermann Open Source Asset Pricing | Long-short returns for 200+ published anomalies, with signal documentation and publication dates |
| Chen & Zimmermann value-weighted deciles | The same predictors as value-weighted decile long-short portfolios (ids with suffix `_VW`), comparable to the French factors |
| Hou-Xue-Zhang | q-factor and q5 returns |
| FRED, NBER, Baker-Wurgler, VIX | Regime and sentiment series for mechanism tests |

### 14.2 Stock layer (provider adapters)

```
Stock layer
      │  depends only on the StockDataProvider interface
      ▼
StockDataProvider (abstract)
  ├── YFinanceProvider   (default, free; current data reliable, history survivorship-biased)
  ├── SharadarProvider   (bring your own; exists from v0.1)
  └── ... community adapters (EODHD, Tiingo, ...)
```

Each provider declares its capability flags, and the warnings in §10 are driven by them:

```python
Capabilities(
    includes_delisted: bool,             # yfinance: False
    point_in_time_fundamentals: bool,    # yfinance: False
    fundamentals_history_years: int,     # yfinance: ~4-5
    price_history_start: date,
    universe: str,                       # "us_equity"
)
```

**yfinance caveats.** It is an unofficial interface to Yahoo Finance. It can break without notice, and Yahoo's terms limit its use to personal and educational purposes. Priors therefore treats it as a replaceable default, never redistributes cached Yahoo data, and keeps all caches out of the repository.

### 14.3 Licensing rules
- Adapter code is open source. Data and caches are never committed.
- Keys live in `.env` or in the session.
- CI tests use synthetic data only.

---

## 15. Tech stack and repository layout

- **Language:** Python 3.11+
- **LLM:** Anthropic and OpenAI SDKs behind one provider interface
- **Computation:** pandas, numpy, statsmodels, scipy
- **Data:** yfinance (default stock provider); Parquet caches
- **UI:** Streamlit. **PDF:** generated from the same report template.
- **Packaging:** `pyproject.toml`. The PyPI name is `priors-quant`, the repo name is `priors`, and the license is MIT.

```
priors/
├── ARCHITECTURE.md, README.md, USER_MANUAL.md, DEPLOY.md
├── pyproject.toml          # package priors-quant; command `priors`
├── src/priors/
│   ├── factors/            # French, Hou-Xue-Zhang, Chen-Zimmermann (original and _VW) loaders; regime series
│   ├── spec/               # strategy spec (factor layer + stock layer); legacy stock-only spec
│   ├── holdout.py          # the calendar holdout rule
│   ├── composite/          # combination weights, factor-layer backtest, composite report (Shapley etc.)
│   ├── registry/           # theory gate, pre-registration records, hash-chained trial ledger
│   ├── literature/         # citations and core library, retrieval log, search, evidence cards, expected priors
│   ├── llm/                # LLM interface, Anthropic client, access (own key / proxy)
│   ├── validation/         # literature benchmark, deflated Sharpe, random-factor placebo, stability
│   ├── zoo/                # spanning regressions, nearest neighbors, factor map
│   ├── mechanisms/         # sentiment, risk-regime, publication-decay templates
│   ├── data/               # stock data providers: yfinance, Sharadar (local files), Sharadar download
│   ├── universe/           # top-3,000 / top-1,000 U.S. universe from the Nasdaq screener
│   ├── stocklayer/         # stock signals, filters, top-N construction, backtest, holdings, gap report
│   ├── engine/             # v0.1 stock-level engine (signals, metrics)
│   ├── stockdata.py        # snapshots, data sources, automatic daily refresh
│   ├── flows/              # session (the three modes), agents and prompts, facts dossier, project files
│   ├── reports/            # charts, report builder, HTML template, PDF export
│   ├── classroom.py        # in-app class-code access with per-student budgets
│   ├── proxy/              # optional classroom proxy service (for a host with a server)
│   ├── app/                # Streamlit app
│   └── cli.py              # priors app | refresh | sharadar-download | proxy
└── tests/                  # synthetic data only (live checks are skipped offline)
```

---

## 16. Roadmap

v1 is built in order of dependencies. There is no fixed course date.

### v1 (built)
1. Factor data loaders and cache (French, Chen-Zimmermann, q-factors, regime series)
2. Factor-layer spec, composite engine, and metrics
3. Theory gate, pre-registration, trial ledger, and exploratory labeling
4. Literature module: Librarian, citation guard, evidence cards, and a core library of about 30 papers (§5.6)
5. Validation: benchmark check, deflated Sharpe, random-factor placebo, stability, holdout
6. Composite report (§9) and factor zoo positioning (§8)
7. Stock layer: yfinance provider, Russell 3000 universe and filters, stock signals, portfolio construction, costs, current holdings, gap report
8. Mechanism tests: sentiment, risk regime, post-publication decay
9. Mentor, Analyst, and Dr. Dong flows with mode upgrade, and the Dr. Dong debate
10. Explanation layer with three depths; report view and PDF export
11. Streamlit app with a mode picker; own-key and class-code access with budgets; daily refresh job
12. User manual (USER_MANUAL.md) and deployment guide (DEPLOY.md)

**End-to-end tests:**
- Reproduce known published results at the factor layer: the momentum and value premia in French data, and a few Chen-Zimmermann anomalies compared with their original-paper t-stats.
- At the stock layer, compare yfinance results with the author's Sharadar results on the same spec, to measure the survivorship bias that students will see.

### v2
- Monthly point-in-time fundamental fields for Sharadar, so financial-statement signals can be backtested
- An OpenAI provider for the LLM interface
- Limits-to-arbitrage and friction mechanism tests (needs survivorship-free stock data)
- More stock-level signals
- One-click paper replication with a fidelity score

### Visibility (alongside v1)
- A Zenodo DOI for each release, so the software is citable
- A hosted demo link in the README
- An optional teaching note, for example in the *Journal of Financial Education* or on SSRN

---

## 17. Risks and open questions

- **LLM knowledge of history.** The model "knows" which factors worked. Mitigations: the sealed holdout, the literature benchmark check, and the citation guard.
- **Literature coverage.** Top journals are paywalled. v1 relies on abstracts, open working papers, and the curated core library.
- **yfinance reliability.** It may break or be rate-limited. Mitigations: the provider interface, daily caching, and graceful degradation (the factor layer still works without stock data).
- **Survivorship bias at the stock layer.** It is disclosed on every affected result. Fundamental-signal backtests are disabled on yfinance.
- **Small-cap costs.** Flat 20 bp understates real costs for small stocks. Mitigations: the default microcap filter, the holdings market-cap display, and the cost sensitivity table.
- **Classroom cost.** Controlled by per-student budgets, fast-tier defaults, and caching.
- **Compliance.** Research and education only. No personalized advice and no single-security recommendations; current holdings are labeled as mechanical rule output.

---

## 18. Change log

**v0.4 (2026-10-04): decisions made while building**
- Purpose restated: learning comes before investing. The flow is: connect, choose a mode, describe a strategy, literature first, test only with theory support, revise with guidance, report (or a literature assessment when the idea is not supported).
- Access: students enter a class code to use the instructor's Claude account with a weekly budget per student, inside the app (no server needed; the course workspace's spend limit is the hard cap). Others use their own key. A separate proxy service remains for hosts with a server.
- Data: free data by default; the hosted app refreshes a prices-only snapshot on the first visit each day. Sharadar is used only on the licensee's own computer, from local files or downloaded with the user's own key; it never goes to a deployment.
- Class settings: `allow_exploratory` (default off for classes) and `required_stage` (`tested` or `dr_dong`).
- Revision loop: free coaching before tests; after results, a revision is a new trial judged on the holdout.
- Reports: a short main report by mode, plus a technical appendix; a literature-assessment report for unsupported ideas.
- Project files for saving and continuing work, verified on upload.
- Teaching assignments and the Failed Strategy Museum were dropped from v1; a user manual replaces them.
- The universe is the 3,000 largest U.S. common stocks from the Nasdaq screener, because the iShares file cannot be downloaded by script.

**v0.3 (2026-10-04)**
- Implementation note: the universe is built from the Nasdaq screener (top 3,000 U.S. common stocks by market cap) because the iShares file cannot be downloaded by script. Stock returns are aligned to the month they are earned (formed at the end of month k, earned in k + 1).
- The top mode is renamed from Referee to **Dr. Dong**. Dr. Dong also replaces the Editor agent, and its standards and disclaimer footer are defined in §5.7.
- Added the stock layer (§10). Factor data gives the evidence; stock data gives the implementation and current holdings.
- Default stock provider is yfinance, with explicit limits: current lists are reliable; price-signal backtests carry a survivorship warning; fundamental-signal backtests are disabled. Users may supply survivorship-free data.
- Universe: Russell 3000 (iShares IWV), with default filters (price ≥ $5, no microcaps below the NYSE 20th percentile, average daily volume ≥ $1M) that students can switch off.
- Defaults: 40 holdings (10–100), equal weight, monthly rebalancing (quarterly optional), 20 bp one-way cost (0–50).
- Holdout fixed at 5 years for both layers.
- Daily server-side data refresh for classroom use; local caching for individual use.
- Core library selection rule: citation count (OpenAlex) first, then newest first (§5.6).

**v0.2 (2026-10-04)**
- Audience changed to students; purpose is teaching and public visibility; MIT license.
- Own Streamlit interface with a mode-picker start screen and a mode upgrade path.
- LLM access through a classroom proxy with per-student budgets, or the user's own key; Anthropic or OpenAI.
- Theory gate enforced in code; exploratory path allowed but labeled.
- Multi-factor composite analysis, with theory priority and empirical contribution shown side by side.
- Placebo redefined as random-factor combinations.

**v0.1 (2026-09-25)** — initial design.

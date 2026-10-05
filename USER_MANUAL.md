# Priors user manual

Priors is a teaching tool for evidence-based factor investing. You describe an investment idea; Priors checks what the research says about it, tests it only if the theory supports it, helps you revise it, and writes a report. Its purpose is learning how factor research is done well. It does not give investment advice, and nothing it produces is a recommendation to buy or sell a security.

This manual is for students and anyone else using Priors. Instructors setting it up for a class should also read [DEPLOY.md](DEPLOY.md).

---

## Contents

1. [The workflow at a glance](#1-the-workflow-at-a-glance)
2. [Getting started](#2-getting-started)
3. [Connecting](#3-connecting)
4. [Choosing a mode](#4-choosing-a-mode)
5. [Mentor mode, step by step](#5-mentor-mode-step-by-step)
6. [Analyst and Dr. Dong modes](#6-analyst-and-dr-dong-modes)
7. [The rules that protect your results](#7-the-rules-that-protect-your-results)
8. [Reading the results](#8-reading-the-results)
9. [Reports](#9-reports)
10. [Saving and continuing your work](#10-saving-and-continuing-your-work)
11. [Data sources and their limits](#11-data-sources-and-their-limits)
12. [Troubleshooting](#12-troubleshooting)
13. [Glossary](#13-glossary)

---

## 1. The workflow at a glance

```
Connect ─► Choose a mode ─► Describe your idea ─► Literature search ─► Theory gate
                                                                          │
                     ┌──────────── not supported ◄────────────────────────┤
                     ▼                                                    ▼ passes
            Literature assessment report                       Pre-register (locks the plan)
                                                                          ▼
                                                                  Tests and explanation
                                                                          ▼
                                                            (optional) Dr. Dong's review
                                                                          ▼
                                                                       Report
```

At any point before testing you can revise freely with the Mentor's help. After you have seen results, a revision counts as a new trial (see [section 7](#7-the-rules-that-protect-your-results)).

---

## 2. Getting started

### In a class

Your instructor gives you a web address (for example https://priors-elon.streamlit.app/) and a class code. Open the address in any browser. Nothing needs to be installed.

### Without a class code

Open https://priors-elon.streamlit.app/ and choose **My own API key** in the sidebar.

### On your own computer

You need Python 3.11 or later. Download the code from the project's GitHub page, then, in its folder:

```bash
pip install -e ".[app]"
```

```bash
priors app
```

The app opens in your browser at `http://localhost:8501`. To get daily stock data for the stock layer on your own computer, run this once a day (it takes a few minutes for prices, about half an hour with company financial data):

```bash
priors refresh
```

---

## 3. Connecting

Open the sidebar on the left.

| Option | When to use it | What it costs you |
|---|---|---|
| **Class code** | You are in a class that uses Priors | Nothing: your instructor's account pays, within a weekly budget per student. The sidebar shows how much of this week's budget you have used. |
| **My own API key** | You are using Priors on your own | Your own Anthropic account. The key stays in your browser session and is never saved. |

For a class code, enter the code and your student id exactly as your instructor gave them. If the app says your id is not on the roster, check it with your instructor.

**Privacy.** What you type is sent to an AI provider (Anthropic). Do not enter personal information.

### Stock data

On the start screen, **Stock data** shows the sources available:

- **Free data (Yahoo Finance)** is the default and needs no setup.
- **Sharadar** appears if Sharadar data is installed on your computer. If you have your own Nasdaq Data Link subscription and are running Priors on your own computer, open **Use your own Sharadar data** on the start screen, enter your key, and download. The data is saved on your computer only; the key is not stored. The same download from the command line:

```bash
priors sharadar-download --key YOUR_KEY
```

See [section 11](#11-data-sources-and-their-limits) for what each source can and cannot do.

---

## 4. Choosing a mode

Enter your name or student id and a project name, then pick a mode.

| Mode | Use it when | How it talks to you |
|---|---|---|
| **Mentor** | You have a rough idea, or you are new to factor investing | Plain language, one question at a time |
| **Analyst** | You can already say which factors you want to combine and how | Standard finance terms, with short explanations |
| **Dr. Dong** | Your strategy is finished and you want a strict review | Technical; a journal-style referee report |

You can switch modes at any time in the sidebar. Everything you have done carries over.

---

## 5. Mentor mode, step by step

### Step 1. Describe your idea

Type your idea in a sentence or two, for example: *"I think stocks with low volatility do better than people expect, because everyone chases exciting stocks."* The Mentor asks a few questions: which stocks you would favor, how long you would hold them, and **why you think it works**. Answer the "why" in your own words; you will need it later.

You cannot ask whether to buy a particular stock. If you mention one, the Mentor asks what characteristic of it you like and turns that into a factor idea.

### Step 2. Search the research

Click **Search the research**. Priors searches its curated library of key papers first, then an academic database, reads the abstracts, and builds an **evidence card**:

- the main claim of the research;
- the papers that support it, and **contrary evidence** that does not;
- **boundary conditions** (where the effect is weaker or absent);
- an **evidence grade**: strong, moderate, weak, or contested.

Every paper on the card was actually found by the search. Priors cannot cite papers from memory.

If the research does not support the idea, you can stop here: click **The research does not support it: conclude** (see step 7).

### Step 3. Propose a strategy

Click **Propose a strategy**. Priors chooses one to four published factors that capture your idea, sets an expected return range from the literature (after the decay that published anomalies usually show), and writes a testable hypothesis.

### Step 4. Explain the mechanism and check the theory gate

Read the hypothesis. In the box, explain **in your own words** why the premium should exist. Then click **Check the theory gate**. The gate passes only if:

1. a mechanism is chosen (risk compensation, behavioral bias, limits to arbitrage, or market friction) and explained in your own words, in a few sentences at least;
2. at least one paper found by the search supports it, with an evidence card attached;
3. there is an expected return range.

You also get advice on how to make the reasoning stronger.

### Step 5. Revise if needed (free before testing)

Open **Revise with the Mentor** to:

- ask the Mentor about your reasoning (it asks questions; it will not write your explanation for you);
- **Search the literature again** with a sharper version of your idea;
- **Propose a different strategy**;
- **Conclude: not supported**, if the research does not back the idea.

Nothing you do before testing counts against you.

### Step 6. Pre-register and run the tests

When the gate passes, click **Pre-register (this locks the plan)**. Your hypothesis, the factors, the expected range, and the tests to run are locked and time-stamped. Then click **Run the tests** (about half a minute). You get an explanation of the results in plain language and the required risk notes.

Some classes allow **Explore without registering**. Exploratory results are labeled as such and kept out of the main report.

### Step 7. Finish

Your project is finished when one of these is true (the sidebar's **Progress** shows which steps are done):

- the pre-registered strategy has been tested and, if your class requires it, reviewed by Dr. Dong; or
- you concluded that the literature does not support the idea. That is a complete, legitimate result: Priors writes a **literature assessment** explaining why the idea was not tested, what evidence would change that, and better-supported directions.

Then download your report from the **Report** tab.

---

## 6. Analyst and Dr. Dong modes

### Building a strategy yourself

In Analyst and Dr. Dong modes you build the strategy in a form:

| Setting | Default | Range | Meaning |
|---|---|---|---|
| Factors | — | 1 to 8 | Published factor portfolios (Ken French, Hou-Xue-Zhang, Chen-Zimmermann). IDs ending in `_VW` are value-weighted versions, closer to an investable portfolio. |
| Combination | equal weight | equal weight, inverse volatility, theory weights, optimized | How the factors are combined. Mentor mode uses equal weight only. Optimized weights are estimated on the first part of the sample and judged only on the rest. |
| Stocks to hold | 40 | 10 to 100 | Size of the stock portfolio built from the strategy's signals |
| Trading cost | 20 bp | 0 to 50 bp | One-way cost on every trade |
| Rebalance | monthly | monthly, quarterly | How often the stock portfolio is rebuilt |
| Exclude microcaps | on | | Drops stocks below the NYSE 20th market-cap percentile |
| Price at least $5 | on | | |
| Daily volume at least $1M | on | | |
| Mechanism tests | publication decay | + sentiment, risk regime | Tests of why the premium might exist |

To pre-register in these modes, open **Pre-register this strategy**, write the hypothesis and your explanation, and click **Build the evidence card and check the gate**.

### Performance Analyst (Analyst mode)

After the tests, **Ask the Performance Analyst** gives a diagnosis and ranked suggestions. Each suggestion says whether acting on it would be a new trial.

### Dr. Dong's review (Dr. Dong mode)

**Request Dr. Dong's review** (about four minutes): an Advocate argues that the premium is real, Reviewer 2 argues that it is not, and Dr. Dong weighs the evidence and writes a referee report with a recommendation: **accept** (suitable for paper trading as specified), **minor revision**, **major revision**, or **reject**. Dr. Dong is strict by design: only formally tested results count, weak results are not rescued, and effort does not earn a better grade.

*Dr. Dong is an AI reviewer persona modeled on the author's research standards. Its reports are generated automatically and are not personally reviewed by Dr. Feng Dong.*

---

## 7. The rules that protect your results

Priors is built to stop the most common mistake in investment research: trying many things and keeping whatever looks best.

- **Theory before data.** A strategy is tested only after it passes the theory gate (unless your class allows exploratory runs, which are labeled).
- **Pre-registration.** The plan is locked before the test. A changed strategy is a different strategy.
- **Every test is counted.** The trial ledger records every specification you test. The more you test, the higher the bar the **deflated Sharpe ratio** sets for every result in your project.
- **Revising after results is a new trial.** You can revise (**Revise the strategy**), but the new version must be registered again, and because you have already seen the in-sample data, only the holdout can judge it.
- **The sealed holdout.** The most recent five years of data are hidden. You can open them **once** per registered hypothesis, as the final out-of-sample check.
- **Numbers come from code.** The AI explains and critiques; every number is computed by code, and any number in an AI's text that is not in the computed results is flagged.
- **Records cannot be edited.** Registrations and the ledger are protected by hashes; a project whose records were edited by hand is rejected.

---

## 8. Reading the results

The **Results** tab and the report show:

| Result | What it tells you |
|---|---|
| Mean, volatility, Sharpe ratio (with 95% confidence interval), t-statistic, maximum drawdown | How the strategy did, before trading costs, at the factor level |
| **Literature benchmark** | Whether the result after the original papers' samples matches what the research leads you to expect. "Far above" can signal an error or overfitting. |
| **Multiple testing** | The t-statistic against the hurdle of 3 suggested by Harvey, Liu and Zhu (2016), and the deflated Sharpe ratio given how many trials you ran |
| **Random-factor placebo** | Whether your combination beats random combinations of published factors, and how good the best of N random combinations looks (a direct picture of data mining) |
| **Stability** | Results by decade and over rolling windows |
| **Contribution of each factor** | How much each factor adds to the Sharpe ratio and to risk; whether a factor is redundant. These are in-sample: acting on them is a new trial. |
| **Factor models** | Whether the strategy earns anything beyond standard factor models (CAPM, Fama-French, q-factors) |
| **Mechanism tests** | Whether returns behave as the proposed mechanism predicts (sentiment, bad times, decay after publication). Results read "consistent with", "inconsistent with", or "inconclusive", never "proves". |
| **Stock portfolio** | The strategy built from real stocks, after costs, with a beta-adjusted comparison to the eligible universe |
| **Current holdings** | The stocks the rules would hold today. This is mechanical output for learning, not a recommendation. |

---

## 9. Reports

On the **Report** tab, click **Build the PDF reports**. You get two files:

- **Main report**, short enough to hand in (about 6 to 8 pages): summary, credibility checks, the factors' contributions, the stock portfolio, the explanation, and (in Dr. Dong mode) Dr. Dong's verdict.
- **Technical appendix**: every exhibit, the full evidence card, the full referee report and the debate, exploratory runs, and your messages.

Exploratory results never appear in the main report. If you concluded that the idea is not supported, the report is a **literature assessment**.

---

## 10. Saving and continuing your work

A hosted app can lose saved files when it restarts. To keep your work:

1. In the sidebar, click **Download project file**. Keep the `.priors.zip` file.
2. To continue later, open **Continue a saved project** on the start screen, upload the file, and click **Open the saved project**.

Download your project file whenever you finish a step you would not want to redo.

---

## 11. Data sources and their limits

### Factor data (always available, free)

Published factor returns from the Ken French Data Library, Hou-Xue-Zhang q-factors, and Chen-Zimmermann's open-source replications of over 200 published anomalies (in original and value-weighted versions). These are the evidence for every strategy.

### Stock data

| | Free data (Yahoo Finance) | Sharadar (your own subscription) |
|---|---|---|
| Firms that were later delisted | Not included: backtests are biased upward (survivorship bias) | Included |
| Universe | Today's 3,000 largest U.S. stocks (approximates the Russell 3000) | The 3,000 largest at each past date |
| Price-signal backtests (momentum, reversal, low volatility, low beta, 52-week high, size) | Yes, with a survivorship warning | Yes |
| Financial-statement signals (book-to-market, earnings yield, ROE, gross profitability, asset growth) | Current holdings only, and only where financial data has been downloaded | Current holdings only in this version |

How large is the survivorship bias? On Sharadar data, a 40-stock momentum portfolio without filters earned about 0.7% a year above its universe (1999 to 2021); the same test on Yahoo data shows 8.7%. Keep the filters on and read Yahoo-based stock backtests with caution.

---

## 12. Troubleshooting

| Message or problem | What to do |
|---|---|
| "Unknown class code" or "not on the class roster" | Check the code and your id with your instructor. |
| "Your weekly budget is used up" | It resets on Monday (UTC). A Dr. Dong review costs the most; plan for it. |
| "In this class, a strategy must pass the theory gate..." | Your class requires pre-registration before testing. Revise until the gate passes, or conclude that the idea is not supported. |
| "...cannot be backtested yet" or "needs company financial data" | Financial-statement signals cannot be backtested with the available data. They are still analyzed at the factor level. |
| "Stock data is not available right now" | The stock layer is skipped; everything else works. Try again later. |
| "The AI declined this request" | Rephrase your message. Priors does not answer questions about buying particular securities. |
| Your project disappeared | Upload your saved project file (section 10). |
| "The project's records do not verify" | The project file was changed after download. Use an unmodified copy. |
| Nasdaq Data Link "rate limit" | The download stopped on purpose to protect your account. Wait, or contact Nasdaq Data Link support. |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Factor** | A characteristic (for example past return or book-to-market) used to sort stocks, and the long-short portfolio built from that sort |
| **Composite** | Several factors combined into one strategy |
| **Sharpe ratio** | Average return divided by its volatility, annualized: reward per unit of risk |
| **t-statistic (Newey-West)** | How many standard errors the average return is from zero, allowing for correlation over time |
| **Deflated Sharpe ratio** | The probability that the true Sharpe ratio beats what the best of your trials would reach by luck |
| **Pre-registration** | Locking the hypothesis and test plan before seeing results |
| **Holdout** | The most recent five years of data, sealed until the final check |
| **Exploratory** | A test run without passing the theory gate; labeled and kept out of the main report |
| **Survivorship bias** | Overstated results from data that leaves out firms that failed or were delisted |
| **Post-publication decay** | The tendency of anomaly returns to shrink after the paper that documents them is published (McLean and Pontiff, 2016) |
| **Spanning regression** | A regression of a strategy's returns on a factor model; the intercept (alpha) is what the model does not explain |
| **Shapley contribution** | A fair split of the composite's Sharpe ratio among its factors |

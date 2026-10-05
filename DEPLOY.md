# Deploying Priors for a class

This guide is for instructors. Students only need the app's address and a class code.

## How access works

| Who | AI model | Stock data |
|---|---|---|
| Students with a class code | The instructor's Anthropic account, with a weekly budget per student | Free data (published factors and Yahoo Finance) |
| Anyone else (from GitHub) | Their own Anthropic API key | Free data, or their own licensed source (for example Sharadar) |

No server of your own is needed. The app runs on Streamlit Community Cloud (free), and holds no licensed data.

## 1. Create a course-only Anthropic key with a spending cap

1. In the Anthropic Console, create a new **workspace** for the course (for example `FIN3500-Fall-2026`).
2. In that workspace's settings, set a **monthly spend limit**. This is the hard cap: the API stops when it is reached, whatever happens in the app.
3. Create an **API key in that workspace**. Do not reuse the key you use for research, so the course key can be disabled on its own.

Rough cost per student action (Claude Opus 5.5 for judgment, Claude Haiku 4.5 for routine work):

| Action | Approximate cost |
|---|---|
| Mentor conversation, literature search, strategy proposal | $0.15 to $0.25 |
| Explanation of results | about $0.05 |
| Dr. Dong review (debate and referee report) | $0.50 to $0.70 |

A weekly budget of $2 to $3 per student covers a full Mentor-to-Dr.-Dong project with room for revisions.

## 2. Deploy the app on Streamlit Community Cloud

1. Push the repository to GitHub.
2. At share.streamlit.io, create an app from the repository with the main file `src/priors/app/streamlit_app.py`.
3. Open the app's **Secrets** and paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."          # the course workspace key from step 1
PRIORS_HOSTED = "1"                         # hides local-only features (licensed-data download, local key)

PRIORS_CLASSROOM = """
classes:
  FIN3500-F26:
    weekly_budget_usd: 2.50
    students: [ab1234, cd5678]           # optional roster of student ids; omit to accept any id
    allowed_models: [claude-opus-5-5, claude-haiku-4-5]
    allow_exploratory: false             # true lets students backtest before passing the theory gate (labeled)
    required_stage: dr_dong              # what counts as finished: "tested" or "dr_dong"
"""
```

4. Give students the app's address and the class code. They sign in with the class code and their student id.

### Class settings

| Setting | Meaning |
|---|---|
| `allow_exploratory` | `false` (default): a strategy must pass the theory gate and be pre-registered before any backtest. `true`: students may also run exploratory backtests, which are labeled and kept out of the main report. |
| `required_stage` | `dr_dong` (default): a project is complete after a Dr. Dong review. `tested`: complete after the pre-registered test. A project that ends with "not supported by the literature" is also complete. |

Users without a class code (their own API key) may run exploratory backtests.

### Students' work

A hosted app can lose its disk on restart. Students should use **Download project file** in the sidebar to save their work, and **Continue a saved project** on the start screen to pick it up again. Uploaded projects are checked: if a registration record or the trial ledger was edited by hand, the project is rejected.

Secrets are never visible to students. `PRIORS_ALLOW_LOCAL_KEY` must not be set on a deployment.

### About the per-student budget

The app records each call's cost (from the token counts the API returns) and stops a student whose spending this week reaches the budget. Streamlit Community Cloud can reset the app's disk when it restarts, which resets the weekly counters. The workspace spend limit in step 1 is what guarantees the total; the per-student budget keeps one student from using everyone's share.

## 3. Stock data

The factor layer (published factor returns) needs no setup.

The stock layer reads a snapshot of daily stock prices. On a hosted deployment, **the first visitor of each day triggers a prices-only refresh** (about three minutes, from 2000 onward; peak memory about 1.3 GB, within the free tier). Other visitors wait for it or see the previous day's data. If Yahoo refuses the download, the previous snapshot is used; if there is none, the stock layer is skipped and everything else still works.

What this covers on a hosted deployment:

| Stock-layer feature | Price signals (momentum, reversal, low volatility, low beta, 52-week high, size) | Financial-statement signals (book-to-market, earnings yield, ROE, gross profitability, asset growth) |
|---|---|---|
| Backtest | Yes, with a survivorship-bias warning | No (no point-in-time history) |
| Current holdings | Yes | No: financial data takes about half an hour to download, too long for a hosted app |

Financial-statement factors are still fully analyzed at the factor layer (published factor returns). On a computer you control, `priors refresh` downloads financial data too, and the app then builds current holdings for those signals:

```bash
priors refresh
```

To turn the automatic refresh off (for example when a scheduled `priors refresh` already runs), set `PRIORS_AUTO_REFRESH=0`.

## 4. Optional: licensed data on a personal computer

A student or instructor with their own Nasdaq Data Link (Sharadar) subscription can run Priors on their own computer and download the data there, either on the app's start screen ("Use your own Sharadar data") or with:

```bash
priors sharadar-download --key YOUR_KEY
```

The key is used for the download only and is not stored. Alternatively, point Priors at existing files by setting, in `.env`:

```
SHARADAR_SEP_PATH=...        # daily prices
SHARADAR_SF1_PATH=...        # fundamentals
SHARADAR_TICKERS_PATH=...    # ticker metadata
```

The app then offers "Sharadar" as a stock-data source, with a point-in-time universe that includes firms that later delisted. Licensed data must never be uploaded to a deployment or committed to the repository.

## 5. Optional: a classroom proxy on a server

If a server becomes available, `priors proxy` runs the same class-code access as a separate service, so the instructor key lives only on that server. Set `PRIORS_PROXY_URL` in the app's secrets to use it instead of the built-in classroom mode.

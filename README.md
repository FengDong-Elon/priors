# Priors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23149421.svg)](https://doi.org/10.5281/zenodo.23149421)

*Start from the literature, not from the chart.*

Priors is an open-source teaching tool for evidence-based factor investing. A student describes an investment idea; Priors searches the academic literature, tests the idea only if theory supports it, guides the student through revisions without data mining, and writes a report. An AI reviewer, Dr. Dong, can then review the strategy the way a journal referee would.

Priors is for education. It does not give investment advice or recommend any security.

**Try it:** https://priors-elon.streamlit.app/ (students sign in with their class code; anyone else can use their own Anthropic API key).

## How it works

1. **Connect.** Use a class code (the instructor's account, with a weekly budget per student) or your own Anthropic API key. Stock data defaults to free Yahoo Finance data; licensed Sharadar data can be used on your own computer.
2. **Choose a mode.** Mentor (guided, plain language), Analyst (build a strategy yourself), or Dr. Dong (strict referee review).
3. **Literature first.** Priors builds an evidence card from a curated library of key papers and a live academic search, with supporting and contrary evidence. Citations come only from papers the search actually found.
4. **Theory gate.** A strategy is tested only after the student states a mechanism in their own words, backed by retrieved papers and an expected return range.
5. **Pre-register and test.** The plan is locked, then tested on published factor returns and on real stocks: literature benchmark, deflated Sharpe ratio, random-factor placebo, stability, factor-model alphas, mechanism tests, costs, and current holdings. Every number comes from code, not from the AI.
6. **Revise honestly.** Revising before testing is free; revising after results is a new trial, counted in a tamper-evident ledger and judged only on a sealed five-year holdout.
7. **Report.** A short main report to hand in and a technical appendix, or a literature assessment when the idea is not supported.

## Documentation

- [USER_MANUAL.md](USER_MANUAL.md): how to use Priors, step by step
- [DEPLOY.md](DEPLOY.md): setting up Priors for a class
- [ARCHITECTURE.md](ARCHITECTURE.md): design and decisions

## Quick start (your own computer)

Python 3.11 or later. In the project folder:

```bash
pip install -e ".[app]"
```

```bash
priors app
```

Optional: daily stock data for the stock layer.

```bash
priors refresh
```

## Data

Priors ships no data. Factor returns are downloaded from the Ken French Data Library, Hou-Xue-Zhang (global-q.org), and Chen-Zimmermann Open Source Asset Pricing. Stock data comes from Yahoo Finance via yfinance, or from your own licensed source. Downloaded and derived data are kept in a local cache and are never committed.

## Development

```bash
pip install -e ".[dev,app,proxy]"
```

```bash
pytest
```

Tests use synthetic data; checks against live sources are skipped when offline.

## Citation

If you use Priors in teaching or research, please cite:

> Dong, Feng. 2026. *Priors: A Literature-First AI Teaching Tool for Evidence-Based Factor Investing* (Version 1.0.0) [Software]. Elon University. https://doi.org/10.5281/zenodo.23149421

GitHub's "Cite this repository" button gives the same citation in APA and BibTeX formats.

## License

MIT

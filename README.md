# Priors

*Start from the literature, not from the chart.*

Priors is a literature-first AI research assistant for systematic investing. It helps investors who don't write much code turn an idea into a testable factor strategy, backtest it, and then audit it the way a journal referee would.

> **Status: early development.** The data layer and the backtest engine work. The literature module, validation suite, and the Mentor / Analyst / Referee agents are not built yet. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

Priors is a research and education tool. It does not give investment advice or recommend individual securities.

## What works today

- **Strategy spec**: a strategy is one human-readable YAML file ([example](examples/momentum_12_1.yaml)). Each spec has a stable fingerprint, so every result can be traced back to exactly what produced it.
- **Backtest engine**: vectorized and deterministic, built for cross-sectional factor strategies.
  - Quantile sorts, long-only or long-short
  - Equal, value, or inverse-volatility weights
  - An explicit gap between the signal date and the trade date
  - Turnover-based transaction costs
  - Sharpe ratios reported with 95% confidence intervals
- **Look-ahead protection**: signals are only given the fields they declare, and forward returns can never be one of those fields. A test checks that weights at any date do not change when later data is removed.
- **Bring your own data**: every data source sits behind a `DataProvider` interface and declares its known limitations, such as survivorship bias or missing delisting returns. These limitations are attached to every result as warnings.
  - `SharadarProvider` reads your local Sharadar Parquet exports.
  - `SyntheticProvider` generates data with a planted effect of known size, for tests and demos.

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

```python
from priors.data import SharadarProvider
from priors.engine import run_backtest
from priors.spec import StrategySpec

spec = StrategySpec.load("examples/momentum_12_1.yaml")
result = run_backtest(spec, SharadarProvider())   # paths read from .env
print(result.summary())
print(result.warnings)
```

## Data licensing

Priors never ships data. Raw data, caches, and derived firm-level datasets are git-ignored. Provide your own licensed data through a provider adapter.

## License

MIT

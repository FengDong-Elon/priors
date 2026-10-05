import numpy as np
import pandas as pd
import pytest

from priors.factors import FactorLibrary


def make_lib(seed: int = 0) -> FactorLibrary:
    """Synthetic factor library: A and B from 1990, C from 2000, B ends 2018-12."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1990-01", "2020-12", freq="M", name="month")
    n = len(idx)
    r = pd.DataFrame(
        {
            "A": 0.006 + 0.02 * rng.standard_normal(n),
            "B": 0.004 + 0.04 * rng.standard_normal(n),
            "C": 0.000 + 0.03 * rng.standard_normal(n),
        },
        index=idx,
    )
    r.loc[:"1999-12", "C"] = np.nan
    r.loc["2019-01":, "B"] = np.nan
    info = pd.DataFrame({"source": "test", "name": list(r.columns)}, index=r.columns)
    return FactorLibrary(returns=r, rf=pd.Series(0.0, index=idx), info=info)


@pytest.fixture
def lib() -> FactorLibrary:
    return make_lib()

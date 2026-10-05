"""The sealed holdout (ARCHITECTURE §7.6).

One calendar rule for every factor and both layers: the most recent
``HOLDOUT_YEARS`` years before the reference month are sealed. The reference
month is fixed when a hypothesis is registered, so the holdout boundary of a
registered hypothesis never moves as time passes.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

HOLDOUT_YEARS = 5


def holdout_start(asof: date | str | pd.Period | None = None, years: int = HOLDOUT_YEARS) -> pd.Period:
    """First sealed month. With asof = 2026-10, the holdout starts 2021-10."""
    ref = pd.Period(asof if asof is not None else pd.Timestamp.today(), "M")
    return ref - 12 * years

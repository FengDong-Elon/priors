"""The strategy spec: the single, human-readable definition of a strategy."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Universe(_Strict):
    asset_class: Literal["us_equity"] = "us_equity"
    min_price: float | None = Field(5.0, description="Minimum unadjusted price at the signal date.")
    min_mcap_pctile: float | None = Field(
        None, ge=0, le=100, description="Exclude stocks below this market-cap percentile."
    )
    mcap_breakpoints: Literal["nyse", "all"] = "nyse"
    min_dollar_volume: float | None = Field(None, description="Minimum mean daily dollar volume.")


class Signal(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    direction: Literal["high", "low"] = Field(
        "high", description="'high': go long the highest signal values; 'low': the lowest."
    )
    source: str | None = Field(None, description="Citation for the signal definition (DOI or URL).")


class Portfolio(_Strict):
    construction: Literal["quantile_sort"] = "quantile_sort"
    quantiles: int = Field(10, ge=2, le=100)
    breakpoints: Literal["all", "nyse"] = Field(
        "all", description="Compute quantile cutoffs from all eligible stocks, or from NYSE stocks only."
    )
    side: Literal["long_short", "long_only"] = "long_short"
    weighting: Literal["equal", "value", "inverse_vol"] = "value"


class Execution(_Strict):
    lag_days: int = Field(1, ge=0, le=10, description="Trading days between signal and trade.")
    cost_bps_one_way: float = Field(15.0, ge=0)


class Sample(_Strict):
    start: date | None = None
    end: date | None = None


class StrategySpec(_Strict):
    spec_version: Literal[1] = 1
    name: str
    hypothesis_id: str | None = None
    universe: Universe = Field(default_factory=Universe)
    signal: Signal
    portfolio: Portfolio = Field(default_factory=Portfolio)
    rebalance: Literal["monthly", "weekly"] = "monthly"
    execution: Execution = Field(default_factory=Execution)
    sample: Sample = Field(default_factory=Sample)

    @model_validator(mode="after")
    def _check(self) -> "StrategySpec":
        if self.sample.start and self.sample.end and self.sample.start >= self.sample.end:
            raise ValueError("sample.start must be before sample.end")
        return self

    # -------------------------------------------------------------- identity

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        """Stable hash of everything that affects results. The name does not count."""
        d = self.model_dump(mode="json")
        d.pop("name")
        d.pop("hypothesis_id")
        raw = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------- I/O

    @classmethod
    def from_yaml(cls, text: str) -> "StrategySpec":
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def load(cls, path: str | Path) -> "StrategySpec":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    def to_yaml(self) -> str:
        # Keep explicit nulls: null can mean "no filter", which differs from the default.
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

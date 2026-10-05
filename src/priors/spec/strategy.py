"""The strategy spec (v3): one human-readable file that defines a student's strategy.

A strategy is a set of factor components plus settings for two layers:

* ``factor_layer``: how the components' published factor returns are combined
  to produce the historical evidence (ARCHITECTURE §9).
* ``stock_layer``: how the components' stock-level signals are turned into a
  portfolio of real stocks and a current holdings list (ARCHITECTURE §10).

Two hashes identify a spec:

* ``fingerprint()`` covers only the settings that change numbers. Two specs with
  the same fingerprint give identical results on the same data.
* ``content_hash()`` covers everything, including citations and theory
  priorities. The pre-registration record locks this hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_COMPONENTS = 8  # Shapley decomposition is exact up to 8 components (2^8 subsets)

Combination = Literal["equal_weight", "inverse_vol", "theory_weights", "optimized"]
Mode = Literal["mentor", "analyst", "dr_dong"]

# Which factor-layer combination methods each mode may use (ARCHITECTURE §9.2).
MODE_COMBINATIONS: dict[str, tuple[str, ...]] = {
    "mentor": ("equal_weight",),
    "analyst": ("equal_weight", "inverse_vol", "theory_weights", "optimized"),
    "dr_dong": ("equal_weight", "inverse_vol", "theory_weights", "optimized"),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _month(v: str | None) -> str | None:
    """Accept 'YYYY-MM' (or a date string) and normalize it to 'YYYY-MM'."""
    if v is None:
        return None
    s = str(v).strip()
    if len(s) >= 7 and s[4] == "-":
        y, m = int(s[:4]), int(s[5:7])
        if 1 <= m <= 12:
            return f"{y:04d}-{m:02d}"
    raise ValueError(f"expected a month as 'YYYY-MM', got {v!r}")


class Component(_Strict):
    factor: str = Field(description="Factor id in the factor library, e.g. 'HML' or 'Mom12m'.")
    stock_signal: str | None = Field(None, description="Stock-layer signal; null means factor layer only.")
    source: str | None = Field(None, description="Citation (DOI or URL) for the factor's original evidence.")
    theory_priority: int | None = Field(None, ge=1, description="1 = strongest theoretical support. Set before testing.")
    weight: float | None = Field(None, description="Fixed weight; only used with combination 'theory_weights'.")


class FactorLayer(_Strict):
    combination: Combination = "equal_weight"
    vol_lookback_months: int = Field(36, ge=12, le=120, description="inverse_vol: trailing window for volatility.")
    train_fraction: float = Field(
        0.5, gt=0.2, lt=0.8,
        description="optimized: share of the in-sample period used to estimate weights; the rest is the test window.",
    )
    start: str | None = Field(None, description="First month of the sample, 'YYYY-MM'. Null = earliest common month.")
    spanning_models: list[Literal["CAPM", "FF3", "FF5", "FF5_UMD", "Q5"]] = Field(
        default_factory=lambda: ["CAPM", "FF3", "FF5_UMD", "Q5"],
        min_length=1,
        description="Factor models for spanning regressions, fixed before testing (§8).",
    )

    _norm_start = field_validator("start", mode="before")(_month)


class StockFilters(_Strict):
    min_price: float | None = Field(5.0, ge=0)
    exclude_microcaps: bool = Field(True, description="Drop stocks below the NYSE 20th market-cap percentile.")
    min_adv_usd: float | None = Field(1_000_000, ge=0, description="Minimum average daily dollar volume.")


class StockLayer(_Strict):
    universe: Literal["russell3000", "russell1000"] = "russell3000"
    filters: StockFilters = Field(default_factory=StockFilters)
    signal_combination: Literal["rank_average"] = "rank_average"
    holdings: int = Field(40, ge=10, le=100)
    weighting: Literal["equal", "inverse_vol"] = "equal"
    rebalance: Literal["monthly", "quarterly"] = "monthly"
    cost_bps_one_way: float = Field(20.0, ge=0, le=50)


class StrategySpec(_Strict):
    spec_version: Literal[3] = 3
    name: str
    hypothesis_id: str | None = None
    status: Literal["preregistered", "exploratory"] = "exploratory"
    components: list[Component] = Field(min_length=1, max_length=MAX_COMPONENTS)
    factor_layer: FactorLayer = Field(default_factory=FactorLayer)
    stock_layer: StockLayer = Field(default_factory=StockLayer)
    mechanism_tests: list[Literal["sentiment", "risk_regime", "publication_decay"]] = Field(
        default_factory=lambda: ["publication_decay"],
        description="Mechanism test templates, fixed before testing (§11).",
    )

    @model_validator(mode="after")
    def _check(self) -> "StrategySpec":
        ids = [c.factor for c in self.components]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate factors in components: {ids}")
        has_w = [c.weight is not None for c in self.components]
        if self.factor_layer.combination == "theory_weights":
            if not all(has_w):
                raise ValueError("combination 'theory_weights' needs a weight on every component")
            total = sum(c.weight for c in self.components)
            if abs(total - 1) > 1e-6:
                raise ValueError(f"theory weights must sum to 1, got {total:.6f}")
        elif any(has_w):
            raise ValueError("component weights are only used with combination 'theory_weights'")
        return self

    @property
    def factor_ids(self) -> list[str]:
        return [c.factor for c in self.components]

    def check_mode(self, mode: Mode) -> None:
        """Raise if the chosen combination method is not available in this mode."""
        allowed = MODE_COMBINATIONS[mode]
        if self.factor_layer.combination not in allowed:
            raise ValueError(
                f"combination {self.factor_layer.combination!r} is not available in {mode} mode; "
                f"allowed: {', '.join(allowed)}"
            )

    # -------------------------------------------------------------- identity

    def _hash(self, d: dict) -> str:
        raw = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def fingerprint(self) -> str:
        """Hash of the settings that change results (not names, citations, priorities, or status)."""
        d = self.model_dump(mode="json")
        for k in ("name", "hypothesis_id", "status"):
            d.pop(k)
        d["components"] = [
            {"factor": c["factor"], "stock_signal": c["stock_signal"], "weight": c["weight"]}
            for c in d["components"]
        ]
        return self._hash(d)

    def factor_fingerprint(self) -> str:
        """Hash of the settings that change factor-layer results only."""
        d = self.model_dump(mode="json")
        return self._hash({
            "components": [{"factor": c["factor"], "weight": c["weight"]} for c in d["components"]],
            "factor_layer": d["factor_layer"],
        })

    def stock_fingerprint(self) -> str:
        """Hash of the settings that change stock-layer results only."""
        d = self.model_dump(mode="json")
        return self._hash({
            "components": [{"stock_signal": c["stock_signal"]} for c in d["components"]],
            "stock_layer": d["stock_layer"],
        })

    def content_hash(self) -> str:
        """Hash of the whole spec, as locked by pre-registration."""
        return self._hash(self.model_dump(mode="json"))

    # ------------------------------------------------------------------- I/O

    @classmethod
    def from_yaml(cls, text: str) -> "StrategySpec":
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def load(cls, path: str | Path) -> "StrategySpec":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

"""Signal library.

A signal maps a Panel to a wide DataFrame of characteristic values, using only
fields known at each signal date. Signals return the raw characteristic. The
spec's ``direction`` decides whether high or low values are bought.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..data.base import FORWARD_FIELDS, Panel


@dataclass(frozen=True)
class SignalDef:
    name: str
    fn: Callable[..., pd.DataFrame]
    description: str
    fields: tuple[str, ...]

    @property
    def defaults(self) -> dict:
        sig = inspect.signature(self.fn)
        return {k: p.default for k, p in list(sig.parameters.items())[1:]}


REGISTRY: dict[str, SignalDef] = {}


def register(name: str, description: str, fields: tuple[str, ...]):
    if FORWARD_FIELDS & set(fields):
        raise ValueError(f"Signal {name!r} may not use forward-looking fields {FORWARD_FIELDS & set(fields)}")

    def deco(fn):
        REGISTRY[name] = SignalDef(name, fn, description, fields)
        return fn

    return deco


def compute_signal(panel: Panel, name: str, params: dict) -> pd.DataFrame:
    if name not in REGISTRY:
        raise KeyError(f"Unknown signal {name!r}. Available: {sorted(REGISTRY)}")
    sd = REGISTRY[name]
    unknown = set(params) - set(sd.defaults)
    if unknown:
        raise ValueError(f"Signal {name!r} got unknown params {sorted(unknown)}; accepts {sorted(sd.defaults)}")
    # Only hand the signal the fields it declared, so it cannot see ret_next.
    view = _FieldView(panel, sd.fields)
    return sd.fn(view, **params)


class _FieldView:
    def __init__(self, panel: Panel, allowed: tuple[str, ...]):
        self._panel, self._allowed = panel, allowed

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self._allowed:
            raise KeyError(f"Signal did not declare field {name!r}")
        return self._panel[name]


# --------------------------------------------------------------------- signals

@register("past_return", "Cumulative return over `lookback` periods, skipping the most recent `skip`.", ("ret_past",))
def past_return(panel, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    if not 0 <= skip < lookback:
        raise ValueError("need 0 <= skip < lookback")
    window = lookback - skip
    logr = np.log1p(panel["ret_past"])
    return np.expm1(logr.shift(skip).rolling(window, min_periods=window).sum())


@register("momentum", "Alias of past_return with the Jegadeesh-Titman 12-1 default.", ("ret_past",))
def momentum(panel, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    return past_return(panel, lookback=lookback, skip=skip)


@register("volatility", "Mean daily return volatility over the last `window` periods.", ("volatility",))
def volatility(panel, window: int = 12) -> pd.DataFrame:
    return panel["volatility"].rolling(window, min_periods=max(1, window // 2)).mean()


@register("size", "Market capitalization at the signal date.", ("mcap",))
def size(panel) -> pd.DataFrame:
    return panel["mcap"]


@register("dollar_volume", "Mean daily dollar volume over the last `window` periods.", ("dollar_volume",))
def dollar_volume(panel, window: int = 1) -> pd.DataFrame:
    return panel["dollar_volume"].rolling(window, min_periods=1).mean()

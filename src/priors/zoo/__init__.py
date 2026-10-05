"""Factor zoo positioning (ARCHITECTURE §8): what is this strategy, really?

* ``spanning``: alpha and loadings against the pre-registered factor models.
  This is the tested part.
* ``nearest_neighbors``: the published factors most correlated with the
  composite. Chosen after the fact, so it is descriptive only.
* ``factor_map``: a 2-D map of the published factors (classical MDS on
  correlation distance) with the composite placed on it, plus the cluster
  ("family") it falls into. Descriptive only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..composite.engine import FactorResult
from ..composite.report import ols_nw
from ..factors import MODELS, FactorLibrary

MIN_OVERLAP = 120
FAMILY_RHO = 0.4  # a family: average-linkage cluster whose members correlate about 0.4 or more


@dataclass
class SpanningResult:
    model: str
    alpha_ann: float
    alpha_t: float
    betas: dict[str, float]
    beta_t: dict[str, float]
    r2: float
    months: int
    start: str
    end: str
    overlap_note: str | None = None


def spanning(result: FactorResult, lib: FactorLibrary) -> list[SpanningResult]:
    y = result.returns["composite"]
    out = []
    for name in result.spec.factor_layer.spanning_models:
        cols = MODELS[name]
        if not set(cols) <= set(lib.returns.columns):
            continue  # the library was loaded without this model's factors
        X = lib.returns[cols].reindex(y.index)
        fit = ols_nw(y, X)
        used = pd.concat([y, X], axis=1).dropna().index
        inside = sorted(set(cols) & set(result.spec.factor_ids))
        if set(result.spec.factor_ids) <= set(cols):
            out.append(SpanningResult(
                model=name, alpha_ann=float("nan"), alpha_t=float("nan"), betas={}, beta_t={}, r2=1.0,
                months=len(y.dropna()), start=str(y.index[0]), end=str(y.index[-1]),
                overlap_note=f"Every component is a {name} factor, so the model explains the composite exactly "
                             "and its alpha is zero by construction. This regression is not informative.",
            ))
            continue
        note = (
            f"The composite contains {', '.join(inside)}, which is also in the {name} model; the alpha measures "
            "only what the model does not explain." if inside else None
        )
        out.append(SpanningResult(
            model=name, alpha_ann=fit["alpha_ann"], alpha_t=fit["alpha_t"], betas=fit["betas"],
            beta_t=fit["beta_t"], r2=fit["r2"], months=fit["months"],
            start=str(used[0]), end=str(used[-1]), overlap_note=note,
        ))
    return out


def _universe(lib: FactorLibrary, exclude: set[str]) -> list[str]:
    """Published factors for neighbor search and the map: French, q-factors, original-construction CZ."""
    keep = lib.info.index[lib.info["source"].isin(["french", "hxz", "cz"])]
    return [f for f in keep if f not in exclude]


def nearest_neighbors(result: FactorResult, lib: FactorLibrary, k: int = 10) -> pd.DataFrame:
    y = result.returns["composite"]
    own = set(result.spec.factor_ids) | {f.removesuffix("_VW") for f in result.spec.factor_ids}
    ids = _universe(lib, own)
    X = lib.returns[ids].reindex(y.index)
    overlap = X.notna().sum()
    X = X.loc[:, overlap >= MIN_OVERLAP]
    corr = X.corrwith(y).dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index)[:k]
    info = lib.info.loc[top.index]
    return pd.DataFrame({
        "correlation": top,
        "name": info["name"],
        "authors": info.get("authors"),
        "year": info["year"],
        "category": info.get("economic_category"),
        "months": overlap[top.index],
    })


@dataclass
class FactorMap:
    coords: pd.DataFrame           # x, y, cluster, name, category; the composite is the row 'YOUR STRATEGY'
    family: str                    # most common category in the composite's cluster ("none" if it has no family)
    family_members: list[str]      # published factors in that cluster, excluding the student's own components
    flags: list[str] = field(default_factory=list)


def classical_mds(d: np.ndarray, dims: int = 2) -> np.ndarray:
    n = len(d)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d**2) @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:dims]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))


def factor_map(result: FactorResult, lib: FactorLibrary, start: str = "1963-07") -> FactorMap:
    """Map published factors by return correlation (from ``start`` to the end of the result's sample)."""
    end = result.sample["end"]
    ids = _universe(lib, set())
    data = lib.returns[ids].loc[pd.Period(start, "M"): end]
    data = data.loc[:, data.notna().sum() >= MIN_OVERLAP]
    label = "YOUR STRATEGY"
    data[label] = result.returns["composite"].reindex(data.index)
    corr = data.corr(min_periods=MIN_OVERLAP).fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    dist = np.sqrt(np.clip(2 * (1 - corr.to_numpy()), 0, None))
    xy = classical_mds(dist)
    cut = float(np.sqrt(2 * (1 - FAMILY_RHO)))
    clusters = fcluster(linkage(squareform(dist, checks=False), method="average"), cut, criterion="distance")

    info = lib.info.reindex(corr.index)
    coords = pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1], "cluster": clusters,
        "name": info["name"].fillna(label), "category": info.get("economic_category"),
    }, index=corr.index)
    mine = coords.loc[label, "cluster"]
    own = set(result.spec.factor_ids)
    members = [f for f in coords.index[coords["cluster"] == mine] if f != label and f not in own]
    cats = coords.loc[members, "category"].dropna()
    if not members:
        family = "none"
        flags = [
            "On the factor map, your strategy does not sit inside any tight family of published factors "
            f"(average correlation {FAMILY_RHO} or more): it mixes several kinds of signals. Descriptive only."
        ]
    else:
        family = cats.mode().iloc[0] if len(cats) else "unclassified"
        flags = [
            f"On the factor map, your strategy falls in a family of {len(members)} published factors, mostly "
            f"'{family}' signals. This is descriptive: the map is drawn after seeing the results."
        ]
    return FactorMap(coords=coords, family=str(family), family_members=members, flags=flags)


@dataclass
class ZooReport:
    spanning: list[SpanningResult]
    neighbors: pd.DataFrame
    map: FactorMap
    flags: list[str] = field(default_factory=list)


def zoo_report(result: FactorResult, lib: FactorLibrary) -> ZooReport:
    span = spanning(result, lib)
    flags = []
    for s in span:
        if s.overlap_note and not np.isfinite(s.alpha_t):
            flags.append(s.overlap_note)
            continue
        verdict = "remains significant" if abs(s.alpha_t) >= 2 else "is not statistically different from zero"
        flags.append(
            f"Against {s.model} ({s.start} to {s.end}), the alpha is {s.alpha_ann:.1%} per year "
            f"(t = {s.alpha_t:.2f}) and {verdict}; R-squared {s.r2:.2f}."
        )
    nn = nearest_neighbors(result, lib)
    if len(nn):
        top = nn.index[0]
        flags.append(
            f"The most similar published factor is {top} ({nn.loc[top, 'name']}), correlation "
            f"{nn.loc[top, 'correlation']:.2f}. Descriptive only."
        )
    fm = factor_map(result, lib)
    return ZooReport(spanning=span, neighbors=nn, map=fm, flags=flags + fm.flags)


__all__ = ["FactorMap", "SpanningResult", "ZooReport", "factor_map", "nearest_neighbors", "spanning", "zoo_report"]

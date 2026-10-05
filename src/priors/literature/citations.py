"""Citations that Priors is allowed to use.

A citation is valid only if it was *retrieved*: it came from the curated core
library shipped with Priors, or from a live literature search in the current
session (added in the literature module). Citations typed in by a student or
produced from a model's memory are never valid on their own (ARCHITECTURE §2).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RetrievedVia = Literal["core_library", "openalex", "semantic_scholar", "arxiv"]


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doi: str | None = None
    url: str | None = None
    title: str
    authors: str
    year: int
    venue: str | None = None
    retrieved_via: RetrievedVia
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def key(self) -> str:
        return (self.doi or self.url or self.title).lower()

    def short(self) -> str:
        """Author-year reference: 'Fama (1970)', 'George and Hwang (2004)', 'Fama et al. (1969)'."""
        names = [_surname(a) for a in self.authors.split(",") if a.strip()]
        if len(names) >= 3:
            who = f"{names[0]} et al."
        elif len(names) == 2:
            who = f"{names[0]} and {names[1]}"
        else:
            who = names[0] if names else "Anonymous"
        return f"{who} ({self.year})"


def _surname(name: str) -> str:
    last = name.strip().split(" ")[-1]
    return last.title() if last.isupper() and len(last) > 1 else last   # OpenAlex sometimes has 'GEORGE'


def normalize_doi(doi: str) -> str:
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d


@lru_cache(maxsize=1)
def core_library() -> dict[str, dict]:
    """The curated core library, keyed by normalized DOI (or title when there is no DOI)."""
    text = resources.files("priors.literature").joinpath("core_library/papers.csv").read_text(encoding="utf-8")
    out = {}
    for row in csv.DictReader(text.splitlines()):
        key = normalize_doi(row["doi"]) if row["doi"] else row["title"].lower()
        out[key] = row
    return out


def from_core_library(doi_or_title: str) -> Citation:
    """Look up a core-library paper. Raises KeyError if it is not in the library."""
    lib = core_library()
    key = normalize_doi(doi_or_title)
    row = lib.get(key) or lib.get(doi_or_title.lower())
    if row is None:
        raise KeyError(f"{doi_or_title!r} is not in the core library")
    return Citation(
        doi=row["doi"] or None,
        title=row["title"],
        authors=row["authors"],
        year=int(row["year"]),
        venue=row["venue"] or None,
        retrieved_via="core_library",
    )

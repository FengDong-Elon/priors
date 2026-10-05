"""The Librarian's search tools: the curated core library first, then OpenAlex.

Every paper returned here is written to the retrieval log, which is what makes
it citable later. The functions are deterministic code; the Librarian agent
only decides which queries to run (``plan_queries``).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from pydantic import BaseModel, Field

from ..llm import LLM
from .citations import Citation, core_library
from .log import RetrievalLog

OPENALEX = "https://api.openalex.org/works"
_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the to with stock stocks return returns".split()
)


@dataclass
class Paper:
    citation: Citation
    abstract: str | None
    cited_by: int | None
    factor_ids: list[str]
    score: float

    def brief(self) -> str:
        c = self.citation
        return f"{c.authors} ({c.year}). {c.title}. {c.venue or ''}"


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}


def search_core(query: str, k: int = 5) -> list[Paper]:
    """Keyword match over the core library's titles, roles, and factor ids."""
    q = _tokens(query)
    hits = []
    for row in core_library().values():
        fields = f"{row['title']} {row['role']} {row['factor_ids'].replace(';', ' ')}"
        overlap = len(q & _tokens(fields))
        if overlap:
            hits.append((overlap, int(row["cites"] or 0), row))
    hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
    return [
        Paper(
            citation=Citation(
                doi=row["doi"] or None, title=row["title"], authors=row["authors"], year=int(row["year"]),
                venue=row["venue"] or None, retrieved_via="core_library",
            ),
            abstract=None,
            cited_by=int(row["cites"]) if row["cites"] else None,
            factor_ids=[f for f in row["factor_ids"].split(";") if f],
            score=float(overlap),
        )
        for overlap, _, row in hits[:k]
    ]


def abstract_from_index(inv: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return None
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def parse_openalex(payload: dict) -> list[Paper]:
    out = []
    for i, w in enumerate(payload.get("results", [])):
        if not w.get("title") or not w.get("publication_year"):
            continue
        authors = ", ".join(a["author"]["display_name"] for a in w.get("authorships", [])[:6])
        if not authors:
            continue
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        doi = w.get("doi")
        out.append(Paper(
            citation=Citation(
                doi=doi.removeprefix("https://doi.org/") if doi else None,
                url=None if doi else w.get("id"),
                title=w["title"], authors=authors, year=int(w["publication_year"]),
                venue=venue, retrieved_via="openalex",
            ),
            abstract=abstract_from_index(w.get("abstract_inverted_index")),
            cited_by=w.get("cited_by_count"),
            factor_ids=[],
            score=1.0 / (1 + i),
        ))
    return out


def search_openalex(query: str, k: int = 8, min_cites: int = 10, retries: int = 4) -> list[Paper]:
    params = {
        "search": query,
        "filter": f"type:article,cited_by_count:>{min_cites}",
        "per_page": k,
        "select": "id,doi,title,publication_year,authorships,primary_location,cited_by_count,abstract_inverted_index",
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(OPENALEX, params=params, timeout=40)
            r.raise_for_status()
            return parse_openalex(r.json())
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenAlex search failed: {last}")


def fill_abstracts(papers: list[Paper], retries: int = 3) -> None:
    """Add OpenAlex abstracts to papers that have a DOI but no abstract (one batched request)."""
    todo = {p.citation.doi.lower(): p for p in papers if p.abstract is None and p.citation.doi}
    if not todo:
        return
    params = {
        "filter": "doi:" + "|".join("https://doi.org/" + d for d in todo),
        "per_page": len(todo),
        "select": "doi,abstract_inverted_index",
    }
    for attempt in range(retries):
        try:
            r = requests.get(OPENALEX, params=params, timeout=40)
            r.raise_for_status()
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        return  # abstracts are a nice-to-have; the search still succeeds without them
    for w in r.json().get("results", []):
        doi = (w.get("doi") or "").removeprefix("https://doi.org/").lower()
        if doi in todo:
            todo[doi].abstract = abstract_from_index(w.get("abstract_inverted_index"))


def search(query: str, log: RetrievalLog | None = None, k_core: int = 5, k_web: int = 8,
           use_web: bool = True) -> list[Paper]:
    """Core library first, then OpenAlex; duplicates (same DOI) are dropped. Results are logged."""
    papers = search_core(query, k_core)
    if use_web:
        fill_abstracts(papers)
        seen = {p.citation.key for p in papers}
        papers += [p for p in search_openalex(query, k_web) if p.citation.key not in seen]
    if log is not None:
        for p in papers:
            log.record(p.citation, query)
    return papers


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="3 to 5 short academic search queries, most important first.")


LIBRARIAN_SYSTEM = """You are the Librarian in Priors, a teaching tool for evidence-based factor investing.
Turn a student's investment idea into short search queries for an academic literature database.
Use the vocabulary of empirical asset pricing papers (for example "cross-section of expected returns",
"momentum", "book-to-market", "accruals anomaly"). Include one query aimed at evidence against the
idea (for example "replication", "post-publication decline", "transaction costs"). Return 3 to 5 queries."""


def plan_queries(llm: LLM, idea: str) -> list[str]:
    plan = llm.structured("fast", LIBRARIAN_SYSTEM, f"Student's idea:\n{idea}", QueryPlan, max_tokens=1024)
    return [q.strip() for q in plan.queries if q.strip()][:5]


def gather_papers(llm: LLM, idea: str, log: RetrievalLog | None = None, max_papers: int = 14) -> list[Paper]:
    """The Librarian's full search: the idea itself against the core library, then planned queries.

    Core-library hits come first (they are curated and highly cited); web results
    fill the remaining slots. Every returned paper is logged.
    """
    papers = search_core(idea, k=5)
    queries = plan_queries(llm, idea)
    for q in queries:
        papers += search_core(q, k=3)
    seen: set[str] = set()
    core = [p for p in papers if not (p.citation.key in seen or seen.add(p.citation.key))]
    fill_abstracts(core)
    web: list[Paper] = []
    for q in queries:
        try:
            web += [p for p in search_openalex(q, k=4) if p.citation.key not in seen and not seen.add(p.citation.key)]
        except RuntimeError:
            continue  # one failed query should not stop the search
    out = core[: max_papers // 2] + web
    out = out[:max_papers]
    if log is not None:
        for p in out:
            log.record(p.citation, idea)
    return out

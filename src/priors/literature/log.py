"""The retrieval log: which papers a literature search actually returned.

The citation guard (ARCHITECTURE §2) checks citations against this log. A
citation counts as retrieved only if a search in this project returned a paper
with the same DOI (or URL) and title.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .citations import Citation, normalize_doi


def _norm_title(t: str) -> str:
    return "".join(ch for ch in t.lower() if ch.isalnum())


class RetrievalLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, citation: Citation, query: str) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query": query,
            "citation": citation.model_dump(mode="json"),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def citations(self) -> list[Citation]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [Citation.model_validate(json.loads(line)["citation"]) for line in f if line.strip()]

    def is_retrieved(self, c: Citation) -> bool:
        title = _norm_title(c.title)
        for r in self.citations():
            same_id = (
                (c.doi and r.doi and normalize_doi(c.doi) == normalize_doi(r.doi))
                or (c.url and r.url and c.url == r.url)
            )
            if same_id and _norm_title(r.title) == title:
                return True
        return False

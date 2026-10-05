"""The trial ledger (ARCHITECTURE §5.4): an append-only, hash-chained log.

Each row stores the hash of the previous row, so any later edit or deletion
breaks the chain and ``verify()`` reports it. The ledger is the source of the
trial count used to deflate Sharpe ratios.

Events:
* ``register``: a hypothesis was pre-registered.
* ``holdout_open``: the sealed holdout was opened for a hypothesis.
* ``run``: a backtest was run (layer, period, status, fingerprints, headline metrics).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

GENESIS = "0" * 64


class LedgerCorrupted(RuntimeError):
    pass


def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


class Ledger:
    def __init__(self, path: str | Path, clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------- writing

    def append(self, event: str, **fields) -> dict:
        rows = self.rows()
        prev = rows[-1]["hash"] if rows else GENESIS
        row = {
            "seq": len(rows) + 1,
            "ts": self._clock().isoformat(timespec="seconds"),
            "event": event,
            **fields,
            "prev_hash": prev,
        }
        row["hash"] = hashlib.sha256(_canonical(row).encode()).hexdigest()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(_canonical(row) + "\n")
        return row

    # ------------------------------------------------------------- reading

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def verify(self) -> None:
        prev = GENESIS
        for i, row in enumerate(self.rows(), start=1):
            body = {k: v for k, v in row.items() if k != "hash"}
            if row.get("seq") != i or row.get("prev_hash") != prev:
                raise LedgerCorrupted(f"ledger row {i} is out of sequence or was removed")
            if hashlib.sha256(_canonical(body).encode()).hexdigest() != row["hash"]:
                raise LedgerCorrupted(f"ledger row {i} was modified")
            prev = row["hash"]

    def runs(self, layer: str | None = None, period: str | None = None) -> list[dict]:
        return [
            r for r in self.rows()
            if r["event"] == "run"
            and (layer is None or r["layer"] == layer)
            and (period is None or r["period"] == period)
        ]

    def trials(self, layer: str = "factor") -> list[dict]:
        """One entry per distinct in-sample specification tested (first run of each)."""
        seen: dict[str, dict] = {}
        for r in self.runs(layer=layer, period="in_sample"):
            seen.setdefault(r["layer_fingerprint"], r)
        return list(seen.values())

    def trial_count(self, layer: str = "factor") -> int:
        return len(self.trials(layer))

    def holdout_opened(self, hypothesis_id: str) -> bool:
        return any(r["event"] == "holdout_open" and r["hypothesis_id"] == hypothesis_id for r in self.rows())

    def summary(self, layer: str = "factor") -> dict:
        trials = self.trials(layer)
        return {
            "trials": len(trials),
            "preregistered_trials": sum(t["status"] == "preregistered" for t in trials),
            "exploratory_trials": sum(t["status"] == "exploratory" for t in trials),
            "runs": len(self.runs(layer=layer)),
            "registrations": sum(r["event"] == "register" for r in self.rows()),
            "holdouts_opened": sum(r["event"] == "holdout_open" for r in self.rows()),
        }

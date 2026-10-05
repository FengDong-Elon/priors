"""Pre-registration and the run gateway (ARCHITECTURE §5.3, §6).

Every backtest a student sees goes through ``Registry.run_factor``, so that:

* the status of a run (pre-registered or exploratory) is decided here, never by
  the ``status`` field a student writes into the spec;
* every run is written to the trial ledger;
* the sealed holdout can be opened once per pre-registered hypothesis, and never
  for exploratory runs.

A pre-registration record is written once and never changed. Its ``record_hash``
is checked whenever it is read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from ..composite import FactorResult, run_factor_backtest
from ..factors import FactorLibrary
from ..holdout import holdout_start
from ..literature.citations import Citation
from ..literature.log import RetrievalLog
from ..spec.strategy import StrategySpec
from .gate import GateResult, Hypothesis, check_gate, in_core_library
from .ledger import Ledger


class GateFailed(ValueError):
    def __init__(self, result: GateResult):
        super().__init__(result.explain())
        self.result = result


class HoldoutError(PermissionError):
    pass


class RecordCorrupted(RuntimeError):
    pass


SEEN_BEFORE = (
    "This specification was tested before it was registered, so its in-sample results are descriptive, "
    "not a test. The confirmatory test for this hypothesis is the sealed holdout."
)


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class Registration:
    id: str
    created_at: str
    holdout_start: pd.Period
    hypothesis: Hypothesis
    spec: StrategySpec
    in_sample_seen: bool
    record_hash: str

    def explain(self) -> str:
        lines = [
            f"{self.id}: {self.hypothesis.statement}",
            f"Registered {self.created_at}. Holdout sealed from {self.holdout_start}.",
            f"Expected annual return {self.hypothesis.expected_low:.1%} to {self.hypothesis.expected_high:.1%}.",
        ]
        if self.in_sample_seen:
            lines.append(SEEN_BEFORE)
        return "\n".join(lines)


@dataclass
class RunOutcome:
    result: FactorResult
    status: Literal["preregistered", "exploratory"]
    hypothesis_id: str | None
    deviation_of: str | None
    ledger_row: dict
    notes: list[str] = field(default_factory=list)


class Registry:
    """One student's project: registrations, ledger, and the run gateway."""

    def __init__(
        self,
        root: str | Path,
        clock: Callable[[], datetime] | None = None,
        require_evidence: bool = True,
    ):
        self.root = Path(root)
        self.reg_dir = self.root / "registrations"
        self.reg_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.require_evidence = require_evidence
        self.ledger = Ledger(self.root / "ledger.jsonl", clock=self._clock)
        self.retrievals = RetrievalLog(self.root / "retrievals.jsonl")
        self.cards_dir = self.root / "evidence"

    def is_retrieved(self, c: Citation) -> bool:
        """A citation is usable if it is a core-library paper or a search in this project returned it."""
        return in_core_library(c) or self.retrievals.is_retrieved(c)

    def card_exists(self, card_id: str) -> bool:
        return (self.cards_dir / f"{card_id}.json").exists()

    # --------------------------------------------------------- registration

    def register(self, hypothesis: Hypothesis, spec: StrategySpec, advisory: str | None = None) -> Registration:
        gate = check_gate(
            hypothesis, advisory=advisory, is_retrieved=self.is_retrieved,
            card_exists=self.card_exists if self.require_evidence else None,
        )
        if not gate.passed:
            raise GateFailed(gate)
        hid = f"H-{len(list(self.reg_dir.glob('H-*.json'))) + 1:04d}"
        now = self._clock()
        locked = spec.model_copy(update={"hypothesis_id": hid, "status": "preregistered"})
        seen = any(r["layer_fingerprint"] == locked.factor_fingerprint() for r in self.ledger.runs(layer="factor"))
        body = {
            "id": hid,
            "created_at": now.isoformat(timespec="seconds"),
            "holdout_start": str(holdout_start(now.date())),
            "hypothesis": hypothesis.model_dump(mode="json"),
            "spec": locked.model_dump(mode="json"),
            "spec_content_hash": locked.content_hash(),
            "in_sample_seen": seen,
        }
        body["record_hash"] = _hash(body)
        path = self.reg_dir / f"{hid}.json"
        if path.exists():
            raise FileExistsError(path)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
        tmp.replace(path)
        self.ledger.append(
            "register", hypothesis_id=hid, spec_content_hash=body["spec_content_hash"],
            layer_fingerprint=locked.factor_fingerprint(), in_sample_seen=seen,
        )
        return self.get(hid)

    def get(self, hid: str) -> Registration:
        path = self.reg_dir / f"{hid}.json"
        if not path.exists():
            raise KeyError(f"No registration {hid}")
        body = json.loads(path.read_text(encoding="utf-8"))
        stored = body.pop("record_hash")
        if _hash(body) != stored:
            raise RecordCorrupted(f"Registration {hid} was modified after it was written.")
        return Registration(
            id=body["id"],
            created_at=body["created_at"],
            holdout_start=pd.Period(body["holdout_start"], "M"),
            hypothesis=Hypothesis.model_validate(body["hypothesis"]),
            spec=StrategySpec.model_validate(body["spec"]),
            in_sample_seen=body["in_sample_seen"],
            record_hash=stored,
        )

    def registrations(self) -> list[Registration]:
        return [self.get(p.stem) for p in sorted(self.reg_dir.glob("H-*.json"))]

    # ---------------------------------------------------------------- runs

    def _classify(self, spec: StrategySpec) -> tuple[str, Registration | None, str | None]:
        """Return (status, registration, deviation_of) for a spec."""
        if not spec.hypothesis_id:
            return "exploratory", None, None
        try:
            reg = self.get(spec.hypothesis_id)
        except KeyError:
            return "exploratory", None, None
        candidate = spec.model_copy(update={"status": "preregistered"})
        if candidate.content_hash() == reg.spec.content_hash():
            return "preregistered", reg, None
        return "exploratory", None, reg.id

    def run_factor(
        self,
        spec: StrategySpec,
        lib: FactorLibrary,
        period: Literal["in_sample", "holdout"] = "in_sample",
    ) -> RunOutcome:
        status, reg, deviation_of = self._classify(spec)
        notes: list[str] = []
        if deviation_of:
            notes.append(
                f"This spec differs from the one registered as {deviation_of}, so the run is exploratory. "
                "Register a new hypothesis to test the changed version."
            )
        if period == "holdout":
            if reg is None:
                raise HoldoutError("The holdout can only be opened for a pre-registered hypothesis.")
            if self.ledger.holdout_opened(reg.id):
                raise HoldoutError(f"The holdout for {reg.id} has already been opened once.")

        effective = spec.model_copy(update={
            "status": status,
            "hypothesis_id": reg.id if reg else None,
        })
        hs = reg.holdout_start if reg else holdout_start(self._clock().date())
        result = run_factor_backtest(effective, lib, period=period, holdout_from=hs)
        if period == "holdout":   # recorded only once a result exists, so a failed run does not use up the holdout
            self.ledger.append("holdout_open", hypothesis_id=reg.id)
        if reg is not None and reg.in_sample_seen and period == "in_sample":
            notes.append(SEEN_BEFORE)
        result.warnings.extend(notes)

        m = result.metrics["composite"]
        row = self.ledger.append(
            "run",
            layer="factor",
            period=period,
            status=status,
            hypothesis_id=reg.id if reg else None,
            deviation_of=deviation_of,
            factors=effective.factor_ids,
            combination=effective.factor_layer.combination,
            spec_fingerprint=effective.fingerprint(),
            layer_fingerprint=effective.factor_fingerprint(),
            spec_content_hash=effective.content_hash(),
            data_snapshot=result.data_snapshot,
            engine_version=result.engine_version,
            holdout_start=str(hs),
            sample_start=str(result.sample["start"]),
            sample_end=str(result.sample["end"]),
            months=result.sample["months"],
            sharpe=round(m["sharpe"], 6),
            mean_ann=round(m["mean_ann"], 6),
            t_stat_nw=round(m["t_stat_mean_nw"], 6),
        )
        return RunOutcome(
            result=result, status=status, hypothesis_id=reg.id if reg else None,
            deviation_of=deviation_of, ledger_row=row, notes=notes,
        )

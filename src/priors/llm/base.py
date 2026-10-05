"""Provider-neutral interface for the LLM calls Priors makes.

Agents ask for structured output: a Pydantic model class goes in, a validated
instance comes out. Two tiers are configurable (ARCHITECTURE §4):

* ``fast``: high-volume, mechanical work (query planning, explanations).
* ``deep``: judgment-heavy work (evidence synthesis, the Dr. Dong debate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, TypeVar

from pydantic import BaseModel

Tier = Literal["fast", "deep"]
T = TypeVar("T", bound=BaseModel)


class LLMRefusal(RuntimeError):
    """The model declined the request (after any provider-side fallback)."""


class LLMTruncated(RuntimeError):
    """The response hit the token limit before the structured output was complete."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


class LLM(Protocol):
    usage: Usage

    def structured(self, tier: Tier, system: str, user: str, schema: type[T], max_tokens: int = 16000) -> T: ...


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests and offline demos.

    ``responder`` receives (tier, system, user, schema) and returns a dict (or a
    JSON string) that must validate against ``schema``. Every call is recorded.
    """

    responder: Callable[[str, str, str, type[BaseModel]], dict | str]
    usage: Usage = field(default_factory=Usage)
    calls: list[dict] = field(default_factory=list)

    def structured(self, tier: Tier, system: str, user: str, schema: type[T], max_tokens: int = 16000) -> T:
        self.calls.append({"tier": tier, "system": system, "user": user, "schema": schema.__name__})
        out = self.responder(tier, system, user, schema)
        data = json.loads(out) if isinstance(out, str) else out
        self.usage.add(len(system + user) // 4, len(json.dumps(data)) // 4)
        return schema.model_validate(data)

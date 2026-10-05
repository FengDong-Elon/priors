"""Anthropic implementation of the LLM interface.

Structured output uses ``output_config.format`` with a JSON schema generated
from the Pydantic model, and the reply is validated with Pydantic. Requests to
models that support it carry the server-side refusal fallback
(``fallbacks: "default"``), which re-runs a declined request on a suitable
fallback model inside the same call.
"""

from __future__ import annotations

import copy
import json
import os

import anthropic
from pydantic import BaseModel, ValidationError

from .base import LLMRefusal, LLMTruncated, T, Tier, Usage

DEFAULT_MODELS = {"fast": "claude-haiku-4-5", "deep": "claude-opus-5-5"}
# Models that accept the server-side fallback parameter.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept output_config.effort.
EFFORT_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5", "claude-sonnet-5"}


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema for structured output: every object closed and every property required."""
    schema = copy.deepcopy(model.model_json_schema())

    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for key in ("title", "default"):
                node.pop(key, None)
            for v in node.values():
                close(v)
        elif isinstance(node, list):
            for v in node:
                close(v)

    close(schema)
    return schema


class AnthropicLLM:
    def __init__(
        self,
        models: dict[str, str] | None = None,
        effort: dict[str, str] | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        self.models = {
            "fast": os.environ.get("PRIORS_FAST_MODEL", DEFAULT_MODELS["fast"]),
            "deep": os.environ.get("PRIORS_DEEP_MODEL", DEFAULT_MODELS["deep"]),
            **(models or {}),
        }
        self.effort = {"fast": "low", "deep": "high", **(effort or {})}
        self.client = client or anthropic.Anthropic()
        self.usage = Usage()
        self.last_call: dict | None = None

    def structured(self, tier: Tier, system: str, user: str, schema: type[T], max_tokens: int = 16000) -> T:
        model = self.models[tier]
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": strict_schema(schema)}},
        }
        if model in EFFORT_MODELS:
            kwargs["output_config"]["effort"] = self.effort[tier]
        if model in FALLBACK_MODELS:
            response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.create(**kwargs)

        self.usage.add(response.usage.input_tokens, response.usage.output_tokens)
        # The model that actually served the call (a server-side fallback may differ) and its token counts.
        self.last_call = {
            "model": getattr(response, "model", model) or model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        }
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise LLMRefusal(f"The model declined this request (category: {category}).")
        if response.stop_reason == "max_tokens":
            raise LLMTruncated(f"The response hit max_tokens={max_tokens} before it was complete.")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ValueError("The response contained no text block.")
        try:
            return schema.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"The model's output did not match the {schema.__name__} schema: {e}") from e

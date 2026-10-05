from .base import LLM, FakeLLM, LLMRefusal, LLMTruncated, Tier, Usage


def get_llm(provider: str | None = None) -> LLM:
    """The configured LLM. Provider from the argument or PRIORS_LLM_PROVIDER (default: anthropic)."""
    import os

    from dotenv import load_dotenv

    load_dotenv()  # reads .env from the working directory upward; never overrides variables already set
    provider = provider or os.environ.get("PRIORS_LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM()
    raise ValueError(f"Unknown LLM provider {provider!r}. Supported: anthropic.")


__all__ = ["LLM", "FakeLLM", "LLMRefusal", "LLMTruncated", "Tier", "Usage", "get_llm"]

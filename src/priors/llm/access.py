"""How the app reaches the LLM (ARCHITECTURE §13.2).

* **Own key** (public use): the student pastes an Anthropic API key. It lives in
  the browser session only and is never written to disk.
* **Classroom**: the student enters the class code and a student id. Requests go
  to the classroom proxy (``priors proxy``), which holds the instructor's key and
  enforces a per-student weekly budget. The student never sees the real key.
"""

from __future__ import annotations

import anthropic

from .anthropic_llm import AnthropicLLM


def classroom_token(class_code: str, student_id: str) -> str:
    """The credential a student sends to the proxy: '<class code>:<student id>'."""
    code, sid = class_code.strip(), student_id.strip().lower()
    if not code or not sid or ":" in code or ":" in sid:
        raise ValueError("Enter a class code and a student id (no colons).")
    return f"{code}:{sid}"


def make_llm(api_key: str | None = None, proxy_url: str | None = None, class_code: str | None = None,
             student_id: str | None = None) -> AnthropicLLM:
    if proxy_url and class_code:
        client = anthropic.Anthropic(api_key=classroom_token(class_code, student_id or ""), base_url=proxy_url.rstrip("/"))
    elif api_key:
        client = anthropic.Anthropic(api_key=api_key.strip())
    else:
        raise ValueError("Provide your own API key, or a class code and student id for the classroom proxy.")
    return AnthropicLLM(client=client)

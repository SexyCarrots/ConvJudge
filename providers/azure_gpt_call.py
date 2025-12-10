#!/usr/bin/env python3
"""
Reusable helper for calling chat completions via OpenAI's Python SDK.

The helper keeps one global OpenAI client, applies retry/backoff, and exposes
`call_chat_completion` so other scripts can make consistent requests. This is
the original structure used across the repo before we experimented with proxy
or raw HTTP variants.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Type

from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_client = OpenAI()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def _chat_create_with_retry(**kwargs):
    return _client.chat.completions.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def _chat_parse_with_retry(**kwargs):
    return _client.beta.chat.completions.parse(**kwargs)


def call_chat_completion(
    model_type: str,
    messages: Iterable[Mapping[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    reasoning_effort: Optional[str] = None,
    response_model: Optional[Type[BaseModel]] = None,
    use_tracing: bool = True,  # retained for compatibility; no-op here
) -> Any:
    if not model_type:
        raise ValueError("model_type is required (e.g., 'gpt-4o').")

    msgs = [dict(m) for m in messages]
    if not msgs or msgs[0].get("role") != "system":
        msgs.insert(0, {"role": "system", "content": system_prompt})

    payload: dict[str, Any] = {"model": model_type, "messages": msgs}
    if reasoning_effort is None and "gpt-5" in model_type:
        payload["reasoning_effort"] = "minimal"
    elif reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    if response_model is not None:
        result = _chat_parse_with_retry(**payload, response_format=response_model)
        choice = result.choices[0]
        return getattr(choice.message, "parsed", None) or choice.message.content.strip()

    result = _chat_create_with_retry(**payload)
    return result.choices[0].message.content.strip()


def send_prompt(prompt: str) -> str:
    """Quick helper for manual testing."""
    return call_chat_completion(
        model_type="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        use_tracing=False,
    )


if __name__ == "__main__":
    demo = "What is the capital of Australia?"
    print("Prompt:", demo)
    print("Response:", send_prompt(demo))

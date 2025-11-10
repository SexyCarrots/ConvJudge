#!/usr/bin/env python3
"""Reusable helper for calling AWS Bedrock chat models.

This mirrors the shape of azure_gpt_call.py: pass a `model_type` (Bedrock
modelId), a list of messages, and get back the assistant text. It supports the
Bedrock Converse API when available and falls back to invoke_model for
Anthropic Claude models.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Iterable, Mapping, MutableMapping
from pathlib import Path

try:
    import boto3
except Exception as exc:  # pragma: no cover
    print("boto3 is required to call Bedrock: pip install boto3", file=sys.stderr)
    raise
try:
    from dotenv import dotenv_values  # lightweight; used by existing scripts
except Exception:
    dotenv_values = None  # type: ignore


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 4096
MAX_RETRIES = 6
RETRY_DELAY_SECONDS = 5


class MissingConfiguration(ValueError):
    """Raised when required environment variables are missing."""


def _prepare_messages(
    messages: Iterable[Mapping[str, Any]], system_prompt: str
) -> list[MutableMapping[str, Any]]:
    message_list = [dict(msg) for msg in messages]
    if not message_list:
        raise ValueError("messages must contain at least one message")
    if message_list[0].get("role") != "system":
        message_list.insert(0, {"role": "system", "content": system_prompt})
    return message_list


def _has_image_content(message_list: Iterable[Mapping[str, Any]]) -> bool:
    for message in message_list:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "image_url":
                    return True
    return False


def _try_load_dotenv() -> None:
    """Load .env variables into process env if available.

    This supports scenarios where AWS credentials/region are stored in .env.
    """
    if dotenv_values is None:
        return
    root = Path(__file__).resolve().parent
    # Project root is repository root (same folder as this file)
    env_path = root / ".env"
    if not env_path.exists():
        # Also try parent for safety when script is run from subdirs
        env_path = Path(__file__).resolve().parents[0] / ".env"
    if env_path.exists():
        values = dotenv_values(env_path)
        for k, v in values.items():
            if k and v and k not in os.environ:
                os.environ[k] = v


def _build_client():
    """Create a Bedrock Runtime client using env/.env credentials.

    Looks for BEDROCK_REGION, AWS_REGION, or AWS_DEFAULT_REGION, defaulting to
    eu-west-1 to match the helper script.
    """
    _try_load_dotenv()
    region = (
        os.environ.get("BEDROCK_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "eu-west-1"
    )
    runtime = boto3.client("bedrock-runtime", region_name=region)
    return runtime


def _as_converse_messages(message_list: list[Mapping[str, Any]]):
    messages: list[dict[str, Any]] = []
    for msg in message_list:
        role = msg.get("role")
        if role == "system":
            # handled separately by Converse API
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # Only text is supported here
            texts = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            content_text = "\n".join(t for t in texts if t)
        else:
            content_text = str(content) if content is not None else ""
        messages.append({
            "role": role,
            "content": [{"text": content_text}],
        })
    return messages


def _converse(runtime, model_id: str, message_list: list[Mapping[str, Any]], *,
              temperature: float | None) -> str:
    system_text = None
    if message_list and message_list[0].get("role") == "system":
        system_text = message_list[0].get("content") or DEFAULT_SYSTEM_PROMPT
    payload: dict[str, Any] = {
        "modelId": model_id,
        "messages": _as_converse_messages(message_list),
        "inferenceConfig": {
            "maxTokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE if temperature is None else float(temperature),
        },
    }
    if system_text:
        payload["system"] = [{"text": str(system_text)}]

    resp = runtime.converse(**payload)
    try:
        parts = resp["output"]["message"]["content"]
        for part in parts:
            if part.get("text"):
                return part["text"].strip()
    except Exception:  # pragma: no cover
        pass
    raise RuntimeError("Unexpected Converse response shape.")


def _invoke_anthropic(runtime, model_id: str, message_list: list[Mapping[str, Any]], *,
                      temperature: float | None) -> str:
    # Convert to Anthropic Bedrock message format
    system_text = None
    if message_list and message_list[0].get("role") == "system":
        system_text = message_list[0].get("content") or DEFAULT_SYSTEM_PROMPT
    converted: list[dict[str, Any]] = []
    for msg in message_list:
        role = msg.get("role")
        if role == "system":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"]
            text = "\n".join(t for t in texts if t)
        else:
            text = str(content) if content is not None else ""
        converted.append({
            "role": role,
            "content": [{"type": "text", "text": text}],
        })

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": converted,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else float(temperature),
    }
    if system_text:
        body["system"] = system_text

    resp = runtime.invoke_model(modelId=model_id, body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    try:
        contents = payload["content"]
        for part in contents:
            if part.get("type") == "text":
                return (part.get("text") or "").strip()
            if part.get("text"):
                return str(part["text"]).strip()
    except Exception:  # pragma: no cover
        pass
    raise RuntimeError("Unexpected Anthropic invoke_model response shape.")


def call_chat_completion(
    model_type: str,
    messages: Iterable[Mapping[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float | None = None,
) -> str:
    """Send a chat completion request to Bedrock and return assistant text.

    - Tries Converse API first (works for many models, including Claude 3.x)
    - Falls back to invoke_model for Anthropic models
    """
    if not model_type:
        raise MissingConfiguration("model_type (Bedrock modelId) is required")

    message_list = _prepare_messages(messages, system_prompt)
    if _has_image_content(message_list):
        raise NotImplementedError("Image content not supported in this helper")

    runtime = _build_client()

    num_attempts = 0
    while True:
        if num_attempts >= MAX_RETRIES:
            raise RuntimeError("Bedrock request failed after retries.")
        try:
            # Prefer Converse when available
            if hasattr(runtime, "converse"):
                return _converse(runtime, model_type, message_list, temperature=temperature)

            # If Converse is not present on the client, fall back
            if model_type.startswith("anthropic."):
                return _invoke_anthropic(runtime, model_type, message_list, temperature=temperature)

            raise RuntimeError("Runtime does not support converse and no fallback handler for this modelId")

        except Exception as exc:  # pylint: disable=broad-except
            # Retry on transient errors
            print(f"Bedrock request failed: {exc}", file=sys.stderr)
            print(f"Retrying in {RETRY_DELAY_SECONDS}s...", file=sys.stderr)
            time.sleep(RETRY_DELAY_SECONDS)
            num_attempts += 1


def main() -> None:
    # Example model: Claude 3.5 Sonnet on Bedrock (update if needed)
    model_id = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-20b-1:0")
    sample_messages = [
        {"role": "user", "content": "Say hello and mention AWS Bedrock."},
    ]
    try:
        response_text = call_chat_completion(model_id, sample_messages)
    except MissingConfiguration as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(response_text)


if __name__ == "__main__":
    main()

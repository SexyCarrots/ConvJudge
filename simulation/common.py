"""Shared helpers for conversation simulation scripts (dental, scan, airline)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, List

ANALYSIS_MARK = "=== ANALYSIS ==="


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_progress(iterable, total=None, desc: str | None = None, enabled: bool = True):
    """Yield items with optional tqdm progress bar."""
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None

    if not enabled:
        for x in iterable:
            yield x
        return
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=desc or "Progress", unit="conv")
    else:
        count = 0
        total_s = f"/{total}" if total is not None else ""
        for x in iterable:
            count += 1
            label = desc or "Progress"
            print(f"{label}: {count}{total_s}", end="\r", flush=True)
            yield x
        print()


def _get_chat_caller(provider: str):
    """Return the chat completion callable for the configured provider."""
    prov = (provider or "azure").strip().lower()
    if prov == "azure":
        from providers.azure_gpt_call import call_chat_completion

        return call_chat_completion
    if prov == "bedrock":
        from providers.bedrock_gpt_call import call_chat_completion

        return call_chat_completion
    raise ValueError("--provider must be 'azure' or 'bedrock'")


def parse_agent_output(text: str) -> tuple[str, dict[str, Any]]:
    """Split agent output into visible reply and private analysis block."""
    raw = text.strip()
    parts = raw.split(ANALYSIS_MARK, 1)
    reply = parts[0].strip()
    analysis_text = parts[1] if len(parts) > 1 else ""

    analysis: dict[str, Any] = {}
    if analysis_text:
        at = analysis_text.strip()
        m = re.search(r"```(?:json|text)?\s*([\s\S]*?)\s*```", at, re.IGNORECASE)
        if m:
            at = m.group(1).strip()

        def find(pattern: str, flags=re.IGNORECASE) -> str | None:
            mm = re.search(pattern, at, flags)
            return mm.group(1).strip() if mm else None

        correctness = find(r"Correctness\s*:\s*(correct|mistake)")
        guideline = find(r"Guideline\s*:\s*(.+)")
        mistake = find(r"Mistake\s*:\s*(.+)")
        term = find(r"Terminate\s*:\s*(true|false)")
        category = find(r"Category\s*:\s*(.+)")
        key = find(r"Key\s*:\s*(.+)")
        phase = find(r"Phase\s*:\s*(-?\d+)")

        if correctness:
            analysis["correctness"] = correctness.lower()
        if guideline:
            analysis["guideline"] = guideline
        if mistake:
            analysis["mistake"] = mistake
        if term:
            analysis["terminate"] = term.lower() == "true"
        if category:
            analysis["category"] = category
        if key:
            analysis["key"] = key
        if phase is not None:
            try:
                analysis["phase"] = int(phase)
            except Exception:
                pass

    return reply, analysis


def call_models_factory(provider: str, agent_model: str, user_model: str):
    """Return callables for agent and user models configured via provider."""
    call_chat = _get_chat_caller(provider)

    def call_agent_model(system_prompt: str, public_messages: List[dict]) -> tuple[str, dict[str, Any]]:
        messages = [{"role": "system", "content": system_prompt}] + public_messages
        text = call_chat(agent_model, messages)
        return parse_agent_output(text)

    def call_user_model(system_prompt: str, public_messages: List[dict]) -> str:
        messages = (
            [{"role": "system", "content": system_prompt}]
            + public_messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Write ONLY the caller’s next utterance in plain text (no labels). "
                        "It is the caller’s turn now. Do not speak as the agent."
                    ),
                }
            ]
        )
        return call_chat(user_model, messages)

    return call_agent_model, call_user_model

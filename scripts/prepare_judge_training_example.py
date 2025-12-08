#!/usr/bin/env python3
"""Convert one simulated conversation into an SFT training example for the judge.

The produced record mirrors the exact evaluation setting in
evaluate_simulated_conversations.py: the model receives the same
system message and user prompt (guidelines + conversation transcript)
and must reply with the strict JSON listing all violations.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "azure_gpt_call" not in sys.modules:
    stub = types.ModuleType("azure_gpt_call")

    def _unused_call_chat_completion(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[name-defined]
        raise RuntimeError("call_chat_completion is not available in training-data prep.")

    stub.call_chat_completion = _unused_call_chat_completion  # type: ignore[attr-defined]
    sys.modules["azure_gpt_call"] = stub

from evaluate_simulated_conversations import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_conversation,
    format_guidelines,
    infer_category_titles,
    normalize_category,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_violation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw mistake entries to the fields expected during evaluation."""

    def _first(*keys: str) -> str:
        for key in keys:
            if key in item and item[key] not in (None, ""):
                return str(item[key])
        return ""

    turn_index = int(item.get("turn_index", -1))
    category = normalize_category(
        _first(
            "guidance category",
            "guideline_category",
            "guideline type",
            "guideline_type",
        )
    )
    key = _first("guidance key", "guideline_key", "key", "guideline_type")
    phase_raw = _first("guideline_phase", "phase")
    try:
        phase = int(phase_raw)
    except Exception:
        phase = -1
    evidence = _first("evidence", "quote", "text")
    return {
        "turn_index": turn_index,
        "guidance_category": category,
        "guidance_key": key,
        "guideline_phase": phase,
        "evidence": evidence,
    }


def build_training_example(
    convo_data: Dict[str, Any],
    convo_path: Path,
    guidelines_text: str,
) -> Dict[str, Any] | None:
    message_list = convo_data.get("message_list") or convo_data.get("conversation") or []
    truth_list = convo_data.get("mistakes", [])
    if not truth_list:
        return None

    conv_id = convo_path.stem
    conversation_text = format_conversation(message_list)
    user_prompt = build_user_prompt(guidelines_text, conversation_text, conv_id)

    normalized_violations = [
        _normalize_violation(it) for it in sorted(truth_list, key=lambda x: x.get("turn_index", -1))
    ]

    target_payload = {
        "violations": normalized_violations,
    }

    return {
        "conversation_id": conv_id,
        "input_messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "target_response": json.dumps(target_payload, ensure_ascii=False, indent=2),
        "source_conversation_file": str(convo_path),
    }


def _write_records(records: Iterable[Dict[str, Any]], output_path: Path | None) -> None:
    data = [json.dumps(rec, ensure_ascii=False) for rec in records]
    if not data:
        return
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")
    else:
        print("\n".join(data))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare judge training examples.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single simulated conversation JSON file.")
    group.add_argument("--input-dir", help="Directory containing conversation JSON files.")
    parser.add_argument(
        "--guidelines",
        default="guidelines/SCAN/oracle.json",
        help="Guidelines JSON (same as evaluation).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write all processed records as JSONL. Defaults to stdout if omitted.",
    )
    args = parser.parse_args()

    guidelines_path = Path(args.guidelines)
    if not guidelines_path.exists():
        raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")

    oracle = _load_json(guidelines_path)
    cat_titles = infer_category_titles(oracle)
    guidelines_text = format_guidelines(oracle, cat_titles)

    records: List[Dict[str, Any]] = []

    def _handle_file(path: Path) -> None:
        convo_data = _load_json(path)
        record = build_training_example(convo_data, path, guidelines_text)
        if record:
            # Retain guidelines file path for traceability.
            record["guidelines_file"] = str(guidelines_path)
            records.append(record)

    if args.input:
        convo_path = Path(args.input)
        if not convo_path.exists():
            raise FileNotFoundError(f"Conversation file not found: {convo_path}")
        _handle_file(convo_path)
    else:
        conv_dir = Path(args.input_dir)
        if not conv_dir.exists():
            raise FileNotFoundError(f"Conversation directory not found: {conv_dir}")
        for path in sorted(conv_dir.glob("*.json")):
            _handle_file(path)

    output_path = Path(args.output) if args.output else None
    _write_records(records, output_path)
    skipped = " (skipped conversations without mistakes)" if args.input_dir else ""
    if not records:
        print(f"No training examples generated{skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

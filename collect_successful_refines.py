#!/usr/bin/env python3
"""Collect successful refinements and dump before/after USER conversations.

Reads a refine run logs directory (containing *_history.json files produced by
loop_refine_conversations.py) and extracts cases where final_outcome.status is
"success". For each, it reconstructs the final conversation by applying the
recorded changes over the original generated conversation, and writes a single
summary JSON with the USER-only turns before and after.

Usage:
  python collect_successful_refines.py \
    --run-logs-dir dump/refine_logs/run_YYYYMMDD_HHMMSS \
    --output dump/refine_logs/run_YYYYMMDD_HHMMSS/successful_refines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Dict


def load_conversation_messages(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if isinstance(data.get("message_list"), list):
            return list(data["message_list"])  # shallow copy
        if isinstance(data.get("conversation"), list):
            return list(data["conversation"])  # shallow copy
    raise ValueError(f"Unsupported conversation schema in {path}")


def extract_user_msgs(message_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in message_list:
        try:
            if str(msg.get("role", "")).lower() == "user":
                out.append({"turn_index": int(msg.get("turn_index", -1)), "content": msg.get("content", "")})
        except Exception:
            continue
    return out


def apply_change_messages(message_list: list[dict[str, Any]], changes: list[dict[str, Any]]) -> None:
    # In-place application of changes: set new_content for matching user turn_index
    by_idx: Dict[int, str] = {}
    for ch in changes:
        try:
            ti = int(ch.get("turn_index"))
            new = str(ch.get("new_content", ""))
            by_idx[ti] = new
        except Exception:
            continue
    for msg in message_list:
        try:
            if str(msg.get("role", "")).lower() != "user":
                continue
            ti = int(msg.get("turn_index"))
            if ti in by_idx:
                msg["content"] = by_idx[ti]
        except Exception:
            continue


def reconstruct_after_from_history(gen_file: Path, history_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    msgs = load_conversation_messages(gen_file)
    # Apply in chronological order the refine changes present in history
    for ev in history_events:
        if isinstance(ev, dict) and "changes" in ev:
            changes = ev.get("changes") or []
            if isinstance(changes, list) and changes:
                # history stores old_content/new_content; we only need new_content
                apply_change_messages(msgs, changes)
    return msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect successful refines into a single JSON report.")
    parser.add_argument("--run-logs-dir", required=True, help="Directory containing *_history.json logs.")
    parser.add_argument("--output", help="Path to write the summary JSON. Defaults to <run-logs-dir>/successful_refines.json")
    args = parser.parse_args()

    run_dir = Path(args.run_logs_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run logs directory not found: {run_dir}")

    out_path = Path(args.output) if args.output else (run_dir / "successful_refines.json")

    cases: list[dict[str, Any]] = []
    for hist_path in sorted(run_dir.glob("*_history.json")):
        try:
            with hist_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        outcome = payload.get("final_outcome") or {}
        if (outcome.get("status") != "success"):
            continue
        gen_file = payload.get("file")
        if not gen_file:
            continue
        gen_path = Path(gen_file)
        try:
            before_msgs = load_conversation_messages(gen_path)
        except Exception:
            # Skip if original file missing or invalid
            continue
        history = payload.get("history", [])
        after_msgs = reconstruct_after_from_history(gen_path, history)
        before_user = extract_user_msgs(before_msgs)
        after_user = extract_user_msgs(after_msgs)
        # Build one-to-one pairs by turn_index, keep only changed entries
        b_map = {int(x.get("turn_index", -1)): str(x.get("content", "")) for x in before_user}
        a_map = {int(x.get("turn_index", -1)): str(x.get("content", "")) for x in after_user}
        all_idx = sorted(set(b_map.keys()) | set(a_map.keys()))
        pairs = []
        for ti in all_idx:
            before = b_map.get(ti, "")
            after = a_map.get(ti, "")
            if before != after:
                pairs.append({"turn_index": ti, "before": before, "after": after})
        if not pairs:
            continue  # skip cases with no actual user-side changes
        cases.append(
            {
                "history_path": str(hist_path),
                "gen_file": str(gen_path),
                "outcome": outcome,
                "user_pairs": pairs,
            }
        )

    summary = {
        "run_logs_dir": str(run_dir),
        "num_cases": len(cases),
        "cases": cases,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} with {len(cases)} successful cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

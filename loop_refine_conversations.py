#!/usr/bin/env python3
"""Pairwise judge real vs. generated conversations and iteratively refine.

This script compares each generated conversation against a sampled real
conversation using an LLM judge (via azure_gpt_call.py). If the judge
correctly identifies which is real vs. generated, it returns a reason.
We then refine the generated conversation by modifying ONLY the user
messages' content to be more realistic based on the judge's reason.

We repeat this loop until the judge either misclassifies, reports
unknown/uncertainty, or a max iteration cap is reached. The final
refined generated conversation is written to the output directory.

Usage example:
  python loop_refine_conversations.py \
    --model gpt-5 \
    --generated-dir dump/simulated_conv \
    --real-dir data_v0 \
    --output-dir dump/refined_conv \
    --max-iters 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from azure_gpt_call import call_chat_completion


# ------------------------------
# Helpers for parsing/model I/O
# ------------------------------

def extract_first_json(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("json"):
                code = code[len("json") :]
            t = code.strip()
    if not t.startswith("{"):
        s = t.find("{")
        e = t.rfind("}")
        if s != -1 and e != -1 and e > s:
            t = t[s : e + 1]
    return json.loads(t)


def load_conversation_messages(path: Path) -> list[dict[str, Any]]:
    """Load a conversation and return a list of message dicts with keys:
    turn_index, role, content. Supports two common layouts:
      - {"message_list": [ ... ]}
      - {"conversation": [ ... ]}
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "message_list" in data and isinstance(data["message_list"], list):
            return list(data["message_list"])  # shallow copy
        if "conversation" in data and isinstance(data["conversation"], list):
            return list(data["conversation"])  # shallow copy
    raise ValueError(f"Unsupported conversation format in {path}")


def format_conversation(message_list: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for msg in message_list:
        idx = msg.get("turn_index")
        role = msg.get("role", "")
        content = msg.get("content", "")
        lines.append(f"{idx} | {str(role).upper()}: {content}")
    return "\n".join(lines)


def build_judge_prompt_text(conv_a: str, conv_b: str) -> str:
    return (
        "You are a forensic conversation analyst. Two USER–ASSISTANT conversations are provided.\n"
        "Important: judge ONLY the USER messages; ignore all ASSISTANT messages and task outcomes.\n"
        "Exactly one conversation's USER messages are written by a real human, and the other's USER messages are written by an LLM.\n"
        "Judge STRICTLY by speaking style; do NOT consider semantic content, factual correctness, intent, or task success.\n"
        "Do not use domain knowledge; rely purely on stylistic signals.\n\n"
        "Return strict JSON only, no commentary.\n"
        "{\n"
        "  \"which_is_real\": \"A\" | \"B\" | \"unknown\",\n"
        "  \"which_is_generated\": \"A\" | \"B\" | \"unknown\",\n"
        "  \"confidence\": <float 0..1>,\n"
        "  \"reason\": \"<brief overall rationale focused on USER style only>\",\n"
        "  \"generated_user_turn_evidence\": [ { \"turn_index\": <int>, \"reason\": \"<concise USER-style cue>\" } ]\n"
        "}\n\n"
        "Notes:\n"
        "- Provide evidence only for the conversation you believe is GENERATED.\n"
        "- Cite only USER-turn indices from that conversation; keep reasons short and STYLE-ONLY (disfluencies, fillers, contractions, punctuation, rhythm).\n"
        "- Do NOT reference semantic content, facts, domain correctness, or task outcomes in your reasons.\n\n"
        "CONVERSATION A:\n" + conv_a + "\n\n"
        "CONVERSATION B:\n" + conv_b + "\n"
    )


def build_refine_prompt_text(issues_json: str, conv_text: str) -> str:
    return (
        "You are improving the realism of ONLY the USER messages in the provided conversation.\n"
        "Keep number of turns, indices, and ALL assistant messages unchanged.\n"
        "Edit ONLY the USER turns listed in TARGET_USER_TURNS below.\n"
        "Important: The reasons describe artifacts that made these USER turns look LLM‑generated (fake).\n"
        "Goal: intentionally reduce polish so the USER sounds more spontaneous and imperfect. Prefer slightly messy, human style over clean, formal writing.\n"
        "Your job is to CORRECT/REDUCE the listed artifacts — do NOT amplify them — but produce output that feels less polished than the original.\n"
        "Allowed edits (keep meaning intact): reorder clauses; split/merge sentences; vary length and rhythm; add light disfluencies and hedges; brief self-corrections; contractions/colloquialisms; remove template/list‑like phrasing; minor non‑critical typos on common words.\n"
        "Required strength: each edited USER turn should include at least TWO such human cues. Avoid purely cosmetic changes.\n"
        "Preserve the user’s original intent and keep all factual tokens EXACT (names, dates, times, flight numbers, addresses, phone numbers, emails, IDs). You may rearrange around them but must not alter them.\n"
        "Do not add or remove turns. Do not change assistant content. Do not introduce new facts or new PII.\n\n"
        "Return strict JSON only with replacements of user content by turn_index. Only include indices you actually changed.\n"
        "{\n"
        "  \"replacements\": [\n"
        "    { \"turn_index\": <int>, \"new_content\": \"<string>\" }\n"
        "  ]\n"
        "}\n\n"
        "TARGET_USER_TURNS (indices and reasons to fix):\n" + issues_json + "\n\n"
        "CONVERSATION (for reference; rewrite ONLY the listed user turns):\n" + conv_text + "\n"
    )


@dataclass
class JudgeResult:
    which_is_real: str  # "A" | "B" | "unknown"
    which_is_generated: str  # "A" | "B" | "unknown"
    confidence: float
    reason: str
    evidence_generated: list[dict[str, Any]]


def call_judge(model: str, conv_a_text: str, conv_b_text: str) -> JudgeResult:
    system = "You are a careful, concise, and skeptical analyst."
    user = build_judge_prompt_text(conv_a_text, conv_b_text)
    resp_text = call_chat_completion(model, [{"role": "system", "content": system}, {"role": "user", "content": user}])
    try:
        d = extract_first_json(resp_text)
    except Exception:
        # Fallback structure if parsing fails
        d = {"which_is_real": "unknown", "which_is_generated": "unknown", "confidence": 0.0, "reason": resp_text[:300], "generated_user_turn_evidence": []}
    which_real = str(d.get("which_is_real", "unknown")).strip().upper()
    which_gen = str(d.get("which_is_generated", "unknown")).strip().upper()
    try:
        conf = float(d.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    reason = str(d.get("reason", "")).strip()
    ev = d.get("generated_user_turn_evidence")
    if ev is None:
        # Back-compat: if the model returned per-side evidence, pick the generated side only.
        ev_all = d.get("user_turn_evidence") or {}
        sel_side = which_gen if which_gen in ("A", "B") else ""
        ev = (ev_all or {}).get(sel_side, [])
    def _normalize_ev(x: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if isinstance(x, list):
            for it in x:
                try:
                    ti = int(it.get("turn_index"))
                    rs = str(it.get("reason", "")).strip()
                    out.append({"turn_index": ti, "reason": rs})
                except Exception:
                    continue
        return out
    ev_gen = _normalize_ev(ev or [])
    return JudgeResult(which_is_real=which_real, which_is_generated=which_gen, confidence=conf, reason=reason, evidence_generated=ev_gen)


def call_refine(model: str, issues: list[dict[str, Any]], conv_text: str) -> list[dict[str, Any]]:
    system = (
        "You aggressively humanize only the specified USER turns. Prefer slightly messy, spontaneous speech over polished writing. "
        "Reduce LLM-like artifacts and allow substantial paraphrase (reorder/split/merge), but preserve intent and keep factual tokens (numbers/emails/IDs) exact. "
        "For each edited turn, include at least two human cues (fillers, hedges, self-correction, repetition, ellipses, varied punctuation)."
    )
    # Serialize issues as compact JSON for the prompt
    try:
        issues_json = json.dumps(issues, ensure_ascii=False)
    except Exception:
        issues_json = "[]"
    user = build_refine_prompt_text(issues_json, conv_text)
    resp_text = call_chat_completion(model, [{"role": "system", "content": system}, {"role": "user", "content": user}])
    try:
        d = extract_first_json(resp_text)
        repl = d.get("replacements", []) or []
        # Validate schema shallowly
        out: list[dict[str, Any]] = []
        for it in repl:
            if not isinstance(it, dict):
                continue
            if "turn_index" in it and "new_content" in it:
                out.append({"turn_index": int(it["turn_index"]), "new_content": str(it["new_content"])})
        return out
    except Exception:
        # If parsing fails, return no changes
        return []


def apply_user_replacements(
    message_list: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    """Apply in-place user content replacements.

    Returns a tuple of (num_changes_applied, changes_detail) where
    changes_detail is a list of {turn_index, old_content, new_content}.
    """
    by_idx = {int(r["turn_index"]): r["new_content"] for r in replacements}
    changes = 0
    detail: list[dict[str, Any]] = []
    for msg in message_list:
        try:
            if str(msg.get("role", "")).lower() == "user":
                ti = int(msg.get("turn_index"))
                if ti in by_idx:
                    old = msg.get("content", "")
                    new = by_idx[ti]
                    if new and new != old:
                        msg["content"] = new
                        changes += 1
                        detail.append({"turn_index": ti, "old_content": old, "new_content": new})
        except Exception:
            continue
    return changes, detail


def conversation_to_text_for_prompt(message_list: list[dict[str, Any]]) -> str:
    # Use a stable textual form to avoid JSON bloat in prompts
    return format_conversation(message_list)


def refine_loop_for_file(
    model: str,
    gen_path: Path,
    real_paths: list[Path],
    max_iters: int,
    conf_threshold: float,
    rng: random.Random,
    logs_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the refine loop for a single generated conversation file.

    Returns a tuple of (final_conversation_dict, history_events).
    Each history event records judge and refine info per iteration.
    """
    with gen_path.open("r", encoding="utf-8") as f:
        gen_data = json.load(f)

    # Derive messages
    gen_messages = gen_data.get("message_list")
    if not isinstance(gen_messages, list):
        # Fallback for alternative schema
        gen_messages = load_conversation_messages(gen_path)
        # Ensure we store back under message_list for consistency
        gen_data["message_list"] = gen_messages

    history: list[dict[str, Any]] = []
    status: str | None = None
    status_detail: str | None = None
    iterations_run = 0

    for it in range(max_iters):
        iterations_run = it + 1
        real_path = rng.choice(real_paths)
        real_messages = load_conversation_messages(real_path)

        # Compose A/B order randomly to avoid position bias
        order = rng.choice(["AR", "RA"])  # A=gen,B=real or A=real,B=gen
        if order == "AR":
            a_text = conversation_to_text_for_prompt(gen_messages)
            b_text = conversation_to_text_for_prompt(real_messages)
            a_is_real = False
        else:
            a_text = conversation_to_text_for_prompt(real_messages)
            b_text = conversation_to_text_for_prompt(gen_messages)
            a_is_real = True

        judge = call_judge(model, a_text, b_text)

        # Determine correctness
        if judge.which_is_real not in {"A", "B"}:
            correct = False
            distinguishable = False
        else:
            predicted_real_is_a = judge.which_is_real == "A"
            actual_real_is_a = a_is_real
            correct = predicted_real_is_a == actual_real_is_a
            distinguishable = True

        event = {
            "iteration": it,
            "gen_file": str(gen_path),
            "real_file": str(real_path),
            "order": order,
            "judge": {
                "which_is_real": judge.which_is_real,
                "which_is_generated": judge.which_is_generated,
                "confidence": judge.confidence,
                "reason": judge.reason,
                "target_side": judge.which_is_generated,
                "generated_user_turn_evidence": judge.evidence_generated,
                "correct": bool(correct),
                "distinguishable": bool(distinguishable),
            },
        }
        history.append(event)
        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)
            with (logs_dir / f"{gen_path.stem}_iter{it}_judge.json").open("w", encoding="utf-8") as f:
                json.dump(event, f, ensure_ascii=False, indent=2)

        # Stopping conditions: judge is not confident or misclassifies
        if not distinguishable or not correct or judge.confidence < conf_threshold:
            # Treat as success stopping condition
            status = "success"
            if not distinguishable:
                status_detail = "not_distinguishable"
            elif not correct:
                status_detail = "misclassified"
            else:
                status_detail = "low_confidence"
            break

        # Judge correctly distinguished; refine user messages of generated
        # Use only the issues for the generated side
        gen_issues = judge.evidence_generated
        # If no indexed issues were provided, there's nothing targeted to fix
        if not gen_issues:
            status = "stalled_no_issues"
            status_detail = "no_indexed_user_turns_from_judge"
            break
        refine_replacements = call_refine(model, gen_issues, conversation_to_text_for_prompt(gen_messages))
        # Filter replacements strictly to the specified target indices
        allowed_indices = {int(x.get("turn_index")) for x in gen_issues if isinstance(x, dict) and "turn_index" in x}
        refine_replacements = [r for r in refine_replacements if int(r.get("turn_index")) in allowed_indices]
        num_changes, change_records = apply_user_replacements(gen_messages, refine_replacements)

        event_refine = {
            "iteration": it,
            "applied_user_changes": num_changes,
            "targets": gen_issues,
            "changes": change_records,
        }
        history.append(event_refine)
        if logs_dir:
            with (logs_dir / f"{gen_path.stem}_iter{it}_refine.json").open("w", encoding="utf-8") as f:
                json.dump(event_refine, f, ensure_ascii=False, indent=2)

        # If no changes were applied, nothing more to refine; stop early
        if num_changes == 0:
            status = "stalled_no_changes"
            status_detail = "model_proposed_no_effective_replacements"
            break

    # If loop exhausted without reaching success or early stop, mark as exceeded
    if status is None:
        status = "exceeded_max_iters"
        status_detail = "reached_iteration_cap_without_success"

    final_outcome = {
        "status": status,
        "detail": status_detail,
        "iterations_run": iterations_run,
        "max_iters": max_iters,
        "confidence_threshold": conf_threshold,
    }

    # Store back updated messages and outcome
    gen_data["message_list"] = gen_messages
    gen_data["refine_outcome"] = final_outcome
    return gen_data, history


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loop-refine generated conversations against real samples.")
    parser.add_argument("--model", default="gpt-4o", help="Model/deployment name to use for judge/refine.")
    parser.add_argument(
        "--generated-dir",
        default="dump/simulated_conv_dental",
        help="Directory OR single JSON file of generated conversations.",
    )
    parser.add_argument(
        "--real-dir",
        default="dump/real_conv",
        help="Directory OR single JSON file of real conversations.",
    )
    parser.add_argument("--output-dir", default="dump/refined_conv", help="Directory to write refined conversations.")
    parser.add_argument("--logs-dir", default="dump/refine_logs", help="Directory to write per-iteration logs.")
    parser.add_argument("--max-iters", type=int, default=5, help="Max refinement iterations per conversation.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Minimum judge confidence to keep refining.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling/order.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of generated files to process (0=all).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    gen_dir = Path(args.generated_dir)
    real_dir = Path(args.real_dir)
    out_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generated path not found: {gen_dir}")
    if not real_dir.exists():
        raise FileNotFoundError(f"Real path not found: {real_dir}")

    rng = random.Random(args.seed)

    # Accept a single file or a directory for both inputs
    if gen_dir.is_file():
        gen_files = [gen_dir]
    else:
        gen_files = sorted([p for p in gen_dir.glob("*.json") if p.is_file()])
    if real_dir.is_file():
        real_files = [real_dir]
    else:
        real_files = sorted([p for p in real_dir.glob("*.json") if p.is_file()])
    if not gen_files:
        print(f"No generated conversations found in {gen_dir}")
        return 0
    if not real_files:
        print(f"No real conversations found in {real_dir}")
        return 0

    # Timestamped run dir under logs for traceability
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_logs_dir = logs_dir / f"run_{timestamp}"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    # Optional limit
    if args.limit and args.limit > 0:
        gen_files = gen_files[: args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, gen_path in enumerate(gen_files, 1):
        try:
            final_data, history = refine_loop_for_file(
                model=args.model,
                gen_path=gen_path,
                real_paths=real_files,
                max_iters=args.max_iters,
                conf_threshold=args.confidence_threshold,
                rng=rng,
                logs_dir=run_logs_dir,
            )

            # Save final refined conversation JSON with same basename
            out_path = out_dir / gen_path.name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            # Save final history for the file, include outcome status
            final_outcome = final_data.get("refine_outcome", {})
            with (run_logs_dir / f"{gen_path.stem}_history.json").open("w", encoding="utf-8") as f:
                json.dump({"file": str(gen_path), "final_outcome": final_outcome, "history": history}, f, ensure_ascii=False, indent=2)

            print(f"[{i}/{len(gen_files)}] Refined {gen_path.name} -> {out_path}")
        except Exception as exc:
            print(f"[{i}/{len(gen_files)}] Failed on {gen_path.name}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

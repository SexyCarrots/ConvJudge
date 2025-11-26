#!/usr/bin/env python3
"""Evaluate LLM-detected violations on simulated conversations.

Reads conversations from dump/simulated_conv (or a provided directory),
asks a model (default gpt-5) to identify guideline violations using only
the oracle guidelines and the message list (with turn indices), and
computes precision/recall/F1 against the ground-truth mistakes embedded
in each simulated file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from azure_gpt_call import call_chat_completion


SYSTEM_PROMPT = (
    "You are a precise compliance evaluator for Celestar Air. "
    "Given the official guidelines and a conversation transcript, identify exactly which assistant turns violate which guideline. "
    "Only assistant (agent) turns can violate guidelines; never mark user turns. "
    "Use the exact Category titles and Keys from the guidelines when possible. "
    "For Category 2 (Intent Triggered), include the specific Phase number (1-based). "
    "Return only the requested strict JSON format with no extra text."
)


@dataclass(frozen=True)
class VKey:
    turn_index: int
    category: str
    key: str
    phase: int

    @classmethod
    def from_pred(cls, item: dict[str, Any]) -> "VKey":
        turn_index = int(item.get("turn_index", -1))
        raw_cat = str(
            item.get("guidance_category")
            or item.get("category")
            or item.get("guidance category", "")
        ).strip()
        norm_cat = normalize_category(raw_cat)
        key = str(item.get("guidance_key") or item.get("key") or item.get("guidance key", "")).strip()
        phase_raw = item.get("guideline_phase", item.get("phase", -1))
        try:
            phase = int(phase_raw)
        except Exception:
            phase = -1
        return cls(turn_index, norm_cat, key, phase)

    @classmethod
    def from_truth(cls, item: dict[str, Any]) -> "VKey":
        turn_index = int(item.get("turn_index", -1))
        cat = normalize_category(str(item.get("guidance category", "")).strip())
        key = str(item.get("guidance key", "")).strip()
        phase = int(item.get("guideline_phase", -1))
        return cls(turn_index, cat, key, phase)


def normalize_category(cat: str) -> str:
    """Map loose category mentions to canonical oracle titles.

    Accepts inputs like 'Category 1', 'Cat1', 'Universal Compliance', etc.
    """
    c = (cat or "").strip().lower()
    if not c:
        return ""
    # Numeric hints
    if c.startswith("category 1") or c.startswith("cat 1") or c.startswith("cat1"):
        return "Category 1: Universal Compliance"
    if c.startswith("category 2") or c.startswith("cat 2") or c.startswith("cat2"):
        return "Category 2: Intent Triggered Guidelines"
    if c.startswith("category 3") or c.startswith("cat 3") or c.startswith("cat3"):
        return "Category 3: Condition Triggered Guidelines"
    # Textual hints
    if "universal" in c or "compliance" in c:
        return "Category 1: Universal Compliance"
    if "intent" in c or "triggered" in c:
        return "Category 2: Intent Triggered Guidelines"
    if "condition" in c or "conditional" in c:
        return "Category 3: Condition Triggered Guidelines"
    return cat


def format_guidelines(oracle: dict[str, Any]) -> str:
    cat1 = oracle.get("Category 1: Universal Compliance", {}) or {}
    cat2 = oracle.get("Category 2: Intent Triggered Guidelines", {}) or {}
    cat3 = oracle.get("Category 3: Condition Triggered Guidelines", {}) or {}

    lines: list[str] = []
    lines.append("CATEGORY 1: Universal Compliance (Keys must match exactly)")
    for k, v in cat1.items():
        lines.append(f"- Key: {k}\n  Text: {v}")
    lines.append("")
    lines.append("CATEGORY 2: Intent Triggered Guidelines (Keys are intents; include Phase number)")
    if isinstance(cat2, dict):
        for intent, phases in cat2.items():
            lines.append(f"- Intent Key: {intent}")
            if isinstance(phases, list):
                for i, p in enumerate(phases, 1):
                    lines.append(f"  Phase {i}: {p}")
    lines.append("")
    lines.append("CATEGORY 3: Condition Triggered Guidelines (Keys must match exactly)")
    for k, v in cat3.items():
        lines.append(f"- Key: {k}\n  Text: {v}")
    return "\n".join(lines)


def format_conversation(message_list: Sequence[dict[str, Any]]) -> str:
    out: list[str] = []
    for msg in message_list:
        idx = msg.get("turn_index")
        role = msg.get("role", "")
        content = msg.get("content", "")
        out.append(f"{idx} | {role.upper()}: {content}")
    return "\n".join(out)


def build_user_prompt(guidelines_text: str, conversation_text: str, conv_id: str) -> str:
    return (
        "TASK:\n"
        "Using the guidelines, identify every assistant (agent) turn that violates a guideline.\n"
        "Only mark assistant turns; never mark user turns.\n"
        "- Use exact field names and values as they appear in the guidelines.\n"
        "- For Category 1 or 3, set guideline_phase to -1.\n"
        "- For Category 2, set guidance_key to the intent name and guideline_phase to the Phase number.\n"
        "RESPONSE (strict JSON only):\n"
        "{\n"
        f"  \"conversation_id\": \"{conv_id}\",\n"
        "  \"violations\": [\n"
        "    {\n"
        "      \"turn_index\": <int>,\n"
        "      \"guidance_category\": \"<string>\",\n"
        "      \"guidance_key\": \"<string>\",\n"
        "      \"guideline_phase\": <int>,\n"
        "      \"evidence\": \"<short quote from the assistant message>\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "GUIDELINES:\n" + guidelines_text + "\n\n"
        "CONVERSATION:\n" + conversation_text
    )


def extract_first_json(text: str) -> dict[str, Any]:
    t = text.strip()
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


def compute_metrics(pred: set[VKey], truth: set[VKey]) -> tuple[float, float, float, list[VKey], list[VKey], list[VKey]]:
    tp_set = pred & truth
    fp_set = pred - truth
    fn_set = truth - pred
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, sorted(tp_set, key=lambda x: x.turn_index), sorted(fp_set, key=lambda x: x.turn_index), sorted(fn_set, key=lambda x: x.turn_index)


def compute_turn_metrics(pred: set[VKey], truth: set[VKey]) -> tuple[float, float, float, list[int], list[int], list[int]]:
    """Looser metrics that only care about picking the right turn index."""
    pred_turns = {p.turn_index for p in pred}
    truth_turns = {t.turn_index for t in truth}
    tp_set = pred_turns & truth_turns
    fp_set = pred_turns - truth_turns
    fn_set = truth_turns - pred_turns
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, sorted(tp_set), sorted(fp_set), sorted(fn_set)


def evaluate_one(model: str, oracle: dict[str, Any], convo_path: Path, out_dir: Path) -> dict[str, Any]:
    with convo_path.open("r", encoding="utf-8") as f:
        convo = json.load(f)

    message_list = convo.get("message_list", [])
    truth_list = convo.get("mistakes", [])
    conv_id = convo_path.stem

    guidelines_text = format_guidelines(oracle)
    conversation_text = format_conversation(message_list)
    user_prompt = build_user_prompt(guidelines_text, conversation_text, conv_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response_text = call_chat_completion(model, messages)
    try:
        resp = extract_first_json(response_text)
    except Exception:
        # If the model returns junk, fail this sample cleanly with empty prediction
        resp = {"conversation_id": conv_id, "violations": []}

    pred_items = resp.get("violations", []) or []
    pred_keys = {VKey.from_pred(it) for it in pred_items}
    truth_keys = {VKey.from_truth(it) for it in truth_list}

    precision, recall, f1, tp, fp, fn = compute_metrics(pred_keys, truth_keys)
    turn_precision, turn_recall, turn_f1, turn_tp, turn_fp, turn_fn = compute_turn_metrics(pred_keys, truth_keys)

    payload = {
        "conversation_file": str(convo_path),
        "model": model,
        "predicted": pred_items,
        "ground_truth": truth_list,
        "true_positive": [it.__dict__ for it in tp],
        "false_positive": [it.__dict__ for it in fp],
        "false_negative": [it.__dict__ for it in fn],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "turn_only_precision": turn_precision,
        "turn_only_recall": turn_recall,
        "turn_only_f1": turn_f1,
        "turn_true_positive": turn_tp,
        "turn_false_positive": turn_fp,
        "turn_false_negative": turn_fn,
        "model_response_text": response_text,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{convo_path.stem}_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate simulated conversations with an LLM.")
    parser.add_argument("--model", default="gpt-5", help="Model/deployment name for evaluation.")
    parser.add_argument("--guidelines", default="guidelines/airlines/oracle.json", help="Path to oracle guidelines.")
    parser.add_argument("--data-dir", default="dump/simulated_conv", help="Directory of simulated conversations.")
    parser.add_argument("--output-dir", default="dump/eval_simulated", help="Directory to write evaluation outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N conversations.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    guidelines_path = Path(args.guidelines)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not guidelines_path.exists():
        raise FileNotFoundError(f"Guidelines not found: {guidelines_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    with guidelines_path.open("r", encoding="utf-8") as f:
        oracle = json.load(f)

    convo_files = sorted(data_dir.glob("*.json"))
    if not convo_files:
        print(f"No simulated conversations in {data_dir}")
        return 0
    if args.limit and args.limit > 0:
        convo_files = convo_files[: args.limit]

    # Create a run-specific directory per model
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = str(args.model or "model")
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name)
    run_dir = output_dir / f"{sanitized}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for path in convo_files:
        try:
            res = evaluate_one(args.model, oracle, path, run_dir)
            results.append(res)
            print(
                f"Evaluated {path.name}: "
                f"P={res['precision']:.2f} R={res['recall']:.2f} F1={res['f1']:.2f} | "
                f"Turn-only P={res['turn_only_precision']:.2f} R={res['turn_only_recall']:.2f} F1={res['turn_only_f1']:.2f}"
            )
        except Exception as exc:  # keep going
            print(f"Failed on {path}: {exc}")

    if not results:
        return 0

    # Macro metrics
    P = [r["precision"] for r in results]
    R = [r["recall"] for r in results]
    F = [r["f1"] for r in results]
    TP = [r["turn_only_precision"] for r in results]
    TR = [r["turn_only_recall"] for r in results]
    TF = [r["turn_only_f1"] for r in results]
    macro = {
        "macro_precision": sum(P) / len(P),
        "macro_recall": sum(R) / len(R),
        "macro_f1": sum(F) / len(F),
        "macro_turn_only_precision": sum(TP) / len(TP),
        "macro_turn_only_recall": sum(TR) / len(TR),
        "macro_turn_only_f1": sum(TF) / len(TF),
        "num_samples": len(results),
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(macro, f, ensure_ascii=False, indent=2)
    # Summary CSV similar to evaluate_gpt_model.py
    csv_path = run_dir / "evaluation_summary.csv"
    columns = [
        "conversation_file",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "turn_precision",
        "turn_recall",
        "turn_f1",
        "turn_tp",
        "turn_fp",
        "turn_fn",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in results:
            tp = len(r.get("true_positive", []))
            fp = len(r.get("false_positive", []))
            fn = len(r.get("false_negative", []))
            turn_tp = len(r.get("turn_true_positive", []))
            turn_fp = len(r.get("turn_false_positive", []))
            turn_fn = len(r.get("turn_false_negative", []))
            writer.writerow(
                {
                    "conversation_file": r.get("conversation_file", ""),
                    "precision": f"{r['precision']:.6f}",
                    "recall": f"{r['recall']:.6f}",
                    "f1": f"{r['f1']:.6f}",
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "turn_precision": f"{r['turn_only_precision']:.6f}",
                    "turn_recall": f"{r['turn_only_recall']:.6f}",
                    "turn_f1": f"{r['turn_only_f1']:.6f}",
                    "turn_tp": turn_tp,
                    "turn_fp": turn_fp,
                    "turn_fn": turn_fn,
                }
            )
        writer.writerow(
            {
                "conversation_file": "AVERAGE",
                "precision": f"{macro['macro_precision']:.6f}",
                "recall": f"{macro['macro_recall']:.6f}",
                "f1": f"{macro['macro_f1']:.6f}",
                "tp": sum(len(r.get("true_positive", [])) for r in results),
                "fp": sum(len(r.get("false_positive", [])) for r in results),
                "fn": sum(len(r.get("false_negative", [])) for r in results),
                "turn_precision": f"{macro['macro_turn_only_precision']:.6f}",
                "turn_recall": f"{macro['macro_turn_only_recall']:.6f}",
                "turn_f1": f"{macro['macro_turn_only_f1']:.6f}",
                "turn_tp": sum(len(r.get("turn_true_positive", [])) for r in results),
                "turn_fp": sum(len(r.get("turn_false_positive", [])) for r in results),
                "turn_fn": sum(len(r.get("turn_false_negative", [])) for r in results),
            }
        )
    print(
        "Macro "
        f"P={macro['macro_precision']:.2f} R={macro['macro_recall']:.2f} F1={macro['macro_f1']:.2f} | "
        f"Turn-only P={macro['macro_turn_only_precision']:.2f} R={macro['macro_turn_only_recall']:.2f} F1={macro['macro_turn_only_f1']:.2f} "
        f"over {macro['num_samples']} files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

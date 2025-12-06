#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
    c = (cat or "").strip().lower()
    if not c:
        return ""
    if c.startswith(("category 1", "cat 1", "cat1")):
        return "Category 1: Universal Compliance"
    if c.startswith(("category 2", "cat 2", "cat2")):
        return "Category 2: Intent Triggered Guidelines"
    if c.startswith(("category 3", "cat 3", "cat3")):
        return "Category 3: Condition Triggered Guidelines"
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


def compute_metrics(pred: set[VKey], truth: set[VKey]):
    tp_set = pred & truth
    fp_set = pred - truth
    fn_set = truth - pred
    tp = len(tp_set); fp = len(fp_set); fn = len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, sorted(tp_set, key=lambda x: x.turn_index), sorted(fp_set, key=lambda x: x.turn_index), sorted(fn_set, key=lambda x: x.turn_index)


def compute_turn_metrics(pred: set[VKey], truth: set[VKey]):
    pred_turns = {p.turn_index for p in pred}
    truth_turns = {t.turn_index for t in truth}
    tp_set = pred_turns & truth_turns
    fp_set = pred_turns - truth_turns
    fn_set = truth_turns - pred_turns
    tp = len(tp_set); fp = len(fp_set); fn = len(fn_set)
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


# ---------- continuation helpers (unchanged logic, scoped per model) ----------

def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_existing_results(out_dir: Path):
    """Load *_eval.json from the given dir; return (results, evaluated_basenames)."""
    results: list[dict[str, Any]] = []
    evaluated_names: set[str] = set()
    if not out_dir.exists():
        return results, evaluated_names
    for path in sorted(out_dir.glob("*_eval.json")):
        data = _safe_read_json(path)
        if not data:
            continue
        results.append(data)
        conv_path = data.get("conversation_file") or ""
        if conv_path:
            evaluated_names.add(Path(conv_path).name)
    return results, evaluated_names


def _compute_macro(results: list[dict[str, Any]]) -> dict[str, Any]:
    P = [r.get("precision", 0.0) for r in results]
    R = [r.get("recall", 0.0) for r in results]
    F = [r.get("f1", 0.0) for r in results]
    TP = [r.get("turn_only_precision", 0.0) for r in results]
    TR = [r.get("turn_only_recall", 0.0) for r in results]
    TF = [r.get("turn_only_f1", 0.0) for r in results]
    return {
        "macro_precision": sum(P) / len(P) if P else 0.0,
        "macro_recall": sum(R) / len(R) if R else 0.0,
        "macro_f1": sum(F) / len(F) if F else 0.0,
        "macro_turn_only_precision": sum(TP) / len(TP) if TP else 0.0,
        "macro_turn_only_recall": sum(TR) / len(TR) if TR else 0.0,
        "macro_turn_only_f1": sum(TF) / len(TF) if TF else 0.0,
        "num_samples": len(results),
    }


def _write_summaries(output_dir: Path, results: list[dict[str, Any]]) -> None:
    macro = _compute_macro(results)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(macro, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "evaluation_summary.csv"
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
                    "precision": f"{r.get('precision', 0.0):.6f}",
                    "recall": f"{r.get('recall', 0.0):.6f}",
                    "f1": f"{r.get('f1', 0.0):.6f}",
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "turn_precision": f"{r.get('turn_only_precision', 0.0):.6f}",
                    "turn_recall": f"{r.get('turn_only_recall', 0.0):.6f}",
                    "turn_f1": f"{r.get('turn_only_f1', 0.0):.6f}",
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate simulated conversations with an LLM.")
    parser.add_argument("--model", default="gpt-5", help="Model/deployment name for evaluation.")
    parser.add_argument("--guidelines", default="guidelines/dental/oracle.json", help="Path to oracle guidelines.")
    parser.add_argument("--data-dir", default="dump/simulated_dental_500_v1", help="Directory of simulated conversations.")
    parser.add_argument("--output-dir", default="dump/eval_dental_500_v1", help="Base directory to write evaluation outputs (per-model subfolder will be used).")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N remaining conversations.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    guidelines_path = Path(args.guidelines)
    data_dir = Path(args.data_dir)
    base_output_dir = Path(args.output_dir)
    if not guidelines_path.exists():
        raise FileNotFoundError(f"Guidelines not found: {guidelines_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    with guidelines_path.open("r", encoding="utf-8") as f:
        oracle = json.load(f)

    # ---- NEW: per-model subdirectory inside the fixed output dir ----
    model_name = str(args.model or "model")
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name)
    output_dir = base_output_dir / sanitized

    # Load existing results for THIS model
    results, evaluated_names = load_existing_results(output_dir)
    if evaluated_names:
        print(f"Loaded {len(results)} existing result files from {output_dir}.")

    # Gather data files
    all_convo_files = sorted(data_dir.glob("*.json"))
    if not all_convo_files:
        print(f"No simulated conversations in {data_dir}")
        if results:
            _write_summaries(output_dir, results)
        return 0

    remaining = [p for p in all_convo_files if p.name not in evaluated_names]
    if not remaining:
        print("All conversations in the data dir already have results for this model.")
    else:
        if args.limit and args.limit > 0:
            remaining = remaining[: args.limit]
        print(f"Evaluating {len(remaining)} new conversation(s); {len(evaluated_names)} already done for this model.")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in remaining:
            try:
                res = evaluate_one(args.model, oracle, path, output_dir)
                results.append(res)
                print(
                    f"Evaluated {path.name}: "
                    f"P={res['precision']:.2f} R={res['recall']:.2f} F1={res['f1']:.2f} | "
                    f"Turn-only P={res['turn_only_precision']:.2f} R={res['turn_only_recall']:.2f} F1={res['turn_only_f1']:.2f}"
                )
            except Exception as exc:
                print(f"Failed on {path}: {exc}")

    if results:
        _write_summaries(output_dir, results)
        macro = _compute_macro(results)
        print(
            "Macro "
            f"P={macro['macro_precision']:.2f} R={macro['macro_recall']:.2f} F1={macro['macro_f1']:.2f} | "
            f"Turn-only P={macro['macro_turn_only_precision']:.2f} R={macro['macro_turn_only_recall']:.2f} F1={macro['macro_turn_only_f1']:.2f} "
            f"over {macro['num_samples']} files."
        )
    else:
        print("No results to summarize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

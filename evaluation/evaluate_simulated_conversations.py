#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from providers.azure_gpt_call import call_chat_completion

if __package__:
    from .shared import (
        VKey,
        build_user_prompt,
        compute_metrics,
        compute_turn_metrics,
        extract_first_json,
        format_conversation,
        format_guidelines,
        infer_category_titles,
    )
else:  # pragma: no cover - script executed directly
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from evaluation.shared import (  # type: ignore[no-redef]
        VKey,
        build_user_prompt,
        compute_metrics,
        compute_turn_metrics,
        extract_first_json,
        format_conversation,
        format_guidelines,
        infer_category_titles,
    )


SYSTEM_PROMPT = (
    "You are an LLM Judge for evaluating whether a call center agent follows the provided guidelines. "
    "Given those guidelines and a conversation transcript, identify exactly which assistant turns violate which clause. "
    "Only assistant (agent) turns can violate guidelines; never mark caller turns. "
    "Each assistant turn may map to at most one violation entry. "
    "Use the exact Category titles and Keys from the guidelines when possible. "
    "Category 1 (Universal Compliance) consists of single-step requirements that apply throughout the call. "
    "Category 2 (Intent Triggered Guidelines) groups intents into ordered phases; phases must be handled in sequence (no skipping), though a phase may be revisited if the flow requires it—only mark a phase when that intent is active and include its 1-based phase number. "
    "Category 3 (Condition Triggered Guidelines) covers conditional or handoff triggers that fire immediately when the condition is met and always use guideline_phase = -1. "
    "Return only the requested strict JSON format with no extra text."
)


def evaluate_one(model: str, oracle: dict[str, Any], convo_path: Path, out_dir: Path) -> dict[str, Any]:
    with convo_path.open("r", encoding="utf-8") as f:
        convo = json.load(f)

    message_list = convo.get("message_list", [])
    truth_list = convo.get("mistakes", [])
    conv_id = convo_path.stem

    cat_titles = infer_category_titles(oracle)
    guidelines_text = format_guidelines(oracle, cat_titles)
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

    raw_pred_items = resp.get("violations", []) or []
    pred_items: list[dict[str, Any]] = []
    seen_turns: set[int] = set()
    for item in raw_pred_items:
        turn_idx = int(item.get("turn_index", -1))
        if turn_idx in seen_turns:
            continue
        seen_turns.add(turn_idx)
        pred_items.append(item)

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
    parser.add_argument("--guidelines", default="guidelines/SCAN/oracle.json", help="Path to oracle guidelines.")
    parser.add_argument("--data-dir", default="dump/simulated_scan_test", help="Directory of simulated conversations.")
    parser.add_argument("--output-dir", default="dump/eval_scan_test", help="Base directory to write evaluation outputs (per-model subfolder will be used).")
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

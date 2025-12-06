#!/usr/bin/env python3
"""Evaluate simulated conversations by judging each user-assistant pair individually.

Instead of sending the full conversation to the model once, this script sends
one request per assistant turn (with history for context) so the model focuses
on the latest pair. Metrics match evaluate_simulated_conversations.py
and include both strict (turn + category/key) and turn-only variants.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, List, Dict, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed  # Option A fallback

from azure_gpt_call import call_chat_completion
from evaluate_simulated_conversations import (
    VKey,
    compute_metrics,
    compute_turn_metrics,
    extract_first_json,
    format_guidelines,
)


PAIRWISE_SYSTEM_PROMPT = (
    "You are a precise compliance evaluator for Celestar Air. "
    "Given the official guidelines and the conversation history, evaluate ONLY the final assistant message shown. "
    "Decide which guidelines that last assistant turn violates. "
    "Return strict JSON only, with accurate turn_index, category, key, and guideline_phase."
)


def format_history_for_turn(messages: Sequence[dict[str, Any]], target_turn_index: int) -> str:
    """Render conversation history up to and including the target turn."""
    lines: list[str] = []
    for msg in messages:
        idx = msg.get("turn_index")
        role = msg.get("role", "")
        content = msg.get("content", "")
        prefix = ">> " if idx == target_turn_index and str(role).lower() == "assistant" else ""
        lines.append(f"{prefix}{idx} | {role.upper()}: {content}")
    return "\n".join(lines)


def build_pair_prompt(guidelines_text: str, history_text: str, conv_id: str, target_turn_index: int) -> str:
    return (
        "TASK:\n"
        f"Evaluate ONLY the final assistant message (turn_index {target_turn_index}) in the history below.\n"
        "- Focus strictly on that assistant turn; earlier turns are context only.\n"
        "- Use exact field names and values as they appear in the guidelines.\n"
        "- For Category 1 or 3, set guideline_phase to -1.\n"
        "- For Category 2, set guidance_key to the intent name and guideline_phase to the Phase number.\n"
        "RESPONSE (strict JSON only):\n"
        "{\n"
        f"  \"conversation_id\": \"{conv_id}\",\n"
        f"  \"target_turn_index\": {target_turn_index},\n"
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
        "CONVERSATION HISTORY (latest assistant turn is prefixed with '>>'):\n" + history_text
    )


async def evaluate_turn(
    model: str,
    guidelines_text: str,
    conv_id: str,
    message_list: Sequence[dict[str, Any]],
    target_turn_index: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, list[dict[str, Any]], str]:
    """Evaluate a single assistant turn and return its predictions and raw model response."""
    history = [m for m in message_list if int(m.get("turn_index", -1)) <= target_turn_index]
    history_text = format_history_for_turn(history, target_turn_index)
    prompt = build_pair_prompt(guidelines_text, history_text, conv_id, target_turn_index)
    async with semaphore:
        response_text = await asyncio.to_thread(
            call_chat_completion,
            model,
            [{"role": "user", "content": prompt}],
            system_prompt=PAIRWISE_SYSTEM_PROMPT,
        )
    try:
        resp = extract_first_json(response_text)
    except Exception:
        resp = {"conversation_id": conv_id, "violations": []}

    pred_items = resp.get("violations", []) or []
    # Keep only predictions for the target turn to avoid double counting.
    filtered = [
        item for item in pred_items if int(item.get("turn_index", -1)) == target_turn_index
    ]
    return target_turn_index, filtered, response_text


async def evaluate_conversation_pairwise(
    model: str,
    oracle: dict[str, Any],
    convo_path: Path,
    out_dir: Path,
    max_concurrency: int,
) -> dict[str, Any]:
    with convo_path.open("r", encoding="utf-8") as f:
        convo = json.load(f)

    message_list = convo.get("message_list", [])
    truth_list = convo.get("mistakes", [])
    conv_id = convo_path.stem

    guidelines_text = format_guidelines(oracle)

    assistant_turns = sorted(
        int(msg.get("turn_index", -1)) for msg in message_list if str(msg.get("role", "")).lower() == "assistant"
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        evaluate_turn(model, guidelines_text, conv_id, message_list, turn_idx, semaphore)
        for turn_idx in assistant_turns
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    pred_items: list[dict[str, Any]] = []
    per_turn_responses: dict[int, str] = {}
    failed_turns: list[str] = []
    for turn_idx, res in zip(assistant_turns, results):
        if isinstance(res, Exception):
            failed_turns.append(f"turn {turn_idx}: {res}")
            continue
        t_idx, preds, response_text = res
        pred_items.extend(preds)
        per_turn_responses[t_idx] = response_text

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
        "failed_turns": failed_turns,
        "per_turn_response_text": {str(k): v for k, v in per_turn_responses.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{convo_path.stem}_pair_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


# ---------- Helpers for Option A fallback ----------

def _print_result_line(res: Dict[str, Any], fname: str) -> None:
    print(
        f"Evaluated {fname}: "
        f"P={res['precision']:.2f} R={res['recall']:.2f} F1={res['f1']:.2f} | "
        f"Turn-only P={res['turn_only_precision']:.2f} R={res['turn_only_recall']:.2f} F1={res['turn_only_f1']:.2f}"
    )
    if res.get("failed_turns"):
        print(f"  Failed turns: {res['failed_turns']}")


def _write_summary_and_csv(results: List[Dict[str, Any]], run_dir: Path) -> None:
    if not results:
        return
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

    # CSV per file + macro row
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


def _eval_one_in_thread(
    model: str,
    oracle: dict[str, Any],
    convo_path: Path,
    run_dir: Path,
    per_turn_concurrency: int,
) -> Dict[str, Any]:
    """Runs one conversation evaluation inside a thread with its own event loop."""
    return asyncio.run(
        evaluate_conversation_pairwise(
            model=model,
            oracle=oracle,
            convo_path=convo_path,
            out_dir=run_dir,
            max_concurrency=max(1, per_turn_concurrency),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pairwise evaluation of simulated conversations.")
    parser.add_argument("--model", default="gpt-5", help="Model/deployment name for evaluation.")
    parser.add_argument("--guidelines", default="guidelines/dental/oracle.json", help="Path to oracle guidelines.")
    parser.add_argument("--data-dir", default="dump/simulated_dental_500_v1", help="Directory of simulated conversations.")
    parser.add_argument("--output-dir", default="dump/eval_dental_500_v1_pair", help="Directory to write evaluation outputs.")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent LLM calls across turns.")
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

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = str(args.model or "model")
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name)
    run_dir = output_dir / f"{sanitized}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    async def runner() -> None:
        # Serial over files; per-file concurrency is controlled inside evaluate_conversation_pairwise
        for path in convo_files:
            try:
                res = await evaluate_conversation_pairwise(
                    args.model, oracle, path, run_dir, max(1, args.concurrency)
                )
                results.append(res)
                _print_result_line(res, path.name)
            except Exception as exc:
                print(f"Failed on {path}: {exc}")

        _write_summary_and_csv(results, run_dir)

    # -------- Option A: Pick execution path based on presence of a running loop --------
    try:
        asyncio.get_running_loop()
        # We are in an active event loop (e.g., Jupyter/IPython) -> use threads fallback.
        # Each thread will create its own event loop to run the coroutine safely.
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            future_map = {
                ex.submit(
                    _eval_one_in_thread,
                    args.model,
                    oracle,
                    path,
                    run_dir,
                    max(1, args.concurrency),
                ): path
                for path in convo_files
            }
            for fut in as_completed(future_map):
                path = future_map[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    _print_result_line(res, path.name)
                except Exception as exc:
                    print(f"Failed on {path}: {exc}")

        _write_summary_and_csv(results, run_dir)

    except RuntimeError:
        # No running loop -> safe to use asyncio.run as usual (CLI path)
        asyncio.run(runner())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

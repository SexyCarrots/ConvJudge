#!/usr/bin/env python3
"""Recompute strict + turn-only metrics for existing evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.shared import VKey, compute_metrics, compute_turn_metrics


def _load_keys(data: dict[str, Any]) -> tuple[set[VKey], set[VKey]]:
    preds = data.get("predicted") or data.get("violations") or []
    truth = data.get("ground_truth") or data.get("mistakes") or []
    pred_keys = {VKey.from_pred(item) for item in preds}
    truth_keys = {VKey.from_truth(item) for item in truth}
    return pred_keys, truth_keys


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _gather_eval_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.rglob(pattern))


def _process_eval_dir(eval_dir: Path, files: list[Path], output_path: Path) -> dict[str, Any]:
    general_tp = general_fp = general_fn = 0
    general_precisions: list[float] = []
    general_recalls: list[float] = []
    general_f1s: list[float] = []

    turn_tp = turn_fp = turn_fn = 0
    turn_precisions: list[float] = []
    turn_recalls: list[float] = []
    turn_f1s: list[float] = []

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        pred_keys, truth_keys = _load_keys(data)
        precision, recall, f1, tp_items, fp_items, fn_items = compute_metrics(pred_keys, truth_keys)
        turn_precision, turn_recall, turn_f1, turn_tp_items, turn_fp_items, turn_fn_items = compute_turn_metrics(
            pred_keys, truth_keys
        )

        data.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": [item.__dict__ for item in tp_items],
                "false_positive": [item.__dict__ for item in fp_items],
                "false_negative": [item.__dict__ for item in fn_items],
                "turn_only_precision": turn_precision,
                "turn_only_recall": turn_recall,
                "turn_only_f1": turn_f1,
                "turn_true_positive": turn_tp_items,
                "turn_false_positive": turn_fp_items,
                "turn_false_negative": turn_fn_items,
            }
        )

        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

        general_precisions.append(precision)
        general_recalls.append(recall)
        general_f1s.append(f1)
        general_tp += len(tp_items)
        general_fp += len(fp_items)
        general_fn += len(fn_items)

        turn_precisions.append(turn_precision)
        turn_recalls.append(turn_recall)
        turn_f1s.append(turn_f1)
        turn_tp += len(turn_tp_items)
        turn_fp += len(turn_fp_items)
        turn_fn += len(turn_fn_items)

    summary = {
        "files_processed": len(files),
        "general": {
            "precision": _mean(general_precisions),
            "recall": _mean(general_recalls),
            "f1": _mean(general_f1s),
            "tp": general_tp,
            "fp": general_fp,
            "fn": general_fn,
        },
        "turn_only": {
            "precision": _mean(turn_precisions),
            "recall": _mean(turn_recalls),
            "f1": _mean(turn_f1s),
            "tp": turn_tp,
            "fp": turn_fp,
            "fn": turn_fn,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def recompute(base_dir: Path, pattern: str) -> None:
    targets: list[tuple[Path, list[Path], Path]] = []

    base_direct = sorted(base_dir.glob(pattern))
    if base_direct:
        targets.append(
            (
                base_dir,
                _gather_eval_files(base_dir, pattern),
                base_dir / "eval_results" / "metrics.json",
            )
        )
    else:
        for child in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            files = _gather_eval_files(child, pattern)
            if files:
                targets.append(
                    (
                        child,
                        files,
                        base_dir / "eval_results" / child.name / "metrics.json",
                    )
                )

    if not targets:
        raise SystemExit(f"No files matching pattern '{pattern}' under {base_dir}")

    for directory, files, dest_path in targets:
        summary = _process_eval_dir(directory, files, dest_path)
        print(f"{directory}: processed {summary['files_processed']} file(s); wrote {dest_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute metrics for existing *_eval.json files (per directory or per immediate subdirectory)."
    )
    parser.add_argument("eval_dir", help="Directory containing eval files or per-model subdirectories")
    parser.add_argument(
        "--pattern",
        default="*_eval.json",
        help="Glob pattern (relative to eval_dir) to select evaluation files",
    )
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists() or not eval_dir.is_dir():
        raise SystemExit(f"Not a directory: {eval_dir}")

    recompute(eval_dir, args.pattern)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

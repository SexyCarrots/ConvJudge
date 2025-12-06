#!/usr/bin/env python3
"""
Utility to visualize aggregate model evaluation metrics.

The script scans each model folder under dump/evaluation, aggregates TP/FP/FN
counts from evaluation_summary.csv, computes precision/recall/F1, and produces
publishable-quality bar charts comparing models.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


def compute_aggregate_metrics(summary_path: Path) -> Dict[str, float]:
    """Compute dataset-level metrics from a single evaluation_summary.csv."""
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        totals = {"turn_tp": 0, "turn_fp": 0, "turn_fn": 0}
        conversation_count = 0

        for row in reader:
            conversation_count += 1
            for key in totals:
                totals[key] += int(row[key])

    tp, fp, fn = totals["turn_tp"], totals["turn_fp"], totals["turn_fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "turn_f1": f1,
        "turn_tp": tp,
        "turn_fp": fp,
        "turn_fn": fn,
        "conversations": conversation_count,
    }


def discover_models(evaluation_root: Path) -> Dict[str, Dict[str, float]]:
    """Walk the evaluation root and collect aggregate metrics per model."""
    metrics_by_model: Dict[str, Dict[str, float]] = {}

    for model_dir in sorted(evaluation_root.iterdir()):
        if not model_dir.is_dir():
            continue
        # if model_dir.name.startswith("o4-mini"):
        #     continue
        summary_path = model_dir / "evaluation_summary.csv"
        if not summary_path.exists():
            continue

        model_metrics = compute_aggregate_metrics(summary_path)
        metrics_by_model[model_dir.name] = model_metrics

    if not metrics_by_model:
        raise FileNotFoundError(
            f"No evaluation_summary.csv files found under {evaluation_root}"
        )

    return metrics_by_model


def plot_metrics(
    metrics_by_model: Dict[str, Dict[str, float]],
    metrics: Sequence[str],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Render grouped bar chart comparing the requested metrics."""
    # Preserve a deterministic ordering for reproducibility.
    model_names = list(metrics_by_model.keys())
    display_names = [
        name.split("_", 1)[0] if "_" in name else name for name in model_names
    ]
    metric_values: Dict[str, List[float]] = {
        metric: [metrics_by_model[name][metric] for name in model_names]
        for metric in metrics
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(min(13, 1.7 * len(model_names) + 4), 6))

    bar_width = 0.8 / len(metrics)
    base_positions = range(len(model_names))

    palette = [
        "#345995",  # deep blue
        "#F28F3B",  # warm orange
        "#56A3A6",  # teal
        "#BC4B51",  # muted red
        "#5E4AE3",  # indigo
        "#705220",  # bright orange
    ]

    for idx, metric in enumerate(metrics):
        offsets = [pos + (idx - (len(metrics) - 1) / 2) * bar_width for pos in base_positions]
        bars = ax.bar(
            offsets,
            metric_values[metric],
            width=bar_width * 0.95,
            label=metric.capitalize(),
            color=palette[idx % len(palette)],
        )

        for bar, value in zip(bars, metric_values[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#252525",
            )

    ax.set_xticks(list(base_positions))
    ax.set_xticklabels(display_names, rotation=20, ha="right", fontsize=10)
    max_metric = max(max(values) for values in metric_values.values())
    ax.set_ylim(0, min(1.05, max_metric + 0.05))
    ax.set_xlabel("Base model name", fontsize=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.margins(x=0.02)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.6)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready plots from evaluation summaries."
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("dump/eval_simulated"),
        help="Root directory containing per-model evaluation folders.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["precision", "recall", "turn_f1"],
        help="Metrics to visualize (must be keys in evaluation_summary.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/evaluation_simulated.png"),
        help="Path for the generated plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution for saved output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_by_model = discover_models(args.evaluation_root)

    missing_metrics = set()
    for metric in args.metrics:
        for model_name, stats in metrics_by_model.items():
            value = stats.get(metric)
            if value is None or not math.isfinite(float(value)):
                missing_metrics.add(metric)
                break
    if missing_metrics:
        raise KeyError(f"Requested metrics not found in dataset: {sorted(missing_metrics)}")

    plot_metrics(metrics_by_model, args.metrics, args.output, args.dpi)

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

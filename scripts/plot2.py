#!/usr/bin/env python3
"""
Utility to visualize aggregate model evaluation metrics.

The script scans each model folder under dump/evaluation (or a provided root),
reads metrics from summary.json, and produces publication-quality bar charts
comparing models. Turn-only metrics use a hatch pattern to distinguish them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


# ---- Data loading -----------------------------------------------------------

EXPECTED_METRICS = [
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_turn_only_precision",
    "macro_turn_only_recall",
    "macro_turn_only_f1",
]


def read_summary_json(summary_path: Path) -> Dict[str, float]:
    """Read metrics from a single summary.json."""
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Keep only expected numeric metrics; coerce to float where possible.
    metrics: Dict[str, float] = {}
    for key in EXPECTED_METRICS:
        val = data.get(key)
        try:
            metrics[key] = float(val) if val is not None else float("nan")
        except (TypeError, ValueError):
            metrics[key] = float("nan")
    return metrics


def discover_models(evaluation_root: Path) -> Dict[str, Dict[str, float]]:
    """Walk the evaluation root and collect metrics per model from summary.json."""
    metrics_by_model: Dict[str, Dict[str, float]] = {}

    for model_dir in sorted(evaluation_root.iterdir()):
        if not model_dir.is_dir():
            continue

        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue

        model_metrics = read_summary_json(summary_path)
        metrics_by_model[model_dir.name] = model_metrics

    if not metrics_by_model:
        raise FileNotFoundError(
            f"No summary.json files found under {evaluation_root}"
        )

    return metrics_by_model


# ---- Plotting ---------------------------------------------------------------

def plot_metrics(
    metrics_by_model: Dict[str, Dict[str, float]],
    metrics: Sequence[str],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Render grouped bar chart comparing the requested metrics."""
    # Deterministic ordering for reproducibility.
    model_names = list(metrics_by_model.keys())
    display_names = [
        name.split("_", 1)[0] if "_" in name else name for name in model_names
    ]

    # Gather values in plotting order.
    metric_values: Dict[str, List[float]] = {
        metric: [metrics_by_model[name].get(metric, float("nan")) for name in model_names]
        for metric in metrics
    }

    # Style and figure.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(min(13, 1.7 * len(model_names) + 4), 6))

    bar_width = 0.8 / max(1, len(metrics))
    base_positions = list(range(len(model_names)))

    palette = [
        "#345995",  # deep blue
        "#F28F3B",  # warm orange
        "#56A3A6",  # teal
        "#BC4B51",  # muted red
        "#5E4AE3",  # indigo
        "#705220",  # golden brown
    ]

    # Use hatch for any "turn_only" metric so it stands out.
    def hatch_for(metric_name: str) -> str | None:
        return "//" if "turn_only" in metric_name else None

    # Make nicer legend labels
    legend_label = {
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "macro_turn_only_precision": "Turn-only Precision",
        "macro_turn_only_recall": "Turn-only Recall",
        "macro_turn_only_f1": "Turn-only F1",
    }

    # Plot groups.
    plotted_handles = []
    for idx, metric in enumerate(metrics):
        offsets = [pos + (idx - (len(metrics) - 1) / 2) * bar_width for pos in base_positions]
        bars = ax.bar(
            offsets,
            metric_values[metric],
            width=bar_width * 0.95,
            label=legend_label.get(metric, metric),
            color=palette[idx % len(palette)],
            hatch=hatch_for(metric),
            edgecolor="#333333",
            linewidth=0.5,
        )
        plotted_handles.append(bars)

        # Annotate values on bars
        for bar, value in zip(bars, metric_values[metric]):
            if value is None or not math.isfinite(float(value)):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#252525",
            )

    # Axes and layout
    ax.set_xticks(base_positions)
    ax.set_xticklabels(display_names, rotation=20, ha="right", fontsize=10)

    # Compute y-limit robustly (ignore NaNs)
    finite_vals = [
        float(v)
        for values in metric_values.values()
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    max_metric = max(finite_vals) if finite_vals else 1.0
    ax.set_ylim(0, min(1.05, max_metric + 0.05))

    ax.set_xlabel("Base model name", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
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


# ---- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready plots from evaluation summaries (summary.json)."
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("dump/eval_dental_500_v1"),
        help="Root directory containing per-model evaluation folders.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=EXPECTED_METRICS,
        help=(
            "Metrics to visualize (keys from summary.json). "
            "Defaults to all six macro and turn-only metrics."
        ),
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

    # Validate requested metrics exist and are finite for at least one model
    missing_or_invalid = set()
    for metric in args.metrics:
        has_valid = any(
            metric in stats and stats[metric] is not None and math.isfinite(float(stats[metric]))
            for stats in metrics_by_model.values()
        )
        if not has_valid:
            missing_or_invalid.add(metric)
    if missing_or_invalid:
        raise KeyError(
            f"Requested metrics missing or invalid across all models: {sorted(missing_or_invalid)}"
        )

    plot_metrics(metrics_by_model, args.metrics, args.output, args.dpi)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

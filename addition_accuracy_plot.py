"""Plot completed one- and two-fact addition accuracy by filler length."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUMMARIES = {
    "One-fact addition": Path(
        "runs/deepseek-v4-flash/one-fact-addition-full-batched/summary.json"
    ),
    "Two-fact addition": Path(
        "runs/deepseek-v4-flash/two-fact-addition-full-under-999/summary.json"
    ),
}

DEFAULT_RESULTS = {
    "One-fact addition": Path(
        "runs/deepseek-v4-flash/one-fact-addition-full-batched/results.json"
    ),
    "Two-fact addition": Path(
        "runs/deepseek-v4-flash/two-fact-addition-full-under-999/results.json"
    ),
}


def wilson_interval(correct: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if count <= 0 or not 0 <= correct <= count:
        raise ValueError(f"invalid binomial counts: correct={correct}, count={count}")
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return center - radius, center + radius


def load_accuracy(summary_path: Path) -> list[dict]:
    summary = json.loads(summary_path.read_text())
    points = []
    for condition, result in summary["conditions"].items():
        filler_length = 0 if condition == "baseline" else int(condition.removeprefix("dots_"))
        correct, count = int(result["correct"]), int(result["count"])
        accuracy = correct / count
        low, high = wilson_interval(correct, count)
        points.append({
            "filler_length": filler_length,
            "correct": correct,
            "count": count,
            "accuracy": accuracy,
            "ci_low": low,
            "ci_high": high,
        })
    return sorted(points, key=lambda point: point["filler_length"])


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return a percentile 95% bootstrap interval for a mean.

    Accuracy changes only take values -1, 0, and 1, so sampling their three
    category counts is equivalent to resampling individual paired items and is
    much less memory intensive.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    counts = np.array([(values == value).sum() for value in (-1, 0, 1)])
    draws = rng.multinomial(len(values), counts / len(values), size=n_bootstrap)
    bootstrap_means = (draws[:, 2] - draws[:, 0]) / len(values)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def load_paired_accuracy_changes(
    results_path: Path,
    *,
    n_bootstrap: int = 20_000,
    seed: int = 42,
) -> list[dict]:
    """Load within-item accuracy changes and paired-bootstrap intervals."""
    results = json.loads(results_path.read_text())
    by_condition: dict[str, dict[str, bool]] = {}
    for result in results:
        condition = result["condition"]
        item_id = result["pair_id"]
        condition_results = by_condition.setdefault(condition, {})
        if item_id in condition_results:
            raise ValueError(f"duplicate result for {condition=}, {item_id=}")
        condition_results[item_id] = bool(result["correct"])

    if "baseline" not in by_condition:
        raise ValueError("results do not contain a baseline condition")
    baseline = by_condition["baseline"]
    rng = np.random.default_rng(seed)
    points = []
    for condition, condition_results in by_condition.items():
        if condition == "baseline":
            continue
        if condition_results.keys() != baseline.keys():
            missing = baseline.keys() - condition_results.keys()
            extra = condition_results.keys() - baseline.keys()
            raise ValueError(
                f"items do not match baseline for {condition}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        changes = np.fromiter(
            (
                int(condition_results[item_id]) - int(baseline_correct)
                for item_id, baseline_correct in baseline.items()
            ),
            dtype=np.int8,
            count=len(baseline),
        )
        low, high = _bootstrap_mean_interval(
            changes, n_bootstrap=n_bootstrap, rng=rng
        )
        points.append({
            "filler_length": int(condition.removeprefix("dots_")),
            "count": len(changes),
            "accuracy_change": float(changes.mean()),
            "ci_low": low,
            "ci_high": high,
            "wrong_to_right": int((changes == 1).sum()),
            "right_to_wrong": int((changes == -1).sum()),
            "unchanged": int((changes == 0).sum()),
        })
    return sorted(points, key=lambda point: point["filler_length"])


def plot_addition_accuracy(
    project_root: str | Path = ".",
    summaries: dict[str, Path] = DEFAULT_SUMMARIES,
):
    """Return ``(figure, axes, plotted_data)`` for easy notebook iteration."""
    project_root = Path(project_root)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_data = {}
    for label, relative_path in summaries.items():
        points = load_accuracy(project_root / relative_path)
        plotted_data[label] = points
        x = [point["filler_length"] for point in points]
        y = [point["accuracy"] for point in points]
        yerr = [
            [point["accuracy"] - point["ci_low"] for point in points],
            [point["ci_high"] - point["accuracy"] for point in points],
        ]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=2, label=label)

    ax.set(xlabel="Filler length (dot tokens)", ylabel="Exact-answer accuracy")
    ax.set_xticks(sorted({p["filler_length"] for points in plotted_data.values() for p in points}))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax, plotted_data


def plot_paired_accuracy_change(
    project_root: str | Path = ".",
    results: dict[str, Path] = DEFAULT_RESULTS,
    *,
    n_bootstrap: int = 20_000,
    seed: int = 42,
):
    """Plot paired accuracy changes from baseline with bootstrap 95% CIs."""
    project_root = Path(project_root)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_data = {}
    for label, relative_path in results.items():
        points = load_paired_accuracy_changes(
            project_root / relative_path,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        plotted_data[label] = points
        x = [point["filler_length"] for point in points]
        y = [point["accuracy_change"] for point in points]
        yerr = [
            [point["accuracy_change"] - point["ci_low"] for point in points],
            [point["ci_high"] - point["accuracy_change"] for point in points],
        ]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=2, label=label)

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set(
        xlabel="Filler length (dot tokens)",
        ylabel="Accuracy change from baseline",
    )
    ax.set_xticks(sorted({p["filler_length"] for points in plotted_data.values() for p in points}))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax, plotted_data

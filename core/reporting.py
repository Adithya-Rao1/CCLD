from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def markdown_table(rows: List[Dict], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    columns = columns or sorted({k for row in rows for k in row.keys()})
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def plot_bar_comparison(
    labels: List[str],
    means: List[float],
    errs: Optional[List[float]] = None,
    title: str = "",
    ylabel: str = "",
    out_path: str = "comparison.png",
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.8), 4))
    ax.bar(labels, means, yerr=errs)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ablation_curve(
    x_values: List[float],
    series: Dict[str, List[float]],
    xlabel: str,
    ylabel: str,
    title: str = "",
    out_path: str = "ablation.png",
    xscale: str = "linear",
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, y_values in series.items():
        ax.plot(x_values, y_values, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_experiment_report(
    experiment_name: str,
    summary_rows: List[Dict],
    significance_rows: List[Dict],
    figure_paths: List[str],
    out_path: str,
) -> str:
    lines = [f"# {experiment_name} -- Results\n"]
    lines.append("## Summary (mean +/- 95% CI, per config)\n")
    lines.append(markdown_table(summary_rows))
    lines.append("\n## Significance vs. baselines (paired Wilcoxon across seeds)\n")
    lines.append(markdown_table(significance_rows))
    if figure_paths:
        lines.append("\n## Figures\n")
        for p in figure_paths:
            lines.append(f"![{os.path.basename(p)}]({p})\n")
    report = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    return out_path
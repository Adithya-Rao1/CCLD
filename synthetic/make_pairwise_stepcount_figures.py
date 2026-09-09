from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITION_LABELS = {
    "mean_field": "mean field", "heterogeneous_eps0.5": "heterogeneous (eps=0.5)",
    "heterogeneous_eps1.0": "heterogeneous (eps=1.0)", "block": "block",
}
CONDITION_COLORS = {
    "mean_field": "#2b7a78", "heterogeneous_eps0.5": "#c1440e",
    "heterogeneous_eps1.0": "#8b1e3f", "block": "#5b5f97",
}
CONDITION_MARKERS = {"mean_field": "o", "heterogeneous_eps0.5": "s", "heterogeneous_eps1.0": "^", "block": "D"}


def load_rows(summary_csv: str) -> list[dict]:
    rows = []
    with open(summary_csv, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("N"):
                continue
            rows.append({
                "N": int(row["N"]), "condition": row["condition"], "n_diff_steps": int(row["n_diff_steps"]),
                "kl_mean": float(row["kl_mean"]), "kl_std": float(row["kl_std"]),
                "corr_pct_of_true": float(row["corr_pct_of_true"]),
                "corr_gen_std": float(row["corr_gen_std"]), "corr_true": float(row["corr_true"]),
            })
    return rows


def make_figures(summary_csv: str, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    rows = load_rows(summary_csv)
    n_values = sorted({r["N"] for r in rows})
    step_values = sorted({r["n_diff_steps"] for r in rows})

    data = defaultdict(dict)
    for r in rows:
        data[(r["N"], r["condition"])][r["n_diff_steps"]] = r

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})
    paths = []

    def conditions_present(N: int) -> list[str]:
        return [c for c in CONDITION_LABELS if (N, c) in data]

    def plot_kl(N: int) -> str:
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        for cond in conditions_present(N):
            xs = sorted(data[(N, cond)].keys())
            ys = [data[(N, cond)][s]["kl_mean"] for s in xs]
            es = [data[(N, cond)][s]["kl_std"] for s in xs]
            ax.errorbar(xs, ys, yerr=es, label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond],
                        marker=CONDITION_MARKERS[cond], markersize=4, capsize=2, linewidth=1.3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(step_values)
        ax.set_xticklabels([str(s) for s in step_values])
        ax.set_xlabel("$n$")
        ax.set_ylabel("KL divergence")
        ax.set_title(f"N={N}")
        ax.grid(alpha=0.3, which="both")
        if N == n_values[0]:
            ax.legend(loc="best", frameon=True, fontsize=7)
        out_path = os.path.join(out_dir, f"pairwise_kl_vs_steps_N{N}.pdf")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_corr_pct(N: int) -> str:
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        for cond in conditions_present(N):
            xs = sorted(data[(N, cond)].keys())
            ys = [data[(N, cond)][s]["corr_pct_of_true"] for s in xs]
            corr_true = data[(N, cond)][xs[0]]["corr_true"]
            es = [100.0 * data[(N, cond)][s]["corr_gen_std"] / corr_true for s in xs]
            ax.errorbar(xs, ys, yerr=es, label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond],
                        marker=CONDITION_MARKERS[cond], markersize=4, capsize=2, linewidth=1.3)
        ax.axhline(100.0, color="black", linestyle="--", linewidth=0.8, label="true (100%)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(step_values)
        ax.set_xticklabels([str(s) for s in step_values])
        ax.set_xlabel("$n$")
        ax.set_ylabel("% of true correlation")
        ax.set_title(f"N={N}")
        ax.grid(alpha=0.3)
        if N == n_values[0]:
            ax.legend(loc="best", frameon=True, fontsize=7)
        out_path = os.path.join(out_dir, f"pairwise_corr_pct_vs_steps_N{N}.pdf")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    for N in n_values:
        paths.append(plot_kl(N))
    for N in n_values:
        paths.append(plot_corr_pct(N))
    return paths


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot KL / %-of-true-correlation vs. step count, one line per coupling condition")
    p.add_argument("--summary-csv", default="results/experiment_3_synthetic/pairwise_stepcount_sweep/pairwise_stepcount_sweep_summary.csv")
    p.add_argument("--out-dir", default="synthetic/figures")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    for p in make_figures(args.summary_csv, args.out_dir):
        print(p)

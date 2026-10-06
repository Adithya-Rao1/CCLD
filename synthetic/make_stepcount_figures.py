from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUMMARY_CSV = os.path.join(
    HERE, "..", "results", "experiment_3_synthetic", "stepcount_sweep_seeds10_iters10000_samples40000",
    "stepcount_sweep_summary.csv",
)
OUT_DIR = os.path.join(HERE, "figures")

METHODS = ["ddpm", "sdm", "ccld_analytic", "ccld_independent"]
METHOD_LABELS = {"ddpm": "DDPM", "sdm": "SGM", "ccld_analytic": "CCLD", "ccld_independent": "CLD"}
METHOD_COLORS = {"ccld_analytic": "#2b7a78", "ddpm": "#c1440e", "sdm": "#5b5f97", "ccld_independent": "#8c8c8c"}
METHOD_MARKERS = {"ccld_analytic": "o", "ddpm": "s", "sdm": "^", "ccld_independent": "d"}
N_VALUES = [2, 3, 4, 5]
STEP_VALUES = [8, 16, 32, 64, 128]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})


def load_data(summary_csv: str):
    rows = []
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            if not row["N"]:
                continue
            rows.append({
                "N": int(row["N"]), "method": row["method"], "n_diff_steps": int(row["n_diff_steps"]),
                "kl_mean": float(row["kl_mean"]), "kl_std": float(row["kl_std"]),
                "corr_pct_of_true": float(row["corr_pct_of_true"]),
                "corr_gen_std": float(row["corr_gen_std"]), "corr_true": float(row["corr_true"]),
            })
    data = defaultdict(dict)
    for r in rows:
        data[(r["N"], r["method"])][r["n_diff_steps"]] = r
    return data


def available_methods(data) -> list:
    return [m for m in METHODS if all((N, m) in data for N in N_VALUES)]


def plot_kl(N: int, data, methods, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    for method in methods:
        xs = STEP_VALUES
        ys = [data[(N, method)][s]["kl_mean"] for s in xs]
        es = [data[(N, method)][s]["kl_std"] for s in xs]
        ax.errorbar(xs, ys, yerr=es, label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method], markersize=4, capsize=2, linewidth=1.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(STEP_VALUES)
    ax.set_xticklabels([str(s) for s in STEP_VALUES])
    ax.set_xlabel("$n$")
    ax.set_ylabel("KL divergence")
    ax.set_title(f"N={N}")
    ax.grid(alpha=0.3, which="both")
    if N == N_VALUES[0]:
        ax.legend(loc="best", frameon=True)
    out_path = os.path.join(out_dir, f"kl_vs_steps_N{N}.pdf")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_corr_pct(N: int, data, methods, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    corr_true = data[(N, "ccld_analytic")][STEP_VALUES[0]]["corr_true"]
    for method in methods:
        xs = STEP_VALUES
        ys = [data[(N, method)][s]["corr_pct_of_true"] for s in xs]
        es = [100.0 * data[(N, method)][s]["corr_gen_std"] / corr_true for s in xs]
        ax.errorbar(xs, ys, yerr=es, label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method], markersize=4, capsize=2, linewidth=1.3)
    ax.axhline(100.0, color="black", linestyle="--", linewidth=0.8, label="true (100%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(STEP_VALUES)
    ax.set_xticklabels([str(s) for s in STEP_VALUES])
    ax.set_xlabel("$n$")
    ax.set_ylabel("% of true correlation")
    ax.set_title(f"N={N}")
    ax.grid(alpha=0.3)
    if N == N_VALUES[0]:
        ax.legend(loc="best", frameon=True, fontsize=7)
    out_path = os.path.join(out_dir, f"corr_pct_vs_steps_N{N}.pdf")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-count sweep figures (KL, %% correlation recovered) vs. n_diff_steps, one panel per N")
    p.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    p.add_argument("--out-dir", default=OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_data(args.summary_csv)
    methods = available_methods(data)
    missing = [m for m in METHODS if m not in methods]
    if missing:
        print(f"Note: {missing} not present for every N in {args.summary_csv} -- plotting only {methods}")

    paths = []
    for N in N_VALUES:
        paths.append(plot_kl(N, data, methods, args.out_dir))
    for N in N_VALUES:
        paths.append(plot_corr_pct(N, data, methods, args.out_dir))
    for p in paths:
        print(p)

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(HERE, "..", "stepcount_sweep_summary (2).csv")
OUT_DIR = os.path.join(HERE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["ddpm", "sdm", "csho_analytic"]
METHOD_LABELS = {"ddpm": "DDPM", "sdm": "SDM", "csho_analytic": "CSHO"}
METHOD_COLORS = {"csho_analytic": "#2b7a78", "ddpm": "#c1440e", "sdm": "#5b5f97"}
METHOD_MARKERS = {"csho_analytic": "o", "ddpm": "s", "sdm": "^"}
N_VALUES = [2, 3, 4, 5]
STEP_VALUES = [8, 16, 32, 64, 128]

rows = []
with open(SUMMARY_CSV) as f:
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

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})


def plot_kl(N: int) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    for method in METHODS:
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
    out_path = os.path.join(OUT_DIR, f"kl_vs_steps_N{N}.pdf")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_corr_pct(N: int) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    corr_true = data[(N, "csho_analytic")][STEP_VALUES[0]]["corr_true"]
    for method in METHODS:
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
    out_path = os.path.join(OUT_DIR, f"corr_pct_vs_steps_N{N}.pdf")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    paths = []
    for N in N_VALUES:
        paths.append(plot_kl(N))
    for N in N_VALUES:
        paths.append(plot_corr_pct(N))
    for p in paths:
        print(p)

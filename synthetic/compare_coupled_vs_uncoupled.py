from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

from core.reporting import write_csv
from core.stats import compare_configs

SIG_METRIC_NAMES = ["kl_divergence", "wasserstein2", "mi_mae", "mean_pairwise_corr_gen"]


def _load_per_seed_by_n(path: str, method: str) -> Dict[int, Dict[int, Dict[str, float]]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    skip = {"N", "method", "seed"}
    out: Dict[int, Dict[int, Dict[str, float]]] = {}
    for row in rows:
        if row["method"] != method:
            continue
        N = int(row["N"])
        seed = int(row["seed"])
        metrics = {k: float(v) for k, v in row.items() if k not in skip and v not in (None, "")}
        out.setdefault(N, {})[seed] = metrics
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("coupled CCLD uncoupled-CLD baseline")
    )
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--step-counts", required=True, help="comma-separated n_diff_steps values, e.g. 8,16,32,64,128")
    p.add_argument("--coupled-dir-template", default="results/experiment_3_synthetic/final_seeds{seeds}_iters{iters}_steps{steps}")
    p.add_argument("--independent-dir-template", default="results/experiment_3_synthetic/final_seeds{seeds}_iters{iters}_steps{steps}_independent")
    p.add_argument("--coupled-method-label", default="ccld_analytic")
    p.add_argument("--independent-method-label", default="ccld_independent")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    step_counts = [int(s) for s in args.step_counts.split(",") if s.strip()]

    rows: List[dict] = []
    for steps in step_counts:
        coupled_dir = args.coupled_dir_template.format(seeds=args.seeds, iters=args.iters, steps=steps)
        independent_dir = args.independent_dir_template.format(seeds=args.seeds, iters=args.iters, steps=steps)
        coupled_by_n = _load_per_seed_by_n(
            os.path.join(coupled_dir, "analytic_n_sweep_per_seed.csv"), args.coupled_method_label
        )
        independent_by_n = _load_per_seed_by_n(
            os.path.join(independent_dir, "analytic_n_sweep_per_seed.csv"), args.independent_method_label
        )

        shared_n = sorted(set(coupled_by_n) & set(independent_by_n))
        if not shared_n:
            print(f"n_diff_steps={steps}: no shared N between {coupled_dir} and {independent_dir}, skipping")
            continue

        for N in shared_n:
            sig = compare_configs(independent_by_n[N], coupled_by_n[N], metric_names=SIG_METRIC_NAMES)
            for metric, s in sig.items():
                rows.append({
                    "N": N, "n_diff_steps": steps,
                    "comparison": "ccld_coupled_vs_ccld_independent", "metric": metric, **s,
                })

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(rows, os.path.join(args.out_dir, "coupled_vs_independent_significance.csv"))

    print(f"{'n_diff_steps':>12} {'N':>3} {'metric':>22} {'independent_mean':>17} {'coupled_mean':>14} {'p_value':>10}")
    for row in sorted(rows, key=lambda r: (r["n_diff_steps"], r["N"], r["metric"])):
        print(
            f"{row['n_diff_steps']:>12} {row['N']:>3} {row['metric']:>22} "
            f"{row['baseline_mean']:>17.4f} {row['treatment_mean']:>14.4f} {row['p_value']:>10.4f}"
        )

    print(f"\nWrote {args.out_dir}/coupled_vs_independent_significance.csv")
from __future__ import annotations

import argparse
import csv
import os

from core.reporting import write_csv


def load_summary_rows(seeds: int, iters: int, steps: int) -> list[dict]:
    path = f"results/experiment_3_synthetic/final_champion_seeds{seeds}_iters{iters}_steps{steps}/tikhonov_n_sweep_summary.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["n_diff_steps"] = steps
    return rows


def load_significance_rows(seeds: int, iters: int, steps: int) -> list[dict]:
    path = f"results/experiment_3_synthetic/final_champion_seeds{seeds}_iters{iters}_steps{steps}/tikhonov_n_sweep_significance.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["n_diff_steps"] = steps
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine per-step-count summary/significance CSVs into one step-count comparison table")
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--step-counts", required=True, help="comma-separated n_diff_steps values, e.g. 8,16,32,64,128")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    step_counts = [int(s) for s in args.step_counts.split(",") if s.strip()]

    all_summary_rows = []
    all_sig_rows = []
    for steps in step_counts:
        all_summary_rows.extend(load_summary_rows(args.seeds, args.iters, steps))
        all_sig_rows.extend(load_significance_rows(args.seeds, args.iters, steps))

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(all_summary_rows, os.path.join(args.out_dir, "stepcount_sweep_summary.csv"))
    write_csv(all_sig_rows, os.path.join(args.out_dir, "stepcount_sweep_significance.csv"))

    print(f"{'n_diff_steps':>12} {'N':>3} {'method':>16} {'KL':>10} {'corr_gen':>10} {'corr_true':>10} {'%true':>8}")
    for row in sorted(all_summary_rows, key=lambda r: (int(r["n_diff_steps"]), int(r["N"]), r["method"])):
        print(f"{row['n_diff_steps']:>12} {row['N']:>3} {row['method']:>16} {float(row['kl_mean']):>10.4f} "
              f"{float(row['corr_gen_mean']):>10.4f} {float(row['corr_true']):>10.4f} {float(row['corr_pct_of_true']):>8.1f}")

    print(f"\nWrote {args.out_dir}/stepcount_sweep_summary.csv and stepcount_sweep_significance.csv")

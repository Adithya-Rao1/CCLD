from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch

import synthetic.pairwise_n_sweep as pns
from core.coupling import build_coupling_matrix
from core.reporting import write_csv
from core.stats import aggregate_over_seeds
from synthetic.skew_coupling import parametrize_skew_matrix

N_SWEEP = [2, 3, 4, 5]
STEP_SWEEP = [8, 16, 32, 64, 128]
SEEDS = [0, 1, 2, 3, 4]
OUT_DIR = "results/experiment_3_synthetic/pairwise_stepcount_sweep"
SKEW_SCALE = 0.5
SKEW_SEED = 0


def _build_skew_matrix(N: int, device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SKEW_SEED)
    W = torch.randn(2 * N, 2 * N, generator=gen) * SKEW_SCALE
    return parametrize_skew_matrix(W).to(device)


def run() -> List[Dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    pns.SEEDS = SEEDS
    summary_rows = []
    per_seed_rows = []

    for n_diff_steps in STEP_SWEEP:
        pns.N_DIFF_STEPS = n_diff_steps
        pns.DT = 1.0 / n_diff_steps

        for N in N_SWEEP:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            conditions = [(label, c, None) for label, c in pns.build_coupling_conditions(N, seed=0, device=device)]
            conditions.append((
                "mean_field+skew", build_coupling_matrix(N, mode="mean_field", device=device),
                _build_skew_matrix(N, device),
            ))
            for label, coupling, skew_matrix in conditions:
                print(f"\n=== n_diff_steps={n_diff_steps} N={N} condition={label} ===")
                per_seed = {}
                for seed in SEEDS:
                    m = pns.train_one_seed(N, coupling, seed, label=f"steps={n_diff_steps}/{label}", skew_matrix=skew_matrix)
                    per_seed[seed] = m
                    per_seed_rows.append({"n_diff_steps": n_diff_steps, "N": N, "condition": label, "seed": seed, **m})
                    print(f"  seed={seed}: kl={m['kl_divergence']:.4f} corr_gen={m['mean_pairwise_corr_gen']:.4f} corr_true={m['mean_pairwise_corr_true']:.4f}")

                cond_summary = aggregate_over_seeds(per_seed)
                summary_rows.append({
                    "n_diff_steps": n_diff_steps, "N": N, "condition": label,
                    "kl_mean": cond_summary["kl_divergence"]["mean"], "kl_std": cond_summary["kl_divergence"]["std"],
                    "corr_gen_mean": cond_summary["mean_pairwise_corr_gen"]["mean"], "corr_gen_std": cond_summary["mean_pairwise_corr_gen"]["std"],
                    "corr_true": cond_summary["mean_pairwise_corr_true"]["mean"],
                    "corr_pct_of_true": cond_summary["mean_pairwise_corr_gen"]["mean"] / cond_summary["mean_pairwise_corr_true"]["mean"] * 100.0,
                })

    write_csv(per_seed_rows, os.path.join(OUT_DIR, "pairwise_stepcount_sweep_per_seed.csv"))
    write_csv(summary_rows, os.path.join(OUT_DIR, "pairwise_stepcount_sweep_summary.csv"))

    print(f"\n{'n_diff_steps':>12} {'N':>3} {'condition':>22} {'KL':>10} {'corr_gen':>10} {'corr_true':>10} {'%true':>8}")
    for row in summary_rows:
        print(f"{row['n_diff_steps']:>12} {row['N']:>3} {row['condition']:>22} {row['kl_mean']:>10.4f} "
              f"{row['corr_gen_mean']:>10.4f} {row['corr_true']:>10.4f} {row['corr_pct_of_true']:>8.1f}")
    print(f"\nWrote results to {OUT_DIR}/")
    return summary_rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-count sensitivity vs. coupling heterogeneity (N x condition x n_diff_steps grid)")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-train-iters", type=int, default=2000)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-sweep", default="2,3,4,5")
    p.add_argument("--step-sweep", default="8,16,32,64,128")
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/pairwise_stepcount_sweep")
    p.add_argument("--quick", action="store_true", help="tiny scale for smoke-testing the pipeline end-to-end")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]
    N_SWEEP = [int(n) for n in args.n_sweep.split(",") if n.strip()]
    STEP_SWEEP = [int(s) for s in args.step_sweep.split(",") if s.strip()]
    OUT_DIR = args.out_dir
    pns.N_TRAIN_ITERS = args.n_train_iters
    pns.N_SAMPLES = args.n_samples
    if args.quick:
        SEEDS = [0]
        pns.N_TRAIN_ITERS = 20
        pns.N_SAMPLES = 128
        N_SWEEP = [2, 4]
        STEP_SWEEP = [4, 8]
    run()

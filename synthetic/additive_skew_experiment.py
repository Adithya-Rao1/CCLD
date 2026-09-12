from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch

import synthetic.additive_skew_pairwise as asp
from core.coupling import build_coupling_matrix
from core.reporting import write_csv
from core.stats import aggregate_over_seeds, compare_configs
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas_spectral
from synthetic.ground_truth_sde import make_asymmetric_ground_truth
from synthetic.skew_coupling import parametrize_skew_matrix

N_SWEEP = [2, 3, 4]
COUPLING_STRENGTH = 0.6
SKEW_SCALE = 0.3
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_ITERS_SWEEP = [300, 2000]
N_SAMPLES = 4000
N_DIFF_STEPS = 32
OUT_DIR = "results/experiment_3_synthetic/additive_skew"


def build_generic_skew_additive(N: int, seed: int, device, scale: float = SKEW_SCALE) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(2 * N, 2 * N, generator=gen) * scale
    return parametrize_skew_matrix(W).to(device)


def build_structured_skew_additive(gt, coupling: torch.Tensor, device, scale: float = SKEW_SCALE) -> torch.Tensor:
    N = gt.N
    theta = gt.theta.to(device)
    theta_antisym = theta - theta.T

    Gamma = calibrate_coupled_gammas_spectral(
        asp.ALPHA_V, asp.BETA, asp.K_REFERENCE, asp.K_REFERENCE, coupling, target_zeta=asp.TARGET_ZETA,
    )
    A0 = asp.base_A0(N, coupling, Gamma, device)
    K = -A0[N:, :N]

    W = torch.zeros(2 * N, 2 * N, device=device)
    W[:N, :N] = theta_antisym * scale  # XX
    W[:N, N:] = (theta_antisym * K) * scale  # XV
    W[N:, :N] = (theta_antisym * K) * scale  # VX
    W[N:, N:] = (theta_antisym * Gamma) * scale  # VV
    return parametrize_skew_matrix(W)


def run() -> List[Dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    asp.N_SAMPLES = N_SAMPLES
    asp.N_DIFF_STEPS = N_DIFF_STEPS
    asp.DT = 1.0 / N_DIFF_STEPS

    summary_rows, per_seed_rows, sig_rows = [], [], []

    for N in N_SWEEP:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt = make_asymmetric_ground_truth(N, COUPLING_STRENGTH, seed=0, device=device)
        coupling = build_coupling_matrix(N, mode="mean_field", device=device)

        conditions: List[Tuple[str, torch.Tensor]] = [
            ("symmetric_only", None),
            ("skew_generic", build_generic_skew_additive(N, seed=0, device=device)),
            ("skew_structured", build_structured_skew_additive(gt, coupling, device)),
        ]

        for n_train_iters in N_TRAIN_ITERS_SWEEP:
            asp.N_TRAIN_ITERS = n_train_iters
            per_seed_by_condition: Dict[str, Dict[int, Dict[str, float]]] = {}
            for label, skew_matrix in conditions:
                print(f"\nN={N} n_train_iters={n_train_iters} condition={label}")
                per_seed = {}
                for seed in SEEDS:
                    metrics = asp.train_one_seed_additive(
                        N, coupling, seed, label=f"additive/{label}/iters{n_train_iters}",
                        skew_matrix=skew_matrix, gt=gt,
                    )
                    per_seed[seed] = metrics
                    per_seed_rows.append({"N": N, "n_train_iters": n_train_iters, "condition": label, "seed": seed, **metrics})
                    print(f"  seed={seed}: kl={metrics['kl_divergence']:.4f} corr_mae={metrics['pairwise_corr_mae']:.4f}")
                per_seed_by_condition[label] = per_seed

                cond_summary = aggregate_over_seeds(per_seed)
                summary_rows.append({
                    "N": N, "n_train_iters": n_train_iters, "condition": label,
                    "kl_mean": cond_summary["kl_divergence"]["mean"], "kl_std": cond_summary["kl_divergence"]["std"],
                    "corr_mae_mean": cond_summary["pairwise_corr_mae"]["mean"], "corr_mae_std": cond_summary["pairwise_corr_mae"]["std"],
                })

            baseline_per_seed = per_seed_by_condition["symmetric_only"]
            sig_metric_names = ["kl_divergence", "pairwise_corr_mae"]
            for label, per_seed in per_seed_by_condition.items():
                if label == "symmetric_only":
                    continue
                sig = compare_configs(baseline_per_seed, per_seed, metric_names=sig_metric_names)
                for metric, s in sig.items():
                    sig_rows.append({"N": N, "n_train_iters": n_train_iters, "comparison": f"{label}_vs_symmetric_only", "metric": metric, **s})

    write_csv(per_seed_rows, os.path.join(OUT_DIR, "additive_skew_per_seed.csv"))
    write_csv(summary_rows, os.path.join(OUT_DIR, "additive_skew_summary.csv"))
    write_csv(sig_rows, os.path.join(OUT_DIR, "additive_skew_significance.csv"))

    print("\n\nSUMMARY: symmetric_only vs skew_generic vs skew_structured (additive injection), vs training budget")
    print(f"{'N':>3} {'iters':>6} {'condition':>16} {'KL':>10} {'corr_mae':>10}")
    for row in summary_rows:
        print(f"{row['N']:>3} {row['n_train_iters']:>6} {row['condition']:>16} {row['kl_mean']:>10.4f} {row['corr_mae_mean']:>10.4f}")
    print(f"\nWrote results to {OUT_DIR}/")
    return summary_rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Additive (non-invariant) skew")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-train-iters-sweep", default="300,2000")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-sweep", default="2,3,4")
    p.add_argument("--n-diff-steps", type=int, default=32)
    p.add_argument("--skew-scale", type=float, default=0.3)
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/additive_skew")
    p.add_argument("--quick", action="store_true",)
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    N_TRAIN_ITERS_SWEEP = [int(s) for s in _args.n_train_iters_sweep.split(",") if s.strip()]
    N_SAMPLES = _args.n_samples
    N_SWEEP = [int(n) for n in _args.n_sweep.split(",") if n.strip()]
    N_DIFF_STEPS = _args.n_diff_steps
    SKEW_SCALE = _args.skew_scale
    OUT_DIR = _args.out_dir
    if _args.quick:
        SEEDS = [0]
        N_TRAIN_ITERS_SWEEP = [10, 20]
        N_SAMPLES = 128
        N_SWEEP = [2, 3]
        N_DIFF_STEPS = 8
    run()

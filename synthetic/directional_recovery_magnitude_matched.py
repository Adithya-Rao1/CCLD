from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch

import synthetic.pairwise_n_sweep as pns
from core.coupling import build_coupling_matrix
from core.reporting import write_csv
from core.stats import aggregate_over_seeds, compare_configs
from synthetic.directional_recovery_experiment import build_generic_skew, build_structured_skew, cross_block_asymmetry_metrics
from synthetic.ground_truth_sde import make_directional_ground_truth

N_ORIG_SWEEP = [2, 3, 4]
COUPLING_STRENGTH = 0.6
LAG_DELTA = 0.5
STRUCTURED_SCALE = 1.0
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_ITERS_SWEEP = [100, 300, 1000, 2000]
N_SAMPLES = 4000
N_DIFF_STEPS = 32
SAMPLER = "exact"
OUT_DIR = "results/experiment_3_synthetic/directional_recovery_magnitude_matched"


def match_frobenius_norm(J: torch.Tensor, target_norm: float) -> torch.Tensor:
    current_norm = torch.linalg.norm(J)
    if current_norm < 1e-12:
        raise ValueError
    return J * (target_norm / current_norm)


def build_conditions_matched(gt, coupling: torch.Tensor, device) -> Tuple[List[Tuple[str, torch.Tensor]], float, float]:
    J_structured = build_structured_skew(gt, coupling, device, scale=STRUCTURED_SCALE)
    target_norm = torch.linalg.norm(J_structured).item()
    J_generic_raw = build_generic_skew(gt.N, seed=0, device=device, scale=1.0)
    J_generic_matched = match_frobenius_norm(J_generic_raw, target_norm)
    generic_norm = torch.linalg.norm(J_generic_matched).item()
    conditions = [
        ("symmetric_only", None),
        ("skew_structured", J_structured),
        ("skew_generic_matched", J_generic_matched),
    ]
    return conditions, target_norm, generic_norm


def run() -> List[Dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    pns.N_SAMPLES = N_SAMPLES
    pns.N_DIFF_STEPS = N_DIFF_STEPS
    pns.DT = 1.0 / N_DIFF_STEPS
    pns.COUPLING_STRENGTH = COUPLING_STRENGTH

    norm_rows, summary_rows, per_seed_rows, sig_rows = [], [], [], []

    for N_orig in N_ORIG_SWEEP:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt = make_directional_ground_truth(N_orig, COUPLING_STRENGTH, seed=0, lag_delta=LAG_DELTA, device=device)
        N_eff = gt.N
        coupling = build_coupling_matrix(N_eff, mode="mean_field", device=device)

        conditions, target_norm, generic_norm = build_conditions_matched(gt, coupling, device)
        norm_rows.append({"N_orig": N_orig, "structured_norm": target_norm, "generic_matched_norm": generic_norm})
        print(f"N_orig={N_orig}: ||J_structured||_F={target_norm:.4f}  ||J_generic_matched||_F={generic_norm:.4f}")

        for n_train_iters in N_TRAIN_ITERS_SWEEP:
            pns.N_TRAIN_ITERS = n_train_iters
            per_seed_by_condition: Dict[str, Dict[int, Dict[str, float]]] = {}
            for label, skew_matrix in conditions:
                print(f"\nN_orig={N_orig} (N_eff={N_eff}) n_train_iters={n_train_iters} condition={label}")
                per_seed = {}
                for seed in SEEDS:
                    metrics, generated = pns.train_one_seed(
                        N_eff, coupling, seed, label=f"matched/{label}/iters{n_train_iters}", skew_matrix=skew_matrix,
                        gt=gt, return_samples=True, sampler=SAMPLER,
                    )
                    metrics.update(cross_block_asymmetry_metrics(generated, gt))
                    per_seed[seed] = metrics
                    per_seed_rows.append({"N_orig": N_orig, "n_train_iters": n_train_iters, "condition": label, "seed": seed, **metrics})
                    print(f"  seed={seed}: kl={metrics['kl_divergence']:.4f} "
                          f"cross_asym_recovery_pct={metrics['cross_asym_recovery_pct']:.1f} "
                          f"cross_asym_mae={metrics['cross_asym_mae']:.4f}")
                per_seed_by_condition[label] = per_seed

                cond_summary = aggregate_over_seeds(per_seed)
                summary_rows.append({
                    "N_orig": N_orig, "n_train_iters": n_train_iters, "condition": label,
                    "kl_mean": cond_summary["kl_divergence"]["mean"], "kl_std": cond_summary["kl_divergence"]["std"],
                    "cross_asym_recovery_pct_mean": cond_summary["cross_asym_recovery_pct"]["mean"],
                    "cross_asym_recovery_pct_std": cond_summary["cross_asym_recovery_pct"]["std"],
                    "cross_asym_mae_mean": cond_summary["cross_asym_mae"]["mean"],
                    "cross_asym_mae_std": cond_summary["cross_asym_mae"]["std"],
                })

            baseline_per_seed = per_seed_by_condition["symmetric_only"]
            sig_metric_names = ["kl_divergence", "cross_asym_mae", "cross_asym_recovery_pct"]
            for label, per_seed in per_seed_by_condition.items():
                if label == "symmetric_only":
                    continue
                sig = compare_configs(baseline_per_seed, per_seed, metric_names=sig_metric_names)
                for metric, s in sig.items():
                    sig_rows.append({"N_orig": N_orig, "n_train_iters": n_train_iters, "comparison": f"{label}_vs_symmetric_only", "metric": metric, **s})

    write_csv(norm_rows, os.path.join(OUT_DIR, "matched_norms.csv"))
    write_csv(per_seed_rows, os.path.join(OUT_DIR, "directional_recovery_per_seed.csv"))
    write_csv(summary_rows, os.path.join(OUT_DIR, "directional_recovery_summary.csv"))
    write_csv(sig_rows, os.path.join(OUT_DIR, "directional_recovery_significance.csv"))

    print("\n\nSUMMARY: symmetric_only vs skew_structured vs skew_generic_matched (equal Frobenius norm), KL vs training budget")
    print(f"{'N_orig':>6} {'iters':>6} {'condition':>22} {'KL':>10} {'asym_recovery%':>16} {'asym_mae':>10}")
    for row in summary_rows:
        print(f"{row['N_orig']:>6} {row['n_train_iters']:>6} {row['condition']:>22} {row['kl_mean']:>10.4f} "
              f"{row['cross_asym_recovery_pct_mean']:>16.1f} {row['cross_asym_mae_mean']:>10.4f}")
    print(f"\nWrote results to {OUT_DIR}/")
    return summary_rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Same magnitude skew matrices but stationary distribution invariant.")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-train-iters-sweep", default="100,300,1000,2000")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-orig-sweep", default="2,3,4")
    p.add_argument("--n-diff-steps", type=int, default=32)
    p.add_argument("--structured-scale", type=float, default=1.0)
    p.add_argument("--lag-delta", type=float, default=0.5)
    p.add_argument("--sampler", default="exact", choices=["euler", "exact"])
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/recovery_magnitude_matched")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    N_TRAIN_ITERS_SWEEP = [int(s) for s in _args.n_train_iters_sweep.split(",") if s.strip()]
    N_SAMPLES = _args.n_samples
    N_ORIG_SWEEP = [int(n) for n in _args.n_orig_sweep.split(",") if n.strip()]
    N_DIFF_STEPS = _args.n_diff_steps
    STRUCTURED_SCALE = _args.structured_scale
    LAG_DELTA = _args.lag_delta
    SAMPLER = _args.sampler
    OUT_DIR = _args.out_dir
    if _args.quick:
        SEEDS = [0]
        N_TRAIN_ITERS_SWEEP = [10, 20]
        N_SAMPLES = 128
        N_ORIG_SWEEP = [2, 3]
        N_DIFF_STEPS = 8
    run()
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
from synthetic.directional_recovery_magnitude_matched import match_frobenius_norm
from synthetic.ground_truth_sde import make_directional_ground_truth

N_ORIG_SWEEP = [2, 3, 4]
COUPLING_STRENGTH = 0.6
LAG_DELTA = 0.5
TARGET_NORM = 0.625
GT_SEEDS = [0, 1, 2]
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_ITERS = 300
N_SAMPLES = 4000
N_DIFF_STEPS = 32
SAMPLER = "exact"
OUT_DIR = "results/experiment_3_synthetic/directional_recovery_gt_seed_robustness"


def build_conditions_fixed_norm(gt, coupling: torch.Tensor, device, target_norm: float) -> List[Tuple[str, torch.Tensor]]:
    J_structured_raw = build_structured_skew(gt, coupling, device, scale=1.0)
    J_structured = match_frobenius_norm(J_structured_raw, target_norm)
    J_generic_raw = build_generic_skew(gt.N, seed=0, device=device, scale=1.0)
    J_generic_matched = match_frobenius_norm(J_generic_raw, target_norm)
    return [
        ("symmetric_only", None),
        ("skew_structured", J_structured),
        ("skew_generic_matched", J_generic_matched),
    ]


def run() -> List[Dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    pns.N_SAMPLES = N_SAMPLES
    pns.N_DIFF_STEPS = N_DIFF_STEPS
    pns.DT = 1.0 / N_DIFF_STEPS
    pns.COUPLING_STRENGTH = COUPLING_STRENGTH
    pns.N_TRAIN_ITERS = N_TRAIN_ITERS

    summary_rows, per_seed_rows, sig_rows, norm_rows = [], [], [], []

    for N_orig in N_ORIG_SWEEP:
        for gt_seed in GT_SEEDS:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gt = make_directional_ground_truth(N_orig, COUPLING_STRENGTH, seed=gt_seed, lag_delta=LAG_DELTA, device=device)
            N_eff = gt.N
            coupling = build_coupling_matrix(N_eff, mode="mean_field", device=device)

            conditions = build_conditions_fixed_norm(gt, coupling, device, TARGET_NORM)
            norm_rows.append({
                "N_orig": N_orig, "gt_seed": gt_seed,
                "structured_norm": torch.linalg.norm(conditions[1][1]).item(),
                "generic_matched_norm": torch.linalg.norm(conditions[2][1]).item(),
            })

            per_seed_by_condition: Dict[str, Dict[int, Dict[str, float]]] = {}
            for label, skew_matrix in conditions:
                print(f"\nN_orig={N_orig} gt_seed={gt_seed} condition={label}")
                per_seed = {}
                for seed in SEEDS:
                    metrics, generated = pns.train_one_seed(
                        N_eff, coupling, seed, label=f"gtrobust/N{N_orig}/gt{gt_seed}/{label}", skew_matrix=skew_matrix,
                        gt=gt, return_samples=True, sampler=SAMPLER,
                    )
                    metrics.update(cross_block_asymmetry_metrics(generated, gt))
                    per_seed[seed] = metrics
                    per_seed_rows.append({"N_orig": N_orig, "gt_seed": gt_seed, "condition": label, "seed": seed, **metrics})
                    print(f"  seed={seed}: kl={metrics['kl_divergence']:.4f} cross_asym_mae={metrics['cross_asym_mae']:.4f}")
                per_seed_by_condition[label] = per_seed

                cond_summary = aggregate_over_seeds(per_seed)
                summary_rows.append({
                    "N_orig": N_orig, "gt_seed": gt_seed, "condition": label,
                    "kl_mean": cond_summary["kl_divergence"]["mean"], "kl_std": cond_summary["kl_divergence"]["std"],
                    "cross_asym_mae_mean": cond_summary["cross_asym_mae"]["mean"], "cross_asym_mae_std": cond_summary["cross_asym_mae"]["std"],
                })

            baseline_per_seed = per_seed_by_condition["symmetric_only"]
            sig_metric_names = ["kl_divergence", "cross_asym_mae"]
            for label, per_seed in per_seed_by_condition.items():
                if label == "symmetric_only":
                    continue
                sig = compare_configs(baseline_per_seed, per_seed, metric_names=sig_metric_names)
                for metric, s in sig.items():
                    sig_rows.append({"N_orig": N_orig, "gt_seed": gt_seed, "comparison": f"{label}_vs_symmetric_only", "metric": metric, **s})

    write_csv(norm_rows, os.path.join(OUT_DIR, "fixed_norms.csv"))
    write_csv(per_seed_rows, os.path.join(OUT_DIR, "gt_seed_robustness_per_seed.csv"))
    write_csv(summary_rows, os.path.join(OUT_DIR, "gt_seed_robustness_summary.csv"))
    write_csv(sig_rows, os.path.join(OUT_DIR, "gt_seed_robustness_significance.csv"))

    print("\n\nSUMMARY: skew_structured vs skew_generic_matched (fixed target norm)")
    print(f"{'N_orig':>6} {'gt_seed':>7} {'condition':>22} {'KL':>10} {'asym_mae':>10}")
    for row in summary_rows:
        print(f"{row['N_orig']:>6} {row['gt_seed']:>7} {row['condition']:>22} {row['kl_mean']:>10.4f} {row['cross_asym_mae_mean']:>10.4f}")
    print(f"\nWrote results to {OUT_DIR}/")
    return summary_rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ground-truth-seed robustness check for the magnitude-matched skew_structured vs skew_generic result, at a fixed Frobenius norm across N and gt seed.")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--gt-seeds", default="0,1,2")
    p.add_argument("--n-train-iters", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-orig-sweep", default="2,3,4")
    p.add_argument("--n-diff-steps", type=int, default=32)
    p.add_argument("--target-norm", type=float, default=0.625)
    p.add_argument("--lag-delta", type=float, default=0.5)
    p.add_argument("--sampler", default="exact", choices=["euler", "exact"])
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/directional_recovery_gt_seed_robustness")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    GT_SEEDS = [int(s) for s in _args.gt_seeds.split(",") if s.strip()]
    N_TRAIN_ITERS = _args.n_train_iters
    N_SAMPLES = _args.n_samples
    N_ORIG_SWEEP = [int(n) for n in _args.n_orig_sweep.split(",") if n.strip()]
    N_DIFF_STEPS = _args.n_diff_steps
    TARGET_NORM = _args.target_norm
    LAG_DELTA = _args.lag_delta
    SAMPLER = _args.sampler
    OUT_DIR = _args.out_dir
    if _args.quick:
        SEEDS = [0]
        GT_SEEDS = [0, 1]
        N_TRAIN_ITERS = 20
        N_SAMPLES = 128
        N_ORIG_SWEEP = [2, 3]
        N_DIFF_STEPS = 8
    run()
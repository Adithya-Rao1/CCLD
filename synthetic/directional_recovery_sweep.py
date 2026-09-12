from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch

import synthetic.pairwise_n_sweep as pns
from core.coupling import build_coupling_matrix
from core.reporting import write_csv
from core.stats import aggregate_over_seeds, compare_configs
from synthetic.directional_recovery_baselines import train_one_seed_ddpm, train_one_seed_sdm
from synthetic.directional_recovery_experiment import build_generic_skew, build_structured_skew, cross_block_asymmetry_metrics
from synthetic.directional_recovery_magnitude_matched import match_frobenius_norm
from synthetic.ground_truth_sde import make_directional_ground_truth

N_ORIG_SWEEP = [2, 3, 4]
COUPLING_STRENGTH = 0.6
LAG_DELTA = 0.5
GT_SEED = 0
TARGET_NORM = 0.625
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_ITERS_SWEEP = [2000]
N_SAMPLES = 4000
N_DIFF_STEPS_SWEEP = [32]
SAMPLER_SWEEP = ["exact"]
CONDITIONS = ["symmetric_only", "skew_structured", "skew_generic_matched", "ddpm", "sdm"]
OUT_DIR = "results/experiment_3_synthetic/directional_recovery_sweep"

CCLD_CONDITIONS = ["symmetric_only", "skew_structured", "skew_generic_matched"]
BASELINE_CONDITIONS = ["ddpm", "sdm"]


def run() -> List[Dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    pns.COUPLING_STRENGTH = COUPLING_STRENGTH

    active_ccld = [c for c in CCLD_CONDITIONS if c in CONDITIONS]
    active_baselines = [c for c in BASELINE_CONDITIONS if c in CONDITIONS]
    sig_metric_names = ["kl_divergence", "cross_asym_mae"]

    norm_rows, summary_rows, per_seed_rows, sig_rows = [], [], [], []

    for N_orig in N_ORIG_SWEEP:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt = make_directional_ground_truth(N_orig, COUPLING_STRENGTH, seed=GT_SEED, lag_delta=LAG_DELTA, device=device)
        N_eff = gt.N
        coupling = build_coupling_matrix(N_eff, mode="mean_field", device=device)

        skew_matrices: Dict[str, torch.Tensor] = {"symmetric_only": None}
        if "skew_structured" in active_ccld:
            skew_matrices["skew_structured"] = match_frobenius_norm(build_structured_skew(gt, coupling, device, scale=1.0), TARGET_NORM)
        if "skew_generic_matched" in active_ccld:
            skew_matrices["skew_generic_matched"] = match_frobenius_norm(build_generic_skew(N_eff, seed=0, device=device, scale=1.0), TARGET_NORM)
        norm_rows.append({
            "N_orig": N_orig,
            "structured_norm": torch.linalg.norm(skew_matrices["skew_structured"]).item() if "skew_structured" in skew_matrices else None,
            "generic_matched_norm": torch.linalg.norm(skew_matrices["skew_generic_matched"]).item() if "skew_generic_matched" in skew_matrices else None,
        })

        for n_diff_steps in N_DIFF_STEPS_SWEEP:
            pns.N_SAMPLES = N_SAMPLES
            pns.N_DIFF_STEPS = n_diff_steps
            pns.DT = 1.0 / n_diff_steps

            for n_train_iters in N_TRAIN_ITERS_SWEEP:
                pns.N_TRAIN_ITERS = n_train_iters
                per_seed_by_key: Dict[str, Dict[int, Dict[str, float]]] = {}

                for label in active_ccld:
                    for sampler in SAMPLER_SWEEP:
                        key = f"{label}@{sampler}"
                        print(f"\nN_orig={N_orig} (N_eff={N_eff}) steps={n_diff_steps} iters={n_train_iters} condition={key}")
                        per_seed = {}
                        for seed in SEEDS:
                            metrics, generated = pns.train_one_seed(
                                N_eff, coupling, seed, label=f"sweep/{key}/steps{n_diff_steps}/iters{n_train_iters}",
                                skew_matrix=skew_matrices[label], gt=gt, return_samples=True, sampler=sampler,
                            )
                            metrics.update(cross_block_asymmetry_metrics(generated, gt))
                            per_seed[seed] = metrics
                            per_seed_rows.append({"N_orig": N_orig, "n_diff_steps": n_diff_steps, "n_train_iters": n_train_iters, "sampler": sampler, "condition": label, "seed": seed, **metrics})
                            print(f"  seed={seed}: kl={metrics['kl_divergence']:.4f} cross_asym_mae={metrics['cross_asym_mae']:.4f}")
                        per_seed_by_key[key] = per_seed

                        cond_summary = aggregate_over_seeds(per_seed)
                        summary_rows.append({
                            "N_orig": N_orig, "n_diff_steps": n_diff_steps, "n_train_iters": n_train_iters, "sampler": sampler, "condition": label,
                            "kl_mean": cond_summary["kl_divergence"]["mean"], "kl_std": cond_summary["kl_divergence"]["std"],
                            "cross_asym_mae_mean": cond_summary["cross_asym_mae"]["mean"], "cross_asym_mae_std": cond_summary["cross_asym_mae"]["std"],
                        })

                for label in active_baselines:
                    key = f"{label}@euler"
                    print(f"\nN_orig={N_orig} (N_eff={N_eff}) steps={n_diff_steps} iters={n_train_iters} condition={key}")
                    per_seed = {}
                    for seed in SEEDS:
                        train_fn = train_one_seed_ddpm if label == "ddpm" else train_one_seed_sdm
                        metrics = train_fn(N_eff, seed, gt, n_diff_steps, n_train_iters, N_SAMPLES, device)
                        per_seed[seed] = metrics
                        per_seed_rows.append({"N_orig": N_orig, "n_diff_steps": n_diff_steps, "n_train_iters": n_train_iters, "sampler": "euler", "condition": label, "seed": seed, **metrics})
                        print(f"  seed={seed}: kl={metrics['kl_divergence']:.4f} cross_asym_mae={metrics['cross_asym_mae']:.4f}")
                    per_seed_by_key[key] = per_seed

                    cond_summary = aggregate_over_seeds(per_seed)
                    summary_rows.append({
                        "N_orig": N_orig, "n_diff_steps": n_diff_steps, "n_train_iters": n_train_iters, "sampler": "euler", "condition": label,
                        "kl_mean": cond_summary["kl_divergence"]["mean"], "kl_std": cond_summary["kl_divergence"]["std"],
                        "cross_asym_mae_mean": cond_summary["cross_asym_mae"]["mean"], "cross_asym_mae_std": cond_summary["cross_asym_mae"]["std"],
                    })

                if "symmetric_only" in active_ccld:
                    for sampler in SAMPLER_SWEEP:
                        baseline_key = f"symmetric_only@{sampler}"
                        if baseline_key not in per_seed_by_key:
                            continue
                        baseline_per_seed = per_seed_by_key[baseline_key]
                        for label in active_ccld:
                            key = f"{label}@{sampler}"
                            if key == baseline_key:
                                continue
                            sig = compare_configs(baseline_per_seed, per_seed_by_key[key], metric_names=sig_metric_names)
                            for metric, s in sig.items():
                                sig_rows.append({"N_orig": N_orig, "n_diff_steps": n_diff_steps, "n_train_iters": n_train_iters, "comparison": f"{key}_vs_{baseline_key}", "metric": metric, **s})

                    euler_baseline_key = "symmetric_only@euler"
                    if euler_baseline_key in per_seed_by_key:
                        euler_baseline_per_seed = per_seed_by_key[euler_baseline_key]
                        for label in active_baselines:
                            key = f"{label}@euler"
                            sig = compare_configs(euler_baseline_per_seed, per_seed_by_key[key], metric_names=sig_metric_names)
                            for metric, s in sig.items():
                                sig_rows.append({"N_orig": N_orig, "n_diff_steps": n_diff_steps, "n_train_iters": n_train_iters, "comparison": f"{key}_vs_{euler_baseline_key}", "metric": metric, **s})

    write_csv(norm_rows, os.path.join(OUT_DIR, "fixed_norms.csv"))
    write_csv(per_seed_rows, os.path.join(OUT_DIR, "sweep_per_seed.csv"))
    write_csv(summary_rows, os.path.join(OUT_DIR, "sweep_summary.csv"))
    write_csv(sig_rows, os.path.join(OUT_DIR, "sweep_significance.csv"))

    print("\n\nSUMMARY")
    print(f"{'N':>3} {'steps':>5} {'iters':>6} {'sampler':>7} {'condition':>22} {'KL':>10} {'asym_mae':>10}")
    for row in summary_rows:
        print(f"{row['N_orig']:>3} {row['n_diff_steps']:>5} {row['n_train_iters']:>6} {row['sampler']:>7} {row['condition']:>22} {row['kl_mean']:>10.4f} {row['cross_asym_mae_mean']:>10.4f}")
    print(f"\nWrote results to {OUT_DIR}/")
    return summary_rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full sweep")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--gt-seed", type=int, default=0)
    p.add_argument("--conditions", default="symmetric_only,skew_structured,skew_generic_matched,ddpm,sdm")
    p.add_argument("--samplers", default="exact")
    p.add_argument("--n-train-iters-sweep", default="300,1000,2000,10000")
    p.add_argument("--n-samples", type=int, default=40000)
    p.add_argument("--n-orig-sweep", default="2,3,4,5")
    p.add_argument("--n-diff-steps-sweep", default="8,16,32,64,128")
    p.add_argument("--target-norm", type=float, default=0.625)
    p.add_argument("--lag-delta", type=float, default=0.5)
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/directional_recovery_sweep")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    GT_SEED = _args.gt_seed
    CONDITIONS = [c for c in _args.conditions.split(",") if c.strip()]
    SAMPLER_SWEEP = [s for s in _args.samplers.split(",") if s.strip()]
    N_TRAIN_ITERS_SWEEP = [int(s) for s in _args.n_train_iters_sweep.split(",") if s.strip()]
    N_SAMPLES = _args.n_samples
    N_ORIG_SWEEP = [int(n) for n in _args.n_orig_sweep.split(",") if n.strip()]
    N_DIFF_STEPS_SWEEP = [int(n) for n in _args.n_diff_steps_sweep.split(",") if n.strip()]
    TARGET_NORM = _args.target_norm
    LAG_DELTA = _args.lag_delta
    OUT_DIR = _args.out_dir
    if _args.quick:
        SEEDS = [0]
        N_TRAIN_ITERS_SWEEP = [20]
        N_SAMPLES = 128
        N_ORIG_SWEEP = [2, 3]
        N_DIFF_STEPS_SWEEP = [8]
        SAMPLER_SWEEP = ["euler", "exact"]
    run()
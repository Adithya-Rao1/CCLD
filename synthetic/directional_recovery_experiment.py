from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch

import synthetic.pairwise_n_sweep as pns
from core.coupling import build_coupling_matrix
from core.reporting import write_csv
from core.stats import aggregate_over_seeds, compare_configs
from synthetic.ground_truth_sde import make_directional_ground_truth
from synthetic.metrics import fit_gaussian
from synthetic.skew_coupling import parametrize_skew_matrix

N_ORIG_SWEEP = [2, 3, 4]
COUPLING_STRENGTH = 0.6
LAG_DELTA = 0.5
SKEW_SCALE = 1.0
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_ITERS = 2000
N_SAMPLES = 4000
N_DIFF_STEPS = 32
OUT_DIR = "results/experiment_3_synthetic/directional_recovery"


def build_generic_skew(N_eff: int, seed: int, device, scale: float = SKEW_SCALE) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(2 * N_eff, 2 * N_eff, generator=gen) * scale
    return parametrize_skew_matrix(W).to(device)


def build_structured_skew(gt, device, scale: float = SKEW_SCALE) -> torch.Tensor:
    N_orig, N_eff = gt.N_orig, gt.N
    theta_antisym = (gt.theta - gt.theta.T).to(device)
    W = torch.zeros(2 * N_eff, 2 * N_eff, device=device)
    W[:N_orig, N_eff + N_orig:N_eff + 2 * N_orig] = theta_antisym * scale
    return parametrize_skew_matrix(W)


def cross_block_asymmetry_metrics(generated: torch.Tensor, gt) -> Dict[str, float]:
    N_orig = gt.N_orig
    _, cov_gen = fit_gaussian(generated.cpu())
    cross_gen = cov_gen[:N_orig, N_orig:]
    cross_true = gt.lagged_cross_covariance()
    asym_gen = cross_gen - cross_gen.T
    asym_true = cross_true - cross_true.T
    asym_mae = (asym_gen - asym_true).abs().mean().item()
    asym_true_mag = asym_true.abs().mean().item()
    asym_gen_mag = asym_gen.abs().mean().item()
    return {
        "cross_asym_mae": asym_mae,
        "cross_asym_true_mag": asym_true_mag,
        "cross_asym_gen_mag": asym_gen_mag,
        "cross_asym_recovery_pct": (asym_gen_mag / asym_true_mag * 100.0) if asym_true_mag > 1e-8 else float("nan"),
    }


def run() -> List[Dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    pns.N_TRAIN_ITERS = N_TRAIN_ITERS
    pns.N_SAMPLES = N_SAMPLES
    pns.N_DIFF_STEPS = N_DIFF_STEPS
    pns.DT = 1.0 / N_DIFF_STEPS
    pns.COUPLING_STRENGTH = COUPLING_STRENGTH

    summary_rows, per_seed_rows, sig_rows = [], [], []

    for N_orig in N_ORIG_SWEEP:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt = make_directional_ground_truth(N_orig, COUPLING_STRENGTH, seed=0, lag_delta=LAG_DELTA, device=device)
        N_eff = gt.N
        coupling = build_coupling_matrix(N_eff, mode="mean_field", device=device)

        conditions: List[Tuple[str, torch.Tensor]] = [
            ("symmetric_only", None),
            ("skew_generic", build_generic_skew(N_eff, seed=0, device=device)),
            ("skew_structured", build_structured_skew(gt, device)),
        ]

        per_seed_by_condition: Dict[str, Dict[int, Dict[str, float]]] = {}
        for label, skew_matrix in conditions:
            print(f"\n=== N_orig={N_orig} (N_eff={N_eff}) condition={label} ===")
            per_seed = {}
            for seed in SEEDS:
                metrics, generated = pns.train_one_seed(
                    N_eff, coupling, seed, label=f"dir/{label}", skew_matrix=skew_matrix,
                    gt=gt, return_samples=True,
                )
                metrics.update(cross_block_asymmetry_metrics(generated, gt))
                per_seed[seed] = metrics
                per_seed_rows.append({"N_orig": N_orig, "condition": label, "seed": seed, **metrics})
                print(f"  seed={seed}: kl={metrics['kl_divergence']:.4f} "
                      f"cross_asym_recovery_pct={metrics['cross_asym_recovery_pct']:.1f} "
                      f"cross_asym_mae={metrics['cross_asym_mae']:.4f}")
            per_seed_by_condition[label] = per_seed

            cond_summary = aggregate_over_seeds(per_seed)
            summary_rows.append({
                "N_orig": N_orig, "condition": label,
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
                sig_rows.append({"N_orig": N_orig, "comparison": f"{label}_vs_symmetric_only", "metric": metric, **s})

    write_csv(per_seed_rows, os.path.join(OUT_DIR, "directional_recovery_per_seed.csv"))
    write_csv(summary_rows, os.path.join(OUT_DIR, "directional_recovery_summary.csv"))
    write_csv(sig_rows, os.path.join(OUT_DIR, "directional_recovery_significance.csv"))

    print("\n\n=== SUMMARY: symmetric_only vs skew_generic vs skew_structured, directional ground truth ===")
    print(f"{'N_orig':>6} {'condition':>16} {'KL':>10} {'asym_recovery%':>16} {'asym_mae':>10}")
    for row in summary_rows:
        print(f"{row['N_orig']:>6} {row['condition']:>16} {row['kl_mean']:>10.4f} "
              f"{row['cross_asym_recovery_pct_mean']:>16.1f} {row['cross_asym_mae_mean']:>10.4f}")
    print(f"\nWrote results to {OUT_DIR}/")
    return summary_rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Directional (asymmetric-theta) ground-truth recovery: symmetric-only vs skew-augmented CCLD")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-train-iters", type=int, default=2000)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-orig-sweep", default="2,3,4")
    p.add_argument("--n-diff-steps", type=int, default=32)
    p.add_argument("--skew-scale", type=float, default=1.0)
    p.add_argument("--lag-delta", type=float, default=0.5)
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/directional_recovery")
    p.add_argument("--quick", action="store_true", help="tiny scale for smoke-testing the pipeline end-to-end")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    N_TRAIN_ITERS = _args.n_train_iters
    N_SAMPLES = _args.n_samples
    N_ORIG_SWEEP = [int(n) for n in _args.n_orig_sweep.split(",") if n.strip()]
    N_DIFF_STEPS = _args.n_diff_steps
    SKEW_SCALE = _args.skew_scale
    LAG_DELTA = _args.lag_delta
    OUT_DIR = _args.out_dir
    if _args.quick:
        SEEDS = [0]
        N_TRAIN_ITERS = 20
        N_SAMPLES = 128
        N_ORIG_SWEEP = [2, 3]
        N_DIFF_STEPS = 8
    run()
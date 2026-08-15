from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch

from core.reporting import render_experiment_report, write_csv, write_json
from core.stats import aggregate_over_seeds, compare_configs
from synthetic import run_experiment as exp


def flatten_summary(summary: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flat = {}
    for metric, stats in summary.items():
        for k, v in stats.items():
            flat[f"{metric}_{k}"] = v
    return flat


def base_argv(args: argparse.Namespace) -> List[str]:
    argv = [
        "--coupling-strength", str(args.coupling_strength),
        "--base-decay", str(args.base_decay),
        "--gt-sigma-scale", str(args.gt_sigma_scale),
        "--sigma", str(args.sigma),
        "--k-reference", str(args.k_reference),
        "--n-diff-steps", str(args.n_diff_steps),
        "--dt", str(args.dt),
        "--batch-size", str(args.batch_size),
        "--n-train-iters", str(args.n_train_iters),
        "--lr", str(args.lr),
        "--hidden-dim", str(args.hidden_dim),
        "--n-layers", str(args.n_layers),
        "--time-embed-dim", str(args.time_embed_dim),
        "--n-samples", str(args.n_samples),
        "--seeds", args.seeds,
        "--device", args.device,
    ]
    if args.gt_seed is not None:
        argv += ["--gt-seed", str(args.gt_seed)]
    return argv


def run_config(args: argparse.Namespace, label: str, extra_argv: List[str], out_subdir: str) -> Dict[int, Dict[str, float]]:
    argv = base_argv(args) + extra_argv + ["--out-dir", os.path.join(args.out_dir, out_subdir)]
    run_args = exp.parse_args(argv)
    per_seed = {}
    for seed in run_args.seeds:
        print(f"[ablation:{label}] seed={seed}")
        per_seed[seed] = exp.train_one_seed(run_args, seed)
    return per_seed


def _record_axis(results, significance_rows, axis, label, extra_fields, per_seed, baseline):
    results.append({"axis": axis, "label": label, **extra_fields, **flatten_summary(aggregate_over_seeds(per_seed))})
    if baseline[0] is None:
        baseline[0], baseline[1] = per_seed, label
    else:
        for metric, sig in compare_configs(baseline[0], per_seed).items():
            significance_rows.append({"axis": axis, "baseline": baseline[1], "treatment": label, "metric": metric, **sig})


def sweep_n_populations(args, results, significance_rows):
    baseline = [None, None]
    for n in range(2, args.max_n + 1):
        label = f"N{n}"
        argv = ["--N", str(n), "--method", "csho", "--damping-regime", args.default_damping_regime,
                "--alpha", str(args.alpha_grid[0]), "--beta", str(args.beta_grid[0])]
        per_seed = run_config(args, label, argv, f"n_populations/{label}")
        _record_axis(results, significance_rows, "n_populations", label, {"N": n}, per_seed, baseline)


def sweep_coupling_mode(args, results, significance_rows):
    baseline = [None, None]
    for method in ["csho", "csho_pairwise", "csho_independent"]:
        argv = ["--N", str(args.default_n), "--method", method, "--damping-regime", args.default_damping_regime]
        per_seed = run_config(args, method, argv, f"coupling_mode/{method}")
        _record_axis(results, significance_rows, "coupling_mode", method, {}, per_seed, baseline)


def sweep_alpha_beta(args, results, significance_rows):
    baseline_alpha = [None, None]
    for alpha_val in args.alpha_grid:
        label = f"alpha{alpha_val}"
        argv = ["--N", str(args.default_n), "--method", "csho", "--damping-regime", args.default_damping_regime,
                "--alpha", str(alpha_val), "--beta", str(args.beta_grid[0])]
        per_seed = run_config(args, label, argv, f"alpha_beta/{label}")
        _record_axis(results, significance_rows, "alpha", label, {"alpha": alpha_val}, per_seed, baseline_alpha)

    baseline_beta = [None, None]
    for beta_val in args.beta_grid:
        label = f"beta{beta_val}"
        argv = ["--N", str(args.default_n), "--method", "csho", "--damping-regime", args.default_damping_regime,
                "--alpha", str(args.alpha_grid[0]), "--beta", str(beta_val)]
        per_seed = run_config(args, label, argv, f"alpha_beta/{label}")
        _record_axis(results, significance_rows, "beta", label, {"beta": beta_val}, per_seed, baseline_beta)


def sweep_damping_regime(args, results, significance_rows):
    baseline = [None, None]
    for regime in ["underdamped", "critically_damped", "overdamped"]:
        label = f"damping_{regime}"
        argv = ["--N", str(args.default_n), "--method", "csho", "--damping-regime", regime]
        per_seed = run_config(args, label, argv, f"damping_regime/{regime}")
        _record_axis(results, significance_rows, "damping_regime", label, {"regime": regime}, per_seed, baseline)


def sweep_diffusion_mode(args, results, significance_rows):
    baseline = [None, None]
    for mode in ["shared", "independent"]:
        label = f"diffusion_{mode}"
        argv = ["--N", str(args.default_n), "--method", "csho", "--damping-regime", args.default_damping_regime,
                "--diffusion-mode", mode]
        per_seed = run_config(args, label, argv, f"diffusion_mode/{label}")
        _record_axis(results, significance_rows, "diffusion_mode", label, {"diffusion_mode": mode}, per_seed, baseline)


def sweep_coupling_strength(args, results, significance_rows):
    baseline = [None, None]
    for cs in args.coupling_strength_grid:
        label = f"cs{cs}"
        argv = ["--N", str(args.default_n), "--method", "csho", "--damping-regime", args.default_damping_regime,
                "--coupling-strength", str(cs)]
        per_seed = run_config(args, label, argv, f"coupling_strength/{label}")
        _record_axis(results, significance_rows, "coupling_strength", label, {"coupling_strength": cs}, per_seed, baseline)


AXIS_FNS = {
    "n_populations": sweep_n_populations,
    "coupling_mode": sweep_coupling_mode,
    "alpha_beta": sweep_alpha_beta,
    "damping_regime": sweep_damping_regime,
    "diffusion_mode": sweep_diffusion_mode,
    "coupling_strength": sweep_coupling_strength,
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablation grid for Experiment 3 (synthetic coupled-OU CSHO diffusion).")
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/ablations")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--coupling-strength", type=float, default=0.6)
    p.add_argument("--base-decay", type=float, default=1.0)
    p.add_argument("--gt-sigma-scale", type=float, default=1.0)
    p.add_argument("--gt-seed", type=int, default=None)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--k-reference", type=float, default=1.0)
    p.add_argument("--n-diff-steps", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-train-iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--time-embed-dim", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--max-n", type=int, default=5)
    p.add_argument("--default-n", type=int, default=3)
    p.add_argument("--default-damping-regime", default="critically_damped")
    p.add_argument("--alpha-grid", default="0.5,1.0,2.0")
    p.add_argument("--beta-grid", default="0.25,0.5,1.0")
    p.add_argument("--coupling-strength-grid", default="0.0,0.3,0.6,0.9")
    p.add_argument("--axes", default=",".join(AXIS_FNS), help=f"comma-separated subset of {list(AXIS_FNS)}")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    args.alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
    args.beta_grid = [float(v) for v in args.beta_grid.split(",") if v.strip()]
    args.coupling_strength_grid = [float(v) for v in args.coupling_strength_grid.split(",") if v.strip()]
    args.axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in args.axes if a not in AXIS_FNS]
    if unknown:
        raise ValueError(f"Unknown axis/axes {unknown}; choose from {list(AXIS_FNS)}")
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results: List[Dict] = []
    significance_rows: List[Dict] = []

    for axis in args.axes:
        AXIS_FNS[axis](args, results, significance_rows)

    write_csv(results, os.path.join(args.out_dir, "ablation_results.csv"))
    write_csv(significance_rows, os.path.join(args.out_dir, "ablation_significance.csv"))
    write_json({"results": results, "significance": significance_rows}, os.path.join(args.out_dir, "ablation_results.json"))
    render_experiment_report(
        experiment_name="Experiment 3 (synthetic) -- ablation grid",
        summary_rows=results, significance_rows=significance_rows, figure_paths=[],
        out_path=os.path.join(args.out_dir, "ablation_report.md"),
    )
    print(f"Ablation grid done. Results in {args.out_dir}")


if __name__ == "__main__":
    main()

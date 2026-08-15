from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch

from core.reporting import render_experiment_report, write_csv, write_json
from core.stats import aggregate_over_seeds, compare_configs
from image import run_experiment as exp

TASK_GROWTH = ["depth", "normals", "segmentation", "edges", "saliency"]


def flatten_summary(summary: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flat = {}
    for metric, stats in summary.items():
        for k, v in stats.items():
            flat[f"{metric}_{k}"] = v
    return flat


def base_argv(args: argparse.Namespace) -> List[str]:
    argv = [
        "--data-root", args.data_root,
        "--source", args.source,
        "--batch-size", str(args.batch_size),
        "--n-epochs", str(args.n_epochs),
        "--lr", str(args.lr),
        "--seeds", args.seeds,
        "--image-size", str(args.image_size),
        "--latent-dim", str(args.latent_dim),
        "--backbone-channels", str(args.backbone_channels),
        "--score-blocks", str(args.score_blocks),
        "--score-heads", str(args.score_heads),
        "--score-spatial-stride", str(args.score_spatial_stride),
        "--n-diff-steps", str(args.n_diff_steps),
        "--dt", str(args.dt),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
    ]
    if not args.pretrained_backbone:
        argv.append("--no-pretrained-backbone")
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


def sweep_n_tasks(args, results, significance_rows):
    baseline = [None, None]
    max_n = min(args.max_n, len(TASK_GROWTH))
    for n in range(2, max_n + 1):
        tasks = TASK_GROWTH[:n]
        label = f"N{n}"
        argv = ["--tasks", ",".join(tasks), "--method", "csho", "--damping-regime", args.default_damping_regime]
        per_seed = run_config(args, label, argv, f"n_tasks/{label}")
        _record_axis(results, significance_rows, "n_tasks", label, {"n_tasks": n, "tasks": "+".join(tasks)}, per_seed, baseline)


def sweep_coupling_mode(args, results, significance_rows):
    baseline = [None, None]
    for method in ["csho", "csho_pairwise", "csho_independent"]:
        argv = ["--tasks", args.coupling_sweep_tasks, "--method", method, "--damping-regime", args.default_damping_regime]
        per_seed = run_config(args, method, argv, f"coupling_mode/{method}")
        _record_axis(results, significance_rows, "coupling_mode", method, {}, per_seed, baseline)


def sweep_alpha_beta(args, results, significance_rows):
    baseline_alpha = [None, None]
    for alpha_val in args.alpha_grid:
        label = f"alpha{alpha_val}"
        argv = ["--tasks", args.default_tasks, "--method", "csho", "--damping-regime", args.default_damping_regime,
                "--alpha", str(alpha_val), "--beta", str(args.beta_grid[0])]
        per_seed = run_config(args, label, argv, f"alpha_beta/{label}")
        _record_axis(results, significance_rows, "alpha", label, {"alpha": alpha_val}, per_seed, baseline_alpha)

    baseline_beta = [None, None]
    for beta_val in args.beta_grid:
        label = f"beta{beta_val}"
        argv = ["--tasks", args.default_tasks, "--method", "csho", "--damping-regime", args.default_damping_regime,
                "--alpha", str(args.alpha_grid[0]), "--beta", str(beta_val)]
        per_seed = run_config(args, label, argv, f"alpha_beta/{label}")
        _record_axis(results, significance_rows, "beta", label, {"beta": beta_val}, per_seed, baseline_beta)


def sweep_damping_regime(args, results, significance_rows):
    baseline = [None, None]
    for regime in ["underdamped", "critically_damped", "overdamped"]:
        label = f"damping_{regime}"
        argv = ["--tasks", args.default_tasks, "--method", "csho", "--damping-regime", regime]
        per_seed = run_config(args, label, argv, f"damping_regime/{regime}")
        _record_axis(results, significance_rows, "damping_regime", label, {"regime": regime}, per_seed, baseline)


def sweep_diffusion_mode(args, results, significance_rows):
    baseline = [None, None]
    for method in ["csho_shared_g", "csho_independent_g"]:
        argv = ["--tasks", args.default_tasks, "--method", method, "--damping-regime", args.default_damping_regime]
        per_seed = run_config(args, method, argv, f"diffusion_mode/{method}")
        _record_axis(results, significance_rows, "diffusion_mode", method, {}, per_seed, baseline)


AXIS_FNS = {
    "n_tasks": sweep_n_tasks,
    "coupling_mode": sweep_coupling_mode,
    "alpha_beta": sweep_alpha_beta,
    "damping_regime": sweep_damping_regime,
    "diffusion_mode": sweep_diffusion_mode,
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablation grid for Experiment 1 (vision multi-task CSHO diffusion).")
    p.add_argument("--data-root", required=True)
    p.add_argument("--source", default="nyudv2", choices=["nyudv2", "pascal", "cityscapes"])
    p.add_argument("--out-dir", default="results/experiment_1_vision/ablations")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--backbone-channels", type=int, default=512)
    p.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true", default=True)
    p.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    p.add_argument("--score-blocks", type=int, default=4)
    p.add_argument("--score-heads", type=int, default=4)
    p.add_argument("--score-spatial-stride", type=int, default=2)
    p.add_argument("--n-diff-steps", type=int, default=2)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--max-n", type=int, default=5)
    p.add_argument("--default-tasks", default="depth,normals")
    p.add_argument("--coupling-sweep-tasks", default="depth,normals,segmentation")
    p.add_argument("--default-damping-regime", default="critically_damped")
    p.add_argument("--alpha-grid", default="0.5,1.0,2.0")
    p.add_argument("--beta-grid", default="0.25,0.5,1.0")
    p.add_argument("--axes", default=",".join(AXIS_FNS), help=f"comma-separated subset of {list(AXIS_FNS)}")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    args.alpha_grid = [float(v) for v in args.alpha_grid.split(",") if v.strip()]
    args.beta_grid = [float(v) for v in args.beta_grid.split(",") if v.strip()]
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
        experiment_name="Experiment 1 (vision) -- ablation grid",
        summary_rows=results, significance_rows=significance_rows, figure_paths=[],
        out_path=os.path.join(args.out_dir, "ablation_report.md"),
    )
    print(f"Ablation grid done. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
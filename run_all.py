from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

from core.reporting import render_experiment_report, write_csv, write_json
from core.stats import aggregate_over_seeds
from image import run_experiment as exp1
from pde import run_experiment as exp2
from synthetic import run_experiment as exp3

def flatten_summary(summary: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {f"{metric}_{k}": v for metric, stats in summary.items() for k, v in stats.items()}


def run_experiment_1(args: argparse.Namespace) -> Optional[Dict]:
    if not args.vision_data_root or not os.path.isdir(args.vision_data_root):
        print(f"[run_all] skipping experiment_1_vision: --vision-data-root ({args.vision_data_root!r}) not found")
        return None
    smoke = args.mode == "smoke"
    argv = [
        "--data-root", args.vision_data_root,
        "--source", args.vision_source,
        "--tasks", args.vision_tasks,
        "--method", "ccld",
        "--seeds", "0" if smoke else args.seeds,
        "--n-epochs", "1" if smoke else str(args.n_epochs),
        "--batch-size", "2" if smoke else str(args.batch_size),
        "--image-size", "32" if smoke else str(args.image_size),
        "--n-diff-steps", str(args.n_diff_steps),
        "--num-workers", "0",
        "--device", args.device,
        "--out-dir", os.path.join(args.out_dir, "experiment_1_vision"),
    ] + (["--no-pretrained-backbone"] if smoke else [])
    run_args = exp1.parse_args(argv)
    per_seed = {seed: exp1.train_one_seed(run_args, seed) for seed in run_args.seeds}
    return {"experiment": "experiment_1_vision", "per_seed": per_seed, "summary": aggregate_over_seeds(per_seed)}


def run_experiment_2(args: argparse.Namespace) -> Optional[Dict]:
    if not args.physics_data_root or not os.path.isdir(args.physics_data_root):
        print(f"[run_all] skipping experiment_2_physics: --physics-data-root ({args.physics_data_root!r}) not found")
        return None
    smoke = args.mode == "smoke"
    argv = [
        "--data-root", args.physics_data_root,
        "--problem", args.physics_problem,
        "--method", "ccld",
        "--seeds", "0" if smoke else args.seeds,
        "--n-epochs", "1" if smoke else str(args.n_epochs),
        "--batch-size", "2" if smoke else str(args.batch_size),
        "--max-samples", "10" if smoke else str(args.physics_max_samples),
        "--image-size", "16" if smoke else "128",
        "--n-diff-steps", str(args.n_diff_steps),
        "--num-workers", "0",
        "--device", args.device,
        "--out-dir", os.path.join(args.out_dir, "experiment_2_physics"),
    ]
    if args.physics_n_tasks:
        argv += ["--n-tasks", str(args.physics_n_tasks)]
    run_args = exp2.parse_args(argv)
    per_seed = {seed: exp2.train_one_seed(run_args, seed) for seed in run_args.seeds}
    return {"experiment": "experiment_2_physics", "per_seed": per_seed, "summary": aggregate_over_seeds(per_seed)}


def run_experiment_3(args: argparse.Namespace) -> Dict:
    smoke = args.mode == "smoke"
    argv = [
        "--N", str(args.synthetic_N),
        "--coupling-strength", str(args.synthetic_coupling_strength),
        "--method", "ccld",
        "--seeds", "0" if smoke else args.seeds,
        "--n-train-iters", "20" if smoke else str(args.synthetic_n_train_iters),
        "--n-samples", "200" if smoke else str(args.synthetic_n_samples),
        "--n-diff-steps", str(args.n_diff_steps),
        "--device", args.device,
        "--out-dir", os.path.join(args.out_dir, "experiment_3_synthetic"),
    ]
    if smoke:
        argv.append("--quick")
    run_args = exp3.parse_args(argv)
    per_seed = {seed: exp3.train_one_seed(run_args, seed) for seed in run_args.seeds}
    return {"experiment": "experiment_3_synthetic", "per_seed": per_seed, "summary": aggregate_over_seeds(per_seed)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run all three CCLD validation experiments (or a fast smoke pass over all of them).")
    p.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    p.add_argument("--out-dir", default="results/run_all")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-diff-steps", type=int, default=2)
    p.add_argument("--device", default="cpu")

    p.add_argument("--vision-data-root", default=None)
    p.add_argument("--vision-source", default="nyudv2")
    p.add_argument("--vision-tasks", default="depth,normals")
    p.add_argument("--image-size", type=int, default=128)

    p.add_argument("--physics-data-root", default=None)
    p.add_argument("--physics-problem", default="NS_heat")
    p.add_argument("--physics-n-tasks", type=int, default=None)
    p.add_argument("--physics-max-samples", type=int, default=500)

    p.add_argument("--synthetic-N", type=int, default=3)
    p.add_argument("--synthetic-coupling-strength", type=float, default=0.6)
    p.add_argument("--synthetic-n-train-iters", type=int, default=2000)
    p.add_argument("--synthetic-n-samples", type=int, default=4000)
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for fn in (run_experiment_1, run_experiment_2, run_experiment_3):
        print(f"[run_all] mode={args.mode} running {fn.__name__}")
        out = fn(args)
        if out is not None:
            results.append(out)
            print(f"[run_all] {out['experiment']} done: {out['summary']}")

    summary_rows = []
    for r in results:
        summary_rows.append({"experiment": r["experiment"], **flatten_summary(r["summary"])})
    write_csv(summary_rows, os.path.join(args.out_dir, "run_all_summary.csv"))
    write_json({"mode": args.mode, "results": results}, os.path.join(args.out_dir, "run_all_results.json"))
    render_experiment_report(
        experiment_name=f"CCLD validation suite -- run_all ({args.mode})",
        summary_rows=summary_rows, significance_rows=[], figure_paths=[],
        out_path=os.path.join(args.out_dir, "run_all_report.md"),
    )
    ran = [r["experiment"] for r in results]
    skipped = [n for n in ["experiment_1_vision", "experiment_2_physics", "experiment_3_synthetic"] if n not in ran]
    print(f"[run_all] done. ran={ran} skipped={skipped}. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
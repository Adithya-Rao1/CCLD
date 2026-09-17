from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from core.reporting import write_csv
from core.stats import aggregate_over_seeds, compare_configs
from pde.run_experiment import _build_pde_skew, make_dataset, parse_args, train_one_seed

CONDITIONS = ["symmetric_only", "skew_structured", "skew_generic_matched", "ddpm", "sdm"]
CCLD_CONDITIONS = ["symmetric_only", "skew_structured", "skew_generic_matched"]
BASELINE_CONDITIONS = ["ddpm", "sdm"]

REL_L2_METRIC_KEYS_BY_PROBLEM = {"TE_heat": ["Re{Ez}_rel_l2", "Im{Ez}_rel_l2", "T_rel_l2"]}
RESIDUAL_METRIC_KEYS_BY_PROBLEM = {"TE_heat": ["e_field_pde_residual", "heat_pde_residual"]}


def _derive_summary_metrics(metrics: Dict[str, float], problem: str) -> Dict[str, float]:
    rel_l2_keys = REL_L2_METRIC_KEYS_BY_PROBLEM.get(problem, [k for k in metrics if k.endswith("_rel_l2")])
    out = dict(metrics)
    vals = [metrics[k] for k in rel_l2_keys if k in metrics]
    out["mean_rel_l2"] = sum(vals) / len(vals) if vals else float("nan")
    return out


def build_sweep_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Skew-coupling condition sweep for PDE")
    p.add_argument("--config", default="pde/config.yaml")
    p.add_argument("--data-root", required=True)
    p.add_argument("--problem", default="TE_heat")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--theta-data-seed", type=int, default=0)
    p.add_argument("--theta-data-n-samples", type=int, default=256)
    p.add_argument("--structured-scale", type=float, default=1.0)
    p.add_argument("--generic-skew-seed", type=int, default=0)
    p.add_argument("--n-diff-steps", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/experiment_2_physics/skew_coupling_sweep")
    return p


def parse_sweep_args(argv=None) -> argparse.Namespace:
    p = build_sweep_arg_parser()
    args, extra_argv = p.parse_known_args(argv)
    args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.extra_argv = extra_argv
    return args


def _base_argv(args: argparse.Namespace) -> List[str]:
    return ["--config", args.config, "--data-root", args.data_root, "--problem", args.problem,
            "--n-diff-steps", str(args.n_diff_steps), "--device", args.device] + args.extra_argv


def _condition_argv(label: str, args: argparse.Namespace, generic_scale: Optional[float]) -> List[str]:
    base = _base_argv(args)
    if label == "symmetric_only":
        return base + ["--method", "ccld_pairwise", "--coupling-family", "mean_field", "--constant-k",
                        "--skew-coupling-family", "none"]
    if label == "skew_structured":
        return base + ["--method", "ccld_pairwise", "--coupling-family", "mean_field", "--constant-k",
                        "--skew-coupling-family", "te_heat_physics", "--skew-coupling-scale", str(args.structured_scale),
                        "--theta-data-seed", str(args.theta_data_seed), "--theta-data-n-samples", str(args.theta_data_n_samples)]
    if label == "skew_generic_matched":
        return base + ["--method", "ccld_pairwise", "--coupling-family", "mean_field", "--constant-k",
                        "--skew-coupling-family", "random", "--skew-coupling-scale", str(generic_scale),
                        "--skew-coupling-seed", str(args.generic_skew_seed)]
    if label in BASELINE_CONDITIONS:
        return base + ["--method", label]
    raise ValueError(label)


def _measure_norms(args: argparse.Namespace, N: int, device, train_ds, task_names) -> Dict[str, Optional[float]]:
    target_norm = None
    generic_scale = None
    generic_matched_norm = None
    if "skew_structured" in args.conditions or "skew_generic_matched" in args.conditions:
        struct_args = parse_args(_condition_argv("skew_structured", args, None))
        J_structured = _build_pde_skew(struct_args, N, device, train_ds=train_ds, task_names=task_names)
        target_norm = torch.linalg.norm(J_structured).item()
    if "skew_generic_matched" in args.conditions:
        if target_norm is None:
            raise ValueError("skew_generic_matched requires skew_structured to define the target norm")
        probe_argv = _base_argv(args) + ["--method", "ccld_pairwise", "--coupling-family", "mean_field", "--constant-k",
                                          "--skew-coupling-family", "random", "--skew-coupling-scale", "1.0",
                                          "--skew-coupling-seed", str(args.generic_skew_seed)]
        probe_args = parse_args(probe_argv)
        J_generic_raw = _build_pde_skew(probe_args, N, device)
        raw_norm = torch.linalg.norm(J_generic_raw).item()
        if raw_norm < 1e-12:
            raise ValueError
        generic_scale = target_norm / raw_norm
        generic_matched_norm = target_norm
    return {"structured_norm": target_norm, "generic_scale": generic_scale, "generic_matched_norm": generic_matched_norm}


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    if args.problem != "TE_heat" and ("skew_structured" in args.conditions or "skew_generic_matched" in args.conditions):
        raise ValueError

    device = torch.device(args.device)
    probe_args = parse_args(_base_argv(args) + ["--method", "ccld_pairwise"])
    train_ds = make_dataset(probe_args, probe_args.split)
    task_names = train_ds.task_names
    N = len(task_names)

    norms = _measure_norms(args, N, device, train_ds, task_names)
    write_csv([norms], os.path.join(args.out_dir, "norms.csv"))

    sig_metrics = ["mean_rel_l2"] + RESIDUAL_METRIC_KEYS_BY_PROBLEM.get(args.problem, [])
    summary_rows, per_seed_rows, sig_rows = [], [], []
    labels = [c for c in CCLD_CONDITIONS if c in args.conditions] + [c for c in BASELINE_CONDITIONS if c in args.conditions]
    cond_args_by_label = {label: parse_args(_condition_argv(label, args, norms["generic_scale"])) for label in labels}
    per_seed_by_condition: Dict[str, Dict[int, Dict[str, float]]] = {label: {} for label in labels}
    run_pairs = [(label, seed) for label in labels for seed in args.seeds]

    for label, seed in tqdm(run_pairs, desc="[skew_coupling_sweep] overall"):
        print(f"\n[skew_coupling_sweep] problem={args.problem} condition={label} seed={seed}")
        metrics = train_one_seed(cond_args_by_label[label], seed)
        metrics = _derive_summary_metrics(metrics, args.problem)
        per_seed_by_condition[label][seed] = metrics
        per_seed_rows.append({"condition": label, "seed": seed, **metrics})
        print(f"  mean_rel_l2={metrics['mean_rel_l2']:.4f}")

    for label in labels:
        cond_summary = aggregate_over_seeds(per_seed_by_condition[label])
        summary_rows.append({
            "condition": label,
            **{f"{m}_mean": cond_summary[m]["mean"] for m in sig_metrics if m in cond_summary},
            **{f"{m}_std": cond_summary[m]["std"] for m in sig_metrics if m in cond_summary},
        })

    if "symmetric_only" in per_seed_by_condition:
        baseline_per_seed = per_seed_by_condition["symmetric_only"]
        for label, per_seed in per_seed_by_condition.items():
            if label == "symmetric_only":
                continue
            sig = compare_configs(baseline_per_seed, per_seed, metric_names=sig_metrics)
            for metric, s in sig.items():
                sig_rows.append({"comparison": f"{label}_vs_symmetric_only", "metric": metric, **s})

    write_csv(summary_rows, os.path.join(args.out_dir, "sweep_summary.csv"))
    write_csv(per_seed_rows, os.path.join(args.out_dir, "sweep_per_seed.csv"))
    write_csv(sig_rows, os.path.join(args.out_dir, "sweep_significance.csv"))
    print(f"\n[skew_coupling_sweep] wrote results to {args.out_dir}")


if __name__ == "__main__":
    run(parse_sweep_args())
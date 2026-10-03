from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

from core.reporting import write_csv
from core.stats import aggregate_over_seeds, compare_configs


def _load_per_seed_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_per_seed_by_condition(rows: List[dict]) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    skip = {"condition", "seed"}
    for row in rows:
        cond = row["condition"]
        seed = int(row["seed"])
        metrics = {}
        for k, v in row.items():
            if k in skip or v in (None, ""):
                continue
            try:
                metrics[k] = float(v)
            except ValueError:
                continue
        out[cond][seed] = metrics
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sweep_per_seed.csv files")
    p.add_argument("--per-seed-csv", action="append", required=True, help="path to a sweep_per_seed.csv; pass multiple times, one per source run")
    p.add_argument("--baseline-condition", default="symmetric_only")
    p.add_argument("--sig-metrics", default="mean_rel_l2,e_field_pde_residual,heat_pde_residual")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    all_rows: List[dict] = []
    for path in args.per_seed_csv:
        all_rows.extend(_load_per_seed_csv(path))

    per_seed_by_condition = _to_per_seed_by_condition(all_rows)
    sig_metrics = [m.strip() for m in args.sig_metrics.split(",") if m.strip()]

    summary_rows = []
    for cond, per_seed in per_seed_by_condition.items():
        agg = aggregate_over_seeds(per_seed)
        row = {"condition": cond, "n_seeds": len(per_seed)}
        for m in sig_metrics:
            if m in agg:
                row[f"{m}_mean"] = agg[m]["mean"]
                row[f"{m}_std"] = agg[m]["std"]
        summary_rows.append(row)

    sig_rows = []
    if args.baseline_condition in per_seed_by_condition:
        baseline_per_seed = per_seed_by_condition[args.baseline_condition]
        for cond, per_seed in per_seed_by_condition.items():
            if cond == args.baseline_condition:
                continue
            sig = compare_configs(baseline_per_seed, per_seed, metric_names=sig_metrics)
            for metric, s in sig.items():
                sig_rows.append({"comparison": f"{cond}_vs_{args.baseline_condition}", "metric": metric, **s})
    else:
        print(f"Warning: baseline condition '{args.baseline_condition}' not found in merged data -- no significance computed")

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(all_rows, os.path.join(args.out_dir, "combined_sweep_per_seed.csv"))
    write_csv(summary_rows, os.path.join(args.out_dir, "combined_sweep_summary.csv"))
    write_csv(sig_rows, os.path.join(args.out_dir, "combined_sweep_significance.csv"))
    print(f"Merged {len(per_seed_by_condition)} conditions: {sorted(per_seed_by_condition.keys())}")
    for cond, per_seed in per_seed_by_condition.items():
        print(f"  {cond}: {sorted(per_seed.keys())}")
    print(f"Wrote combined results to {args.out_dir}/")

from __future__ import annotations

import argparse
import csv
import os
from typing import List

from core.reporting import write_csv


def _load_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _dedup_key(row: dict) -> tuple:
    if "method" in row:
        return (row.get("N"), row.get("method"), row.get("n_diff_steps"))
    return (row.get("N"), row.get("comparison"), row.get("metric"), row.get("n_diff_steps"))


def _merge(csv_paths: List[str]) -> List[dict]:
    merged: List[dict] = []
    seen: dict = {}
    for path in csv_paths:
        for row in _load_csv(path):
            key = _dedup_key(row)
            if key in seen:
                prev_row, prev_path = seen[key]
                mismatches = [k for k in row if k not in ("N", "method", "n_diff_steps") and row.get(k) != prev_row.get(k)]
                if mismatches:
                    print(f"Warning: duplicate key {key} in both {prev_path} and {path} with differing values for {mismatches} "
                          f"-- keeping the value from {prev_path} (earlier --summary-csv/--significance-csv wins)")
                continue
            seen[key] = (row, path)
            merged.append(row)
    return merged


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple already-aggregated stepcount_sweep_summary.csv")
    p.add_argument("--summary-csv", action="append", required=True, help="repeatable -- one stepcount_sweep_summary.csv per source sweep")
    p.add_argument("--significance-csv", action="append", default=[], help="repeatable -- one stepcount_sweep_significance.csv per source sweep (optional, merged if given)")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    merged_summary = _merge(args.summary_csv)
    write_csv(merged_summary, os.path.join(args.out_dir, "stepcount_sweep_summary.csv"))
    print(f"Merged {len(args.summary_csv)} summary CSVs -> {len(merged_summary)} rows")

    if args.significance_csv:
        merged_sig = _merge(args.significance_csv)
        write_csv(merged_sig, os.path.join(args.out_dir, "stepcount_sweep_significance.csv"))
        print(f"Merged {len(args.significance_csv)} significance CSVs -> {len(merged_sig)} rows")

    print(f"Wrote merged results to {args.out_dir}/")
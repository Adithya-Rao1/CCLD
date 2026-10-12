from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch

from core.reporting import write_csv
from core.stats import compare_configs
from pde.run_experiment import parse_args as pde_parse_args, parse_float_list
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas
from synthetic.exact_dsm import analytic_score_precision_n

N_TE_HEAT = 2  # TE_heat's task count (Re{Ez}, T)
Q_GRID = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 2.0]  


def _calibrate_gammas(args: argparse.Namespace, beta_override: Optional[float] = None) -> tuple:
    alpha_list = parse_float_list(args.alpha, N_TE_HEAT)
    beta = args.beta if beta_override is None else str(beta_override)
    beta_list = parse_float_list(beta, N_TE_HEAT)
    return calibrate_coupled_gammas(
        alpha_list[0], beta_list[0], args.k_reference, args.k_reference,
        N_TE_HEAT, regime=args.damping_regime, target_zeta=args.target_zeta,
    )


def precision_diagonal_sweep(gamma_self: float, gamma_couple: float) -> Dict[float, float]:
    out = {}
    for q in Q_GRID:
        precision, _ = analytic_score_precision_n(N_TE_HEAT, gamma_self, gamma_couple, torch.tensor(q))
        out[q] = precision[0, 0].item()
    return out


def run_precision_comparison(args: argparse.Namespace) -> List[dict]:
    gamma_self_ccld, gamma_couple_ccld = _calibrate_gammas(args)
    gamma_self_cld, gamma_couple_cld = _calibrate_gammas(args, beta_override=0.0)
    assert abs(gamma_self_cld - gamma_self_ccld) < 1e-9, "matched calibration requires identical gamma_self"
    assert gamma_couple_cld == 0.0

    ccld_precision = precision_diagonal_sweep(gamma_self_ccld, gamma_couple_ccld)
    cld_precision = precision_diagonal_sweep(gamma_self_cld, gamma_couple_cld)

    rows = []
    for q in Q_GRID:
        pct_lower = (cld_precision[q] - ccld_precision[q]) / cld_precision[q] * 100.0
        rows.append({
            "q": q, "ccld_precision_diag": ccld_precision[q], "cld_precision_diag": cld_precision[q],
            "ccld_pct_lower_than_cld": pct_lower,
        })
    return rows


def load_per_seed_metric(results_dir: str, method: str, metric: str) -> Dict[int, Dict[str, float]]:
    path = os.path.join(results_dir, f"{method}_results.json")
    with open(path) as f:
        d = json.load(f)
    return {int(s): {metric: m[metric]} for s, m in d["per_seed"].items()}


def run_explosion_comparison(results_dir: str) -> List[dict]:
    cld_per_seed = load_per_seed_metric(results_dir, "ccld_independent", "explosion_events")
    ccld_per_seed = load_per_seed_metric(results_dir, "ccld", "explosion_events")
    sig = compare_configs(cld_per_seed, ccld_per_seed, metric_names=["explosion_events"])
    rows = []
    for metric, s in sig.items():
        rows.append({
            "metric": metric, "cld_mean": s["baseline_mean"], "ccld_mean": s["treatment_mean"],
            "delta": s["delta"], "p_value": s["p_value"], "n": s["n"],
        })
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TE_heat CCLD-vs-CLD training-stability"
    )
    p.add_argument("--results-dir", default=None, help="directory containing ccld_results.json and ccld_independent_results.json.")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pde_args = pde_parse_args([
        "--config", "pde/config.yaml", "--data-root", "pde/multiphysics-bench",
        "--problem", "TE_heat", "--method", "ccld",
    ])

    precision_rows = run_precision_comparison(pde_args)
    write_csv(precision_rows, os.path.join(args.out_dir, "te_heat_precision_comparison.csv"))
    print(f"{'q':>10} {'ccld_prec':>12} {'cld_prec':>12} {'pct_lower':>10}")
    for row in precision_rows:
        print(f"{row['q']:>10.4g} {row['ccld_precision_diag']:>12.4f} "
              f"{row['cld_precision_diag']:>12.4f} {row['ccld_pct_lower_than_cld']:>9.1f}%")

    if args.results_dir is not None:
        explosion_rows = run_explosion_comparison(args.results_dir)
        write_csv(explosion_rows, os.path.join(args.out_dir, "te_heat_explosion_significance.csv"))
        print(f"\n{'metric':>18} {'cld_mean':>10} {'ccld_mean':>10} {'delta':>10} {'p_value':>10} {'n':>4}")
        for row in explosion_rows:
            print(f"{row['metric']:>18} {row['cld_mean']:>10.4f} {row['ccld_mean']:>10.4f} "
                  f"{row['delta']:>10.4f} {row['p_value']:>10.4f} {row['n']:>4}")
    else:
        print("\n--results-dir not given -- skipping explosion-event significance comparison.")

    print(f"\nWrote {args.out_dir}/te_heat_precision_comparison.csv"
          + (f" and te_heat_explosion_significance.csv" if args.results_dir is not None else ""))
from __future__ import annotations

import argparse

import torch

from synthetic.drift_coupled_gamma import calibrate_coupled_gammas
from synthetic.exact_dsm import calibrate_sigma_for_leak
from synthetic.ground_truth_sde import make_ground_truth
from synthetic.run_experiment import _vp_linear_time_scale

ALPHA_V = 1.0
K_REFERENCE = 1.0
TARGET_ZETA = 1.0
TIME_SCALE_FN = _vp_linear_time_scale


def calibrate(N: int, n_diff_steps: int, dt: float, coupling_strength: float, leak_fraction: float, gt_seed: int) -> float:
    gamma_self, gamma_couple = calibrate_coupled_gammas(ALPHA_V, 0.0, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
    gt = make_ground_truth(N, coupling_strength, seed=gt_seed, base_decay=1.0, sigma_scale=1.0)
    cov_data = gt.stationary_covariance()
    return calibrate_sigma_for_leak(
        N, gamma_self, gamma_couple, [ALPHA_V] * N, K_REFERENCE, cov_data,
        n_diff_steps, dt, TIME_SCALE_FN, leak_fraction=leak_fraction,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact (matrix-recursion, no simulation) sigma_N calibration for a target correlation-leak fraction")
    p.add_argument("--n-sweep", default="2,3,4,5", help="comma-separated N values")
    p.add_argument("--n-diff-steps", type=int, required=True)
    p.add_argument("--dt", type=float, default=None, help="defaults to 1/n_diff_steps")
    p.add_argument("--coupling-strength", type=float, default=0.6)
    p.add_argument("--leak-fraction", type=float, default=0.01)
    p.add_argument("--gt-seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    n_sweep = [int(n) for n in args.n_sweep.split(",") if n.strip()]
    dt = args.dt if args.dt is not None else 1.0 / args.n_diff_steps

    print(f"n_diff_steps={args.n_diff_steps} dt={dt} leak_fraction={args.leak_fraction} coupling_strength={args.coupling_strength}")
    sigma_by_n = {}
    for N in n_sweep:
        sigma = calibrate(N, args.n_diff_steps, dt, args.coupling_strength, args.leak_fraction, args.gt_seed)
        sigma_by_n[N] = round(sigma, 4)
        print(f"N={N}: sigma={sigma:.4f}")
    print("\nSIGMA_BY_N =", sigma_by_n)

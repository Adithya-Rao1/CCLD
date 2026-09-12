from __future__ import annotations

import argparse

from synthetic.drift_coupled_gamma import calibrate_coupled_gammas, calibrate_sigma_fdt_coupled

ALPHA_V = 1.0
K_REFERENCE = 1.0
TARGET_ZETA = 1.0

def calibrate(N: int, beta: float, target_variance: float = 1.0):
    gamma_self, gamma_couple = calibrate_coupled_gammas(ALPHA_V, beta, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
    return calibrate_sigma_fdt_coupled(gamma_self, gamma_couple, N, target_variance=target_variance)

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-sweep", default="2,3,4,5", help="comma-separated N values")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--target-variance", type=float, default=1.0,)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    n_sweep = [int(n) for n in args.n_sweep.split(",") if n.strip()]

    print(f"beta={args.beta} target_variance={args.target_variance}  (G = a*I + b*C)")
    sigma_ab_by_n = {}
    for N in n_sweep:
        a, b = calibrate(N, args.beta, args.target_variance)
        sigma_ab_by_n[N] = (round(a, 4), round(b, 4))
        gamma_self, gamma_couple = calibrate_coupled_gammas(ALPHA_V, args.beta, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
        print(f"N={N}: a={a:.4f} b={b:.4f}  (gamma_self={gamma_self:.4f}, gamma_couple={gamma_couple:.4f})")
    print("\nSIGMA_AB_BY_N =", sigma_ab_by_n)
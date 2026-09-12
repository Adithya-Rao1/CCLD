from __future__ import annotations

import argparse
import os

import torch

from core.coupling import build_coupling_matrix, random_heterogeneous_coupling
from core.hypoellipticity import hypoellipticity_check_coupled_gamma
from core.reporting import write_csv
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas_spectral, calibrate_sigma_fdt_spectral

ALPHA_V = 1.0
BETA = 1.0
K_REFERENCE = 1.0
TARGET_ZETA = 1.0
EPSILON_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SWEEP = [3, 5, 8]
N_SEEDS_PER_EPSILON = 3


def _make_K(N: int, k_ref: float):
    K_self = [[torch.tensor([[k_ref]])] for _ in range(N)]
    K_global = [torch.tensor([[k_ref]]) for _ in range(N)]
    return K_self, K_global


def run(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    K_self, K_global = None, None
    t, T = torch.tensor(5.0), 10

    for N in N_SWEEP:
        K_self, K_global = _make_K(N, K_REFERENCE)
        for epsilon in EPSILON_SWEEP:
            for seed in range(N_SEEDS_PER_EPSILON):
                if epsilon == 0.0:
                    C = build_coupling_matrix(N, mode="mean_field")
                    if seed > 0:
                        continue  # epsilon=0 --> mean-field
                else:
                    C = random_heterogeneous_coupling(N, epsilon=epsilon, seed=2000 + seed)
                L = torch.eye(N, dtype=C.dtype) - C
                mu = torch.linalg.eigvalsh(L)
                spread = (mu[1:].max() - mu[1:].min()).item()

                Gamma = calibrate_coupled_gammas_spectral(ALPHA_V, BETA, K_REFERENCE, K_REFERENCE, C, target_zeta=TARGET_ZETA)
                G = calibrate_sigma_fdt_spectral(Gamma, C, target_variance=1.0)
                res = hypoellipticity_check_coupled_gamma(
                    N, K_self, K_global, t, T, [ALPHA_V] * N, [BETA] * N, C, True, G, damping_matrix=Gamma,
                )
                rows.append({
                    "N": N, "epsilon": epsilon, "seed": seed, "eigenvalue_spread": spread,
                    "min_eig": res["min_eig"], "passed": res["passed"],
                })
                print(f"N={N} epsilon={epsilon:.2f} seed={seed}: spread={spread:.4f} min_eig={res['min_eig']:.6f} passed={res['passed']}")

    write_csv(rows, os.path.join(out_dir, "hypoellipticity_vs_heterogeneity.csv"))
    print(f"\nWrote {os.path.join(out_dir, 'hypoellipticity_vs_heterogeneity.csv')}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controllability margin vs. coupling heterogeneity epsilon")
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/hypoellipticity_vs_heterogeneity")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.out_dir)

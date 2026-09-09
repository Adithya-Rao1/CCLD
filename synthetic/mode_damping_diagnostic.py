from __future__ import annotations

import argparse
import os

import torch

from core.coupling import block_coupling, build_coupling_matrix, random_heterogeneous_coupling
from core.reporting import write_csv
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas, calibrate_coupled_gammas_spectral
from synthetic.exact_dsm import _extract_Avx_Avv_coupled_gamma

ALPHA_V = 1.0
BETA = 1.0
K_REFERENCE = 1.0
TARGET_ZETA = 1.0


def per_mode_zeta(N: int, C: torch.Tensor, Gamma: torch.Tensor) -> torch.Tensor:
    const_one = lambda t, T: torch.tensor(1.0)
    A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(
        N, None, None, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, C, t=1, T=1, constant_k=True,
        time_scale_fn=const_one, damping_matrix=Gamma,
    )
    L = torch.eye(N, dtype=C.dtype) - C
    _, U = torch.linalg.eigh(L)
    Omega_k = torch.sqrt((-torch.diagonal(U.T @ A_vx @ U)).clamp_min(0.0))
    Gamma_k = -torch.diagonal(U.T @ A_vv @ U)
    return Gamma_k / (2.0 * Omega_k.clamp_min(1e-12))


def naive_gamma_matrix(N: int, C: torch.Tensor) -> torch.Tensor:
    gamma_self, gamma_couple = calibrate_coupled_gammas(ALPHA_V, BETA, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
    eye = torch.eye(N, dtype=C.dtype)
    return gamma_self * eye + gamma_couple * (eye - C)


def run(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    configs = [
        (4, "mean_field", lambda N: build_coupling_matrix(N, mode="mean_field")),
        (4, "heterogeneous_eps1.0", lambda N: random_heterogeneous_coupling(N, epsilon=1.0, seed=11)),
        (5, "heterogeneous_eps1.0", lambda N: random_heterogeneous_coupling(N, epsilon=1.0, seed=11)),
        (5, "block_w20_0.01", lambda N: block_coupling(N, [2, 3], w_in=20.0, w_out=0.01)),
        (8, "block_w20_0.01", lambda N: block_coupling(N, [4, 4], w_in=20.0, w_out=0.01)),
    ]

    for N, label, coupling_fn in configs:
        C = coupling_fn(N)
        Gamma_spectral = calibrate_coupled_gammas_spectral(ALPHA_V, BETA, K_REFERENCE, K_REFERENCE, C, target_zeta=TARGET_ZETA)
        Gamma_naive = naive_gamma_matrix(N, C)
        zeta_spectral = per_mode_zeta(N, C, Gamma_spectral)
        zeta_naive = per_mode_zeta(N, C, Gamma_naive)
        for k in range(N):
            rows.append({
                "N": N, "condition": label, "mode": k,
                "zeta_spectral": zeta_spectral[k].item(), "zeta_naive": zeta_naive[k].item(),
                "target_zeta": TARGET_ZETA,
            })
        max_dev_spectral = (zeta_spectral - TARGET_ZETA).abs().max().item()
        max_dev_naive = (zeta_naive - TARGET_ZETA).abs().max().item()
        print(f"N={N:>2} {label:>22}: max|zeta-1| spectral={max_dev_spectral:.2e}  naive={max_dev_naive:.4f}")

    write_csv(rows, os.path.join(out_dir, "mode_damping_diagnostic.csv"))
    print(f"\nWrote {os.path.join(out_dir, 'mode_damping_diagnostic.csv')}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-mode critical-damping validation: spectral vs naive affine calibration")
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/mode_damping_diagnostic")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.out_dir)

from __future__ import annotations

import argparse
import math

import torch

from synthetic.drift_coupled_gamma import calibrate_coupled_gammas
from synthetic.exact_dsm import calibrate_sigma_for_leak

N_DIFF_STEPS = 20
DT = 1.0 / N_DIFF_STEPS  # A4
K_REFERENCE = 1.0
ALPHA = 1.0
BETA = 0.5  # pde/run_experiment.py's --beta default (A3)
LEAK_FRACTION = 0.01  # pde/run_experiment.py's --leak-fraction default
N_TRIALS_PER_CONFIG = 100
VAR_LO, VAR_HI = 0.5, 2.0  # A1


def _calibrate_or_classify(N, gamma_self, cov_data):
    try:
        sigma = calibrate_sigma_for_leak(
            N, gamma_self, [ALPHA] * N, K_REFERENCE, cov_data,
            N_DIFF_STEPS, DT, None, leak_fraction=LEAK_FRACTION,
        )
    except Exception:
        return "error", float("nan")
    if isinstance(sigma, complex):
        return "complex", abs(sigma)
    if isinstance(sigma, float) and math.isnan(sigma):
        return "error", float("nan")
    return "real", abs(sigma)


def _mean_offdiag(corr: torch.Tensor) -> float:
    N = corr.shape[0]
    off = corr - torch.diag(torch.diag(corr))
    return off.sum().item() / (N * (N - 1))


def _random_variances(N: int, generator: torch.Generator) -> torch.Tensor:
    return VAR_LO + (VAR_HI - VAR_LO) * torch.rand(N, generator=generator)


def _make_equicorrelated_cov(N: int, rho: float, generator: torch.Generator) -> torch.Tensor:
    std = _random_variances(N, generator)
    corr = torch.full((N, N), rho)
    corr.fill_diagonal_(1.0)
    return corr * std.view(-1, 1) * std.view(1, -1)


def sweep_equicorrelated(N: int, gamma_self: float, generator: torch.Generator) -> None:
    print(f"\n--- Sweep 1 (control): equicorrelated, N={N} ---")
    print(f"{'rho_target':>10s}  {'complex_frac':>13s}  {'error_frac':>11s}  {'mean_|sigma|':>13s}")
    lo = -1.0 / (N - 1) + 0.1  # PSD bound for a compound-symmetric matrix
    rhos = [round(lo + 0.1 * i, 3) for i in range(int(round((0.9 - lo) / 0.1)) + 1)]
    for rho in rhos:
        counts = {"complex": 0, "error": 0, "real": 0}
        sigmas = []
        for _ in range(N_TRIALS_PER_CONFIG):
            cov_data = _make_equicorrelated_cov(N, rho, generator)
            kind, mag = _calibrate_or_classify(N, gamma_self, cov_data)
            counts[kind] += 1
            if not math.isnan(mag):
                sigmas.append(mag)
        mean_sigma = sum(sigmas) / len(sigmas) if sigmas else float("nan")
        print(f"{rho:10.3f}  {counts['complex'] / N_TRIALS_PER_CONFIG:13.1%}  "
              f"{counts['error'] / N_TRIALS_PER_CONFIG:11.1%}  {mean_sigma:13.6f}")

def _shrink_to_valid_correlation(corr: torch.Tensor, eps: float = 1e-3):
    """Shrinks `corr` toward the identity by the minimal amount needed for it to be a valid
    (PSD) correlation matrix, preserving the unit diagonal exactly (A2)."""
    N = corr.shape[0]
    min_eig = torch.linalg.eigvalsh(corr).min().item()
    if min_eig >= eps:
        return corr, 0.0
    lam = (eps - min_eig) / (1.0 - min_eig)
    lam = min(max(lam, 0.0), 1.0)
    return (1 - lam) * corr + lam * torch.eye(N), lam


def _make_block_cov(N: int, n1: int, rho_within: float, rho_between: float, generator: torch.Generator):
    group = torch.zeros(N, dtype=torch.long)
    group[n1:] = 1
    corr = torch.where(
        (group.view(-1, 1) == group.view(1, -1)), torch.tensor(rho_within), torch.tensor(rho_between),
    ).float()
    corr.fill_diagonal_(1.0)
    corr, lam = _shrink_to_valid_correlation(corr)
    std = _random_variances(N, generator) 
    cov = corr * std.view(-1, 1) * std.view(1, -1)
    return cov, corr, lam


def sweep_mixed_sign(N: int, n1: int, gamma_self: float, generator: torch.Generator) -> None:
    rho_within = 0.7
    print(f"\n--- Sweep 2 (stress test): mixed-sign 2-block + heterogeneous variance, N={N}, "
          f"split=({n1},{N - n1}), rho_within={rho_within} ---")
    print(f"{'rho_between':>11s}  {'shrink_lam':>10s}  {'realized_rho_true':>18s}  "
          f"{'complex_frac':>13s}  {'error_frac':>11s}  {'mean_|sigma|':>13s}")
    for rho_between in [round(-0.9 + 0.1 * i, 2) for i in range(19)]:
        counts = {"complex": 0, "error": 0, "real": 0}
        sigmas, lams, realized_rhos = [], [], []
        for _ in range(N_TRIALS_PER_CONFIG):
            cov_data, corr, lam = _make_block_cov(N, n1, rho_within, rho_between, generator)
            lams.append(lam)
            realized_rhos.append(_mean_offdiag(corr))
            kind, mag = _calibrate_or_classify(N, gamma_self, cov_data)
            counts[kind] += 1
            if not math.isnan(mag):
                sigmas.append(mag)
        mean_sigma = sum(sigmas) / len(sigmas) if sigmas else float("nan")
        print(f"{rho_between:11.2f}  {sum(lams) / len(lams):10.3f}  "
              f"{sum(realized_rhos) / len(realized_rhos):18.4f}  "
              f"{counts['complex'] / N_TRIALS_PER_CONFIG:13.1%}  "
              f"{counts['error'] / N_TRIALS_PER_CONFIG:11.1%}  {mean_sigma:13.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)

    for N in (3, 12):
        gamma_self, _ = calibrate_coupled_gammas(
            ALPHA, BETA, K_REFERENCE, K_REFERENCE, N, regime="critically_damped", target_zeta=None,
        )
        print(f"\n=== N={N}  (gamma_self={gamma_self:.4f}, dt={DT}, n_diff_steps={N_DIFF_STEPS}, "
              f"{N_TRIALS_PER_CONFIG} trials/config) ===")
        sweep_equicorrelated(N, gamma_self, generator)
        # n1=1: one outlier task (meant to simulate a single-driver problem like E_flow's ec_V vs u_flow/v_flow)
        sweep_mixed_sign(N, 1, gamma_self, generator)
        if N > 3:
            sweep_mixed_sign(N, N // 2, gamma_self, generator)

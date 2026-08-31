"""Investigates how often synthetic/exact_dsm.py::calibrate_sigma_for_leak returns a complex
sigma, discovered while building tests/test_pde_reverse_sde_stability.py (see pde/README.md's
"FNO reverse-SDE divergence" section, "Separately flagged" paragraph).

The formula computes (k / snr_target) ** 0.5, where k is proportional to
snr_ref = roundtrip_leak_snr(...) = x / (1 - x), x = rho_leak / rho_true, and rho_true is
cov_data's mean pairwise off-diagonal task correlation. Nothing guards against rho_true <= 0 or
against x falling outside (0, 1) -- either makes snr_ref (and therefore k) negative, so
`** 0.5` on a Python float silently returns a complex number. Downstream, build_g_matrix_n casts
that into a real tensor, discarding the imaginary part (the "Casting complex values to real
discards the imaginary part" UserWarning some runs print) and leaving sigma's real part, which
can be near-zero -- silently corrupting the calibrated diffusion strength with no visible error.

This script constructs synthetic cov_data with a CONTROLLED target mean pairwise correlation
(equicorrelated structure: every off-diagonal task pair has the same correlation rho, task
variances randomized per trial) and sweeps rho across a grid, at N=3 (TE_heat/E_flow's task
count) and N=12 (VA's task count), using the corrected dt=1/n_diff_steps production config
(pde/run_experiment.py's dt/T_max=1.0 fix). Reports, per rho, the fraction of trials that produce
a complex sigma, quantifying how much of a plausible real-data correlation regime is at risk --
not just whether the bug is theoretically reachable (already established: pde/README.md).

Usage: python -m tests.investigate_complex_sigma_bug
"""

from __future__ import annotations

import argparse
import math

import torch

from synthetic.drift_coupled_gamma import calibrate_coupled_gammas
from synthetic.exact_dsm import calibrate_sigma_for_leak

N_DIFF_STEPS = 20
DT = 1.0 / N_DIFF_STEPS  # corrected T_max=1.0 convention, matches pde/run_experiment.py's fix
K_REFERENCE = 1.0
ALPHA = 1.0
BETA = 0.5  # matches pde/run_experiment.py's --beta default
LEAK_FRACTION = 0.01  # matches pde/run_experiment.py's --leak-fraction default
N_TRIALS_PER_RHO = 200
RHO_STEP = 0.1


def _rho_grid(N: int) -> list:
    # Compound-symmetric (equicorrelated) N x N correlation matrices are only PSD for
    # rho > -1/(N-1); stay strictly inside that bound.
    lo = -1.0 / (N - 1) + RHO_STEP
    n_steps = int(round((0.9 - lo) / RHO_STEP))
    return [round(lo + RHO_STEP * i, 3) for i in range(n_steps + 1)]


def _make_equicorrelated_cov(N: int, rho: float, generator: torch.Generator) -> torch.Tensor:
    # Variances random in [0.5, 2.0] -- arbitrary, physically-agnostic scale; only the
    # correlation structure (rho) is what this investigation controls and cares about.
    std = 0.5 + 1.5 * torch.rand(N, generator=generator)
    corr = torch.full((N, N), rho)
    corr.fill_diagonal_(1.0)
    return corr * std.view(-1, 1) * std.view(1, -1)


def sweep(N: int, seed: int) -> None:
    gamma_self, gamma_couple = calibrate_coupled_gammas(
        ALPHA, BETA, K_REFERENCE, K_REFERENCE, N, regime="critically_damped", target_zeta=None,
    )
    generator = torch.Generator().manual_seed(seed)
    print(f"\n=== N={N}  (gamma_self={gamma_self:.4f}, gamma_couple={gamma_couple:.4f}, "
          f"dt={DT}, n_diff_steps={N_DIFF_STEPS}, {N_TRIALS_PER_RHO} trials/rho) ===")
    print(f"{'rho_true':>10s}  {'complex_frac':>13s}  {'error_frac':>11s}  {'mean_|sigma|_real':>18s}")
    for rho in _rho_grid(N):
        n_complex, n_error = 0, 0
        sigmas = []
        for _ in range(N_TRIALS_PER_RHO):
            cov_data = _make_equicorrelated_cov(N, rho, generator)
            try:
                sigma = calibrate_sigma_for_leak(
                    N, gamma_self, gamma_couple, [ALPHA] * N, K_REFERENCE, cov_data,
                    N_DIFF_STEPS, DT, None, leak_fraction=LEAK_FRACTION,
                )
            except Exception:
                n_error += 1
                continue
            if isinstance(sigma, complex):
                n_complex += 1
                sigmas.append(abs(sigma))
            elif isinstance(sigma, float) and math.isnan(sigma):
                n_error += 1
            else:
                sigmas.append(abs(sigma))
        mean_sigma = sum(sigmas) / len(sigmas) if sigmas else float("nan")
        print(f"{rho:10.3f}  {n_complex / N_TRIALS_PER_RHO:13.1%}  {n_error / N_TRIALS_PER_RHO:11.1%}  "
              f"{mean_sigma:18.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for N in (3, 12):
        sweep(N, seed=args.seed)

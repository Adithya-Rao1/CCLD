from __future__ import annotations

import argparse

import torch

from synthetic.drift_coupled_gamma import calibrate_coupled_gammas
from synthetic.exact_dsm import roundtrip_propagator

N_DIFF_STEPS = 20
DT = 1.0 / N_DIFF_STEPS  # A4
K_REFERENCE = 1.0
ALPHA = 1.0
BETA = 0.5  # pde/run_experiment.py's --beta default (A3)
LEAK_FRACTION = 0.01  # pde/run_experiment.py's --leak-fraction default
SNR_TARGET = LEAK_FRACTION / (1 - LEAK_FRACTION)  # matches calibrate_sigma_for_leak
N_TRIALS_PER_CONFIG = 2000
VAR_LO, VAR_HI = 0.5, 2.0  # A1


def _random_variances_batch(N: int, B: int, generator: torch.Generator) -> torch.Tensor:
    return VAR_LO + (VAR_HI - VAR_LO) * torch.rand(B, N, generator=generator)


def _batched_classify(Phi_x: torch.Tensor, Sigma_x: torch.Tensor, cov_data: torch.Tensor):
    # cov_data: (B, N, N). Phi_x, Sigma_x: (N, N).
    N = Phi_x.shape[0]
    cov_final = Phi_x @ cov_data @ Phi_x.T + Sigma_x

    def corr_matrix(cov: torch.Tensor) -> torch.Tensor:
        std = cov.diagonal(dim1=-2, dim2=-1).sqrt()
        return cov / (std.unsqueeze(-1) * std.unsqueeze(-2))

    off_mask = ~torch.eye(N, dtype=torch.bool)
    corr_true_off = corr_matrix(cov_data)[:, off_mask]
    corr_leak_off = corr_matrix(cov_final)[:, off_mask]

    eps = 1e-6
    valid = corr_true_off.abs() >= eps
    safe_denom = torch.where(valid, corr_true_off, torch.ones_like(corr_true_off))
    ratio = torch.where(valid, corr_leak_off / safe_denom, torch.zeros_like(corr_leak_off))
    counts = valid.sum(dim=-1)
    x = ratio.sum(dim=-1) / counts.clamp_min(1)
    snr_ref = torch.where(counts == 0, torch.zeros_like(x), x / (1 - x))

    is_complex = snr_ref < 0
    is_error = torch.isnan(snr_ref) | torch.isinf(snr_ref)
    sigma_mag = torch.sqrt(snr_ref.abs() / SNR_TARGET)
    return is_complex, is_error, sigma_mag


def _mean_offdiag(corr: torch.Tensor) -> float:
    N = corr.shape[0]
    off = corr - torch.diag(torch.diag(corr))
    return off.sum().item() / (N * (N - 1))


def _make_equicorrelated_cov_batch(N: int, rho: float, B: int, generator: torch.Generator) -> torch.Tensor:
    std = _random_variances_batch(N, B, generator)
    corr = torch.full((N, N), rho)
    corr.fill_diagonal_(1.0)
    return corr.unsqueeze(0) * std.unsqueeze(-1) * std.unsqueeze(-2)


def sweep_equicorrelated(N: int, Phi_x: torch.Tensor, Sigma_x: torch.Tensor, generator: torch.Generator) -> None:
    print(f"\n--- Sweep 1 (control): equicorrelated, N={N} ---")
    print(f"{'rho_target':>10s}  {'complex_frac':>13s}  {'error_frac':>11s}  {'mean_|sigma|':>13s}")
    lo = -1.0 / (N - 1) + 0.1  # PSD bound for a compound-symmetric matrix
    rhos = [round(lo + 0.1 * i, 3) for i in range(int(round((0.9 - lo) / 0.1)) + 1)]
    for rho in rhos:
        cov_data = _make_equicorrelated_cov_batch(N, rho, N_TRIALS_PER_CONFIG, generator)
        is_complex, is_error, sigma_mag = _batched_classify(Phi_x, Sigma_x, cov_data)
        valid_sigma = sigma_mag[~is_error]
        mean_sigma = valid_sigma.mean().item() if valid_sigma.numel() else float("nan")
        print(f"{rho:10.3f}  {is_complex.float().mean().item():13.1%}  "
              f"{is_error.float().mean().item():11.1%}  {mean_sigma:13.6f}")


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


def _make_block_cov_batch(N: int, n1: int, rho_within: float, rho_between: float, B: int, generator: torch.Generator):
    group = torch.zeros(N, dtype=torch.long)
    group[n1:] = 1
    corr = torch.where(
        (group.view(-1, 1) == group.view(1, -1)), torch.tensor(rho_within), torch.tensor(rho_between),
    ).float()
    corr.fill_diagonal_(1.0)
    corr, lam = _shrink_to_valid_correlation(corr)
    std = _random_variances_batch(N, B, generator) 
    cov = corr.unsqueeze(0) * std.unsqueeze(-1) * std.unsqueeze(-2)
    return cov, corr, lam


def sweep_mixed_sign(N: int, n1: int, Phi_x: torch.Tensor, Sigma_x: torch.Tensor, generator: torch.Generator) -> None:
    rho_within = 0.7
    print(f"\n--- Sweep 2 (stress test): mixed-sign 2-block + heterogeneous variance, N={N}, "
          f"split=({n1},{N - n1}), rho_within={rho_within} ---")
    print(f"{'rho_between':>11s}  {'shrink_lam':>10s}  {'realized_rho_true':>18s}  "
          f"{'complex_frac':>13s}  {'error_frac':>11s}  {'mean_|sigma|':>13s}")
    for rho_between in [round(-0.9 + 0.1 * i, 2) for i in range(19)]:
        cov_data, corr, lam = _make_block_cov_batch(N, n1, rho_within, rho_between, N_TRIALS_PER_CONFIG, generator)
        is_complex, is_error, sigma_mag = _batched_classify(Phi_x, Sigma_x, cov_data)
        valid_sigma = sigma_mag[~is_error]
        mean_sigma = valid_sigma.mean().item() if valid_sigma.numel() else float("nan")
        print(f"{rho_between:11.2f}  {lam:10.3f}  {_mean_offdiag(corr):18.4f}  "
              f"{is_complex.float().mean().item():13.1%}  {is_error.float().mean().item():11.1%}  "
              f"{mean_sigma:13.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)

    for N in (3, 12):
        gamma_self, _ = calibrate_coupled_gammas(
            ALPHA, BETA, K_REFERENCE, K_REFERENCE, N, regime="critically_damped", target_zeta=None,
        )
        Phi_x, Sigma_x = roundtrip_propagator(N, gamma_self, [ALPHA] * N, K_REFERENCE, N_DIFF_STEPS, DT, None)
        print(f"\n=== N={N}  (gamma_self={gamma_self:.4f}, dt={DT}, n_diff_steps={N_DIFF_STEPS}, "
              f"{N_TRIALS_PER_CONFIG} trials/config) ===")
        sweep_equicorrelated(N, Phi_x, Sigma_x, generator)
        # n1=1: one outlier task (meant to simulate a single-driver problem like E_flow's ec_V vs u_flow/v_flow)
        sweep_mixed_sign(N, 1, Phi_x, Sigma_x, generator)
        if N > 3:
            sweep_mixed_sign(N, N // 2, Phi_x, Sigma_x, generator)
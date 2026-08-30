from __future__ import annotations

from typing import List, Tuple

import torch

from synthetic.drift_coupled_gamma import drift_fn_coupled_gamma

def _extract_Avx_Avv_coupled_gamma(
    N: int, gamma_self: float, gamma_couple: float, alpha: List[float], beta: List[float],
    k_reference: float, coupling, t: float, T: int, constant_k: bool, time_scale_fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = coupling.device if coupling is not None else torch.device("cpu")
    K_self = [[torch.tensor([[k_reference]], device=device)] for _ in range(N)]
    K_global = [torch.tensor([[k_reference]], device=device) for _ in range(N)]
    t_tensor = torch.tensor(float(t), device=device)

    def eval_dV(x_vals: List[float], v_vals: List[float]) -> torch.Tensor:
        X = [[torch.tensor([[x_vals[i]]], device=device)] for i in range(N)]
        V = [[torch.tensor([[v_vals[i]]], device=device)] for i in range(N)]
        dV = drift_fn_coupled_gamma(
            X, V, K_self, K_global, t_tensor, T, alpha, beta, gamma_self, gamma_couple,
            coupling_matrix=coupling, constant_k=constant_k,
            scale_damping_with_time=True, time_scale_fn=time_scale_fn,
        )
        return torch.tensor([dV[i][0].item() for i in range(N)], device=device)

    A_vx = torch.zeros(N, N, device=device)
    A_vv = torch.zeros(N, N, device=device)
    zeros = [0.0] * N
    for k in range(N):
        e = list(zeros)
        e[k] = 1.0
        A_vx[:, k] = eval_dV(e, zeros)
        A_vv[:, k] = eval_dV(zeros, e)
    return A_vx, A_vv


def _forward_step_matrix(A_vx: torch.Tensor, A_vv: torch.Tensor, dt: float) -> torch.Tensor:
    Nd = A_vx.shape[0]
    I = torch.eye(Nd, device=A_vx.device)
    M = torch.zeros(2 * Nd, 2 * Nd, device=A_vx.device)
    M[:Nd, :Nd] = I + dt**2 * A_vx
    M[:Nd, Nd:] = dt * (I + dt * A_vv)
    M[Nd:, :Nd] = dt * A_vx
    M[Nd:, Nd:] = I + dt * A_vv
    return M


def _noise_injection_matrix(G: torch.Tensor, dt: float) -> torch.Tensor:
    Nd = G.shape[0]
    sqrt_dt = dt**0.5
    Nmat = torch.zeros(2 * Nd, Nd, device=G.device)
    Nmat[:Nd, :] = dt * sqrt_dt * G
    Nmat[Nd:, :] = sqrt_dt * G
    return Nmat


def _reverse_step_matrix(A_vx: torch.Tensor, A_vv: torch.Tensor, dt: float) -> torch.Tensor:
    Nd = A_vx.shape[0]
    I = torch.eye(Nd, device=A_vx.device, dtype=A_vx.dtype)
    M = torch.zeros(2 * Nd, 2 * Nd, device=A_vx.device, dtype=A_vx.dtype)
    M[:Nd, :Nd] = I + dt**2 * A_vx
    M[:Nd, Nd:] = -dt * (I - dt * A_vv)
    M[Nd:, :Nd] = -dt * A_vx
    M[Nd:, Nd:] = I - dt * A_vv
    return M


def _reverse_noise_injection_matrix(G: torch.Tensor, dt: float) -> torch.Tensor:
    Nd = G.shape[0]
    sqrt_dt = dt**0.5
    Nmat = torch.zeros(2 * Nd, Nd, device=G.device, dtype=G.dtype)
    Nmat[:Nd, :] = -dt * sqrt_dt * G
    Nmat[Nd:, :] = sqrt_dt * G
    return Nmat


def roundtrip_leak_snr(
    N: int, gamma_self: float, gamma_couple: float, alpha: List[float], k_reference: float,
    cov_data: torch.Tensor, T: int, dt: float, time_scale_fn, sigma_ref: float = 1.0,
) -> float:
    dtype = cov_data.dtype
    beta_zero = [0.0] * N
    Phi = torch.eye(2 * N, dtype=dtype)
    Sigma = torch.zeros(2 * N, 2 * N, dtype=dtype)
    G = sigma_ref * torch.eye(N, dtype=dtype)

    for t in range(1, T + 1):
        A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(N, gamma_self, gamma_couple, alpha, beta_zero, k_reference, None, t, T, False, time_scale_fn)
        M = _forward_step_matrix(A_vx, A_vv, dt)
        Nmat = _noise_injection_matrix(G, dt)
        Phi = M @ Phi
        Sigma = M @ Sigma @ M.T + Nmat @ Nmat.T

    for t in range(T, 0, -1):
        A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(N, gamma_self, gamma_couple, alpha, beta_zero, k_reference, None, t, T, False, time_scale_fn)
        M = _reverse_step_matrix(A_vx, A_vv, dt)
        Nmat = _reverse_noise_injection_matrix(G, dt)
        Phi = M @ Phi
        Sigma = M @ Sigma @ M.T + Nmat @ Nmat.T

    Phi_x = Phi[:N, :N]
    cov_final_x = Phi_x @ cov_data @ Phi_x.T + Sigma[:N, :N]

    def mean_pairwise_corr(cov: torch.Tensor) -> float:
        std = cov.diagonal().sqrt()
        corr = cov / (std[:, None] * std[None, :])
        off = corr - torch.diag(torch.diag(corr))
        return off.sum().item() / (N * (N - 1))

    rho_true = mean_pairwise_corr(cov_data)
    rho_leak = mean_pairwise_corr(cov_final_x)
    x = rho_leak / rho_true
    return x / (1 - x)


def calibrate_sigma_for_leak(
    N: int, gamma_self: float, gamma_couple: float, alpha: List[float], k_reference: float,
    cov_data: torch.Tensor, T: int, dt: float, time_scale_fn, leak_fraction: float = 0.01,
) -> float:
    snr_ref = roundtrip_leak_snr(N, gamma_self, gamma_couple, alpha, k_reference, cov_data, T, dt, time_scale_fn, sigma_ref=1.0)
    k = snr_ref * 1.0**2
    snr_target = leak_fraction / (1 - leak_fraction)
    return (k / snr_target) ** 0.5


def precompute_transition_params(
    N: int, gamma_self: float, gamma_couple: float, alpha: List[float], beta: List[float],
    k_reference: float, coupling, T: int, dt: float, g_fn, constant_k: bool, time_scale_fn,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    device = coupling.device if coupling is not None else torch.device("cpu")
    Phi = torch.eye(2 * N, device=device)
    Sigma = torch.zeros(2 * N, 2 * N, device=device)
    params = []
    for t in range(1, T + 1):
        A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, t, T, constant_k, time_scale_fn)
        M = _forward_step_matrix(A_vx, A_vv, dt)
        G = g_fn(t, T)
        Nmat = _noise_injection_matrix(G, dt)
        Q = Nmat @ Nmat.T
        Phi = M @ Phi
        Sigma = M @ Sigma @ M.T + Q
        params.append((Phi.clone(), Sigma.clone()))
    return params


def sample_and_tikhonov_score_target(Z0: torch.Tensor, Phi_t: torch.Tensor, Sigma_t: torch.Tensor, N: int, lam: float, jitter: float = 1e-6):
    B = Z0.shape[0]
    device = Z0.device
    mean = Z0 @ Phi_t.T
    scale = Sigma_t.diagonal().abs().max().clamp_min(1.0)
    Sigma_reg = Sigma_t + jitter * scale * torch.eye(2 * N, device=device)
    L = torch.linalg.cholesky(Sigma_reg)
    eps = torch.randn(B, 2 * N, device=device)
    Zt = mean + eps @ L.T

    Sxx = Sigma_t[:N, :N]
    Sxv = Sigma_t[:N, N:]
    Svx = Sigma_t[N:, :N]
    Svv = Sigma_t[N:, N:]
    Sxx_inv = torch.linalg.inv(Sxx + jitter * scale * torch.eye(N, device=device))
    Var_V_given_X = Svv - Svx @ Sxx_inv @ Sxv

    diff_x = Zt[:, :N] - mean[:, :N]
    diff_v = Zt[:, N:] - mean[:, N:]
    E_V_given_X_dev = diff_x @ (Sxx_inv @ Sxv).clone()
    resid = diff_v - E_V_given_X_dev

    reg_precision = torch.linalg.inv(Var_V_given_X + lam * torch.eye(N, device=device))
    score_v_reg = -resid @ reg_precision.T
    return Zt, score_v_reg

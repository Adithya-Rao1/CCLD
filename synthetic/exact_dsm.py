from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from core.drift import _time_scale
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


def _forward_step_matrix(A_vx: torch.Tensor, A_vv: torch.Tensor, dt: float, time_scale=1.0) -> torch.Tensor:
    Nd = A_vx.shape[0]
    I = torch.eye(Nd, device=A_vx.device)
    M = torch.zeros(2 * Nd, 2 * Nd, device=A_vx.device)
    M[:Nd, :Nd] = I + time_scale * dt**2 * A_vx
    M[:Nd, Nd:] = time_scale * dt * (I + dt * A_vv)
    M[Nd:, :Nd] = dt * A_vx
    M[Nd:, Nd:] = I + dt * A_vv
    return M


def _noise_injection_matrix(G: torch.Tensor, dt: float, time_scale=1.0) -> torch.Tensor:
    Nd = G.shape[0]
    sqrt_dt = dt**0.5
    Nmat = torch.zeros(2 * Nd, Nd, device=G.device)
    Nmat[:Nd, :] = time_scale * dt * sqrt_dt * G
    Nmat[Nd:, :] = sqrt_dt * G
    return Nmat


def _reverse_step_matrix(A_vx: torch.Tensor, A_vv: torch.Tensor, dt: float, time_scale=1.0) -> torch.Tensor:
    Nd = A_vx.shape[0]
    I = torch.eye(Nd, device=A_vx.device, dtype=A_vx.dtype)
    M = torch.zeros(2 * Nd, 2 * Nd, device=A_vx.device, dtype=A_vx.dtype)
    M[:Nd, :Nd] = I + time_scale * dt**2 * A_vx
    M[:Nd, Nd:] = -time_scale * dt * (I - dt * A_vv)
    M[Nd:, :Nd] = -dt * A_vx
    M[Nd:, Nd:] = I - dt * A_vv
    return M


def _reverse_noise_injection_matrix(G: torch.Tensor, dt: float, time_scale=1.0) -> torch.Tensor:
    Nd = G.shape[0]
    sqrt_dt = dt**0.5
    Nmat = torch.zeros(2 * Nd, Nd, device=G.device, dtype=G.dtype)
    Nmat[:Nd, :] = -time_scale * dt * sqrt_dt * G
    Nmat[Nd:, :] = sqrt_dt * G
    return Nmat


def roundtrip_propagator(
    N: int, gamma_self: float, alpha: List[float], k_reference: float,
    T: int, dt: float, time_scale_fn, sigma_ref: float = 1.0, dtype=torch.float32,
    scale_kinematics_with_time: bool = True, scale_diffusion_with_time: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    beta_zero = [0.0] * N
    gamma_couple_zero = 0.0
    Phi = torch.eye(2 * N, dtype=dtype)
    Sigma = torch.zeros(2 * N, 2 * N, dtype=dtype)
    G0 = sigma_ref * torch.eye(N, dtype=dtype)

    for t in range(1, T + 1):
        A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(N, gamma_self, gamma_couple_zero, alpha, beta_zero, k_reference, None, t, T, True, time_scale_fn)
        time_scale = _time_scale(torch.as_tensor(float(t), dtype=dtype), T, time_scale_fn)
        kin_scale = time_scale if scale_kinematics_with_time else 1.0
        G = G0 * time_scale.clamp_min(0).sqrt() if scale_diffusion_with_time else G0
        M = _forward_step_matrix(A_vx, A_vv, dt, time_scale=kin_scale)
        Nmat = _noise_injection_matrix(G, dt, time_scale=kin_scale)
        Phi = M @ Phi
        Sigma = M @ Sigma @ M.T + Nmat @ Nmat.T

    for t in range(T, 0, -1):
        A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(N, gamma_self, gamma_couple_zero, alpha, beta_zero, k_reference, None, t, T, True, time_scale_fn)
        time_scale = _time_scale(torch.as_tensor(float(t), dtype=dtype), T, time_scale_fn)
        kin_scale = time_scale if scale_kinematics_with_time else 1.0
        G = G0 * time_scale.clamp_min(0).sqrt() if scale_diffusion_with_time else G0
        M = _reverse_step_matrix(A_vx, A_vv, dt, time_scale=kin_scale)
        Nmat = _reverse_noise_injection_matrix(G, dt, time_scale=kin_scale)
        Phi = M @ Phi
        Sigma = M @ Sigma @ M.T + Nmat @ Nmat.T

    return Phi[:N, :N], Sigma[:N, :N]


def roundtrip_leak_snr(
    N: int, gamma_self: float, alpha: List[float], k_reference: float,
    cov_data: torch.Tensor, T: int, dt: float, time_scale_fn, sigma_ref: float = 1.0,
    scale_kinematics_with_time: bool = True, scale_diffusion_with_time: bool = True,
) -> float:
    Phi_x, Sigma_x = roundtrip_propagator(N, gamma_self, alpha, k_reference, T, dt, time_scale_fn,
                                           sigma_ref=sigma_ref, dtype=cov_data.dtype,
                                           scale_kinematics_with_time=scale_kinematics_with_time,
                                           scale_diffusion_with_time=scale_diffusion_with_time)
    cov_final_x = Phi_x @ cov_data @ Phi_x.T + Sigma_x

    def corr_matrix(cov: torch.Tensor) -> torch.Tensor:
        std = cov.diagonal().sqrt()
        return cov / (std[:, None] * std[None, :])

    off_mask = ~torch.eye(N, dtype=torch.bool)
    corr_true_off = corr_matrix(cov_data)[off_mask]
    corr_leak_off = corr_matrix(cov_final_x)[off_mask]

    eps = 1e-6
    valid = corr_true_off.abs() >= eps
    if not torch.any(valid):
        return 0.0
    x = (corr_leak_off[valid] / corr_true_off[valid]).mean().item()
    return x / (1 - x)


def calibrate_sigma_for_leak(
    N: int, gamma_self: float, alpha: List[float], k_reference: float,
    cov_data: torch.Tensor, T: int, dt: float, time_scale_fn, leak_fraction: float = 0.01,
    scale_kinematics_with_time: bool = True, scale_diffusion_with_time: bool = True,
) -> float:
    snr_ref = roundtrip_leak_snr(N, gamma_self, alpha, k_reference, cov_data, T, dt, time_scale_fn, sigma_ref=1.0,
                                  scale_kinematics_with_time=scale_kinematics_with_time,
                                  scale_diffusion_with_time=scale_diffusion_with_time)
    k = snr_ref * 1.0**2
    snr_target = leak_fraction / (1 - leak_fraction)
    return (k / snr_target) ** 0.5


def precompute_transition_params(
    N: int, gamma_self: float, gamma_couple: float, alpha: List[float], beta: List[float],
    k_reference: float, coupling, T: int, dt: float, g_fn, constant_k: bool, time_scale_fn,
    scale_kinematics_with_time: bool = True, scale_diffusion_with_time: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    device = coupling.device if coupling is not None else torch.device("cpu")
    Phi = torch.eye(2 * N, device=device)
    Sigma = torch.zeros(2 * N, 2 * N, device=device)
    params = []
    for t in range(1, T + 1):
        A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, t, T, constant_k, time_scale_fn)
        time_scale = _time_scale(torch.as_tensor(float(t), device=device), T, time_scale_fn)
        kin_scale = time_scale if scale_kinematics_with_time else 1.0
        M = _forward_step_matrix(A_vx, A_vv, dt, time_scale=kin_scale)
        G0 = g_fn(t, T)
        G = G0 * time_scale.clamp_min(0).sqrt() if scale_diffusion_with_time else G0
        Nmat = _noise_injection_matrix(G, dt, time_scale=kin_scale)
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


def tau_hat_default_schedule(T: int, dt: float) -> float:
    return T * dt * (2.0 * math.log(2.0) - 1.0)


def tau_hat_vp_linear_schedule(T: int, dt: float) -> float:
    return T * dt * 0.5


def closed_form_propagator(
    N: int, gamma_self: float, gamma_couple: float, alpha: List[float], beta: List[float],
    k_reference: float, coupling: Optional[torch.Tensor], tau_hat: float,
    constant_k: bool = False, sigma_ref: float = 1.0, dtype=torch.float32,
    G0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not constant_k:
        raise ValueError(
            "closed_form_propagator requires constant_k=True"
        )
    const_one = lambda t, T: torch.tensor(1.0)
    A_vx0, A_vv0 = _extract_Avx_Avv_coupled_gamma(
        N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling,
        t=1, T=1, constant_k=constant_k, time_scale_fn=const_one,
    )
    A_vx0 = A_vx0.to(dtype)
    A_vv0 = A_vv0.to(dtype)

    J = torch.zeros(2 * N, 2 * N, dtype=dtype)
    J[:N, N:] = torch.eye(N, dtype=dtype)
    J[N:, :N] = A_vx0
    J[N:, N:] = A_vv0

    L = torch.zeros(2 * N, N, dtype=dtype)
    L[N:, :] = G0.to(dtype) if G0 is not None else sigma_ref * torch.eye(N, dtype=dtype)
    LLT = L @ L.T

    M = torch.zeros(4 * N, 4 * N, dtype=dtype)
    M[:2 * N, :2 * N] = J
    M[:2 * N, 2 * N:] = LLT
    M[2 * N:, 2 * N:] = -J.T

    max_real_eig = torch.linalg.eigvals(J).real.abs().max().item()
    if tau_hat * max_real_eig > 600.0:
        raise OverflowError(
            f"tau_hat={tau_hat:.3g} * max|Re(eig(J))|={max_real_eig:.3g} = {tau_hat * max_real_eig:.3g} is too large for a stable matrix_exp"
        )

    Mexp = torch.matrix_exp(tau_hat * M)
    Phi = Mexp[:2 * N, :2 * N]
    Phi_Sigma = Mexp[:2 * N, 2 * N:]
    Sigma = Phi_Sigma @ Phi.T
    return Phi, Sigma

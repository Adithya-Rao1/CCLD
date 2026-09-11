from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from core.coupling import build_coupling_matrix
from core.drift import _time_scale
from synthetic.drift_coupled_gamma import antisymmetric_mode_damping, calibrate_sigma_fdt, drift_fn_coupled_gamma
from synthetic.skew_coupling import inject_skew_coupling, reference_stationary_covariance

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def elapsed_time_at_step(t_idx: int, T: int, dt: float, time_scale_fn=None, scale_kinematics_with_time: bool = True) -> float:
    if not scale_kinematics_with_time:
        return t_idx * dt
    q = 0.0
    for t in range(1, t_idx + 1):
        kin_scale = _time_scale(torch.as_tensor(float(t)), T, time_scale_fn)
        q += float(kin_scale.item()) * dt
    return q


def _extract_Avx_Avv_coupled_gamma(
    N: int, gamma_self: Optional[float], gamma_couple: Optional[float], alpha: List[float], beta: List[float],
    k_reference: float, coupling, t: float, T: int, constant_k: bool, time_scale_fn,
    damping_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = coupling.device if coupling is not None else _DEVICE
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
            damping_matrix=damping_matrix,
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
    N: int, gamma_self: Optional[float], gamma_couple: Optional[float], alpha: List[float], beta: List[float],
    k_reference: float, coupling, T: int, dt: float, g_fn, constant_k: bool, time_scale_fn,
    scale_kinematics_with_time: bool = True, scale_diffusion_with_time: bool = True,
    damping_matrix: Optional[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    device = coupling.device if coupling is not None else _DEVICE
    Phi = torch.eye(2 * N, device=device)
    Sigma = torch.zeros(2 * N, 2 * N, device=device)
    params = []
    for t in range(1, T + 1):
        A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, t, T, constant_k, time_scale_fn, damping_matrix=damping_matrix)
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


def _sxx_scalar(Omega: torch.Tensor, g: torch.Tensor, q: torch.Tensor, x_threshold: float = 0.5) -> torch.Tensor:
    x = Omega * q
    series = (g**2 * q**3 / 3 - Omega * g**2 * q**4 / 2 + 2 * Omega**2 * g**2 * q**5 / 5
              - 2 * Omega**3 * g**2 * q**6 / 9 + 2 * Omega**4 * g**2 * q**7 / 21 - Omega**5 * g**2 * q**8 / 30)
    e2x = torch.exp(2 * x)
    direct = (g**2 / (4 * Omega**3)) * (e2x - 1 - 2 * x - 2 * x**2) * torch.exp(-2 * x)
    return torch.where(x < x_threshold, series, direct)


def _sxv_scalar(Omega: torch.Tensor, g: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (g**2 * q**2 / 2) * torch.exp(-2 * Omega * q)


def _svv_scalar(Omega: torch.Tensor, g: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    x = Omega * q
    e2x = torch.exp(2 * x)
    return (g**2 / (4 * Omega)) * (-2 * x**2 + 2 * x + e2x - 1) * torch.exp(-2 * x)


def _var_v_given_x_scalar(Omega: torch.Tensor, g: torch.Tensor, q: torch.Tensor, x_threshold: float = 0.5) -> torch.Tensor:
    Sxx = _sxx_scalar(Omega, g, q, x_threshold)
    Sxv = _sxv_scalar(Omega, g, q)
    Svv = _svv_scalar(Omega, g, q)
    return Svv - Sxv**2 / Sxx.clamp_min(1e-30)


def _regression_coeff_scalar(Omega: torch.Tensor, g: torch.Tensor, q: torch.Tensor, x_threshold: float = 0.5) -> torch.Tensor:
    Sxx = _sxx_scalar(Omega, g, q, x_threshold)
    Sxv = _sxv_scalar(Omega, g, q)
    return Sxv / Sxx.clamp_min(1e-30)


def _reconstruct_from_modes(val_sym: torch.Tensor, val_anti: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    d = (N - 1) / N * (val_sym - val_anti)
    c = val_sym - d
    return c, d


def _mode_quantities(N: int, gamma_self: float, gamma_couple: float, q: torch.Tensor, target_variance: float, device, dtype):
    q = torch.as_tensor(q, dtype=dtype, device=device)
    Omega_sym = torch.as_tensor(gamma_self / 2.0, dtype=dtype, device=device)
    g_sym = torch.as_tensor(calibrate_sigma_fdt(gamma_self, target_variance), dtype=dtype, device=device)
    if N == 1:
        return q, (Omega_sym, g_sym), None
    gamma_anti = antisymmetric_mode_damping(gamma_self, gamma_couple, N)
    Omega_anti = torch.as_tensor(gamma_anti / 2.0, dtype=dtype, device=device)
    g_anti = torch.as_tensor(calibrate_sigma_fdt(gamma_anti, target_variance), dtype=dtype, device=device)
    return q, (Omega_sym, g_sym), (Omega_anti, g_anti)


def analytic_score_precision_n(
    N: int, gamma_self: float, gamma_couple: float, q: torch.Tensor, target_variance: float = 1.0,
    x_threshold: float = 0.5, device=None, dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, (Omega_sym, g_sym), anti = _mode_quantities(N, gamma_self, gamma_couple, q, target_variance, device, dtype)
    var_sym = _var_v_given_x_scalar(Omega_sym, g_sym, q, x_threshold)
    coeff_sym = _regression_coeff_scalar(Omega_sym, g_sym, q, x_threshold)

    if anti is None:
        precision = (1.0 / var_sym.clamp_min(1e-30)).reshape(1, 1)
        coeff = coeff_sym.reshape(1, 1)
        return precision, coeff

    Omega_anti, g_anti = anti
    var_anti = _var_v_given_x_scalar(Omega_anti, g_anti, q, x_threshold)
    coeff_anti = _regression_coeff_scalar(Omega_anti, g_anti, q, x_threshold)

    prec_sym, prec_anti = 1.0 / var_sym.clamp_min(1e-30), 1.0 / var_anti.clamp_min(1e-30)
    c_prec, d_prec = _reconstruct_from_modes(prec_sym, prec_anti, N)
    c_coeff, d_coeff = _reconstruct_from_modes(coeff_sym, coeff_anti, N)

    I = torch.eye(N, dtype=dtype, device=device)
    C = build_coupling_matrix(N, mode="mean_field", device=device, dtype=dtype)
    precision = c_prec * I + d_prec * C
    coeff = c_coeff * I + d_coeff * C
    return precision, coeff


def analytic_conditional_covariance_n(
    N: int, gamma_self: float, gamma_couple: float, q: torch.Tensor, target_variance: float = 1.0,
    x_threshold: float = 0.5, device=None, dtype=torch.float32,
) -> torch.Tensor:
    q, (Omega_sym, g_sym), anti = _mode_quantities(N, gamma_self, gamma_couple, q, target_variance, device, dtype)
    sxx_sym = _sxx_scalar(Omega_sym, g_sym, q, x_threshold)
    sxv_sym = _sxv_scalar(Omega_sym, g_sym, q)
    svv_sym = _svv_scalar(Omega_sym, g_sym, q)

    I = torch.eye(N, dtype=dtype, device=device)
    if anti is None:
        Sxx, Sxv, Svv = sxx_sym.reshape(1, 1), sxv_sym.reshape(1, 1), svv_sym.reshape(1, 1)
    else:
        Omega_anti, g_anti = anti
        sxx_anti = _sxx_scalar(Omega_anti, g_anti, q, x_threshold)
        sxv_anti = _sxv_scalar(Omega_anti, g_anti, q)
        svv_anti = _svv_scalar(Omega_anti, g_anti, q)
        C = build_coupling_matrix(N, mode="mean_field", device=device, dtype=dtype)
        c_xx, d_xx = _reconstruct_from_modes(sxx_sym, sxx_anti, N)
        c_xv, d_xv = _reconstruct_from_modes(sxv_sym, sxv_anti, N)
        c_vv, d_vv = _reconstruct_from_modes(svv_sym, svv_anti, N)
        Sxx, Sxv, Svv = c_xx * I + d_xx * C, c_xv * I + d_xv * C, c_vv * I + d_vv * C

    Sigma = torch.zeros(2 * N, 2 * N, dtype=dtype, device=device)
    Sigma[:N, :N], Sigma[:N, N:], Sigma[N:, :N], Sigma[N:, N:] = Sxx, Sxv, Sxv.T, Svv
    return Sigma


def _mode_quantities_spectral(
    Gamma: torch.Tensor, C: torch.Tensor, q: torch.Tensor, target_variance: float = 1.0, zeta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = C.shape[0]
    device, dtype = C.device, C.dtype
    L = torch.eye(N, device=device, dtype=dtype) - C
    mu, U = torch.linalg.eigh(L)
    gamma_modes = torch.diagonal(U.T @ Gamma @ U)
    Omega_modes = gamma_modes / (2.0 * zeta)
    g_modes = torch.sqrt((2.0 * gamma_modes * target_variance).clamp_min(0.0))
    q_t = torch.as_tensor(q, dtype=dtype, device=device)
    return q_t, Omega_modes, g_modes, U


def analytic_score_precision_n_spectral(
    Gamma: torch.Tensor, C: torch.Tensor, q: torch.Tensor, target_variance: float = 1.0,
    x_threshold: float = 0.5, zeta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = C.shape[0]
    q_t, Omega_modes, g_modes, U = _mode_quantities_spectral(Gamma, C, q, target_variance, zeta)
    var_modes = torch.stack([_var_v_given_x_scalar(Omega_modes[k], g_modes[k], q_t, x_threshold) for k in range(N)])
    coeff_modes = torch.stack([_regression_coeff_scalar(Omega_modes[k], g_modes[k], q_t, x_threshold) for k in range(N)])
    prec_modes = 1.0 / var_modes.clamp_min(1e-30)
    precision = U @ torch.diag(prec_modes) @ U.T
    coeff = U @ torch.diag(coeff_modes) @ U.T
    return precision, coeff


def analytic_conditional_covariance_n_spectral(
    Gamma: torch.Tensor, C: torch.Tensor, q: torch.Tensor, target_variance: float = 1.0,
    x_threshold: float = 0.5, zeta: float = 1.0,
) -> torch.Tensor:
    N = C.shape[0]
    device, dtype = C.device, C.dtype
    q_t, Omega_modes, g_modes, U = _mode_quantities_spectral(Gamma, C, q, target_variance, zeta)
    sxx_modes = torch.stack([_sxx_scalar(Omega_modes[k], g_modes[k], q_t, x_threshold) for k in range(N)])
    sxv_modes = torch.stack([_sxv_scalar(Omega_modes[k], g_modes[k], q_t) for k in range(N)])
    svv_modes = torch.stack([_svv_scalar(Omega_modes[k], g_modes[k], q_t) for k in range(N)])
    Sxx = U @ torch.diag(sxx_modes) @ U.T
    Sxv = U @ torch.diag(sxv_modes) @ U.T
    Svv = U @ torch.diag(svv_modes) @ U.T

    Sigma = torch.zeros(2 * N, 2 * N, dtype=dtype, device=device)
    Sigma[:N, :N], Sigma[:N, N:], Sigma[N:, :N], Sigma[N:, N:] = Sxx, Sxv, Sxv.T, Svv
    return Sigma


def sample_and_analytic_score_target_spectral(
    Z0: torch.Tensor, Phi_t: torch.Tensor, Gamma: torch.Tensor, C: torch.Tensor,
    q: torch.Tensor, target_variance: float = 1.0, jitter: float = 1e-6, zeta: float = 1.0,
):
    N = C.shape[0]
    B = Z0.shape[0]
    device = Z0.device
    mean = Z0 @ Phi_t.T
    Sigma_t = analytic_conditional_covariance_n_spectral(Gamma, C, q, target_variance, zeta=zeta)
    scale = Sigma_t.diagonal().abs().max().clamp_min(1.0)
    Sigma_reg = Sigma_t + jitter * scale * torch.eye(2 * N, device=device)
    L_chol = torch.linalg.cholesky(Sigma_reg)
    eps = torch.randn(B, 2 * N, device=device)
    Zt = mean + eps @ L_chol.T

    reg_precision, regression_coeff = analytic_score_precision_n_spectral(
        Gamma, C, q, target_variance, zeta=zeta,
    )

    diff_x = Zt[:, :N] - mean[:, :N]
    diff_v = Zt[:, N:] - mean[:, N:]
    E_V_given_X_dev = diff_x @ regression_coeff.T
    resid = diff_v - E_V_given_X_dev

    score_v_reg = -resid @ reg_precision.T
    return Zt, score_v_reg


def score_target_from_covariance(
    Sigma: torch.Tensor, N: int, jitter: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Sxx = Sigma[:N, :N]
    Sxv = Sigma[:N, N:]
    Svv = Sigma[N:, N:]
    eye = torch.eye(N, device=Sigma.device, dtype=Sigma.dtype)
    Sxx_inv = torch.linalg.inv(Sxx + jitter * eye)
    coeff = Sxv.T @ Sxx_inv
    var_v_given_x = Svv - Sxv.T @ Sxx_inv @ Sxv
    precision = torch.linalg.inv(var_v_given_x + jitter * eye)
    return precision, coeff


def sample_and_analytic_score_target_skew(
    Z0: torch.Tensor, Phi_t: torch.Tensor, Sigma_t: torch.Tensor, jitter: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = Phi_t.shape[0] // 2
    B = Z0.shape[0]
    device = Z0.device
    mean = Z0 @ Phi_t.T
    scale = Sigma_t.diagonal().abs().max().clamp_min(1.0)
    Sigma_reg = Sigma_t + jitter * scale * torch.eye(2 * N, device=device, dtype=Sigma_t.dtype)
    L_chol = torch.linalg.cholesky(Sigma_reg)
    eps = torch.randn(B, 2 * N, device=device)
    Zt = mean + eps @ L_chol.T

    reg_precision, regression_coeff = score_target_from_covariance(Sigma_t, N)

    diff_x = Zt[:, :N] - mean[:, :N]
    diff_v = Zt[:, N:] - mean[:, N:]
    E_V_given_X_dev = diff_x @ regression_coeff.T
    resid = diff_v - E_V_given_X_dev

    score_v_reg = -resid @ reg_precision.T
    return Zt, score_v_reg


def sample_and_analytic_score_target(
    Z0: torch.Tensor, Phi_t: torch.Tensor, N: int,
    gamma_self: float, gamma_couple: float, q: torch.Tensor, target_variance: float = 1.0, jitter: float = 1e-6,
):
    B = Z0.shape[0]
    device = Z0.device
    mean = Z0 @ Phi_t.T
    Sigma_t = analytic_conditional_covariance_n(N, gamma_self, gamma_couple, q, target_variance, device=device, dtype=Z0.dtype)
    scale = Sigma_t.diagonal().abs().max().clamp_min(1.0)
    Sigma_reg = Sigma_t + jitter * scale * torch.eye(2 * N, device=device)
    L = torch.linalg.cholesky(Sigma_reg)
    eps = torch.randn(B, 2 * N, device=device)
    Zt = mean + eps @ L.T

    reg_precision, regression_coeff = analytic_score_precision_n(
        N, gamma_self, gamma_couple, q, target_variance, device=device, dtype=Sigma_t.dtype,
    )

    diff_x = Zt[:, :N] - mean[:, :N]
    diff_v = Zt[:, N:] - mean[:, N:]
    E_V_given_X_dev = diff_x @ regression_coeff.T
    resid = diff_v - E_V_given_X_dev

    score_v_reg = -resid @ reg_precision.T
    return Zt, score_v_reg


def tau_hat_default_schedule(T: int, dt: float) -> float:
    return T * dt * (2.0 * math.log(2.0) - 1.0)


def tau_hat_vp_linear_schedule(T: int, dt: float) -> float:
    return T * dt * 0.5


def closed_form_propagator(
    N: int, gamma_self: Optional[float], gamma_couple: Optional[float], alpha: List[float], beta: List[float],
    k_reference: float, coupling: Optional[torch.Tensor], tau_hat: float,
    constant_k: bool = False, sigma_ref: float = 1.0, dtype=torch.float32,
    G0: Optional[torch.Tensor] = None,
    damping_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not constant_k:
        raise ValueError
    const_one = lambda t, T: torch.tensor(1.0)
    A_vx0, A_vv0 = _extract_Avx_Avv_coupled_gamma(
        N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling,
        t=1, T=1, constant_k=constant_k, time_scale_fn=const_one,
        damping_matrix=damping_matrix,
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


def closed_form_propagator_skew(
    N: int, gamma_self: Optional[float], gamma_couple: Optional[float], alpha: List[float], beta: List[float],
    k_reference: float, coupling: Optional[torch.Tensor], tau_hat: float,
    constant_k: bool = False, sigma_ref: float = 1.0, dtype=torch.float32,
    G0: Optional[torch.Tensor] = None,
    damping_matrix: Optional[torch.Tensor] = None,
    skew_matrix: Optional[torch.Tensor] = None,
    target_variance: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not constant_k:
        raise ValueError
    const_one = lambda t, T: torch.tensor(1.0)
    A_vx0, A_vv0 = _extract_Avx_Avv_coupled_gamma(
        N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling,
        t=1, T=1, constant_k=constant_k, time_scale_fn=const_one,
        damping_matrix=damping_matrix,
    )
    A_vx0 = A_vx0.to(dtype)
    A_vv0 = A_vv0.to(dtype)

    A0_block = torch.zeros(2 * N, 2 * N, dtype=dtype)
    A0_block[:N, N:] = torch.eye(N, dtype=dtype)
    A0_block[N:, :N] = A_vx0
    A0_block[N:, N:] = A_vv0

    if skew_matrix is not None:
        K = -A_vx0
        Sigma_ref = reference_stationary_covariance(K, target_variance)
        A0_block = inject_skew_coupling(A0_block, Sigma_ref, skew_matrix.to(dtype))

    L = torch.zeros(2 * N, N, dtype=dtype)
    L[N:, :] = G0.to(dtype) if G0 is not None else sigma_ref * torch.eye(N, dtype=dtype)
    LLT = L @ L.T

    M = torch.zeros(4 * N, 4 * N, dtype=dtype)
    M[:2 * N, :2 * N] = A0_block
    M[:2 * N, 2 * N:] = LLT
    M[2 * N:, 2 * N:] = -A0_block.T

    max_real_eig = torch.linalg.eigvals(A0_block).real.abs().max().item()
    if tau_hat * max_real_eig > 600.0:
        raise OverflowError(
            f"tau_hat={tau_hat:.3g} * max|Re(eig(A0))|={max_real_eig:.3g} = {tau_hat * max_real_eig:.3g} is too large for a stable matrix_exp"
        )

    Mexp = torch.matrix_exp(tau_hat * M)
    Phi = Mexp[:2 * N, :2 * N]
    Phi_Sigma = Mexp[:2 * N, 2 * N:]
    Sigma = Phi_Sigma @ Phi.T
    return Phi, Sigma

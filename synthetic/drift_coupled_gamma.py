from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch

from core.coupling import build_coupling_matrix
from core.damping import _DEFAULT_TARGET_ZETA, DampingRegime
from core.drift import _broadcast_per_sample, _mean_frob_norm, _mean_per_sample, _time_scale

def mode_frequencies_sq(alpha: float, beta: float, k_self_reference: float, k_global_reference: float, N: int) -> Tuple[float, Optional[float]]:
    Omega_sq = k_self_reference + k_global_reference
    omega_sym_sq = alpha * Omega_sq
    if N == 1:
        return omega_sym_sq, None
    omega_anti_sq = omega_sym_sq + (N / (N - 1)) * beta * k_global_reference
    return omega_sym_sq, omega_anti_sq


def calibrate_coupled_gammas(
    alpha: float, beta: float, k_self_reference: float, k_global_reference: float, N: int,
    regime: DampingRegime = "critically_damped", target_zeta: Optional[float] = None,
) -> Tuple[float, float]:
    zeta = target_zeta if target_zeta is not None else _DEFAULT_TARGET_ZETA[regime]
    omega_sym_sq, omega_anti_sq = mode_frequencies_sq(alpha, beta, k_self_reference, k_global_reference, N)
    Omega_sym = math.sqrt(max(omega_sym_sq, 0.0))
    if N == 1:
        return 2.0 * zeta * Omega_sym, 0.0
    Omega_anti = math.sqrt(max(omega_anti_sq, 0.0))
    gamma_self = 2.0 * zeta * Omega_sym
    gamma_couple = (2.0 * zeta * (N - 1) / N) * (Omega_anti - Omega_sym)
    return gamma_self, gamma_couple


def antisymmetric_mode_damping(gamma_self: float, gamma_couple: float, N: int) -> float:
    if N < 2:
        raise ValueError
    return gamma_self + gamma_couple * N / (N - 1)


def mode_frequencies_spectral(
    alpha: float, beta: float, k_self_reference: float, k_global_reference: float, C: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = C.shape[0]
    if not torch.allclose(C, C.T, atol=1e-5):
        raise ValueError("mode_frequencies_spectral requires a symmetric coupling matrix C")
    L = torch.eye(N, device=C.device, dtype=C.dtype) - C
    mu = torch.linalg.eigvalsh(L)
    Omega_sq_base = alpha * (k_self_reference + k_global_reference)
    Omega_sq = Omega_sq_base + beta * k_global_reference * mu
    return mu, Omega_sq


def calibrate_coupled_gammas_spectral(
    alpha: float, beta: float, k_self_reference: float, k_global_reference: float, C: torch.Tensor,
    regime: DampingRegime = "critically_damped", target_zeta: Optional[float] = None,
) -> torch.Tensor:
    zeta = target_zeta if target_zeta is not None else _DEFAULT_TARGET_ZETA[regime]
    N = C.shape[0]
    L = torch.eye(N, device=C.device, dtype=C.dtype) - C
    mu, U = torch.linalg.eigh(L)
    Omega_sq = alpha * (k_self_reference + k_global_reference) + beta * k_global_reference * mu
    Omega = torch.sqrt(Omega_sq.clamp_min(0.0))
    gamma_modes = 2.0 * zeta * Omega
    Gamma = U @ torch.diag(gamma_modes) @ U.T
    return Gamma


def calibrate_sigma_fdt_spectral(
    Gamma: torch.Tensor, C: torch.Tensor, target_variance: float = 1.0,
) -> torch.Tensor:
    N = C.shape[0]
    L = torch.eye(N, device=C.device, dtype=C.dtype) - C
    _, U = torch.linalg.eigh(L)
    gamma_modes = torch.diagonal(U.T @ Gamma @ U)
    g_modes = torch.sqrt((2.0 * gamma_modes * target_variance).clamp_min(0.0))
    G = U @ torch.diag(g_modes) @ U.T
    return G


def calibrate_sigma_fdt(gamma_self: float, target_variance: float = 1.0) -> float:
    return math.sqrt(2.0 * gamma_self * target_variance)


def calibrate_sigma_fdt_coupled(gamma_self: float, gamma_couple: float, N: int, target_variance: float = 1.0) -> Tuple[float, float]:
    g_sym = calibrate_sigma_fdt(gamma_self, target_variance)
    if N < 2:
        return g_sym, 0.0
    gamma_anti = antisymmetric_mode_damping(gamma_self, gamma_couple, N)
    g_anti = calibrate_sigma_fdt(gamma_anti, target_variance)
    b = (N - 1) / N * (g_sym - g_anti)
    a = (g_sym + (N - 1) * g_anti) / N
    return a, b


def drift_fn_coupled_gamma(
    X: List[List[torch.Tensor]],
    V: List[List[torch.Tensor]],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma_self: Optional[float] = None,
    gamma_couple: Optional[float] = None,
    coupling_matrix: Optional[torch.Tensor] = None,
    constant_k: bool = False,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    k_global_reference: float = 1.0,
    damping_matrix: Optional[torch.Tensor] = None,
) -> List[List[torch.Tensor]]:
    N = len(X)
    if N < 1:
        raise ValueError
    if not (len(V) == len(K_self) == len(alpha) == len(beta) == N):
        raise ValueError
    if damping_matrix is not None and (gamma_self is not None or gamma_couple is not None):
        raise ValueError
    if damping_matrix is None and gamma_self is None:
        raise ValueError

    time_scale = _time_scale(t, T, time_scale_fn)
    norm_f_k_self = [_mean_frob_norm(K_self[i]) for i in range(N)]
    if not constant_k:
        norm_f_k_global = time_scale * _mean_frob_norm(K_global)
    else:
        norm_f_k_global = torch.as_tensor(k_global_reference, device=X[0][0].device)
    omega_sq = [-(norm_f_k_self[i] + norm_f_k_global) for i in range(N)]

    if N == 1:
        if gamma_self is None:
            raise ValueError
        # Standard CLD
        dv_i = []
        for x, v in zip(X[0], V[0]):
            confine = time_scale * alpha[0] * omega_sq[0] * x
            damping = -gamma_self * v
            if scale_damping_with_time:
                damping = time_scale * damping
            dv_i.append(damping + confine)
        return [dv_i]

    if coupling_matrix is None:
        coupling_matrix = build_coupling_matrix(N, mode="mean_field", device=X[0][0].device, dtype=torch.float32)

    m_x = [_mean_per_sample(X[i]) for i in range(N)]
    m_v = [_mean_per_sample(V[i]) for i in range(N)]
    target_x = [sum(coupling_matrix[i, j] * m_x[j] for j in range(N) if j != i) for i in range(N)]

    if damping_matrix is not None:
        Gamma = damping_matrix
    else:
        gamma_couple_val = gamma_couple if gamma_couple is not None else 0.0
        eye_N = torch.eye(N, device=coupling_matrix.device, dtype=coupling_matrix.dtype)
        Gamma = gamma_self * eye_N + gamma_couple_val * (eye_N - coupling_matrix)
    off_diag_v = [sum(Gamma[i, j] * m_v[j] for j in range(N) if j != i) for i in range(N)]

    dV: List[List[torch.Tensor]] = []
    for i in range(N):
        dv_i = []
        for x, v in zip(X[i], V[i]):
            target_xi = _broadcast_per_sample(target_x[i], x)
            confine = time_scale * (alpha[i] * omega_sq[i] * x + beta[i] * norm_f_k_global * (target_xi - x))
            off_diag_vi = _broadcast_per_sample(off_diag_v[i], v)
            damping = -Gamma[i, i] * v - off_diag_vi
            if scale_damping_with_time:
                damping = time_scale * damping
            dv_i.append(damping + confine)
        dV.append(dv_i)
    return dV


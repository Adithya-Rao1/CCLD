from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch

from core.coupling import build_coupling_matrix
from core.damping import _DEFAULT_TARGET_ZETA, DampingRegime
from core.drift import _broadcast_per_sample, _mean_frob_norm, _mean_per_sample

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


def drift_fn_coupled_gamma(
    X: List[List[torch.Tensor]],
    V: List[List[torch.Tensor]],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma_self: float,
    gamma_couple: float,
    coupling_matrix: Optional[torch.Tensor] = None,
    constant_k: bool = False,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> List[List[torch.Tensor]]:
    N = len(X)
    if N < 1:
        raise ValueError(f"drift_fn_coupled_gamma requires N >= 1, got N={N}")
    if not (len(V) == len(K_self) == len(alpha) == len(beta) == N):
        raise ValueError("X, V, K_self, alpha, beta must all have length N")

    time_scale = time_scale_fn(t, T) if time_scale_fn is not None else (T - t) / (t + T)
    norm_f_k_self = [_mean_frob_norm(K_self[i]) for i in range(N)]
    if not constant_k:
        norm_f_k_global = time_scale * _mean_frob_norm(K_global)
    else:
        norm_f_k_global = torch.as_tensor(1.0, device=X[0][0].device)
    omega_sq = [-(norm_f_k_self[i] + norm_f_k_global) for i in range(N)]

    if N == 1:
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
    target_v = [sum(coupling_matrix[i, j] * m_v[j] for j in range(N) if j != i) for i in range(N)]

    dV: List[List[torch.Tensor]] = []
    for i in range(N):
        dv_i = []
        for x, v in zip(X[i], V[i]):
            target_xi = _broadcast_per_sample(target_x[i], x)
            confine = time_scale * (alpha[i] * omega_sq[i] * x + beta[i] * norm_f_k_global * (target_xi - x))
            target_vi = _broadcast_per_sample(target_v[i], v)
            damping = -gamma_self * v + gamma_couple * (target_vi - v)
            if scale_damping_with_time:
                damping = time_scale * damping
            dv_i.append(damping + confine)
        dV.append(dv_i)
    return dV


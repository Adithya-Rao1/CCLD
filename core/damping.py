from __future__ import annotations

import math
from typing import List, Literal, Optional, Tuple

import torch

from core.drift import _mean_frob_norm, _time_scale

DampingRegime = Literal["underdamped", "critically_damped", "overdamped"]

_DEFAULT_TARGET_ZETA = {
    "underdamped": 0.5,
    "critically_damped": 1.0,
    "overdamped": 2.0,
}


def effective_k(alpha_i: float, k_i: float) -> float:
    return alpha_i * k_i


def effective_omega(alpha_i: float, k_i: float) -> float:
    return math.sqrt(max(effective_k(alpha_i, k_i), 0.0))


def effective_damping_ratio(alpha_i: float, gamma_i: float, k_i: float) -> float:
    omega_i = effective_omega(alpha_i, k_i)
    if omega_i < 1e-12:
        return float("inf") if gamma_i > 0 else 0.0
    return gamma_i / (2.0 * omega_i)


def classify_regime(zeta: float, tol: float = 0.05) -> DampingRegime:
    if abs(zeta - 1.0) <= tol:
        return "critically_damped"
    return "underdamped" if zeta < 1.0 else "overdamped"


def calibrate_gamma_for_regime(
    alpha_i: float,
    k_i_reference: float,
    regime: DampingRegime,
    target_zeta: Optional[float] = None,
) -> float:
    zeta = target_zeta if target_zeta is not None else _DEFAULT_TARGET_ZETA[regime]
    omega_i = effective_omega(alpha_i, k_i_reference)
    return 2.0 * zeta * omega_i


def calibrate_gammas_for_regime(
    alpha: List[float],
    k_reference: List[float],
    regime: DampingRegime,
    target_zeta: Optional[float] = None,
) -> List[float]:
    return [calibrate_gamma_for_regime(alpha[i], k_reference[i], regime, target_zeta) for i in range(len(alpha))]


def realized_k_i(K_self_i: List[torch.Tensor], K_global: List[torch.Tensor], t: torch.Tensor, T: int, constant_k: bool = False,
                  k_global_reference: float = 1.0) -> float:
    norm_self = _mean_frob_norm(K_self_i)
    if constant_k:
        norm_global = torch.as_tensor(k_global_reference)
    else:
        time_scale = _time_scale(t, T)
        norm_global = time_scale * _mean_frob_norm(K_global)
    return float((norm_self + norm_global).item())


def realized_damping_ratios(
    alpha: List[float],
    gamma: List[float],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    constant_k: bool = False,
    k_global_reference: float = 1.0,
) -> List[float]:
    N = len(alpha)
    k_vals = [realized_k_i(K_self[i], K_global, t, T, constant_k, k_global_reference) for i in range(N)]
    return [effective_damping_ratio(alpha[i], gamma[i], k_vals[i]) for i in range(N)]
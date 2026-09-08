from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch

from core.drift import _time_scale
from core.sde import _primary_noise_and_sigma
from synthetic.drift_coupled_gamma import drift_fn_coupled_gamma

def anderson_em_step_coupled_gamma(
    X: List[List[torch.Tensor]],
    V: List[List[torch.Tensor]],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma_self: Optional[float],
    gamma_couple: Optional[float],
    coupling_matrix_drift: Optional[torch.Tensor],
    constant_k: bool,
    dt: float,
    G: torch.Tensor,
    primary_index: int = 0,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    scale_kinematics_with_time: bool = True,
    scale_diffusion_with_time: bool = True,
    k_global_reference: float = 1.0,
    damping_matrix: Optional[torch.Tensor] = None,
):
    N = len(X)
    dV = drift_fn_coupled_gamma(
        X, V, K_self, K_global, t, T, alpha, beta, gamma_self, gamma_couple,
        coupling_matrix=coupling_matrix_drift, constant_k=constant_k,
        scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
        k_global_reference=k_global_reference, damping_matrix=damping_matrix,
    )
    time_scale = _time_scale(t, T, time_scale_fn)
    kin_scale = time_scale if scale_kinematics_with_time else 1.0
    if scale_diffusion_with_time:
        G = G * time_scale.clamp_min(0).sqrt()

    mu = [[v + dv * dt for v, dv in zip(V[i], dV[i])] for i in range(N)]

    primary_shape = mu[0][primary_index].shape
    device, dtype = mu[0][primary_index].device, mu[0][primary_index].dtype
    primary_noise, primary_sigma = _primary_noise_and_sigma(G, N, primary_shape, device, dtype)

    sqrt_dt = math.sqrt(dt)
    V_next: List[List[torch.Tensor]] = []
    for i in range(N):
        v_i = []
        for k in range(len(mu[i])):
            if k == primary_index:
                v_i.append(mu[i][k] + sqrt_dt * primary_noise[i])
            else:
                v_i.append(mu[i][k] + sqrt_dt * G[i, i] * torch.randn_like(mu[i][k]))
        V_next.append(v_i)

    X_next = [[x + v * kin_scale * dt for x, v in zip(X[i], V_next[i])] for i in range(N)]

    z_list = [primary_noise[i] / primary_sigma[i] for i in range(N)]
    sigma_list = [sqrt_dt * primary_sigma[i] for i in range(N)]

    return X_next, V_next, mu, z_list, sigma_list

def anderson_reverse_step_coupled_gamma(
    X: List[List[torch.Tensor]],
    V: List[List[torch.Tensor]],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    score_outputs: List[List[torch.Tensor]],
    t: int,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma_self: Optional[float],
    gamma_couple: Optional[float],
    coupling_matrix_drift: Optional[torch.Tensor],
    constant_k: bool,
    dt: float,
    G: torch.Tensor,
    score_scale: float = 1.0,
    primary_index: int = 0,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    scale_kinematics_with_time: bool = True,
    scale_diffusion_with_time: bool = True,
    k_global_reference: float = 1.0,
    damping_matrix: Optional[torch.Tensor] = None,
):
    N = len(X)
    t_tensor = torch.as_tensor(t, device=X[0][0].device, dtype=X[0][0].dtype)
    dV = drift_fn_coupled_gamma(
        X, V, K_self, K_global, t_tensor, T, alpha, beta, gamma_self, gamma_couple,
        coupling_matrix=coupling_matrix_drift, constant_k=constant_k,
        scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
        k_global_reference=k_global_reference, damping_matrix=damping_matrix,
    )
    time_scale = _time_scale(t_tensor, T, time_scale_fn)
    kin_scale = time_scale if scale_kinematics_with_time else 1.0
    if scale_diffusion_with_time:
        G = G * time_scale.clamp_min(0).sqrt()
    Sigma = G @ G.T

    sqrt_dt = math.sqrt(dt)
    primary_shape = V[0][primary_index].shape
    device, dtype = V[0][primary_index].device, V[0][primary_index].dtype
    eps = [torch.randn(primary_shape, device=device, dtype=dtype) for _ in range(N)]

    V_new: List[List[torch.Tensor]] = []
    for i in range(N):
        v_i = []
        for k in range(len(V[i])):
            dv = dV[i][k] * dt
            if k == primary_index:
                score_correction = sum(
                    Sigma[i, j] * score_outputs[j][primary_index].float() for j in range(N)
                ) * score_scale
                noise = sqrt_dt * sum(G[i, j] * eps[j] for j in range(N))
            else:
                score_correction = Sigma[i, i] * score_outputs[i][k].float() * score_scale
                noise = sqrt_dt * G[i, i] * torch.randn_like(V[i][k].float())
            v_i.append(V[i][k].float() - dv + score_correction * dt + noise)
        V_new.append(v_i)

    X_new = [[X[i][k].float() - V_new[i][k] * kin_scale * dt for k in range(len(X[i]))] for i in range(N)]
    return X_new, V_new

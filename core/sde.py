from __future__ import annotations

import math
from typing import Callable, List, Literal, Optional, Tuple

import torch

from core.drift import _time_scale, drift_fn_n

DiffusionMode = Literal["shared", "independent"]


def build_g_matrix_n(
    sigma_t: torch.Tensor,
    N: int,
    diffusion_mode: DiffusionMode = "shared",
    g_per_task: Optional[List[float]] = None,
    coupling_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = sigma_t.device if torch.is_tensor(sigma_t) else None
    G = torch.zeros((N, N), device=device, dtype=torch.float32)

    if diffusion_mode == "shared":
        for i in range(N):
            G[i, i] = sigma_t
    elif diffusion_mode == "independent":
        if g_per_task is None or len(g_per_task) != N:
            raise ValueError
        for i in range(N):
            G[i, i] = g_per_task[i]
    else:
        raise ValueError

    if coupling_matrix is not None:
        off_diag = coupling_matrix.to(device=device, dtype=torch.float32).clone()
        off_diag.fill_diagonal_(0.0)
        G = G + off_diag

    return G


def _primary_noise_and_sigma(G: torch.Tensor, N: int, primary_shape, device, dtype) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    z = [torch.randn(primary_shape, device=device, dtype=dtype) for _ in range(N)]
    Sigma = G @ G.T
    noise = []
    sigma_eff = []
    for i in range(N):
        n_i = sum(G[i, j] * z[j] for j in range(N))
        noise.append(n_i)
        sigma_eff.append(torch.sqrt(Sigma[i, i].clamp(min=1e-12)))
    return noise, sigma_eff


def em_step_n(
    X: List[List[torch.Tensor]],
    V: List[List[torch.Tensor]],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma: List[float],
    coupling_matrix_drift: Optional[torch.Tensor],
    use_gamma: bool,
    constant_k: bool,
    dt: float,
    G: torch.Tensor,
    primary_index: int = 0,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    scale_kinematics_with_time: bool = True,
    scale_diffusion_with_time: bool = True,
):
    N = len(X)
    dV = drift_fn_n(
        X, V, K_self, K_global, t, T, alpha, beta, gamma,
        coupling_matrix=coupling_matrix_drift, use_gamma=use_gamma, constant_k=constant_k,
        scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
    )
    time_scale = _time_scale(t, T, time_scale_fn)
    kin_scale = time_scale if scale_kinematics_with_time else 1.0
    if scale_diffusion_with_time:
        G = G * time_scale.clamp_min(0).sqrt()

    mu = [[v - dv * dt for v, dv in zip(V[i], dV[i])] for i in range(N)]

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


def reverse_step_n(
    X: List[List[torch.Tensor]],
    V: List[List[torch.Tensor]],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    score_outputs: List[List[torch.Tensor]],
    t: int,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma: List[float],
    coupling_matrix_drift: Optional[torch.Tensor],
    use_gamma: bool,
    constant_k: bool,
    dt: float,
    G: torch.Tensor,
    score_scale: float = 1.0,
    primary_index: int = 0,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    scale_kinematics_with_time: bool = True,
    scale_diffusion_with_time: bool = True,
):
    N = len(X)
    t_tensor = torch.as_tensor(t, device=X[0][0].device, dtype=X[0][0].dtype)
    dV = drift_fn_n(
        X, V, K_self, K_global, t_tensor, T, alpha, beta, gamma,
        coupling_matrix=coupling_matrix_drift, use_gamma=use_gamma, constant_k=constant_k,
        scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
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
            v_i.append(V[i][k].float() + dv - score_correction * dt + noise)
        V_new.append(v_i)

    X_new = [[X[i][k].float() + V_new[i][k] * kin_scale * dt for k in range(len(X[i]))] for i in range(N)]
    return X_new, V_new


def _score_norm_sq(score_list: List[torch.Tensor]) -> torch.Tensor:
    acc = None
    for s in score_list:
        sq = (s.float() ** 2).flatten(start_dim=1).sum(dim=-1)
        acc = sq if acc is None else acc + sq
    return acc


def ndsm_loss_n(
    X: List[List[torch.Tensor]],
    score_fn,
    y_n_list: List[List[torch.Tensor]],
    mu_nm1_list: List[List[torch.Tensor]],
    z_n_list: List[torch.Tensor],
    sigma_nm1_list: List[torch.Tensor],
    t_n,
    skip_mu_diff: bool = False,
    primary_index: int = 0,
) -> torch.Tensor:
    N = len(X)
    score_state = score_fn(X, y_n_list, t_n)
    yn_loss = 0.5 * sum(_score_norm_sq(score_state[i]) for i in range(N)).mean()

    sigma_is_zero = all(bool((s == 0).all().item()) for s in sigma_nm1_list) if sigma_nm1_list else False
    if skip_mu_diff or sigma_is_zero:
        return yn_loss

    score_mu = score_fn(X, mu_nm1_list, t_n)

    mu_diff_terms = []
    for i in range(N):
        diff = score_state[i][primary_index] - score_mu[i][primary_index]
        noise_scale = z_n_list[i] / sigma_nm1_list[i]
        term = (noise_scale.float() * diff.float()).flatten(start_dim=1).sum(dim=-1)
        mu_diff_terms.append(term)
    mu_diff_loss = torch.stack(mu_diff_terms, dim=0).sum(dim=0).mean()

    return yn_loss + mu_diff_loss

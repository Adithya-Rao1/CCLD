from __future__ import annotations

import math
from typing import List, Tuple

import torch

from core.coupling import build_coupling_matrix

def cosine_beta_schedule(T, s=0.008):
    steps = torch.linspace(0, T, T + 1)
    t = steps / T
 
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
 
    return alphas_cumprod

def per_step_betas_from_cumprod(alphas_cumprod: torch.Tensor, max_beta: float = 0.999) -> torch.Tensor:
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=0.0, max=max_beta)

def independent_coupling(N: int, **kwargs) -> torch.Tensor:
    return build_coupling_matrix(N, mode="independent", **kwargs)

# Baseline 1: DDPM + Cosine
def make_ddpm_schedule(T: int, device=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alphas_cumprod = cosine_beta_schedule(T).to(device)
    betas = per_step_betas_from_cumprod(alphas_cumprod).to(device)
    alphas = 1.0 - betas
    return alphas_cumprod, betas, alphas


def ddpm_forward_marginal(x0: torch.Tensor, t: torch.Tensor, alphas_cumprod: torch.Tensor):
    ac_t = alphas_cumprod[t]
    noise = torch.randn_like(x0)
    x_t = torch.sqrt(ac_t) * x0 + torch.sqrt(1.0 - ac_t) * noise
    return x_t, noise


def ddpm_forward_n(X0: List[torch.Tensor], t: torch.Tensor, alphas_cumprod: torch.Tensor):
    out = [ddpm_forward_marginal(x0, t, alphas_cumprod) for x0 in X0]
    X_t = [o[0] for o in out]
    noise = [o[1] for o in out]
    return X_t, noise


def ddpm_ancestral_step(
    x_t: torch.Tensor, eps_pred: torch.Tensor, t: int,
    betas: torch.Tensor, alphas: torch.Tensor, alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    beta_t, alpha_t, ac_t = betas[t], alphas[t], alphas_cumprod[t + 1]
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - ac_t)) * eps_pred)
    if t == 0:
        return mean
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(beta_t) * noise


def ddpm_reverse_step_n(
    X_t: List[torch.Tensor], eps_pred: List[torch.Tensor], t: int,
    betas: torch.Tensor, alphas: torch.Tensor, alphas_cumprod: torch.Tensor,
) -> List[torch.Tensor]:
    return [ddpm_ancestral_step(X_t[i], eps_pred[i], t, betas, alphas, alphas_cumprod) for i in range(len(X_t))]


# Baseline 2: SDM
def vp_beta_t(t: torch.Tensor, T: int, beta_min: float = 0.1, beta_max: float = 20.0) -> torch.Tensor:
    return beta_min + (t / T) * (beta_max - beta_min)


def vp_sde_drift_n(X: List[torch.Tensor], beta_t: torch.Tensor) -> List[torch.Tensor]:
    return [-0.5 * beta_t * x for x in X]


def vp_alpha_bar(t: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> torch.Tensor:
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2
    return torch.exp(-integral)


def vp_sde_forward_marginal(x0: torch.Tensor, t: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0):
    ac_t = vp_alpha_bar(t, beta_min, beta_max)
    noise = torch.randn_like(x0)
    x_t = torch.sqrt(ac_t) * x0 + torch.sqrt(1.0 - ac_t) * noise
    return x_t, noise


def vp_sde_forward_marginal_n(X0: List[torch.Tensor], t: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0):
    out = [vp_sde_forward_marginal(x0, t, beta_min, beta_max) for x0 in X0]
    X_t = [o[0] for o in out]
    noise = [o[1] for o in out]
    return X_t, noise


def vp_sde_reverse_step_n(
    X: List[torch.Tensor], score: List[torch.Tensor], beta_t: torch.Tensor, dt: float,
) -> List[torch.Tensor]:
    drift = vp_sde_drift_n(X, beta_t)
    g_sq = beta_t
    g = torch.sqrt(beta_t)
    out = []
    for x, d, s in zip(X, drift, score):
        reverse_drift = d - g_sq * s
        noise = g * math.sqrt(dt) * torch.randn_like(x)
        out.append(x - reverse_drift * dt + noise)
    return out
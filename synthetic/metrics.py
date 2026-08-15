from __future__ import annotations

from typing import Tuple

import torch


def fit_gaussian(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim == 1:
        samples = samples.unsqueeze(-1)
    mean = samples.mean(dim=0)
    N = samples.shape[1]
    if N == 1:
        cov = samples.var(dim=0, unbiased=True).reshape(1, 1)
    else:
        cov = torch.cov(samples.T)
    return mean, cov


def gaussian_kl(
    mean_p: torch.Tensor, cov_p: torch.Tensor, mean_q: torch.Tensor, cov_q: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    k = mean_p.shape[-1]
    I = torch.eye(k, dtype=cov_q.dtype, device=cov_q.device)
    cov_p = (cov_p + cov_p.T) / 2 + eps * I
    cov_q = (cov_q + cov_q.T) / 2 + eps * I
    cov_q_inv = torch.linalg.inv(cov_q)
    diff = (mean_q - mean_p).reshape(-1, 1)
    trace_term = torch.trace(cov_q_inv @ cov_p)
    mean_term = (diff.T @ cov_q_inv @ diff).squeeze()
    logdet_q = torch.linalg.slogdet(cov_q)[1]
    logdet_p = torch.linalg.slogdet(cov_p)[1]
    kl = 0.5 * (trace_term + mean_term - k + logdet_q - logdet_p)
    return kl.clamp_min(0.0)


def _sqrtm_psd(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mat = (mat + mat.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp_min(eps)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def gaussian_wasserstein2(
    mean_a: torch.Tensor, cov_a: torch.Tensor, mean_b: torch.Tensor, cov_b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    k = mean_a.shape[-1]
    I = torch.eye(k, dtype=cov_a.dtype, device=cov_a.device)
    cov_a = (cov_a + cov_a.T) / 2 + eps * I
    cov_b = (cov_b + cov_b.T) / 2 + eps * I
    mean_term = ((mean_a - mean_b) ** 2).sum()
    sqrt_b = _sqrtm_psd(cov_b)
    inner = sqrt_b @ cov_a @ sqrt_b
    sqrt_inner = _sqrtm_psd(inner)
    trace_term = torch.trace(cov_a) + torch.trace(cov_b) - 2.0 * torch.trace(sqrt_inner)
    sq_dist = (mean_term + trace_term).clamp_min(0.0)
    return torch.sqrt(sq_dist)


def gaussian_mutual_information_matrix(cov: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    std = torch.sqrt(torch.diagonal(cov).clamp_min(eps))
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
    corr = corr.clamp(-1.0 + eps, 1.0 - eps)
    mi = -0.5 * torch.log1p(-(corr ** 2))
    mi.fill_diagonal_(0.0)
    return mi


def lag_k_autocorrelation(series: torch.Tensor, k: int = 1) -> float:
    series = series.flatten().float()
    n = series.numel()
    if n <= k:
        return float("nan")
    x = series - series.mean()
    denom = (x ** 2).sum()
    if denom < 1e-12:
        return 0.0
    num = (x[: n - k] * x[k:]).sum()
    return float((num / denom).item())


def integrated_autocorrelation_time(series: torch.Tensor, max_lag: int = None) -> float:
    series = series.flatten().float()
    n = series.numel()
    max_lag = max_lag or max(1, n // 4)
    tau = 1.0
    for k in range(1, max_lag + 1):
        rho = lag_k_autocorrelation(series, k)
        if rho <= 0:
            break
        tau += 2.0 * rho
    return tau
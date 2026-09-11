from __future__ import annotations

from typing import Optional

import torch
from scipy.linalg import solve_continuous_lyapunov

from synthetic.metrics import gaussian_mutual_information_matrix

def _sample_from_covariance(cov: torch.Tensor, N: int, n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    jitter = 1e-6 * torch.eye(N, dtype=cov.dtype, device=cov.device)
    mean = torch.zeros(N, dtype=cov.dtype, device=cov.device)
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov + jitter)
    return dist.sample((n_samples,))


class GroundTruthCoupledOU:
    def __init__(self, theta: torch.Tensor, sigma_gt: torch.Tensor):
        if theta.shape[0] != theta.shape[1] or sigma_gt.shape[0] != sigma_gt.shape[1] or theta.shape[0] != sigma_gt.shape[0]:
            raise ValueError
        self.theta = theta
        self.sigma_gt = sigma_gt
        self.N = theta.shape[0]

    def stationary_covariance(self) -> torch.Tensor:
        A = (-self.theta).double().cpu().numpy()
        Q = (-(self.sigma_gt @ self.sigma_gt.T)).double().cpu().numpy()
        cov = solve_continuous_lyapunov(A, Q)
        cov = torch.as_tensor((cov + cov.T) / 2, dtype=self.theta.dtype)
        return cov

    def pairwise_mutual_information(self) -> torch.Tensor:
        return gaussian_mutual_information_matrix(self.stationary_covariance())

    def sample_stationary(self, n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        return _sample_from_covariance(self.stationary_covariance(), self.N, n_samples, seed)


class DirectionalGroundTruthOU:
    def __init__(self, theta: torch.Tensor, sigma_gt: torch.Tensor, lag_delta: float):
        if theta.shape[0] != theta.shape[1] or sigma_gt.shape != theta.shape:
            raise ValueError
        self.theta = theta
        self.sigma_gt = sigma_gt
        self.lag_delta = lag_delta
        self.N_orig = theta.shape[0]
        self.N = 2 * self.N_orig

    def _base_stationary_covariance(self) -> torch.Tensor:
        A = (-self.theta).double().cpu().numpy()
        Q = (-(self.sigma_gt @ self.sigma_gt.T)).double().cpu().numpy()
        cov = solve_continuous_lyapunov(A, Q)
        return torch.as_tensor((cov + cov.T) / 2, dtype=self.theta.dtype)

    def lagged_cross_covariance(self) -> torch.Tensor:
        """Cov(X(tau), X(tau+lag_delta)) = Sigma @ Phi(lag_delta)^T, Phi(d) = exp(-theta*d)."""
        Sigma = self._base_stationary_covariance()
        Phi = torch.matrix_exp((-self.theta).double().cpu() * self.lag_delta).to(Sigma.dtype)
        return Sigma @ Phi.T

    def stationary_covariance(self) -> torch.Tensor:
        Sigma = self._base_stationary_covariance()
        cross = self.lagged_cross_covariance()
        top = torch.cat([Sigma, cross], dim=1)
        bottom = torch.cat([cross.T, Sigma], dim=1)
        return torch.cat([top, bottom], dim=0)

    def pairwise_mutual_information(self) -> torch.Tensor:
        return gaussian_mutual_information_matrix(self.stationary_covariance())

    def sample_stationary(self, n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        return _sample_from_covariance(self.stationary_covariance(), self.N, n_samples, seed)


def make_ground_truth(
    N: int,
    coupling_strength: float = 0.5,
    seed: int = 0,
    base_decay: float = 1.0,
    sigma_scale: float = 1.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> GroundTruthCoupledOU:
    if N < 2 or not (0.0 <= coupling_strength < 1.0):
        raise ValueError

    g = torch.Generator().manual_seed(seed)
    w = torch.rand((N, N), generator=g, dtype=dtype)
    w = (w + w.T) / 2
    w.fill_diagonal_(0.0)
    row_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    w = w / row_sum

    theta = base_decay * torch.eye(N, dtype=dtype) - coupling_strength * base_decay * w
    theta = ((theta + theta.T) / 2).to(device=device)
    sigma_gt = (sigma_scale * torch.eye(N, dtype=dtype)).to(device=device)
    return GroundTruthCoupledOU(theta, sigma_gt)


def make_ground_truth_from_coupling(
    N: int,
    C: torch.Tensor,
    coupling_strength: float = 0.5,
    base_decay: float = 1.0,
    sigma_scale: float = 1.0,
) -> GroundTruthCoupledOU:
    if N < 2 or not (0.0 <= coupling_strength < 1.0):
        raise ValueError
    if C.shape != (N, N):
        raise ValueError

    theta = base_decay * torch.eye(N, dtype=C.dtype, device=C.device) - coupling_strength * base_decay * C
    theta = (theta + theta.T) / 2
    sigma_gt = sigma_scale * torch.eye(N, dtype=C.dtype, device=C.device)
    return GroundTruthCoupledOU(theta, sigma_gt)


def make_directional_ground_truth(
    N: int,
    coupling_strength: float = 0.5,
    seed: int = 0,
    base_decay: float = 1.0,
    sigma_scale: float = 1.0,
    lag_delta: float = 0.5,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> DirectionalGroundTruthOU:
    if N < 2 or not (0.0 <= coupling_strength < 1.0):
        raise ValueError

    g = torch.Generator().manual_seed(seed)
    w = torch.rand((N, N), generator=g, dtype=dtype)
    w.fill_diagonal_(0.0)
    row_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    w = w / row_sum

    theta = (base_decay * torch.eye(N, dtype=dtype) - coupling_strength * base_decay * w).to(device=device)
    sigma_gt = (sigma_scale * torch.eye(N, dtype=dtype)).to(device=device)
    return DirectionalGroundTruthOU(theta, sigma_gt, lag_delta)
from __future__ import annotations

from typing import Optional

import torch
from scipy.linalg import solve_continuous_lyapunov

from synthetic.metrics import gaussian_mutual_information_matrix


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
        if seed is not None:
            torch.manual_seed(seed)
        cov = self.stationary_covariance()
        jitter = 1e-6 * torch.eye(self.N, dtype=cov.dtype)
        mean = torch.zeros(self.N, dtype=cov.dtype)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov + jitter)
        return dist.sample((n_samples,))


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
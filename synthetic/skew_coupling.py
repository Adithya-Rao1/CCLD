from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def parametrize_skew_matrix(W: torch.Tensor) -> torch.Tensor:
    if W.shape[0] != W.shape[1]:
        raise ValueError
    return W - W.T


def reference_stationary_covariance(K: torch.Tensor, target_variance: float = 1.0) -> torch.Tensor:
    N = K.shape[0]
    I = torch.eye(N, device=K.device, dtype=K.dtype)
    return target_variance * torch.block_diag(torch.linalg.inv(K), I)


def inject_skew_coupling(A0: torch.Tensor, Sigma_ref: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    if not torch.allclose(J, -J.T, atol=1e-5):
        raise ValueError
    return A0 - J @ torch.linalg.inv(Sigma_ref)


def inject_skew_coupling_additive(A0: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    if not torch.allclose(J, -J.T, atol=1e-5):
        raise ValueError
    return A0 - J


def stationary_covariance_from_drift(A0_new: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    from scipy.linalg import solve_continuous_lyapunov
    max_real_eig = torch.linalg.eigvals(A0_new).real.max().item()
    if max_real_eig >= 0:
        raise ValueError
    a = A0_new.double().cpu().numpy()
    q = (-Q).double().cpu().numpy()
    cov = solve_continuous_lyapunov(a, q)
    return torch.as_tensor((cov + cov.T) / 2, dtype=A0_new.dtype, device=A0_new.device)


def skew_drift_correction(
    X: List[List[torch.Tensor]], V: List[List[torch.Tensor]],
    J: torch.Tensor, Sigma_ref: torch.Tensor,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    from core.drift import _broadcast_per_sample, _mean_per_sample

    N = len(X)
    M = -J @ torch.linalg.inv(Sigma_ref)
    m_x = [_mean_per_sample(X[i]) for i in range(N)]
    m_v = [_mean_per_sample(V[i]) for i in range(N)]
    m = m_x + m_v  

    dX: List[List[torch.Tensor]] = []
    dV: List[List[torch.Tensor]] = []
    for i in range(N):
        correction_x = sum(M[i, k] * m[k] for k in range(2 * N))
        correction_v = sum(M[N + i, k] * m[k] for k in range(2 * N))
        dX.append([_broadcast_per_sample(correction_x, x) + torch.zeros_like(x) for x in X[i]])
        dV.append([_broadcast_per_sample(correction_v, v) + torch.zeros_like(v) for v in V[i]])
    return dX, dV

from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch

from core.coupling import build_coupling_matrix


def _mean_frob_norm(tensors: List[torch.Tensor]) -> torch.Tensor:
    norms = []
    for t in tensors:
        tf = t.float()
        if tf.ndim >= 2:
            n = torch.linalg.norm(tf, ord="fro", dim=(-2, -1)) / math.sqrt(tf.shape[-2] * tf.shape[-1])
            n = n.mean() if n.ndim > 0 else n
        else:
            n = torch.linalg.norm(tf, ord="fro") / math.sqrt(tf.numel())
        norms.append(n)
    return torch.stack(norms).mean()


def _mean_per_sample(tensors: List[torch.Tensor]) -> torch.Tensor:
    means = [t.float().reshape(t.shape[0], -1).mean(dim=1) for t in tensors]
    return torch.stack(means, dim=0).mean(dim=0)


def _broadcast_per_sample(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return v.view(v.shape[0], *([1] * (like.ndim - 1)))


def drift_fn_n(
    X: List[List[torch.Tensor]],
    V: List[List[torch.Tensor]],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma: List[float],
    coupling_matrix: Optional[torch.Tensor] = None,
    use_gamma: bool = True,
    constant_k: bool = False,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> List[List[torch.Tensor]]:
    N = len(X)
    if N < 2:
        raise ValueError(f"drift_fn_n requires N >= 2 populations, got N={N}")
    if not (len(V) == len(K_self) == len(alpha) == len(beta) == len(gamma) == N):
        raise ValueError("X, V, K_self, alpha, beta, gamma must all have length N")

    if coupling_matrix is None:
        coupling_matrix = build_coupling_matrix(N, mode="mean_field", device=X[0][0].device, dtype=torch.float32)
    time_scale = time_scale_fn(t, T) if time_scale_fn is not None else (T - t) / (t + T)

    norm_f_k_self = [_mean_frob_norm(K_self[i]) for i in range(N)]

    if not constant_k:
        norm_f_k_global = time_scale * _mean_frob_norm(K_global)
    else:
        norm_f_k_global = torch.as_tensor(1.0, device=X[0][0].device)

    omega_sq = [-(norm_f_k_self[i] + norm_f_k_global) for i in range(N)]
    m = [_mean_per_sample(X[i]) for i in range(N)]  # each m[i] has shape (B,)

    target = [
        sum(coupling_matrix[i, j] * m[j] for j in range(N) if j != i)
        for i in range(N)
    ]  # each target[i] has shape (B,)

    dV: List[List[torch.Tensor]] = []
    for i in range(N):
        dv_i = []
        for x, v in zip(X[i], V[i]):
            target_i = _broadcast_per_sample(target[i], x)
            confine = time_scale * (alpha[i] * omega_sq[i] * x + beta[i] * norm_f_k_global * (target_i - x))
            if use_gamma:
                damping = -gamma[i] * v
                if scale_damping_with_time:
                    damping = time_scale * damping
                dv_i.append(damping + confine)
            else:
                dv_i.append(confine)
        dV.append(dv_i)

    return dV
from __future__ import annotations

import math
from typing import List, Optional

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
    s: float = 1e-8,
) -> List[List[torch.Tensor]]:
    N = len(X)
    if N < 2:
        raise ValueError(f"drift_fn_n requires N >= 2 populations, got N={N}")
    if not (len(V) == len(K_self) == len(alpha) == len(beta) == len(gamma) == N):
        raise ValueError("X, V, K_self, alpha, beta, gamma must all have length N")

    if coupling_matrix is None:
        coupling_matrix = build_coupling_matrix(N, mode="mean_field", device=X[0][0].device, dtype=torch.float32)

    time_scale = (T - t) / (t + T)

    norm_f_k_self = [_mean_frob_norm(K_self[i]) for i in range(N)]

    if not constant_k:
        norm_f_k_global = time_scale * _mean_frob_norm(K_global)
    else:
        norm_f_k_global = torch.as_tensor(1.0, device=X[0][0].device)

    omega_sq = [-(norm_f_k_self[i] + norm_f_k_global) for i in range(N)]
    m = [torch.stack([x.float().mean() for x in X[i]]).mean() for i in range(N)]

    target = [
        sum(coupling_matrix[i, j] * m[j] for j in range(N) if j != i)
        for i in range(N)
    ]

    dV: List[List[torch.Tensor]] = []
    for i in range(N):
        dv_i = []
        for x, v in zip(X[i], V[i]):
            core = alpha[i] * omega_sq[i] * x + beta[i] * norm_f_k_global * (target[i] - x)
            if use_gamma:
                core = -gamma[i] * v + core
            dv_i.append(time_scale * core)
        dV.append(dv_i)

    return dV
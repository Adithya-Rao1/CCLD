from __future__ import annotations

from typing import List, Literal, Optional

import torch

CouplingMode = Literal["mean_field", "pairwise", "independent"]

def build_coupling_matrix(
    N: int,
    mode: CouplingMode = "mean_field",
    weights: Optional[torch.Tensor] = None,
    normalize_rows: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if N < 2:
        raise ValueError

    if mode == "mean_field":
        C = torch.full((N, N), 1.0 / (N - 1), device=device, dtype=dtype)
        C.fill_diagonal_(0.0)
        return C

    if mode == "independent":
        return torch.zeros((N, N), device=device, dtype=dtype)

    if mode == "pairwise":
        if weights is None:
            raise ValueError
        if weights.shape != (N, N):
            raise ValueError
        C = weights.to(device=device, dtype=dtype).clone()
        C.fill_diagonal_(0.0)
        if normalize_rows:
            row_sums = C.sum(dim=-1, keepdim=True)
            safe_row_sums = torch.where(row_sums.abs() > 1e-12, row_sums, torch.ones_like(row_sums))
            C = C / safe_row_sums
        return C

    raise ValueError


def task_similarity_to_coupling(
    similarity: torch.Tensor,
    normalize_rows: bool = True,
) -> torch.Tensor:
    N = similarity.shape[0]
    return build_coupling_matrix(
        N=N, mode="pairwise", weights=similarity, normalize_rows=normalize_rows,
        device=similarity.device, dtype=similarity.dtype,
    )


def sinkhorn_symmetric_doubly_stochastic(
    W: torch.Tensor, n_iter: int = 200, tol: float = 1e-10,
) -> torch.Tensor:
    if W.shape[0] != W.shape[1]:
        raise ValueError
    if not torch.allclose(W, W.T, atol=1e-6):
        raise ValueError
    N = W.shape[0]
    d = torch.ones(N, device=W.device, dtype=W.dtype)
    for _ in range(n_iter):
        row_sums = (d[:, None] * W * d[None, :]).sum(dim=-1)
        row_sums = row_sums.clamp_min(1e-30)
        d = d / row_sums.sqrt()
        C = d[:, None] * W * d[None, :]
        if (C.sum(dim=-1) - 1.0).abs().max().item() < tol:
            break
    C = d[:, None] * W * d[None, :]
    C.fill_diagonal_(0.0)
    return C


def random_heterogeneous_coupling(
    N: int, epsilon: float, seed: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError
    if N < 2:
        raise ValueError
    J_offdiag = torch.ones((N, N), device=device, dtype=dtype)
    J_offdiag.fill_diagonal_(0.0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.rand((N, N), generator=gen).to(device=device, dtype=dtype)
    R = (raw + raw.T) / 2.0
    R.fill_diagonal_(0.0)
    W = (1.0 - epsilon) * J_offdiag + epsilon * R
    return sinkhorn_symmetric_doubly_stochastic(W)


def block_coupling(
    N: int, block_sizes: List[int], w_in: float, w_out: float,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if sum(block_sizes) != N:
        raise ValueError
    if w_in <= 0 or w_out <= 0:
        raise ValueError
    labels = torch.repeat_interleave(torch.arange(len(block_sizes)), torch.tensor(block_sizes))
    same_block = labels[:, None] == labels[None, :]
    W = torch.where(same_block, torch.tensor(float(w_in)), torch.tensor(float(w_out))).to(device=device, dtype=dtype)
    W.fill_diagonal_(0.0)
    return sinkhorn_symmetric_doubly_stochastic(W)
from __future__ import annotations

from typing import Literal, Optional

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
        raise ValueError(f"build_coupling_matrix requires N >= 2 coupled populations, got N={N}")

    if mode == "mean_field":
        C = torch.full((N, N), 1.0 / (N - 1), device=device, dtype=dtype)
        C.fill_diagonal_(0.0)
        return C

    if mode == "independent":
        return torch.zeros((N, N), device=device, dtype=dtype)

    if mode == "pairwise":
        if weights is None:
            raise ValueError('mode="pairwise" requires an explicit (N,N) `weights` tensor')
        if weights.shape != (N, N):
            raise ValueError(f"weights must have shape ({N},{N}), got {tuple(weights.shape)}")
        C = weights.to(device=device, dtype=dtype).clone()
        C.fill_diagonal_(0.0)
        if normalize_rows:
            row_sums = C.sum(dim=-1, keepdim=True)
            safe_row_sums = torch.where(row_sums.abs() > 1e-12, row_sums, torch.ones_like(row_sums))
            C = C / safe_row_sums
        return C

    raise ValueError(f"Unknown coupling mode: {mode!r}")


def task_similarity_to_coupling(
    similarity: torch.Tensor,
    normalize_rows: bool = True,
) -> torch.Tensor:
    N = similarity.shape[0]
    return build_coupling_matrix(
        N=N, mode="pairwise", weights=similarity, normalize_rows=normalize_rows,
        device=similarity.device, dtype=similarity.dtype,
    )
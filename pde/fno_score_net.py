from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from neuralop.models import FNO


class FNOScoreNetwork(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        cond_in_ch: int,
        n_diff_steps: int,
        n_modes=(12, 12),
        hidden_channels: int = 128,
        projection_channel_ratio: int = 2,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_diff_steps = n_diff_steps
        in_ch = 2 * n_tasks + cond_in_ch + 1
        self.fno = FNO(
            n_modes=n_modes, in_channels=in_ch, out_channels=n_tasks,
            hidden_channels=hidden_channels, projection_channel_ratio=projection_channel_ratio,
        )
        self.output_gain = nn.Parameter(torch.ones(n_tasks))

    def forward(
        self,
        conditioning: torch.Tensor,
        X_query: List[List[torch.Tensor]],
        V_query: List[List[torch.Tensor]],
        t,
    ) -> List[List[torch.Tensor]]:
        B, _, H, W = conditioning.shape
        x_stack = torch.cat([x[0] for x in X_query], dim=1)
        v_stack = torch.cat([v[0] for v in V_query], dim=1)
        t_channel = torch.full((B, 1, H, W), float(t) / self.n_diff_steps,
                                device=conditioning.device, dtype=conditioning.dtype)
        out = self.fno(torch.cat([x_stack, v_stack, conditioning, t_channel], dim=1))
        out = out * self.output_gain.view(1, -1, 1, 1)
        return [[out[:, i:i + 1]] for i in range(self.n_tasks)]


class FlatFNOScoreNetwork(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        cond_in_ch: int,
        n_diff_steps: int,
        n_modes=(12, 12),
        hidden_channels: int = 128,
        projection_channel_ratio: int = 2,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_diff_steps = n_diff_steps
        in_ch = n_tasks + cond_in_ch + 1
        self.fno = FNO(
            n_modes=n_modes, in_channels=in_ch, out_channels=n_tasks,
            hidden_channels=hidden_channels, projection_channel_ratio=projection_channel_ratio,
        )
        self.output_gain = nn.Parameter(torch.ones(n_tasks))

    def forward(self, conditioning: torch.Tensor, X: List[torch.Tensor], t) -> List[torch.Tensor]:
        B, _, H, W = conditioning.shape
        t_channel = torch.full((B, 1, H, W), float(t) / self.n_diff_steps,
                                device=conditioning.device, dtype=conditioning.dtype)
        out = self.fno(torch.cat(X + [conditioning, t_channel], dim=1))
        out = out * self.output_gain.view(1, -1, 1, 1)
        return list(out.split(1, dim=1))


def make_spatial_score_fn(score_net: FNOScoreNetwork, conditioning: torch.Tensor):
    return lambda X, V_query, t: score_net(conditioning, X, V_query, t)


def make_flat_fno_score_fn(score_net: FlatFNOScoreNetwork, conditioning: torch.Tensor):
    return lambda X, t: score_net(conditioning, X, t)

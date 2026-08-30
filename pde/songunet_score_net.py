from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from pde.vendored_songunet import SongUNet


def _parse_channel_mult(s) -> List[int]:
    if isinstance(s, (list, tuple)):
        return [int(v) for v in s]
    return [int(v) for v in str(s).split(",") if v.strip()]


class SongUNetScoreNetwork(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        cond_in_ch: int,
        n_diff_steps: int,
        img_resolution: int,
        model_channels: int = 32,
        channel_mult=(1, 2, 2),
        num_blocks: int = 2,
        attn_resolutions=(),
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_diff_steps = n_diff_steps
        in_ch = 2 * n_tasks + cond_in_ch  # X channels + V channels + conditioning (t via noise_labels)
        self.net = SongUNet(
            img_resolution=img_resolution, in_channels=in_ch, out_channels=n_tasks, label_dim=0,
            model_channels=model_channels, channel_mult=_parse_channel_mult(channel_mult),
            channel_mult_emb=4, num_blocks=num_blocks, attn_resolutions=list(attn_resolutions),
        )

    def forward(
        self,
        conditioning: torch.Tensor,
        X_query: List[List[torch.Tensor]],
        V_query: List[List[torch.Tensor]],
        t,
    ) -> List[List[torch.Tensor]]:
        B = conditioning.shape[0]
        x_stack = torch.cat([x[0] for x in X_query], dim=1)
        v_stack = torch.cat([v[0] for v in V_query], dim=1)
        noise_labels = torch.full((B,), float(t) / self.n_diff_steps,
                                   device=conditioning.device, dtype=conditioning.dtype)
        out = self.net(torch.cat([x_stack, v_stack, conditioning], dim=1), noise_labels, None)
        return [[out[:, i:i + 1]] for i in range(self.n_tasks)]


class FlatSongUNetScoreNetwork(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        cond_in_ch: int,
        n_diff_steps: int,
        img_resolution: int,
        model_channels: int = 32,
        channel_mult=(1, 2, 2),
        num_blocks: int = 2,
        attn_resolutions=(),
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_diff_steps = n_diff_steps
        in_ch = n_tasks + cond_in_ch
        self.net = SongUNet(
            img_resolution=img_resolution, in_channels=in_ch, out_channels=n_tasks, label_dim=0,
            model_channels=model_channels, channel_mult=_parse_channel_mult(channel_mult),
            channel_mult_emb=4, num_blocks=num_blocks, attn_resolutions=list(attn_resolutions),
        )

    def forward(self, conditioning: torch.Tensor, X: List[torch.Tensor], t) -> List[torch.Tensor]:
        B = conditioning.shape[0]
        noise_labels = torch.full((B,), float(t) / self.n_diff_steps,
                                   device=conditioning.device, dtype=conditioning.dtype)
        out = self.net(torch.cat(X + [conditioning], dim=1), noise_labels, None)
        return list(out.split(1, dim=1))


def make_songunet_score_fn(score_net: SongUNetScoreNetwork, conditioning: torch.Tensor):
    return lambda X, V_query, t: score_net(conditioning, X, V_query, t)


def make_flat_songunet_score_fn(score_net: FlatSongUNetScoreNetwork, conditioning: torch.Tensor):
    return lambda X, t: score_net(conditioning, X, t)
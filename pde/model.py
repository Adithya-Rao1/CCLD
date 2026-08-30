from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class PhysicsBackbone(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 32, out_ch: int = 128, n_downsample: int = 3):
        super().__init__()
        self.out_ch = out_ch
        chs = [base_ch * (2 ** i) for i in range(n_downsample)]
        layers = [ConvBlock(in_ch, chs[0])]
        for i in range(n_downsample - 1):
            layers.append(ConvBlock(chs[i], chs[i + 1], stride=2))
        layers.append(ConvBlock(chs[-1], out_ch, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        global_cond = feat.mean(dim=(-2, -1))
        return feat, global_cond


class FieldHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, latent_dim: int, out_hw: Tuple[int, int], base_ch: int = 64):
        super().__init__()
        self.out_hw = out_hw
        self.latent_dim = latent_dim
        self.trunk = nn.Sequential(ConvBlock(in_ch, base_ch), ConvBlock(base_ch, base_ch))
        self.y0_head = nn.Conv2d(base_ch, out_ch, 1)
        self.to_latent = nn.Linear(base_ch, latent_dim)

        readout_ch = max(latent_dim // 4, out_ch)
        self.readout_seed_hw = (out_hw[0] // 8, out_hw[1] // 8)
        self.readout_proj = nn.Linear(latent_dim, readout_ch * self.readout_seed_hw[0] * self.readout_seed_hw[1])
        self.readout_conv = nn.Sequential(
            ConvBlock(readout_ch, readout_ch), ConvBlock(readout_ch, readout_ch),
        )
        self.readout_ch = readout_ch
        self.readout_out = nn.Conv2d(readout_ch, out_ch, 1)

    def encode(self, shared_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(shared_feat)
        pooled = feat.mean(dim=(-2, -1))
        latent = self.to_latent(pooled)
        y0 = F.interpolate(self.y0_head(feat), size=self.out_hw, mode="bilinear", align_corners=False)
        return latent, y0

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.shape[0]
        x = self.readout_proj(latent).view(B, self.readout_ch, *self.readout_seed_hw)
        x = self.readout_conv(x)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.readout_out(x)


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t, device, dtype) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / half)
        t_val = torch.as_tensor(float(t), device=device, dtype=dtype).view(1)
        args = t_val * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class CrossFieldAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, tokens: torch.Tensor, cond_token: torch.Tensor) -> torch.Tensor:
        context = self.norm_ctx(torch.cat([tokens, cond_token], dim=1))
        attn_out, _ = self.attn(self.norm_q(tokens), context, context)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class MultiPhysicsScoreNetwork(nn.Module):
    def __init__(self, n_tasks: int, latent_dim: int, cond_dim: int, n_blocks: int = 3, n_heads: int = 4, time_dim: int = 64):
        super().__init__()
        self.n_tasks = n_tasks
        self.latent_dim = latent_dim
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.time_proj = nn.Linear(time_dim, latent_dim)
        self.cond_proj = nn.Linear(cond_dim, latent_dim)
        self.blocks = nn.ModuleList([CrossFieldAttentionBlock(latent_dim, n_heads) for _ in range(n_blocks)])
        self.out_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
            for _ in range(n_tasks)
        ])

    def forward(self, global_cond: torch.Tensor, V_query: List[List[torch.Tensor]], t) -> List[List[torch.Tensor]]:
        B = V_query[0][0].shape[0]
        device, dtype = V_query[0][0].device, V_query[0][0].dtype
        t_emb = self.time_proj(self.time_embed(t, device, dtype)).view(1, 1, -1)

        tokens = torch.stack([v[0] for v in V_query], dim=1) + t_emb
        cond_token = (self.cond_proj(global_cond) + t_emb.squeeze(1)).unsqueeze(1)

        for block in self.blocks:
            tokens = block(tokens, cond_token)

        out = []
        for i in range(self.n_tasks):
            out.append([self.out_heads[i](tokens[:, i, :])])
        return out


def make_score_fn(score_net: MultiPhysicsScoreNetwork, global_cond: torch.Tensor):
    return lambda X, V_query, t: score_net(global_cond, V_query, t)


class FlatScoreNetwork(nn.Module):
    def __init__(self, n_tasks: int, latent_dim: int, cond_dim: int, **kwargs):
        super().__init__()
        self.inner = MultiPhysicsScoreNetwork(n_tasks, latent_dim, cond_dim, **kwargs)

    def forward(self, global_cond: torch.Tensor, X: List[torch.Tensor], t) -> List[torch.Tensor]:
        out = self.inner(global_cond, [[x] for x in X], t)
        return [o[0] for o in out]


def make_flat_score_fn(score_net: FlatScoreNetwork, global_cond: torch.Tensor):
    return lambda X, t: score_net(global_cond, X, t)


class PhysicsModel(nn.Module):
    def __init__(
        self,
        task_names: List[str],
        cond_in_ch: int,
        out_hw: Tuple[int, int] = (128, 128),
        latent_dim: int = 64,
        backbone_ch: int = 128,
        base_ch: int = 32,
        n_downsample: int = 3,
        task_out_ch: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.latent_dim = latent_dim
        self.backbone = PhysicsBackbone(cond_in_ch, base_ch=base_ch, out_ch=backbone_ch, n_downsample=n_downsample)
        task_out_ch = task_out_ch or {}
        self.heads = nn.ModuleDict({
            name: FieldHead(self.backbone.out_ch, task_out_ch.get(name, 1), latent_dim, out_hw)
            for name in self.task_names
        })

    def encode(self, conditioning: torch.Tensor, k_reference: float = 1.0):
        feat, global_cond = self.backbone(conditioning)
        B = conditioning.shape[0]
        k_const = torch.full((B, 1), k_reference, device=conditioning.device, dtype=conditioning.dtype)
        X, K_self, K_global, y0 = [], [], [], {}
        for name in self.task_names:
            latent, y0_t = self.heads[name].encode(feat)
            X.append([latent])
            K_self.append([k_const])
            K_global.append(k_const)
            y0[name] = y0_t
        return X, K_self, K_global, global_cond, y0

    def decode(self, task_name: str, latent: torch.Tensor) -> torch.Tensor:
        return self.heads[task_name].decode(latent)


class SpatialFieldModel(nn.Module):
    def __init__(
        self,
        task_names: List[str],
        cond_in_ch: int,
        out_hw: Tuple[int, int] = (128, 128),
        backbone_ch: int = 128,
        base_ch: int = 32,
        n_downsample: int = 3,
        init_ch: int = 32,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.out_hw = out_hw
        self.backbone = PhysicsBackbone(cond_in_ch, base_ch=base_ch, out_ch=backbone_ch, n_downsample=n_downsample)
        self.init_heads = nn.ModuleDict({
            name: nn.Sequential(ConvBlock(self.backbone.out_ch, init_ch), nn.Conv2d(init_ch, 1, 1))
            for name in self.task_names
        })

    def encode(self, conditioning: torch.Tensor, k_reference: float = 1.0):
        feat, _ = self.backbone(conditioning)
        B = conditioning.shape[0]
        k_const = torch.full((B, 1), k_reference, device=conditioning.device, dtype=conditioning.dtype)
        X0 = {
            name: F.interpolate(self.init_heads[name](feat), size=self.out_hw, mode="bilinear", align_corners=False)
            for name in self.task_names
        }
        K_self = [[k_const] for _ in self.task_names]
        K_global = [k_const for _ in self.task_names]
        return X0, K_self, K_global
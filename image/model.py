from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiTaskBackbone(nn.Module):
    def __init__(self, out_ch: int = 512, pretrained: bool = True):
        super().__init__()
        self.out_ch = out_ch
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.reduce = nn.Sequential(
            nn.Conv2d(64 + 128 + 256 + 512, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.mean) / self.std
        x0 = self.stem(x)
        f1 = self.layer1(x0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        size = f1.shape[-2:]
        f2u = F.interpolate(f2, size=size, mode="bilinear", align_corners=False)
        f3u = F.interpolate(f3, size=size, mode="bilinear", align_corners=False)
        f4u = F.interpolate(f4, size=size, mode="bilinear", align_corners=False)
        shared = self.reduce(torch.cat([f1, f2u, f3u, f4u], dim=1))
        global_cond = shared.mean(dim=(-2, -1))
        return shared, global_cond


class ResidualConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class TaskHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, latent_dim: int = 64, out_hw: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.out_hw = out_hw
        self.res_blocks = nn.Sequential(ResidualConvBlock(in_ch), ResidualConvBlock(in_ch))
        self.pred_head = nn.Conv2d(in_ch, out_ch, 1)
        self.to_latent = nn.Conv2d(out_ch, latent_dim, 1)
        self.to_cond = nn.Conv2d(in_ch, latent_dim, 1)
        self.readout = nn.Conv2d(latent_dim, out_ch, 1)

    def encode(self, shared_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.res_blocks(shared_feat)
        y0 = self.pred_head(feat)
        latent = self.to_latent(y0)
        cond = self.to_cond(feat)
        return latent, cond, y0

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.readout(latent)
        return F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)


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

# TODO: Implement Metis-like CrossTaskAttentionBlock to mirror the Ewald sum decomposition with MHA node-level embedding refinement
class CrossTaskAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, task_tokens: List[torch.Tensor], cond_tokens: torch.Tensor) -> List[torch.Tensor]:
        context = self.norm_ctx(torch.cat(task_tokens + [cond_tokens], dim=1))
        out = []
        for tok in task_tokens:
            attn_out, _ = self.attn(self.norm_q(tok), context, context)
            tok = tok + attn_out
            tok = tok + self.mlp(self.norm2(tok))
            out.append(tok)
        return out


class MultiTaskScoreNetwork(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        latent_dim: int,
        img_ch: int,
        n_blocks: int = 4,
        n_heads: int = 4,
        time_dim: int = 128,
        spatial_stride: int = 2,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.latent_dim = latent_dim
        self.spatial_stride = max(spatial_stride, 1)
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.time_proj = nn.Linear(time_dim, latent_dim)
        self.cond_proj = nn.Conv2d(img_ch, latent_dim, 1)
        self.blocks = nn.ModuleList([CrossTaskAttentionBlock(latent_dim, n_heads) for _ in range(n_blocks)])
        self.task_conv_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(latent_dim, latent_dim, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(latent_dim, latent_dim, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(latent_dim, latent_dim, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(latent_dim, latent_dim, 1),
            ) for _ in range(n_tasks)
        ])

    def _tokenize(self, x: torch.Tensor, hw: Tuple[int, int], t_emb: torch.Tensor) -> torch.Tensor:
        small = F.adaptive_avg_pool2d(x, hw)
        return small.flatten(2).transpose(1, 2) + t_emb

    def forward(self, img_feat: torch.Tensor, V_query: List[List[torch.Tensor]], t) -> List[List[torch.Tensor]]:
        B, _, H, W = V_query[0][0].shape
        s = self.spatial_stride
        Hs, Ws = max(H // s, 1), max(W // s, 1)

        t_emb = self.time_proj(self.time_embed(t, img_feat.device, img_feat.dtype)).view(1, 1, -1)
        cond_tok = self._tokenize(self.cond_proj(img_feat), (Hs, Ws), t_emb)
        tokens = [self._tokenize(v[0], (Hs, Ws), t_emb) for v in V_query]

        for block in self.blocks:
            tokens = block(tokens, cond_tok)

        out = []
        for i, tok in enumerate(tokens):
            spatial = tok.transpose(1, 2).view(B, self.latent_dim, Hs, Ws)
            spatial = F.interpolate(spatial, size=(H, W), mode="bilinear", align_corners=False)
            out.append([self.task_conv_heads[i](spatial)])
        return out


def make_score_fn(score_net: MultiTaskScoreNetwork, img_feat: torch.Tensor):
    return lambda X, V_query, t: score_net(img_feat, V_query, t)


class FlatScoreNetwork(nn.Module):
    def __init__(self, n_tasks: int, latent_dim: int, img_ch: int, **kwargs):
        super().__init__()
        self.inner = MultiTaskScoreNetwork(n_tasks, latent_dim, img_ch, **kwargs)

    def forward(self, img_feat: torch.Tensor, X: List[torch.Tensor], t) -> List[torch.Tensor]:
        out = self.inner(img_feat, [[x] for x in X], t)
        return [o[0] for o in out]


def make_flat_score_fn(score_net: FlatScoreNetwork, img_feat: torch.Tensor):
    return lambda X, t: score_net(img_feat, X, t)


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        tasks: List[str],
        task_channels: Dict[str, int],
        latent_dim: int = 64,
        out_hw: Tuple[int, int] = (128, 128),
        backbone_ch: int = 512,
        pretrained: bool = True,
        backbone: Optional[MultiTaskBackbone] = None,
    ):
        super().__init__()
        self.tasks = list(tasks)
        self.latent_dim = latent_dim
        self.backbone = backbone or MultiTaskBackbone(out_ch=backbone_ch, pretrained=pretrained)
        self.heads = nn.ModuleDict({
            t: TaskHead(self.backbone.out_ch, task_channels[t], latent_dim=latent_dim, out_hw=out_hw)
            for t in tasks
        })

    def encode(self, image: torch.Tensor):
        feat, _ = self.backbone(image)
        X, K_self, K_global, y0 = [], [], [], {}
        for t in self.tasks:
            latent, cond, y0_t = self.heads[t].encode(feat)
            X.append([latent])
            K_self.append([cond])
            K_global.append(feat)
            y0[t] = y0_t
        return X, K_self, K_global, feat, y0

    def decode(self, task: str, latent: torch.Tensor) -> torch.Tensor:
        return self.heads[task].decode(latent)
from __future__ import annotations

import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.t()


def _sobel_grad(x: torch.Tensor):
    kx = _SOBEL_X.to(x.device, x.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(x.device, x.dtype).view(1, 1, 3, 3)
    return F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)


def _gaussian_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g).view(1, 1, size, size)


def depth_to_normals(depth: torch.Tensor) -> torch.Tensor:
    d = depth
    if d.ndim == 2:
        d = d.unsqueeze(0)
    d = d.unsqueeze(0).float()
    gx, gy = _sobel_grad(d)
    ones = torch.ones_like(gx)
    normal = torch.cat([-gx, -gy, ones], dim=1)
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return normal.squeeze(0)


def image_to_edges(image: torch.Tensor) -> torch.Tensor:
    img = image
    gray = img.mean(dim=0, keepdim=True) if img.shape[0] > 1 else img
    gray = gray.unsqueeze(0).float()
    gx, gy = _sobel_grad(gray)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).squeeze(0)
    return mag / mag.amax().clamp_min(1e-8)


def saliency_proxy(image: torch.Tensor) -> torch.Tensor:
    img = image
    gray = img.mean(dim=0, keepdim=True) if img.shape[0] > 1 else img
    gray = gray.unsqueeze(0).float()
    H, W = gray.shape[-2:]
    kernel = _gaussian_kernel(7, sigma=2.0, device=gray.device, dtype=gray.dtype)
    blurred = F.conv2d(gray, kernel, padding=3)
    contrast = (gray - blurred).abs()

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=gray.device, dtype=gray.dtype),
        torch.linspace(-1, 1, W, device=gray.device, dtype=gray.dtype),
        indexing="ij",
    )
    center_bias = torch.exp(-(xx ** 2 + yy ** 2) / 0.5).view(1, 1, H, W)

    sal = (contrast * center_bias).squeeze(0)
    return sal / sal.amax().clamp_min(1e-8)
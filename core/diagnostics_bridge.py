from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from core.diagnose_drift import DriftDetector, RollingStats, GradientTracker, MagnitudeTracker, format_hook_report

__all__ = [
    "DriftDetector",
    "RollingStats",
    "GradientTracker",
    "MagnitudeTracker",
    "format_hook_report",
    "attach_hooks_generic",
    "attach_grad_hooks_generic",
    "make_drift_detector",
]


def _named_leaf_modules(model: nn.Module, max_modules: Optional[int] = None) -> Iterable[Tuple[nn.Module, str]]:
    count = 0
    for name, module in model.named_modules():
        if name == "":
            continue
        if len(list(module.children())) > 0:
            continue
        yield module, name
        count += 1
        if max_modules is not None and count >= max_modules:
            return


def attach_hooks_generic(model: nn.Module, max_modules: Optional[int] = None) -> MagnitudeTracker:
    tracker = MagnitudeTracker()
    for module, name in _named_leaf_modules(model, max_modules):
        tracker.attach(module, name)
    return tracker


def attach_grad_hooks_generic(model: nn.Module, max_modules: Optional[int] = None) -> GradientTracker:
    tracker = GradientTracker()
    for module, name in _named_leaf_modules(model, max_modules):
        tracker.attach(module, name)
    return tracker


def make_drift_detector(bucket_names: List[str], decay: float = 0.98, k: float = 6.0, min_samples: int = 20) -> DriftDetector:
    return DriftDetector(bucket_names=bucket_names, decay=decay, k=k, min_samples=min_samples)


def grad_norm_buckets(model: nn.Module) -> Dict[str, float]:
    buckets: Dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        top = name.split(".")[0]
        g = p.grad.detach().float().norm().item()
        buckets[top] = buckets.get(top, 0.0) + g * g
    return {k: v ** 0.5 for k, v in buckets.items()}


def check_nan_inf(tensors: Dict[str, torch.Tensor]) -> Dict[str, bool]:
    return {name: bool(not torch.isfinite(t).all().item()) for name, t in tensors.items()}
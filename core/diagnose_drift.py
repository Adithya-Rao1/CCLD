from __future__ import annotations

import math
from typing import Dict, Optional
import torch

class RollingStats:
    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self.ema: Optional[float] = None
        self.ema_var: float = 0.0
        self.n_updates: int = 0

    def update(self, value: float) -> tuple[float, float]:
        if not math.isfinite(value):
            return (self.ema or 0.0), math.sqrt(self.ema_var)
        self.n_updates += 1
        if self.ema is None:
            self.ema = value
            self.ema_var = 0.0
        else:
            delta = value - self.ema
            self.ema = self.decay * self.ema + (1 - self.decay) * value
            self.ema_var = self.decay * self.ema_var + (1 - self.decay) * delta * delta
        return self.ema, math.sqrt(self.ema_var)


class DriftDetector:
    def __init__(self, bucket_names: list[str], decay: float = 0.98, k: float = 6.0, min_samples: int = 20):
        self.trackers: Dict[str, RollingStats] = {name: RollingStats(decay) for name in bucket_names}
        self.k = k
        self.min_samples = min_samples

    def update_and_check(self, values: Dict[str, float], explode_threshold: float) -> tuple[bool, Dict[str, str]]:
        triggered = False
        reasons: Dict[str, str] = {}
        for name, value in values.items():
            tracker = self.trackers.setdefault(name, RollingStats())
            ema, std = tracker.update(value)
            if not math.isfinite(value) or value > explode_threshold:
                triggered = True
                reasons[name] = f"value={value:.4g} exceeds absolute threshold={explode_threshold:.4g}"
            elif tracker.n_updates >= self.min_samples and std > 0 and value > ema + self.k * std:
                triggered = True
                reasons[name] = f"value={value:.4g} exceeds drift bound ema={ema:.4g}+{self.k}*std={std:.4g}"
        return triggered, reasons


class MagnitudeTracker:
    def __init__(self):
        self.stats: dict[str, float] = {}
        self.handles = []

    def _hook(self, name):
        def fn(module, inputs, output):
            if isinstance(output, dict):
                vals = [v for v in output.values() if torch.is_tensor(v)]
            elif isinstance(output, (tuple, list)):
                vals = [v for v in output if torch.is_tensor(v)]
            elif torch.is_tensor(output):
                vals = [output]
            else:
                return
            for v in vals:
                if v.numel() == 0:
                    continue
                is_finite = torch.isfinite(v).all().item()
                self.stats[name] = float("nan") if not is_finite else v.detach().abs().max().item()
        return fn

    def attach(self, module: torch.nn.Module, name: str):
        self.handles.append(module.register_forward_hook(self._hook(name)))

    def reset(self):
        self.stats.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def report(self, top_k: int = 30) -> str:
        return format_hook_report(self.stats, top_k=top_k)


class GradientTracker:
    def __init__(self):
        self.stats: dict[str, float] = {}
        self.handles = []

    def _make_grad_hook(self, name):
        def grad_hook(grad):
            if grad is None or grad.numel() == 0:
                return
            is_finite = torch.isfinite(grad).all().item()
            self.stats[name] = float("nan") if not is_finite else grad.detach().norm().item()
        return grad_hook

    def _forward_hook(self, name):
        def fn(module, inputs, output):
            if isinstance(output, dict):
                vals = [v for v in output.values() if torch.is_tensor(v)]
            elif isinstance(output, (tuple, list)):
                vals = [v for v in output if torch.is_tensor(v)]
            elif torch.is_tensor(output):
                vals = [output]
            else:
                return
            for v in vals:
                if v.requires_grad:
                    v.register_hook(self._make_grad_hook(name))
        return fn

    def attach(self, module: torch.nn.Module, name: str):
        self.handles.append(module.register_forward_hook(self._forward_hook(name)))

    def reset(self):
        self.stats.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def report(self, top_k: int = 30) -> str:
        return format_hook_report(self.stats, top_k=top_k)


def format_hook_report(stats: dict, top_k: int = 30) -> str:
    non_finite = sorted(n for n, v in stats.items() if v != v)
    finite = sorted(((n, v) for n, v in stats.items() if v == v), key=lambda kv: -abs(kv[1]))
    lines = [f"  {name:60s} max|.|=nan  [NON-FINITE]" for name in non_finite]
    lines += [f"  {name:60s} max|.|={val:.6g}" for name, val in finite[:top_k]]
    return "\n".join(lines)
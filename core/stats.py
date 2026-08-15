from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class SeedAggregate:
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1)) if self.n > 1 else 0.0

    def confidence_interval(self, confidence: float = 0.95) -> tuple:
        if self.n < 2:
            return (self.mean, self.mean)
        sem = self.std / np.sqrt(self.n)
        h = sem * scipy_stats.t.ppf((1 + confidence) / 2.0, self.n - 1)
        return (self.mean - h, self.mean + h)

    def summary(self, confidence: float = 0.95) -> Dict[str, float]:
        lo, hi = self.confidence_interval(confidence)
        return {"n": self.n, "mean": self.mean, "std": self.std, "ci_lo": lo, "ci_hi": hi}


def paired_significance(
    values_a: List[float], values_b: List[float], alternative: str = "two-sided"
) -> Dict[str, float]:
    if len(values_a) != len(values_b):
        raise ValueError("paired_significance requires equal-length seed-matched lists")
    if len(values_a) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": len(values_a)}
    diffs = np.array(values_a) - np.array(values_b)
    if np.allclose(diffs, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "n": len(values_a)}
    stat, p = scipy_stats.wilcoxon(values_a, values_b, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p), "n": len(values_a)}


def aggregate_over_seeds(per_seed_results: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    metric_names = set()
    for seed_result in per_seed_results.values():
        metric_names.update(seed_result.keys())

    aggregates: Dict[str, SeedAggregate] = {name: SeedAggregate() for name in metric_names}
    for seed_result in per_seed_results.values():
        for name in metric_names:
            if name in seed_result:
                aggregates[name].add(seed_result[name])

    return {name: agg.summary() for name, agg in aggregates.items()}


def compare_configs(
    baseline_per_seed: Dict[int, Dict[str, float]],
    treatment_per_seed: Dict[int, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    seeds = sorted(set(baseline_per_seed.keys()) & set(treatment_per_seed.keys()))
    if metric_names is None:
        metric_names = sorted(set().union(*[set(v.keys()) for v in baseline_per_seed.values()]))

    out: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        a = [baseline_per_seed[s][metric] for s in seeds if metric in baseline_per_seed[s]]
        b = [treatment_per_seed[s][metric] for s in seeds if metric in treatment_per_seed[s]]
        if len(a) != len(b) or len(a) < 2:
            continue
        sig = paired_significance(a, b)
        out[metric] = {
            "baseline_mean": float(np.mean(a)),
            "treatment_mean": float(np.mean(b)),
            "delta": float(np.mean(b) - np.mean(a)),
            **sig,
        }
    return out

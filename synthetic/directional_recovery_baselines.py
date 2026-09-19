from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

import torch

from synthetic.directional_recovery_experiment import cross_block_asymmetry_metrics
from synthetic.run_experiment import evaluate_sampling_quality, sample_ddpm, sample_sdm, train_ddpm, train_sdm

DEFAULT_HIDDEN_DIM = 64
DEFAULT_N_LAYERS = 3
DEFAULT_TIME_EMBED_DIM = 16
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_GRAD_CLIP_NORM = 1.0


def _baseline_args(N: int, n_diff_steps: int, n_train_iters: int, n_samples: int) -> SimpleNamespace:
    return SimpleNamespace(
        N=N, n_diff_steps=n_diff_steps, n_train_iters=n_train_iters, n_samples=n_samples,
        hidden_dim=DEFAULT_HIDDEN_DIM, n_layers=DEFAULT_N_LAYERS, time_embed_dim=DEFAULT_TIME_EMBED_DIM,
        batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR, grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
        siloed_score_net=False,
    )


def train_one_seed_ddpm(N: int, seed: int, gt, n_diff_steps: int, n_train_iters: int, n_samples: int,
                         device, return_samples: bool = False) -> Dict:
    torch.manual_seed(seed)
    args = _baseline_args(N, n_diff_steps, n_train_iters, n_samples)
    score_net, ac, betas, alphas = train_ddpm(args, gt, device)
    generated = sample_ddpm(args, score_net, ac, betas, alphas, device)
    metrics = evaluate_sampling_quality(generated, gt)
    metrics.update(cross_block_asymmetry_metrics(generated, gt))
    if return_samples:
        return metrics, generated
    return metrics


def train_one_seed_sdm(N: int, seed: int, gt, n_diff_steps: int, n_train_iters: int, n_samples: int,
                        device, return_samples: bool = False) -> Dict:
    torch.manual_seed(seed)
    args = _baseline_args(N, n_diff_steps, n_train_iters, n_samples)
    score_net = train_sdm(args, gt, device)
    generated = sample_sdm(args, score_net, device)
    metrics = evaluate_sampling_quality(generated, gt)
    metrics.update(cross_block_asymmetry_metrics(generated, gt))
    if return_samples:
        return metrics, generated
    return metrics

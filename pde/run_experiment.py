from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from core.baselines import (
    ddpm_forward_n, ddpm_reverse_step_n, make_ddpm_schedule, vp_alpha_bar, vp_beta_t, vp_sde_forward_marginal_n,
    vp_sde_reverse_step_n,
)
from core.coupling import build_coupling_matrix
from core.diagnostics_bridge import attach_grad_hooks_generic, grad_norm_buckets, make_drift_detector
from core.reporting import plot_bar_comparison, render_experiment_report, write_csv, write_json
from core.sde import build_g_matrix_n
from core.stats import aggregate_over_seeds
from synthetic.anderson_sde import anderson_em_step_coupled_gamma, anderson_reverse_step_coupled_gamma
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas, calibrate_sigma_fdt, calibrate_sigma_fdt_coupled
from synthetic.exact_dsm import elapsed_time_at_step, precompute_transition_params, sample_and_analytic_score_target
from pde.dataset import ALL_PROBLEMS, MultiPhysicsFieldDataset, collate_fn, _get_or_create_target_norm_stats
from pde.pde_residuals import e_flow_residual, pde_residual_metric, te_heat_normalize_mater, te_heat_residual, va_residual


def _normalize_targets(targets, target_mean, target_std):
    return [(t - mean) / std for t, mean, std in zip(targets, target_mean, target_std)]

def _denormalize_preds(preds, target_mean, target_std):
    return [p * std + mean for p, mean, std in zip(preds, target_mean, target_std)]


def _model_input_conditioning(args, batch, conditioning):
    if args.problem == "TE_heat":
        return te_heat_normalize_mater(conditioning[:, 0], batch["elliptic_params"]).unsqueeze(1)
    return conditioning
from pde.model import (
    FlatScoreNetwork, MultiPhysicsScoreNetwork, PhysicsModel, SpatialFieldModel, make_flat_score_fn, make_score_fn,
)
from pde.fno_score_net import FNOScoreNetwork, FlatFNOScoreNetwork, make_spatial_score_fn, make_flat_fno_score_fn
from pde.unet_model_score_net import (
    UNetModelScoreNetwork, FlatUNetModelScoreNetwork, make_unet_model_score_fn, make_flat_unet_model_score_fn,
)

METHOD_CONFIGS = {
    "csho": {"coupling_mode": "mean_field", "diffusion_mode": "shared"},
    "csho_independent": {"coupling_mode": "independent", "diffusion_mode": "shared"},
    "csho_shared_g": {"coupling_mode": "mean_field", "diffusion_mode": "shared"},
    "csho_independent_g": {"coupling_mode": "mean_field", "diffusion_mode": "independent"},
    "csho_pairwise": {"coupling_mode": "pairwise", "diffusion_mode": "shared"},
}
BASELINE_METHODS = {"ddpm", "sdm"}
ALL_METHODS = sorted(set(METHOD_CONFIGS) | BASELINE_METHODS)


def parse_float_list(s, n: int) -> List[float]:
    vals = [float(v) for v in s] if isinstance(s, (list, tuple)) else [float(v) for v in str(s).split(",") if v.strip()]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"Expected 1 or {n} float values, got {len(vals)}")
    return vals


def load_config_defaults(config_path: str) -> Dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment 2 (physics): does CSHO cross-field coupling help multi-physics coupled-PDE prediction?")
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--problem", default="TE_heat", choices=ALL_PROBLEMS)
    p.add_argument("--split", default="training")
    p.add_argument("--val-split", default="testing")
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--task-subset", default=None, help="comma-separated explicit task-name override")
    p.add_argument("--method", default="csho", choices=ALL_METHODS)
    p.add_argument("--damping-regime", default="critically_damped",
                    choices=["underdamped", "critically_damped", "overdamped"])
    p.add_argument("--target-zeta", type=float, default=None)
    p.add_argument("--alpha", default="1.0")
    p.add_argument("--beta", default="0.5")
    p.add_argument("--k-reference", type=float, default=1.0,)
    p.add_argument("--constant-k", action="store_true")
    p.add_argument("--n-diff-steps", type=int, default=32)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--backbone-channels", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--n-downsample", type=int, default=3)
    p.add_argument("--score-blocks", type=int, default=3)
    p.add_argument("--score-heads", type=int, default=4)
    p.add_argument("--score-arch", default="attention", choices=["attention", "fno", "unet_model"],
                    help="'attention' = existing pooled-latent MultiPhysicsScoreNetwork; "
                         "'fno'/'unet_model' = native-pixel-diffusion (X+V-conditioned) score networks")
    p.add_argument("--fno-modes", default="12,12")
    p.add_argument("--fno-hidden-channels", type=int, default=128)
    p.add_argument("--fno-init-channels", type=int, default=32,
                    help="SpatialFieldModel's init-head hidden channels -- shared by fno and unet_model, "
                         "both use the same native-pixel-diffusion state representation")
    p.add_argument("--unet-model-channels", type=int, default=32)
    p.add_argument("--unet-channel-mult", default="1,2,2")
    p.add_argument("--unet-num-blocks", type=int, default=2)
    p.add_argument("--unet-attn-resolutions", default="16")
    p.add_argument("--lambda-dsm", type=float, default=1.0)
    p.add_argument("--explode-threshold", type=float, default=1e3)
    p.add_argument("--max-hook-modules", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/experiment_2_physics")
    p.add_argument("--debug-rollout", action="store_true",
                    help="print trained score_net.output_gain (if present) and the CSHO reverse "
                         "rollout's per-step state norm for the first eval batch -- diagnostic "
                         "for isolating where a reverse-SDE divergence originates")
    p.add_argument("--encoder-only", action="store_true",
                    help="train only the encoder (init_loss/readout_loss) per seed in --seeds, save a "
                         "checkpoint per seed to --save-encoder-dir, and exit -- no score net, no eval. "
                         "Used to produce one shared, frozen encoder per seed that csho/ddpm/sdm all "
                         "train their own score net on top of via --frozen-encoder-dir.")
    p.add_argument("--save-encoder-dir", default=None,
                    help="output directory for --encoder-only checkpoints (one seed{N}.pt per seed)")
    p.add_argument("--frozen-encoder-dir", default=None,
                    help="directory of per-seed encoder checkpoints (from --encoder-only) to load and "
                         "freeze instead of training the encoder jointly with the score net")
    p.add_argument("--csho-tau", type=float, default=None,
                    help="if set, CSHO's elapsed dynamical time schedule is held at this constant value "
                         "(time_scale_fn) instead of the default (T-t)/(t+T) schedule -- see pde/README.md "
                         "on the corruption-severity confound between CSHO and DDPM/SDM this addresses")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    parser = build_arg_parser()
    if pre_args.config:
        parser.set_defaults(**load_config_defaults(pre_args.config))
    args = parser.parse_args(argv)

    if args.data_root is None:
        raise ValueError("--data-root is required (directly, or via data.data_root in --config)")

    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    args.task_subset = [t.strip() for t in args.task_subset.split(",")] if args.task_subset else None
    if args.dt is None:
        args.dt = 1.0 / args.n_diff_steps
    return args


def _csho_time_scale_fn(args):
    if getattr(args, "csho_tau", None) is not None:
        tau = args.csho_tau
        return lambda t, T: torch.tensor(tau, dtype=torch.float32)
    return None


def build_method_state(args, N: int, device, sigma_ab=None):
    if args.method in METHOD_CONFIGS:
        cfg = METHOD_CONFIGS[args.method]
        coupling = build_coupling_matrix(N, mode=cfg["coupling_mode"], device=device)
        a, b = sigma_ab
        if args.method == "csho":
            g_per_task = None
            g_matrix = build_g_matrix_n(
                torch.tensor(a, device=device), N, diffusion_mode="shared",
                coupling_matrix=b * coupling,
            )
        else:
            g_per_task = None
            if cfg["diffusion_mode"] == "independent":
                g_per_task = [a * (0.8 + 0.4 * i / max(N - 1, 1)) for i in range(N)]
            g_matrix = build_g_matrix_n(
                torch.tensor(a, device=device), N, diffusion_mode=cfg["diffusion_mode"],
                g_per_task=g_per_task, coupling_matrix=coupling if cfg["diffusion_mode"] == "independent" else None,
            )
        return {"is_csho": True, "cfg": cfg, "coupling": coupling, "g_per_task": g_per_task,
                "ddpm_sched": None, "sigma": a, "g_matrix": g_matrix}
    if args.method == "ddpm":
        return {"is_csho": False, "cfg": None, "coupling": None, "g_per_task": None,
                "ddpm_sched": make_ddpm_schedule(args.n_diff_steps, device=device)}
    if args.method == "sdm":
        return {"is_csho": False, "cfg": None, "coupling": None, "g_per_task": None, "ddpm_sched": None}
    raise ValueError(f"Unknown method {args.method!r}")


def relative_l2_error_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.linalg.norm((pred - target).flatten(1), dim=-1)
    den = torch.linalg.norm(target.flatten(1), dim=-1).clamp_min(eps)
    return num / den


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    return relative_l2_error_per_sample(pred, target, eps).mean().item()


def spectral_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_fft = torch.fft.fft2(pred.float()).abs()
    target_fft = torch.fft.fft2(target.float()).abs()
    num = torch.linalg.norm((pred_fft - target_fft).flatten(1), dim=-1)
    den = torch.linalg.norm(target_fft.flatten(1), dim=-1).clamp_min(eps)
    return (num / den).mean().item()


def make_dataset(args, split: str) -> MultiPhysicsFieldDataset:
    return MultiPhysicsFieldDataset(
        args.data_root, args.problem, split=split, n_tasks=args.n_tasks, task_subset=args.task_subset,
        image_size=(args.image_size, args.image_size), max_samples=args.max_samples,
    )


def _encode_state(args, model, task_names, model_conditioning, is_spatial: bool):
    if is_spatial:
        X0_dict, K_self, K_global = model.encode(model_conditioning, k_reference=args.k_reference)
        X = [[X0_dict[name]] for name in task_names]
        return X, K_self, K_global, model_conditioning, X0_dict
    X, K_self, K_global, feat, y0 = model.encode(model_conditioning, k_reference=args.k_reference)
    return X, K_self, K_global, feat, y0


def _build_score_fns(args, score_net, conditioning):
    if args.score_arch == "fno":
        return make_spatial_score_fn(score_net, conditioning), make_flat_fno_score_fn(score_net, conditioning)
    if args.score_arch == "unet_model":
        return make_unet_model_score_fn(score_net, conditioning), make_flat_unet_model_score_fn(score_net, conditioning)
    return make_score_fn(score_net, conditioning), make_flat_score_fn(score_net, conditioning)


def _calibrate_csho_sigma(model, train_loader, N: int, gamma_self: float, gamma_couple: float,
                           args: argparse.Namespace, device, task_names, is_spatial: bool) -> Tuple[float, float]:
    batch = next(iter(train_loader))
    conditioning = batch["conditioning"].to(device)
    if "elliptic_params" in batch:
        batch["elliptic_params"] = batch["elliptic_params"].to(device)
    with torch.no_grad():
        model_conditioning = _model_input_conditioning(args, batch, conditioning)
        X, _, _, _, _ = _encode_state(args, model, task_names, model_conditioning, is_spatial)
    X_flat = torch.cat([X[i][0].reshape(-1, 1) for i in range(N)], dim=-1).detach().cpu()
    cov_data = torch.cov(X_flat.T)
    if args.method == "csho":
        a, b = calibrate_sigma_fdt_coupled(gamma_self, gamma_couple, N)
    else:
        a, b = calibrate_sigma_fdt(gamma_self), 0.0
    if args.debug_rollout:
        std = cov_data.diagonal().clamp_min(1e-12).sqrt()
        corr = cov_data / (std[:, None] * std[None, :])
        off = corr - torch.diag(torch.diag(corr))
        rho_true = off.sum().item() / (N * (N - 1))
        print(f"[debug-rollout] cov_data diag (per-task variance) = {dict(zip(task_names, std.pow(2).tolist()))}")
        print(f"[debug-rollout] cov_data mean pairwise corr (rho_true) = {rho_true:.6f}")
        print(f"[debug-rollout] calibrated sigma (a, b) = ({a}, {b})")
    return a, b


def train_encoder_only(args: argparse.Namespace, seed: int) -> None:
    torch.manual_seed(seed)
    device = torch.device(args.device)

    train_ds = make_dataset(args, args.split)
    task_names = train_ds.task_names
    N = len(task_names)

    target_norm_stats = _get_or_create_target_norm_stats(train_ds, args.problem, task_names)
    target_mean = [target_norm_stats[name]["mean"] for name in task_names]
    target_std = [target_norm_stats[name]["std"] for name in task_names]

    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps = args.n_epochs * steps_per_epoch
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=total_steps * args.batch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
                               persistent_workers=args.num_workers > 0)

    is_spatial = args.score_arch in ("fno", "unet_model")
    cond_in_ch = train_ds[0]["conditioning"].shape[0]
    if is_spatial:
        model = SpatialFieldModel(task_names, cond_in_ch=cond_in_ch, out_hw=(args.image_size, args.image_size),
                                   backbone_ch=args.backbone_channels, base_ch=args.base_channels,
                                   n_downsample=args.n_downsample, init_ch=args.fno_init_channels).to(device)
    else:
        model = PhysicsModel(task_names, cond_in_ch=cond_in_ch, out_hw=(args.image_size, args.image_size),
                              latent_dim=args.latent_dim, backbone_ch=args.backbone_channels,
                              base_ch=args.base_channels, n_downsample=args.n_downsample).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    pbar = tqdm(total=total_steps, desc=f"[{args.problem}/encoder-only/{args.score_arch}] seed={seed}")
    for step, batch in enumerate(train_loader):
        conditioning = batch["conditioning"].to(device)
        if "elliptic_params" in batch:
            batch["elliptic_params"] = batch["elliptic_params"].to(device)
        targets = [t.to(device) for t in batch["tasks"]]
        targets_norm = _normalize_targets(targets, target_mean, target_std)
        model_conditioning = _model_input_conditioning(args, batch, conditioning)
        X, K_self, K_global, feat, y0 = _encode_state(args, model, task_names, model_conditioning, is_spatial)

        if is_spatial:
            loss = torch.zeros((), device=device)
            for name, target_n in zip(task_names, targets_norm):
                loss = loss + F.l1_loss(y0[name], target_n)
        else:
            loss = torch.zeros((), device=device)
            for name, y0_t, target_n in zip(task_names, [y0[n] for n in task_names], targets_norm):
                loss = loss + F.l1_loss(y0_t, target_n)
            for name, x, target_n in zip(task_names, X, targets_norm):
                pred = model.decode(name, x[0])
                loss = loss + F.l1_loss(pred, target_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)
    pbar.close()

    os.makedirs(args.save_encoder_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_encoder_dir, f"seed{seed}.pt")
    torch.save({"model_state_dict": model.state_dict(), "is_spatial": is_spatial,
                "task_names": task_names, "cond_in_ch": cond_in_ch}, ckpt_path)
    print(f"[encoder-only] seed={seed}: saved to {ckpt_path}")


def train_one_seed(args: argparse.Namespace, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    device = torch.device(args.device)

    train_ds = make_dataset(args, args.split)
    try:
        val_ds = make_dataset(args, args.val_split)
    except (FileNotFoundError, ValueError):
        val_ds = train_ds

    task_names = train_ds.task_names
    N = len(task_names)
    args.alpha_list = parse_float_list(args.alpha, N)
    args.beta_list = parse_float_list(args.beta, N)

    target_norm_stats = _get_or_create_target_norm_stats(train_ds, args.problem, task_names)
    target_mean = [target_norm_stats[name]["mean"] for name in task_names]
    target_std = [target_norm_stats[name]["std"] for name in task_names]

    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps = args.n_epochs * steps_per_epoch
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=total_steps * args.batch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
                               persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn,
                             persistent_workers=args.num_workers > 0)

    is_spatial = args.score_arch in ("fno", "unet_model")
    fno_modes = tuple(int(v) for v in args.fno_modes.split(","))
    unet_channel_mult = tuple(int(v) for v in args.unet_channel_mult.split(","))
    unet_attn_res = tuple(int(v) for v in args.unet_attn_resolutions.split(",")) if args.unet_attn_resolutions else ()
    cond_in_ch = train_ds[0]["conditioning"].shape[0]
    if is_spatial:
        model = SpatialFieldModel(task_names, cond_in_ch=cond_in_ch, out_hw=(args.image_size, args.image_size),
                                   backbone_ch=args.backbone_channels, base_ch=args.base_channels,
                                   n_downsample=args.n_downsample, init_ch=args.fno_init_channels).to(device)
    else:
        model = PhysicsModel(task_names, cond_in_ch=cond_in_ch, out_hw=(args.image_size, args.image_size),
                              latent_dim=args.latent_dim, backbone_ch=args.backbone_channels,
                              base_ch=args.base_channels, n_downsample=args.n_downsample).to(device)

    frozen_encoder = args.frozen_encoder_dir is not None
    if frozen_encoder:
        ckpt_path = os.path.join(args.frozen_encoder_dir, f"seed{seed}.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    is_csho = args.method in METHOD_CONFIGS
    if args.score_arch == "fno":
        if is_csho:
            score_net = FNOScoreNetwork(N, cond_in_ch, args.n_diff_steps, n_modes=fno_modes,
                                         hidden_channels=args.fno_hidden_channels).to(device)
        else:
            score_net = FlatFNOScoreNetwork(N, cond_in_ch, args.n_diff_steps, n_modes=fno_modes,
                                             hidden_channels=args.fno_hidden_channels).to(device)
    elif args.score_arch == "unet_model":
        if is_csho:
            score_net = UNetModelScoreNetwork(N, cond_in_ch, args.n_diff_steps, img_resolution=args.image_size,
                                              model_channels=args.unet_model_channels,
                                              channel_mult=unet_channel_mult, num_blocks=args.unet_num_blocks,
                                              attn_resolutions=unet_attn_res).to(device)
        else:
            score_net = FlatUNetModelScoreNetwork(N, cond_in_ch, args.n_diff_steps, img_resolution=args.image_size,
                                                  model_channels=args.unet_model_channels,
                                                  channel_mult=unet_channel_mult, num_blocks=args.unet_num_blocks,
                                                  attn_resolutions=unet_attn_res).to(device)
    elif is_csho:
        score_net = MultiPhysicsScoreNetwork(N, args.latent_dim, model.backbone.out_ch,
                                              n_blocks=args.score_blocks, n_heads=args.score_heads).to(device)
    else:
        score_net = FlatScoreNetwork(N, args.latent_dim, model.backbone.out_ch,
                                      n_blocks=args.score_blocks, n_heads=args.score_heads).to(device)

    if frozen_encoder:
        optimizer = torch.optim.Adam(score_net.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(score_net.parameters()), lr=args.lr)

    gamma_self, gamma_couple = calibrate_coupled_gammas(
        args.alpha_list[0], args.beta_list[0], args.k_reference, args.k_reference, N,
        regime=args.damping_regime, target_zeta=args.target_zeta,
    )

    if is_csho:
        sigma_ab = _calibrate_csho_sigma(model, train_loader, N, gamma_self, gamma_couple, args, device,
                                          task_names, is_spatial)
        state = build_method_state(args, N, device, sigma_ab=sigma_ab)
        coupling = state["coupling"]
        g_matrix = state["g_matrix"]
        params = precompute_transition_params(
            N, gamma_self, gamma_couple, args.alpha_list, args.beta_list, args.k_reference, coupling,
            args.n_diff_steps, args.dt, lambda t, T: g_matrix, args.constant_k, _csho_time_scale_fn(args),
        )
    else:
        state = build_method_state(args, N, device)

    drift_detector = make_drift_detector(bucket_names=[])
    attach_grad_hooks_generic(model, max_modules=args.max_hook_modules)
    attach_grad_hooks_generic(score_net, max_modules=args.max_hook_modules)

    nan_events, explosion_events, n_steps = 0, 0, 0

    if not frozen_encoder:
        model.train()
    score_net.train()
    pbar = tqdm(total=total_steps, desc=f"[{args.problem}/{args.method}/{args.score_arch}] seed={seed}")
    for step, batch in enumerate(train_loader):
        conditioning = batch["conditioning"].to(device)
        if "elliptic_params" in batch:
            batch["elliptic_params"] = batch["elliptic_params"].to(device)
        targets = [t.to(device) for t in batch["tasks"]]
        targets_norm = _normalize_targets(targets, target_mean, target_std)
        model_conditioning = _model_input_conditioning(args, batch, conditioning)
        X, K_self, K_global, feat, y0 = _encode_state(args, model, task_names, model_conditioning, is_spatial)

        init_loss = torch.zeros((), device=device)
        for name, y0_t, target_n in zip(task_names, [y0[n] for n in task_names], targets_norm):
            init_loss = init_loss + F.l1_loss(y0_t, target_n)

        t_idx = torch.randint(1, args.n_diff_steps + 1, (1,)).item()

        if is_csho:
            diffuse_state = targets_norm if is_spatial else [x[0] for x in X]
            orig_shape = diffuse_state[0].shape
            X_flat = torch.cat([diffuse_state[i].reshape(-1, 1) for i in range(N)], dim=-1)
            Z0 = torch.cat([X_flat, torch.zeros_like(X_flat)], dim=-1)
            Phi_t, _ = params[t_idx - 1]
            q = elapsed_time_at_step(t_idx, args.n_diff_steps, args.dt, _csho_time_scale_fn(args))
            Zt, score_target = sample_and_analytic_score_target(Z0, Phi_t, N, gamma_self, gamma_couple, q)
            X_t = [[Zt[:, i:i + 1].reshape(*orig_shape)] for i in range(N)]
            V_t = [[Zt[:, N + i:N + i + 1].reshape(*orig_shape)] for i in range(N)]

            score_fn, _ = _build_score_fns(args, score_net, feat)
            score_pred = score_fn(X_t, V_t, t_idx)
            dsm_loss = torch.zeros((), device=device)
            for i in range(N):
                target_i = score_target[:, i:i + 1].reshape(*orig_shape)
                dsm_loss = dsm_loss + F.mse_loss(score_pred[i][0], target_i)

            if is_spatial:
                loss = init_loss + args.lambda_dsm * dsm_loss
            else:
                readout_loss = torch.zeros((), device=device)
                for name, x, target_n in zip(task_names, X, targets_norm):
                    pred = model.decode(name, x[0])
                    readout_loss = readout_loss + F.l1_loss(pred, target_n)
                loss = init_loss + readout_loss + args.lambda_dsm * dsm_loss
        else:
            X0_flat = targets_norm if is_spatial else [x[0] for x in X]
            _, flat_fn = _build_score_fns(args, score_net, feat)
            if args.method == "ddpm":
                ac, _, _ = state["ddpm_sched"]
                Xt, noise = ddpm_forward_n(X0_flat, torch.tensor(t_idx, device=device), ac)
                eps_pred = flat_fn(Xt, t_idx)
            else:
                t_cont = t_idx / args.n_diff_steps
                Xt, noise = vp_sde_forward_marginal_n(X0_flat, torch.tensor(t_cont, device=device))
                eps_pred = flat_fn(Xt, t_cont)
            diffusion_loss = sum(F.mse_loss(p, n) for p, n in zip(eps_pred, noise))

            if is_spatial:
                loss = init_loss + args.lambda_dsm * diffusion_loss
            else:
                readout_loss = torch.zeros((), device=device)
                for name, x, target_n in zip(task_names, X, targets_norm):
                    pred = model.decode(name, x[0])
                    readout_loss = readout_loss + F.l1_loss(pred, target_n)
                loss = init_loss + readout_loss + args.lambda_dsm * diffusion_loss

        optimizer.zero_grad()
        loss.backward()
        buckets = {}
        for prefix, mod in (("model", model), ("score_net", score_net)):
            for k, v in grad_norm_buckets(mod).items():
                buckets[f"{prefix}.{k}"] = v
        triggered, _ = drift_detector.update_and_check(buckets, args.explode_threshold)
        if triggered:
            explosion_events += 1
        if torch.isfinite(loss):
            optimizer.step()
        else:
            nan_events += 1
        n_steps += 1
        pbar.set_postfix(epoch=f"{step // steps_per_epoch + 1}/{args.n_epochs}", loss=f"{loss.item():.4f}",
                          nan=nan_events, explosion=explosion_events)
        pbar.update(1)
    pbar.close()

    if args.debug_rollout and hasattr(score_net, "output_gain"):
        gain = score_net.output_gain.detach().cpu().tolist()
        print(f"[debug-rollout] trained output_gain per task ({list(zip(task_names, gain))})")

    metrics, per_sample_rel_l2, sample_maps = evaluate(
        args, model, score_net, val_loader, device, task_names, state, gamma_self, gamma_couple,
        is_spatial, target_mean, target_std,
    )
    metrics["nan_events"] = float(nan_events)
    metrics["explosion_events"] = float(explosion_events)
    metrics["n_train_steps"] = float(n_steps)

    os.makedirs(args.out_dir, exist_ok=True)
    write_json(per_sample_rel_l2, os.path.join(args.out_dir, f"{args.method}_per_sample_rel_l2_seed{seed}.json"))
    np.savez(os.path.join(args.out_dir, f"{args.method}_sample_maps_seed{seed}.npz"), **sample_maps)

    return metrics


@torch.no_grad()
def evaluate(args, model, score_net, val_loader, device, task_names, state, gamma_self, gamma_couple,
             is_spatial: bool = False, target_mean=None, target_std=None,
             ) -> Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, np.ndarray]]:
    model.eval()
    score_net.eval()
    is_csho = state["is_csho"]
    N = len(task_names)
    is_elder = args.problem == "Elder"
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    rollout_errs: List[List[float]] = []
    per_sample_rel_l2: Dict[str, List[float]] = {}
    sample_maps: Dict[str, np.ndarray] = {}
    first_batch = True

    for batch in val_loader:
        conditioning = batch["conditioning"].to(device)
        if "elliptic_params" in batch:
            batch["elliptic_params"] = batch["elliptic_params"].to(device)
        targets = [t.to(device) for t in batch["tasks"]]
        model_conditioning = _model_input_conditioning(args, batch, conditioning)
        X, K_self, K_global, feat, _ = _encode_state(args, model, task_names, model_conditioning, is_spatial)

        if is_csho:
            coupling, G = state["coupling"], state["g_matrix"]
            X_init = [[torch.randn_like(x[0])] for x in X] if is_spatial else X
            V = [[torch.zeros_like(x[0])] for x in X_init]
            X_cur, V_cur = X_init, V
            score_fn, _ = _build_score_fns(args, score_net, feat)
            debug_this_batch = args.debug_rollout and first_batch
            for t_idx in reversed(range(1, args.n_diff_steps + 1)):
                score_outputs = score_fn(X_cur, V_cur, t_idx)
                if debug_this_batch:
                    x_norms = [round(X_cur[i][0].flatten(1).norm(dim=1).mean().item(), 4) for i in range(N)]
                    score_norms = [round(score_outputs[i][0].flatten(1).norm(dim=1).mean().item(), 4) for i in range(N)]
                    print(f"[debug-rollout] t_idx={t_idx:3d}  X_norm={dict(zip(task_names, x_norms))}  "
                          f"score_norm={dict(zip(task_names, score_norms))}")
                X_cur, V_cur = anderson_reverse_step_coupled_gamma(
                    X_cur, V_cur, K_self, K_global, score_outputs, t_idx, args.n_diff_steps,
                    args.alpha_list, args.beta_list, gamma_self, gamma_couple,
                    coupling, args.constant_k, args.dt, G, time_scale_fn=_csho_time_scale_fn(args),
                )
            final_latents = [X_cur[i][0] for i in range(N)]
        elif args.method == "ddpm":
            ac, betas_s, alphas_s = state["ddpm_sched"]
            T_eff = max(args.n_diff_steps - 1, 1)
            if is_spatial:
                X_flat = [torch.randn_like(x[0]) for x in X]
            else:
                X0_flat = [x[0] for x in X]
                X_flat, _ = ddpm_forward_n(X0_flat, torch.tensor(T_eff, device=device), ac)
            _, flat_fn = _build_score_fns(args, score_net, feat)
            for t_idx in reversed(range(1, T_eff)):
                eps_pred = flat_fn(X_flat, t_idx)
                X_flat = ddpm_reverse_step_n(X_flat, eps_pred, t_idx, betas_s, alphas_s, ac)
            eps_pred = flat_fn(X_flat, 1)
            ac1 = ac[1]
            final_latents = [(x - torch.sqrt(1 - ac1) * e) / torch.sqrt(ac1) for x, e in zip(X_flat, eps_pred)]
        else:
            if is_spatial:
                X_flat = [torch.randn_like(x[0]) for x in X]
            else:
                X0_flat = [x[0] for x in X]
                X_flat, _ = vp_sde_forward_marginal_n(X0_flat, torch.tensor(1.0, device=device))
            _, flat_fn = _build_score_fns(args, score_net, feat)
            dt_step = 1.0 / args.n_diff_steps
            for t_idx in reversed(range(1, args.n_diff_steps + 1)):
                t_cont = t_idx / args.n_diff_steps
                beta_t = vp_beta_t(torch.tensor(t_cont, device=device), 1.0)
                ac_t = vp_alpha_bar(torch.tensor(t_cont, device=device))
                eps_pred = flat_fn(X_flat, t_cont)
                score = [-e / torch.sqrt(1.0 - ac_t).clamp_min(1e-8) for e in eps_pred]
                X_flat = vp_sde_reverse_step_n(X_flat, score, beta_t, dt_step)
            final_latents = X_flat

        if is_spatial:
            preds_norm = final_latents
        else:
            preds_norm = [model.decode(name, final_latents[i]) for i, name in enumerate(task_names)]
        preds = _denormalize_preds(preds_norm, target_mean, target_std)

        for name, pred, target in zip(task_names, preds, targets):
            rel_per_sample = relative_l2_error_per_sample(pred, target)
            rel = rel_per_sample.mean().item()
            spec = spectral_l2_error(pred, target)
            for k, v in ((f"{name}_rel_l2", rel), (f"{name}_spectral_l2", spec)):
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
            per_sample_rel_l2.setdefault(name, []).extend(rel_per_sample.detach().cpu().tolist())

        if first_batch:
            n_show = min(4, targets[0].shape[0])
            for name, pred, target in zip(task_names, preds, targets):
                sample_maps[f"{name}_pred"] = pred[:n_show].detach().cpu().numpy()
                sample_maps[f"{name}_target"] = target[:n_show].detach().cpu().numpy()

        if args.problem in ("TE_heat", "E_flow", "VA"):
            pred_by_name = dict(zip(task_names, [p.squeeze(1) for p in preds]))
            if args.problem == "TE_heat":
                residuals = te_heat_residual({
                    "mater": conditioning[:, 0], "T": pred_by_name["T"],
                    "Ez_re": pred_by_name["Re{Ez}"], "Ez_im": pred_by_name["Im{Ez}"],
                }, batch["elliptic_params"])
            elif args.problem == "E_flow":
                residuals = e_flow_residual({
                    "kappa": conditioning[:, 0],
                    "ec_V": pred_by_name["ec_V"], "u_flow": pred_by_name["u_flow"], "v_flow": pred_by_name["v_flow"],
                })
            else:
                residuals = va_residual({
                    "rho_water": conditioning[:, 0],
                    "p_t_re": pred_by_name["Re{p_t}"], "p_t_im": pred_by_name["Im{p_t}"],
                    "Sxx_re": pred_by_name["Re{Sxx}"], "Sxx_im": pred_by_name["Im{Sxx}"],
                    "Sxy_re": pred_by_name["Re{Sxy}"], "Sxy_im": pred_by_name["Im{Sxy}"],
                    "Syy_re": pred_by_name["Re{Syy}"], "Syy_im": pred_by_name["Im{Syy}"],
                    "x_u_re": pred_by_name["Re{x_u}"], "x_u_im": pred_by_name["Im{x_u}"],
                    "x_v_re": pred_by_name["Re{x_v}"], "x_v_im": pred_by_name["Im{x_v}"],
                })
            for eq_name, residual in residuals.items():
                k = f"{eq_name}_pde_residual"
                sums[k] = sums.get(k, 0.0) + pde_residual_metric(residual)
                counts[k] = counts.get(k, 0) + 1

        if is_elder:
            n_t, n_f = 10, 3
            per_step = []
            for step in range(n_t):
                errs = [relative_l2_error(preds[step * n_f + f], targets[step * n_f + f]) for f in range(n_f)]
                per_step.append(sum(errs) / n_f)
            rollout_errs.append(per_step)

        first_batch = False

    metrics = {k: sums[k] / counts[k] for k in sums}
    if rollout_errs:
        for step in range(len(rollout_errs[0])):
            metrics[f"elder_rollout_step{step + 1}_rel_l2"] = sum(r[step] for r in rollout_errs) / len(rollout_errs)
    return metrics, per_sample_rel_l2, sample_maps


def main():
    args = parse_args()

    if args.encoder_only:
        if not args.save_encoder_dir:
            raise ValueError("--encoder-only requires --save-encoder-dir")
        for seed in args.seeds:
            train_encoder_only(args, seed)
        print(f"Done. Encoder checkpoints written to {args.save_encoder_dir}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    per_seed_results: Dict[int, Dict[str, float]] = {}
    for seed in args.seeds:
        print(f"[{args.method}] seed={seed} starting")
        per_seed_results[seed] = train_one_seed(args, seed)
        print(f"[{args.method}] seed={seed} done: {per_seed_results[seed]}")

    summary = aggregate_over_seeds(per_seed_results)
    summary_rows = [{"metric": k, **v} for k, v in summary.items()]
    write_csv(summary_rows, os.path.join(args.out_dir, f"{args.method}_summary.csv"))
    write_json(
        {"args": vars(args), "per_seed": per_seed_results, "summary": summary},
        os.path.join(args.out_dir, f"{args.method}_results.json"),
    )

    labels = list(summary.keys())
    fig_path = plot_bar_comparison(
        labels, [summary[k]["mean"] for k in labels], [summary[k]["std"] for k in labels],
        title=f"{args.method} -- metric summary", ylabel="value",
        out_path=os.path.join(args.out_dir, f"{args.method}_summary.png"),
    )
    render_experiment_report(
        experiment_name=f"Experiment 2 (physics) -- {args.method}",
        summary_rows=summary_rows, significance_rows=[], figure_paths=[fig_path],
        out_path=os.path.join(args.out_dir, f"{args.method}_report.md"),
    )
    print(f"Done. Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
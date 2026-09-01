from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from core.baselines import (
    ddpm_forward_n, ddpm_reverse_step_n, make_ddpm_schedule, vp_beta_t, vp_sde_drift_n, vp_sde_reverse_step_n,
)
from core.coupling import build_coupling_matrix
from core.diagnostics_bridge import attach_grad_hooks_generic, grad_norm_buckets, make_drift_detector
from core.reporting import plot_bar_comparison, render_experiment_report, write_csv, write_json
from core.sde import build_g_matrix_n
from core.stats import aggregate_over_seeds
from synthetic.anderson_sde import anderson_em_step_coupled_gamma, anderson_reverse_step_coupled_gamma
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas
from synthetic.exact_dsm import calibrate_sigma_for_leak, precompute_transition_params, sample_and_tikhonov_score_target
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
from pde.songunet_score_net import (
    SongUNetScoreNetwork, FlatSongUNetScoreNetwork, make_songunet_score_fn, make_flat_songunet_score_fn,
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
    p.add_argument("--leak-fraction", type=float, default=0.01,
                    help="target correlation-leak fraction for CSHO's exact matrix-recursion sigma calibration "
                         "(synthetic/exact_dsm.py::calibrate_sigma_for_leak); replaces the old flat --sigma")
    p.add_argument("--lam", type=float, default=0.1, help="Tikhonov regularization strength for the DSM score target")
    p.add_argument("--k-reference", type=float, default=1.0,
                    help="fixed confinement/coupling stiffness ensuring critical damping -- NOT learned from data "
                         "(see PhysicsModel.encode's k_reference arg); this is what makes the exact closed-form "
                         "transition kernel valid, the same role K_REFERENCE plays in synthetic/")
    p.add_argument("--constant-k", action="store_true")
    p.add_argument("--n-diff-steps", type=int, default=2)
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
    p.add_argument("--score-arch", default="attention", choices=["attention", "fno", "songunet"],
                    help="'attention' = existing pooled-latent MultiPhysicsScoreNetwork; "
                         "'fno'/'songunet' = native-pixel-diffusion (X+V-conditioned) score networks")
    p.add_argument("--fno-modes", default="12,12")
    p.add_argument("--fno-hidden-channels", type=int, default=128)
    p.add_argument("--fno-init-channels", type=int, default=32,
                    help="SpatialFieldModel's init-head hidden channels -- shared by fno and songunet, "
                         "both use the same native-pixel-diffusion state representation")
    p.add_argument("--songunet-model-channels", type=int, default=32)
    p.add_argument("--songunet-channel-mult", default="1,2,2")
    p.add_argument("--songunet-num-blocks", type=int, default=2)
    p.add_argument("--songunet-attn-resolutions", default="16")
    p.add_argument("--lambda-tikhonov", type=float, default=1.0)
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


def build_method_state(args, N: int, device, sigma: float = None):
    if args.method in METHOD_CONFIGS:
        cfg = METHOD_CONFIGS[args.method]
        coupling = build_coupling_matrix(N, mode=cfg["coupling_mode"], device=device)
        g_per_task = None
        if cfg["diffusion_mode"] == "independent":
            g_per_task = [sigma * (0.8 + 0.4 * i / max(N - 1, 1)) for i in range(N)]
        return {"is_csho": True, "cfg": cfg, "coupling": coupling, "g_per_task": g_per_task, "ddpm_sched": None, "sigma": sigma}
    if args.method == "ddpm":
        return {"is_csho": False, "cfg": None, "coupling": None, "g_per_task": None,
                "ddpm_sched": make_ddpm_schedule(args.n_diff_steps, device=device)}
    if args.method == "sdm":
        return {"is_csho": False, "cfg": None, "coupling": None, "g_per_task": None, "ddpm_sched": None}
    raise ValueError(f"Unknown method {args.method!r}")


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    num = torch.linalg.norm((pred - target).flatten(1), dim=-1)
    den = torch.linalg.norm(target.flatten(1), dim=-1).clamp_min(eps)
    return (num / den).mean().item()


def spectral_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_fft = torch.fft.fft2(pred.float()).abs()
    target_fft = torch.fft.fft2(target.float()).abs()
    num = torch.linalg.norm((pred_fft - target_fft).flatten(1), dim=-1)
    den = torch.linalg.norm(target_fft.flatten(1), dim=-1).clamp_min(eps)
    return (num / den).mean().item()


def vp_forward_step_with_noise(X: List[torch.Tensor], beta_t: torch.Tensor, dt: float):
    drift = vp_sde_drift_n(X, beta_t)
    g = torch.sqrt(beta_t)
    noise = [torch.randn_like(x) for x in X]
    X_next = [x + d * dt + g * math.sqrt(dt) * n for x, d, n in zip(X, drift, noise)]
    return X_next, noise


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
    if args.score_arch == "songunet":
        return make_songunet_score_fn(score_net, conditioning), make_flat_songunet_score_fn(score_net, conditioning)
    return make_score_fn(score_net, conditioning), make_flat_score_fn(score_net, conditioning)


def _calibrate_csho_sigma(model, train_loader, N: int, gamma_self: float,
                           args: argparse.Namespace, device, task_names, is_spatial: bool) -> float:
    batch = next(iter(train_loader))
    conditioning = batch["conditioning"].to(device)
    if "elliptic_params" in batch:
        batch["elliptic_params"] = batch["elliptic_params"].to(device)
    with torch.no_grad():
        model_conditioning = _model_input_conditioning(args, batch, conditioning)
        X, _, _, _, _ = _encode_state(args, model, task_names, model_conditioning, is_spatial)
    X_flat = torch.cat([X[i][0].reshape(-1, 1) for i in range(N)], dim=-1).detach().cpu()
    cov_data = torch.cov(X_flat.T)
    sigma = calibrate_sigma_for_leak(
        N, gamma_self, args.alpha_list, args.k_reference, cov_data,
        args.n_diff_steps, args.dt, None, leak_fraction=args.leak_fraction,
    )
    if args.debug_rollout:
        std = cov_data.diagonal().clamp_min(1e-12).sqrt()
        corr = cov_data / (std[:, None] * std[None, :])
        off = corr - torch.diag(torch.diag(corr))
        rho_true = off.sum().item() / (N * (N - 1))
        print(f"[debug-rollout] cov_data diag (per-task variance) = {dict(zip(task_names, std.pow(2).tolist()))}")
        print(f"[debug-rollout] cov_data mean pairwise corr (rho_true) = {rho_true:.6f}")
        print(f"[debug-rollout] calibrated sigma = {sigma}")
    return sigma


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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    is_spatial = args.score_arch in ("fno", "songunet")
    fno_modes = tuple(int(v) for v in args.fno_modes.split(","))
    songunet_channel_mult = tuple(int(v) for v in args.songunet_channel_mult.split(","))
    songunet_attn_res = tuple(int(v) for v in args.songunet_attn_resolutions.split(",")) if args.songunet_attn_resolutions else ()
    cond_in_ch = train_ds[0]["conditioning"].shape[0]
    if is_spatial:
        model = SpatialFieldModel(task_names, cond_in_ch=cond_in_ch, out_hw=(args.image_size, args.image_size),
                                   backbone_ch=args.backbone_channels, base_ch=args.base_channels,
                                   n_downsample=args.n_downsample, init_ch=args.fno_init_channels).to(device)
    else:
        model = PhysicsModel(task_names, cond_in_ch=cond_in_ch, out_hw=(args.image_size, args.image_size),
                              latent_dim=args.latent_dim, backbone_ch=args.backbone_channels,
                              base_ch=args.base_channels, n_downsample=args.n_downsample).to(device)

    is_csho = args.method in METHOD_CONFIGS
    if args.score_arch == "fno":
        if is_csho:
            score_net = FNOScoreNetwork(N, cond_in_ch, args.n_diff_steps, n_modes=fno_modes,
                                         hidden_channels=args.fno_hidden_channels).to(device)
        else:
            score_net = FlatFNOScoreNetwork(N, cond_in_ch, args.n_diff_steps, n_modes=fno_modes,
                                             hidden_channels=args.fno_hidden_channels).to(device)
    elif args.score_arch == "songunet":
        if is_csho:
            score_net = SongUNetScoreNetwork(N, cond_in_ch, args.n_diff_steps, img_resolution=args.image_size,
                                              model_channels=args.songunet_model_channels,
                                              channel_mult=songunet_channel_mult, num_blocks=args.songunet_num_blocks,
                                              attn_resolutions=songunet_attn_res).to(device)
        else:
            score_net = FlatSongUNetScoreNetwork(N, cond_in_ch, args.n_diff_steps, img_resolution=args.image_size,
                                                  model_channels=args.songunet_model_channels,
                                                  channel_mult=songunet_channel_mult, num_blocks=args.songunet_num_blocks,
                                                  attn_resolutions=songunet_attn_res).to(device)
    elif is_csho:
        score_net = MultiPhysicsScoreNetwork(N, args.latent_dim, model.backbone.out_ch,
                                              n_blocks=args.score_blocks, n_heads=args.score_heads).to(device)
    else:
        score_net = FlatScoreNetwork(N, args.latent_dim, model.backbone.out_ch,
                                      n_blocks=args.score_blocks, n_heads=args.score_heads).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(score_net.parameters()), lr=args.lr)

    gamma_self, gamma_couple = calibrate_coupled_gammas(
        args.alpha_list[0], args.beta_list[0], args.k_reference, args.k_reference, N,
        regime=args.damping_regime, target_zeta=args.target_zeta,
    )

    if is_csho:
        sigma = _calibrate_csho_sigma(model, train_loader, N, gamma_self, args, device,
                                       task_names, is_spatial)
        state = build_method_state(args, N, device, sigma=sigma)
        coupling = state["coupling"]
        params = precompute_transition_params(
            N, gamma_self, gamma_couple, args.alpha_list, args.beta_list, args.k_reference, coupling,
            args.n_diff_steps, args.dt, lambda t, T: build_g_matrix_n(
                torch.tensor(sigma, device=device), N, diffusion_mode=state["cfg"]["diffusion_mode"],
                g_per_task=state["g_per_task"], coupling_matrix=coupling if state["cfg"]["diffusion_mode"] == "independent" else None,
            ), args.constant_k, None,
        )
    else:
        state = build_method_state(args, N, device)

    drift_detector = make_drift_detector(bucket_names=[])
    attach_grad_hooks_generic(model, max_modules=args.max_hook_modules)
    attach_grad_hooks_generic(score_net, max_modules=args.max_hook_modules)

    nan_events, explosion_events, n_steps = 0, 0, 0

    model.train()
    score_net.train()
    for _epoch in range(args.n_epochs):
        for batch in train_loader:
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
                orig_shape = X[0][0].shape
                X_flat = torch.cat([X[i][0].reshape(-1, 1) for i in range(N)], dim=-1)
                Z0 = torch.cat([X_flat, torch.zeros_like(X_flat)], dim=-1)
                Phi_t, Sigma_t = params[t_idx - 1]
                Zt, score_target = sample_and_tikhonov_score_target(Z0, Phi_t, Sigma_t, N, args.lam)
                X_t = [[Zt[:, i:i + 1].reshape(*orig_shape)] for i in range(N)]
                V_t = [[Zt[:, N + i:N + i + 1].reshape(*orig_shape)] for i in range(N)]

                score_fn, _ = _build_score_fns(args, score_net, feat)
                score_pred = score_fn(X_t, V_t, t_idx)
                tikhonov = torch.zeros((), device=device)
                for i in range(N):
                    target_i = score_target[:, i:i + 1].reshape(*orig_shape)
                    tikhonov = tikhonov + F.mse_loss(score_pred[i][0], target_i)

                if is_spatial:
                    loss = init_loss + args.lambda_tikhonov * tikhonov
                else:
                    readout_loss = torch.zeros((), device=device)
                    for name, x, target_n in zip(task_names, X, targets_norm):
                        pred = model.decode(name, x[0])
                        readout_loss = readout_loss + F.l1_loss(pred, target_n)
                    loss = init_loss + readout_loss + args.lambda_tikhonov * tikhonov
            else:
                X0_flat = [x[0] for x in X]
                _, flat_fn = _build_score_fns(args, score_net, feat)
                if args.method == "ddpm":
                    ac, _, _ = state["ddpm_sched"]
                    Xt, noise = ddpm_forward_n(X0_flat, torch.tensor(t_idx, device=device), ac)
                    eps_pred = flat_fn(Xt, t_idx)
                else:
                    beta_t = vp_beta_t(torch.tensor(t_idx / args.n_diff_steps, device=device), 1.0)
                    Xt, noise = vp_forward_step_with_noise(X0_flat, beta_t, 1.0 / args.n_diff_steps)
                    eps_pred = flat_fn(Xt, t_idx / args.n_diff_steps)
                diffusion_loss = sum(F.mse_loss(p, n) for p, n in zip(eps_pred, noise))

                if is_spatial:
                    loss = init_loss + args.lambda_tikhonov * diffusion_loss
                else:
                    readout_loss = torch.zeros((), device=device)
                    for name, x, target_n in zip(task_names, X, targets_norm):
                        pred = model.decode(name, x[0])
                        readout_loss = readout_loss + F.l1_loss(pred, target_n)
                    loss = init_loss + readout_loss + args.lambda_tikhonov * diffusion_loss

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

    if args.debug_rollout and hasattr(score_net, "output_gain"):
        gain = score_net.output_gain.detach().cpu().tolist()
        print(f"[debug-rollout] trained output_gain per task ({list(zip(task_names, gain))})")

    metrics = evaluate(args, model, score_net, val_loader, device, task_names, state, gamma_self, gamma_couple,
                        is_spatial, target_mean, target_std)
    metrics["nan_events"] = float(nan_events)
    metrics["explosion_events"] = float(explosion_events)
    metrics["n_train_steps"] = float(n_steps)
    return metrics


@torch.no_grad()
def evaluate(args, model, score_net, val_loader, device, task_names, state, gamma_self, gamma_couple,
             is_spatial: bool = False, target_mean=None, target_std=None) -> Dict[str, float]:
    model.eval()
    score_net.eval()
    is_csho = state["is_csho"]
    N = len(task_names)
    is_elder = args.problem == "Elder"
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    rollout_errs: List[List[float]] = []
    first_batch = True

    for batch in val_loader:
        conditioning = batch["conditioning"].to(device)
        if "elliptic_params" in batch:
            batch["elliptic_params"] = batch["elliptic_params"].to(device)
        targets = [t.to(device) for t in batch["tasks"]]
        model_conditioning = _model_input_conditioning(args, batch, conditioning)
        X, K_self, K_global, feat, _ = _encode_state(args, model, task_names, model_conditioning, is_spatial)

        if is_csho:
            cfg, coupling, g_per_task, sigma = state["cfg"], state["coupling"], state["g_per_task"], state["sigma"]
            V = [[torch.zeros_like(x[0])] for x in X]
            X_cur, V_cur = X, V
            G = build_g_matrix_n(torch.tensor(sigma, device=device), N, diffusion_mode=cfg["diffusion_mode"],
                                  g_per_task=g_per_task, coupling_matrix=coupling if cfg["diffusion_mode"] == "independent" else None)
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
                    coupling, args.constant_k, args.dt, G,
                )
            final_latents = [X_cur[i][0] for i in range(N)]
        elif args.method == "ddpm":
            ac, betas_s, alphas_s = state["ddpm_sched"]
            T_eff = max(args.n_diff_steps - 1, 1)
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
            X_flat = [x[0] for x in X]
            _, flat_fn = _build_score_fns(args, score_net, feat)
            dt_step = 1.0 / args.n_diff_steps
            for t_idx in reversed(range(1, args.n_diff_steps + 1)):
                t_cont = t_idx / args.n_diff_steps
                beta_t = vp_beta_t(torch.tensor(t_cont, device=device), 1.0)
                eps_pred = flat_fn(X_flat, t_cont)
                g_t = torch.sqrt(beta_t)
                score = [-e / (g_t * math.sqrt(dt_step) + 1e-8) for e in eps_pred]
                X_flat = vp_sde_reverse_step_n(X_flat, score, beta_t, dt_step)
            final_latents = X_flat

        if is_spatial:
            preds_norm = final_latents
        else:
            preds_norm = [model.decode(name, final_latents[i]) for i, name in enumerate(task_names)]
        preds = _denormalize_preds(preds_norm, target_mean, target_std)

        for name, pred, target in zip(task_names, preds, targets):
            rel = relative_l2_error(pred, target)
            spec = spectral_l2_error(pred, target)
            for k, v in ((f"{name}_rel_l2", rel), (f"{name}_spectral_l2", spec)):
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1

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
    return metrics


def main():
    args = parse_args()
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
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
    ddpm_reverse_step_n, ddpm_forward_n, make_ddpm_schedule, vp_beta_t, vp_sde_drift_n, vp_sde_reverse_step_n,
)
from core.coupling import build_coupling_matrix
from core.damping import calibrate_gammas_for_regime
from core.diagnostics_bridge import attach_grad_hooks_generic, grad_norm_buckets, make_drift_detector
from core.reporting import plot_bar_comparison, render_experiment_report, write_csv, write_json
from core.sde import build_g_matrix_n, em_step_n, ndsm_loss_n, reverse_step_n
from core.stats import aggregate_over_seeds
from image.dataset import (
    SEG_IGNORE_INDEX, SOURCE_TASKS, MultiTaskVisionDataset, collate_fn, task_output_channels,
)
from image.model import (
    FlatScoreNetwork, MultiTaskModel, MultiTaskScoreNetwork, make_flat_score_fn, make_score_fn,
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

DEFAULT_TASK_SIMILARITY = {
    ("depth", "normals"): 0.9, ("normals", "depth"): 0.9,
    ("depth", "segmentation"): 0.5, ("segmentation", "depth"): 0.5,
    ("normals", "segmentation"): 0.5, ("segmentation", "normals"): 0.5,
    ("segmentation", "edges"): 0.6, ("edges", "segmentation"): 0.6,
    ("depth", "edges"): 0.4, ("edges", "depth"): 0.4,
    ("normals", "edges"): 0.6, ("edges", "normals"): 0.6,
    ("depth", "saliency"): 0.3, ("saliency", "depth"): 0.3,
    ("normals", "saliency"): 0.3, ("saliency", "normals"): 0.3,
    ("segmentation", "saliency"): 0.4, ("saliency", "segmentation"): 0.4,
    ("edges", "saliency"): 0.5, ("saliency", "edges"): 0.5,
}


def build_task_similarity_matrix(tasks: List[str], path: Optional[str] = None) -> torch.Tensor:
    if path:
        import json
        with open(path) as f:
            raw = json.load(f)
        sim = {(a, b): float(w) for a, row in raw.items() for b, w in row.items()}
    else:
        sim = DEFAULT_TASK_SIMILARITY
    N = len(tasks)
    W = torch.zeros(N, N)
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            if i != j:
                W[i, j] = sim.get((a, b), 0.1)
    return W


def apply_task_activation(task: str, raw: torch.Tensor) -> torch.Tensor:
    if task in ("edges", "saliency"):
        return torch.sigmoid(raw)
    if task == "depth":
        return F.softplus(raw)
    return raw


def compute_task_loss(task: str, raw_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task == "segmentation":
        return F.cross_entropy(raw_pred, target, ignore_index=SEG_IGNORE_INDEX)
    pred = apply_task_activation(task, raw_pred)
    if task == "normals":
        pred_n = F.normalize(pred, dim=1, eps=1e-6)
        target_n = F.normalize(target, dim=1, eps=1e-6)
        return (1.0 - (pred_n * target_n).sum(dim=1)).mean()
    return F.l1_loss(pred, target)


def compute_miou(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = pred_logits.argmax(dim=1)
    valid = target != SEG_IGNORE_INDEX
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c) & valid
        target_c = (target == c) & valid
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append((pred_c & target_c).sum().item() / union)
    return sum(ious) / len(ious) if ious else float("nan")


def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    diff = pred - target
    rmse = torch.sqrt((diff ** 2).mean()).item()
    rel = (diff.abs() / target.clamp_min(eps)).mean().item()
    return rmse, rel


def compute_normal_angular_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_n = F.normalize(pred, dim=1, eps=eps)
    target_n = F.normalize(target, dim=1, eps=eps)
    cos = (pred_n * target_n).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return (torch.acos(cos) * (180.0 / math.pi)).mean().item()


def compute_edge_fscore(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8) -> float:
    pred_b = (pred > threshold).float()
    target_b = (target > threshold).float()
    tp = (pred_b * target_b).sum().item()
    fp = (pred_b * (1 - target_b)).sum().item()
    fn = ((1 - pred_b) * target_b).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def compute_saliency_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


def compute_metric(task: str, pred: torch.Tensor, target: torch.Tensor, num_classes: Optional[int]) -> Dict[str, float]:
    if task == "segmentation":
        return {"segmentation_miou": compute_miou(pred, target, num_classes)}
    if task == "depth":
        rmse, rel = compute_depth_metrics(pred, target)
        return {"depth_rmse": rmse, "depth_rel_err": rel}
    if task == "normals":
        return {"normals_angular_error_deg": compute_normal_angular_error(pred, target)}
    if task == "edges":
        return {"edges_fscore": compute_edge_fscore(pred, target)}
    if task == "saliency":
        return {"saliency_mae": compute_saliency_mae(pred, target)}
    return {}


def combined_grad_buckets(named_modules) -> Dict[str, float]:
    out = {}
    for name, module in named_modules:
        for k, v in grad_norm_buckets(module).items():
            out[f"{name}.{k}"] = v
    return out


def vp_forward_step_with_noise(X: List[torch.Tensor], beta_t: torch.Tensor, dt: float):
    drift = vp_sde_drift_n(X, beta_t)
    g = torch.sqrt(beta_t)
    noise = [torch.randn_like(x) for x in X]
    X_next = [x + d * dt + g * math.sqrt(dt) * n for x, d, n in zip(X, drift, noise)]
    return X_next, noise


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
    p = argparse.ArgumentParser(description="Experiment 1 (vision): does CSHO cross-task coupling help multi-task diffusion?")
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--tasks", default="depth,normals")
    p.add_argument("--source", default="nyudv2", choices=list(SOURCE_TASKS))
    p.add_argument("--method", default="csho", choices=ALL_METHODS)
    p.add_argument("--damping-regime", default="critically_damped",
                    choices=["underdamped", "critically_damped", "overdamped"])
    p.add_argument("--target-zeta", type=float, default=None)
    p.add_argument("--alpha", default="1.0")
    p.add_argument("--beta", default="0.5")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--k-reference", type=float, default=1.0)
    p.add_argument("--constant-k", action="store_true")
    p.add_argument("--task-similarity-json", default=None)
    p.add_argument("--n-diff-steps", type=int, default=2)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--backbone-channels", type=int, default=512)
    p.add_argument("--pretrained-backbone", dest="pretrained_backbone", action="store_true", default=True)
    p.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    p.add_argument("--score-blocks", type=int, default=4)
    p.add_argument("--score-heads", type=int, default=4)
    p.add_argument("--score-spatial-stride", type=int, default=2)
    p.add_argument("--lambda-ndsm", type=float, default=1.0)
    p.add_argument("--explode-threshold", type=float, default=1e3)
    p.add_argument("--max-hook-modules", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/experiment_1_vision")
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

    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    args.alpha = parse_float_list(args.alpha, len(args.tasks))
    args.beta = parse_float_list(args.beta, len(args.tasks))
    return args


def build_method_state(args, tasks: List[str], device):
    N = len(tasks)
    if args.method in METHOD_CONFIGS:
        cfg = METHOD_CONFIGS[args.method]
        if cfg["coupling_mode"] == "pairwise":
            weights = build_task_similarity_matrix(tasks, args.task_similarity_json)
            coupling = build_coupling_matrix(N, mode="pairwise", weights=weights, device=device)
        else:
            coupling = build_coupling_matrix(N, mode=cfg["coupling_mode"], device=device)
        g_per_task = None
        if cfg["diffusion_mode"] == "independent":
            g_per_task = [args.sigma * (0.8 + 0.4 * i / max(N - 1, 1)) for i in range(N)]
        return {"is_csho": True, "cfg": cfg, "coupling": coupling, "g_per_task": g_per_task, "ddpm_sched": None}
    if args.method == "ddpm":
        return {"is_csho": False, "cfg": None, "coupling": None, "g_per_task": None,
                "ddpm_sched": make_ddpm_schedule(args.n_diff_steps, device=device)}
    if args.method == "sdm":
        return {"is_csho": False, "cfg": None, "coupling": None, "g_per_task": None, "ddpm_sched": None}
    raise ValueError(f"Unknown method {args.method!r}")


def train_one_seed(args: argparse.Namespace, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    tasks = args.tasks
    N = len(tasks)

    train_ds = MultiTaskVisionDataset(args.data_root, tasks, args.source, split="train",
                                       height=args.image_size, width=args.image_size)
    val_ds = MultiTaskVisionDataset(args.data_root, tasks, args.source, split="val",
                                     height=args.image_size, width=args.image_size,
                                     num_classes=train_ds.num_classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    num_classes = train_ds.num_classes
    task_channels = {t: task_output_channels(t, num_classes if t == "segmentation" else None) for t in tasks}

    model = MultiTaskModel(tasks, task_channels, latent_dim=args.latent_dim,
                            out_hw=(args.image_size, args.image_size), backbone_ch=args.backbone_channels,
                            pretrained=args.pretrained_backbone).to(device)

    state = build_method_state(args, tasks, device)
    is_csho = state["is_csho"]
    if is_csho:
        score_net = MultiTaskScoreNetwork(N, args.latent_dim, model.backbone.out_ch, n_blocks=args.score_blocks,
                                           n_heads=args.score_heads, spatial_stride=args.score_spatial_stride).to(device)
    else:
        score_net = FlatScoreNetwork(N, args.latent_dim, model.backbone.out_ch, n_blocks=args.score_blocks,
                                      n_heads=args.score_heads, spatial_stride=args.score_spatial_stride).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(score_net.parameters()), lr=args.lr)

    gamma = calibrate_gammas_for_regime(args.alpha, [args.k_reference] * N, args.damping_regime, args.target_zeta)
    drift_detector = make_drift_detector(bucket_names=[])
    attach_grad_hooks_generic(model, max_modules=args.max_hook_modules)
    attach_grad_hooks_generic(score_net, max_modules=args.max_hook_modules)

    nan_events, explosion_events, n_steps = 0, 0, 0

    model.train()
    score_net.train()
    for _epoch in range(args.n_epochs):
        for batch in train_loader:
            image = batch["image"].to(device)
            X, K_self, K_global, feat, y0 = model.encode(image)

            init_loss = torch.zeros((), device=device)
            for t in tasks:
                y0_full = F.interpolate(y0[t], size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
                init_loss = init_loss + compute_task_loss(t, y0_full, batch[t].to(device))

            t_idx = torch.randint(1, args.n_diff_steps + 1, (1,)).item()

            if is_csho:
                cfg, coupling, g_per_task = state["cfg"], state["coupling"], state["g_per_task"]
                V = [[torch.zeros_like(x[0])] for x in X]
                G = build_g_matrix_n(torch.tensor(args.sigma, device=device), N, diffusion_mode=cfg["diffusion_mode"],
                                      g_per_task=g_per_task, coupling_matrix=coupling if cfg["diffusion_mode"] == "independent" else None)
                X_next, V_next, mu, z_list, sigma_list = em_step_n(
                    X, V, K_self, K_global, t_idx, args.n_diff_steps, alpha=args.alpha, beta=args.beta, gamma=gamma,
                    coupling_matrix_drift=coupling, use_gamma=True, constant_k=args.constant_k, dt=args.dt, G=G,
                )
                score_fn = make_score_fn(score_net, feat)
                ndsm = ndsm_loss_n(X, score_fn, V_next, mu, z_list, sigma_list, t_n=t_idx)

                readout_loss = torch.zeros((), device=device)
                for t, x in zip(tasks, X):
                    pred = model.decode(t, x[0])
                    readout_loss = readout_loss + compute_task_loss(t, pred, batch[t].to(device))

                loss = init_loss + readout_loss + args.lambda_ndsm * ndsm
            else:
                X0_flat = [x[0] for x in X]
                flat_fn = make_flat_score_fn(score_net, feat)
                if args.method == "ddpm":
                    ac, _, _ = state["ddpm_sched"]
                    Xt, noise = ddpm_forward_n(X0_flat, torch.tensor(t_idx, device=device), ac)
                    eps_pred = flat_fn(Xt, t_idx)
                else:
                    beta_t = vp_beta_t(torch.tensor(t_idx / args.n_diff_steps, device=device), 1.0)
                    Xt, noise = vp_forward_step_with_noise(X0_flat, beta_t, 1.0 / args.n_diff_steps)
                    eps_pred = flat_fn(Xt, t_idx / args.n_diff_steps)
                diffusion_loss = sum(F.mse_loss(p, n) for p, n in zip(eps_pred, noise))

                readout_loss = torch.zeros((), device=device)
                for t, x in zip(tasks, X):
                    pred = model.decode(t, x[0])
                    readout_loss = readout_loss + compute_task_loss(t, pred, batch[t].to(device))

                loss = init_loss + readout_loss + args.lambda_ndsm * diffusion_loss

            optimizer.zero_grad()
            loss.backward()
            buckets = combined_grad_buckets([("model", model), ("score_net", score_net)])
            triggered, _ = drift_detector.update_and_check(buckets, args.explode_threshold)
            if triggered:
                explosion_events += 1
            if torch.isfinite(loss):
                optimizer.step()
            else:
                nan_events += 1
            n_steps += 1

    metrics = evaluate(args, model, score_net, val_loader, device, tasks, num_classes, state)
    metrics["nan_events"] = float(nan_events)
    metrics["explosion_events"] = float(explosion_events)
    metrics["n_train_steps"] = float(n_steps)
    return metrics


@torch.no_grad()
def evaluate(args, model, score_net, val_loader, device, tasks, num_classes, state) -> Dict[str, float]:
    model.eval()
    score_net.eval()
    is_csho = state["is_csho"]
    N = len(tasks)
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for batch in val_loader:
        image = batch["image"].to(device)
        X, K_self, K_global, feat, _ = model.encode(image)

        if is_csho:
            cfg, coupling, g_per_task = state["cfg"], state["coupling"], state["g_per_task"]
            V = [[torch.zeros_like(x[0])] for x in X]
            X_cur, V_cur = X, V
            G = build_g_matrix_n(torch.tensor(args.sigma, device=device), N, diffusion_mode=cfg["diffusion_mode"],
                                  g_per_task=g_per_task, coupling_matrix=coupling if cfg["diffusion_mode"] == "independent" else None)
            score_fn = make_score_fn(score_net, feat)
            for t_idx in reversed(range(1, args.n_diff_steps + 1)):
                score_outputs = score_fn(X_cur, V_cur, t_idx)
                X_cur, V_cur = reverse_step_n(
                    X_cur, V_cur, K_self, K_global, score_outputs, t_idx, args.n_diff_steps,
                    alpha=args.alpha, beta=args.beta, gamma=calibrate_gammas_for_regime(
                        args.alpha, [args.k_reference] * N, args.damping_regime, args.target_zeta),
                    coupling_matrix_drift=coupling, use_gamma=True, constant_k=args.constant_k, dt=args.dt, G=G,
                )
            final_latents = {t: X_cur[i][0] for i, t in enumerate(tasks)}
        elif args.method == "ddpm":
            ac, betas_s, alphas_s = state["ddpm_sched"]
            T_eff = max(args.n_diff_steps - 1, 1)
            X0_flat = [x[0] for x in X]
            X_flat, _ = ddpm_forward_n(X0_flat, torch.tensor(T_eff, device=device), ac)
            flat_fn = make_flat_score_fn(score_net, feat)
            for t_idx in reversed(range(1, T_eff)):
                eps_pred = flat_fn(X_flat, t_idx)
                X_flat = ddpm_reverse_step_n(X_flat, eps_pred, t_idx, betas_s, alphas_s, ac)
            eps_pred = flat_fn(X_flat, 1)
            ac1 = ac[1]
            X_flat = [(x - torch.sqrt(1 - ac1) * e) / torch.sqrt(ac1) for x, e in zip(X_flat, eps_pred)]
            final_latents = {t: X_flat[i] for i, t in enumerate(tasks)}
        else:
            X_flat = [x[0] for x in X]
            flat_fn = make_flat_score_fn(score_net, feat)
            dt_step = 1.0 / args.n_diff_steps
            for t_idx in reversed(range(1, args.n_diff_steps + 1)):
                t_cont = t_idx / args.n_diff_steps
                beta_t = vp_beta_t(torch.tensor(t_cont, device=device), 1.0)
                eps_pred = flat_fn(X_flat, t_cont)
                g_t = torch.sqrt(beta_t)
                score = [-e / (g_t * math.sqrt(dt_step) + 1e-8) for e in eps_pred]
                X_flat = vp_sde_reverse_step_n(X_flat, score, beta_t, dt_step)
            final_latents = {t: X_flat[i] for i, t in enumerate(tasks)}

        for t in tasks:
            pred_raw = model.decode(t, final_latents[t])
            pred = pred_raw if t == "segmentation" else apply_task_activation(t, pred_raw)
            target = batch[t].to(device)
            for k, v in compute_metric(t, pred, target, num_classes).items():
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1

    return {k: sums[k] / counts[k] for k in sums}


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
        experiment_name=f"Experiment 1 (vision) -- {args.method}",
        summary_rows=summary_rows, significance_rows=[], figure_paths=[fig_path],
        out_path=os.path.join(args.out_dir, f"{args.method}_report.md"),
    )
    print(f"Done. Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
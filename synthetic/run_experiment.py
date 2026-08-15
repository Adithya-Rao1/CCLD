from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from core.baselines import (
    ddpm_forward_n, ddpm_reverse_step_n, make_ddpm_schedule, vp_beta_t, vp_sde_forward_step_n, vp_sde_reverse_step_n,
)
from core.coupling import build_coupling_matrix
from core.damping import calibrate_gammas_for_regime
from core.hypoellipticity import hypoellipticity_check
from core.reporting import plot_bar_comparison, render_experiment_report, write_csv, write_json
from core.sde import build_g_matrix_n, em_step_n, ndsm_loss_n, reverse_step_n
from core.stats import aggregate_over_seeds
from synthetic.ground_truth_sde import GroundTruthCoupledOU, make_ground_truth
from synthetic.metrics import (
    fit_gaussian, gaussian_kl, gaussian_mutual_information_matrix, gaussian_wasserstein2,
    integrated_autocorrelation_time, lag_k_autocorrelation,
)

CSHO_METHODS = {"csho": "mean_field", "csho_independent": "independent", "csho_pairwise": "pairwise"}
ALL_METHODS = sorted(list(CSHO_METHODS) + ["ddpm", "sdm"])


def _default_pairwise_weights(N: int, device=None) -> torch.Tensor:
    # No obvious "more/less related" prior between arbitrary OU populations (unlike e.g. vision's
    # depth<->normals), so pairwise defaults to a flat off-diagonal weight, matching image/pde's
    # csho_pairwise default -- override by editing this if a real similarity structure is desired.
    W = torch.ones((N, N), device=device)
    W.fill_diagonal_(0.0)
    return W

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t, batch: int, device, dtype) -> torch.Tensor:
        half = max(self.dim // 2, 1)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / half)
        t_val = torch.as_tensor(float(t), device=device, dtype=dtype).view(1)
        args = t_val * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb.unsqueeze(0).expand(batch, -1)


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(max(n_layers - 1, 1)):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class CoupledScoreNet(nn.Module):
    def __init__(self, N: int, hidden: int = 64, n_layers: int = 3, time_dim: int = 16):
        super().__init__()
        self.N = N
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.net = _mlp(2 * N + time_dim, N, hidden, n_layers)

    def forward(self, X: List[List[torch.Tensor]], V_query: List[List[torch.Tensor]], t_n) -> List[List[torch.Tensor]]:
        B = V_query[0][0].shape[0]
        device, dtype = V_query[0][0].device, V_query[0][0].dtype
        x_cat = torch.cat([X[i][0] for i in range(self.N)], dim=-1)
        v_cat = torch.cat([V_query[i][0] for i in range(self.N)], dim=-1)
        t_emb = self.time_embed(t_n, B, device, dtype)
        out = self.net(torch.cat([x_cat, v_cat, t_emb], dim=-1))
        return [[out[:, i : i + 1]] for i in range(self.N)]


class FlatScoreNet(nn.Module):
    def __init__(self, N: int, hidden: int = 64, n_layers: int = 3, time_dim: int = 16):
        super().__init__()
        self.N = N
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.net = _mlp(N + time_dim, N, hidden, n_layers)

    def forward(self, X: List[torch.Tensor], t_n) -> List[torch.Tensor]:
        B = X[0].shape[0]
        device, dtype = X[0].device, X[0].dtype
        x_cat = torch.cat(X, dim=-1)
        t_emb = self.time_embed(t_n, B, device, dtype)
        out = self.net(torch.cat([x_cat, t_emb], dim=-1))
        return [out[:, i : i + 1] for i in range(self.N)]


def _make_conditioning(N: int, B: int, k_reference: float, device, dtype=torch.float32):
    K_self = [[torch.full((B, 1), k_reference, device=device, dtype=dtype)] for _ in range(N)]
    K_global = [torch.full((B, 1), k_reference, device=device, dtype=dtype) for _ in range(N)]
    return K_self, K_global


def _make_hypo_conditioning(N: int, seed: int, shape=(2, 2)):
    g = torch.Generator().manual_seed(seed)
    K_self = [[torch.rand(shape, generator=g) + 0.1] for _ in range(N)]
    K_global = [torch.rand(shape, generator=g) + 0.1 for _ in range(N)]
    return K_self, K_global


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
    p = argparse.ArgumentParser(description="Experiment 3 (synthetic): does CSHO-N recover a known coupled-OU joint distribution?")
    p.add_argument("--config", default=None)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--coupling-strength", type=float, default=0.6)
    p.add_argument("--base-decay", type=float, default=1.0)
    p.add_argument("--gt-sigma-scale", type=float, default=1.0)
    p.add_argument("--gt-seed", type=int, default=None)

    p.add_argument("--method", default="csho", choices=ALL_METHODS)
    p.add_argument("--damping-regime", default="critically_damped", choices=["underdamped", "critically_damped", "overdamped"])
    p.add_argument("--target-zeta", type=float, default=None)
    p.add_argument("--alpha", default="1.0")
    p.add_argument("--beta", default="0.5")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--k-reference", type=float, default=1.0)
    p.add_argument("--constant-k", action="store_true")
    p.add_argument("--diffusion-mode", default="shared", choices=["shared", "independent"])

    p.add_argument("--n-diff-steps", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-train-iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--time-embed-dim", type=int, default=16)

    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--prior-probe-samples", type=int, default=512)
    p.add_argument("--mixing-n-steps", type=int, default=2000)
    p.add_argument("--mixing-t-fixed", type=float, default=None)

    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/experiment_3_synthetic")
    p.add_argument("--quick", action="store_true")
    return p


def apply_quick_overrides(args: argparse.Namespace) -> None:
    args.N = min(args.N, 3)
    args.n_diff_steps = 5
    args.batch_size = 32
    args.n_train_iters = 20
    args.n_samples = 200
    args.prior_probe_samples = 64
    args.mixing_n_steps = 100
    args.seeds = "0"


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    parser = build_arg_parser()
    if pre_args.config:
        parser.set_defaults(**load_config_defaults(pre_args.config))
    args = parser.parse_args(argv)

    if args.quick:
        apply_quick_overrides(args)

    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    args.alpha = parse_float_list(args.alpha, args.N)
    args.beta = parse_float_list(args.beta, args.N)
    if args.mixing_t_fixed is None:
        args.mixing_t_fixed = args.n_diff_steps / 2.0
    return args


def _diffusion_mode_g(args, N: int, coupling: torch.Tensor, device) -> torch.Tensor:
    if args.diffusion_mode == "independent":
        g_per_task = [args.sigma * (0.8 + 0.4 * i / max(N - 1, 1)) for i in range(N)]
        return build_g_matrix_n(torch.tensor(args.sigma, device=device), N, diffusion_mode="independent",
                                 g_per_task=g_per_task, coupling_matrix=coupling)
    return build_g_matrix_n(torch.tensor(args.sigma, device=device), N, diffusion_mode="shared")


def _build_csho_state(args: argparse.Namespace, device):
    N = args.N
    mode = CSHO_METHODS[args.method]
    if mode == "pairwise":
        # No task-identity prior between anonymous populations (unlike image/pde's named
        # tasks) -- default to a uniform off-diagonal weight matrix, same convention as
        # image/pde's "no obvious more/less related prior" default.
        weights = torch.ones((N, N), device=device)
        coupling = build_coupling_matrix(N, mode="pairwise", weights=weights, device=device)
    else:
        coupling = build_coupling_matrix(N, mode=mode, device=device)
    gamma = calibrate_gammas_for_regime(args.alpha, [args.k_reference] * N, args.damping_regime, args.target_zeta)
    G = _diffusion_mode_g(args, N, coupling, device)
    return coupling, gamma, G


def _estimate_prior_std(args, gt: GroundTruthCoupledOU, K_self, K_global, coupling, gamma, G, device) -> List[float]:
    B = args.prior_probe_samples
    X0 = gt.sample_stationary(B).to(device)
    X = [[X0[:, i : i + 1].clone()] for i in range(args.N)]
    V = [[torch.zeros_like(X0[:, i : i + 1])] for i in range(args.N)]
    K_self_p, K_global_p = _make_conditioning(args.N, B, args.k_reference, device)
    # Chain the FULL trajectory (not a single step at t=n_diff_steps, whose
    # time_scale=(T-t)/(t+T)=0 makes the drift vanish and would grossly
    # underestimate the true T-step accumulated noise) -- matches the fix in
    # train_csho below.
    for step in range(1, args.n_diff_steps + 1):
        X, V, _, _, _ = em_step_n(
            X, V, K_self_p, K_global_p, step, args.n_diff_steps,
            args.alpha, args.beta, gamma, coupling, True, args.constant_k, args.dt, G,
        )
    return [max(X[i][0].std().item(), 1e-3) for i in range(args.N)]


def train_csho(args, gt: GroundTruthCoupledOU, device) -> Tuple[nn.Module, List[float], torch.Tensor, List[float]]:
    N = args.N
    coupling, gamma, G = _build_csho_state(args, device)
    K_self, K_global = _make_conditioning(N, args.batch_size, args.k_reference, device)
    score_net = CoupledScoreNet(N, args.hidden_dim, args.n_layers, args.time_embed_dim).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=args.lr)

    for _ in range(args.n_train_iters):
        X0 = gt.sample_stationary(args.batch_size).to(device)
        X = [[X0[:, i : i + 1]] for i in range(N)]
        V = [[torch.zeros_like(X0[:, i : i + 1])] for i in range(N)]
        t_idx = torch.randint(1, args.n_diff_steps + 1, (1,)).item()

        # Chain t_idx real em_step_n calls from data (not a single jump straight to
        # t_idx). A single Euler-Maruyama step only approximates a SHORT timespan
        # well (the assumption NDSM's own derivation relies on); jumping directly
        # from data to t_idx in one step badly misrepresents the true t_idx-step
        # marginal once t_idx>1, increasingly so as t_idx grows -- this was
        # confirmed empirically (KL divergence grew ~40x per doubling of
        # n_diff_steps at fixed training budget with the single-jump version, and
        # got WORSE with more training as the network fit the flawed target more
        # precisely). Matches production's _simulate_and_sample_ndsm, which chains
        # through the real trajectory rather than jumping.
        for step in range(1, t_idx + 1):
            X_next, V_next, mu, z_list, sigma_list = em_step_n(
                X, V, K_self, K_global, step, args.n_diff_steps,
                args.alpha, args.beta, gamma, coupling, True, args.constant_k, args.dt, G,
            )
            X, V = X_next, V_next
        loss = ndsm_loss_n(X_next, score_net, V_next, mu, z_list, sigma_list, t_n=t_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prior_std = _estimate_prior_std(args, gt, K_self, K_global, coupling, gamma, G, device)
    return score_net, gamma, coupling, prior_std


@torch.no_grad()
def sample_csho(args, score_net, gamma, coupling, prior_std: List[float], device) -> torch.Tensor:
    N = args.N
    G = _diffusion_mode_g(args, N, coupling, device)
    K_self, K_global = _make_conditioning(N, args.n_samples, args.k_reference, device)

    X = [[prior_std[i] * torch.randn(args.n_samples, 1, device=device)] for i in range(N)]
    V = [[prior_std[i] * torch.randn(args.n_samples, 1, device=device)] for i in range(N)]

    for t_idx in reversed(range(1, args.n_diff_steps + 1)):
        score_outputs = score_net(X, V, t_idx)
        X, V = reverse_step_n(
            X, V, K_self, K_global, score_outputs, t_idx, args.n_diff_steps,
            args.alpha, args.beta, gamma, coupling, True, args.constant_k, args.dt, G,
        )
    return torch.cat([X[i][0] for i in range(N)], dim=-1)


def train_ddpm(args, gt: GroundTruthCoupledOU, device) -> Tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = args.N
    ac, betas, alphas = make_ddpm_schedule(args.n_diff_steps, device=device)
    score_net = FlatScoreNet(N, args.hidden_dim, args.n_layers, args.time_embed_dim).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=args.lr)

    for _ in range(args.n_train_iters):
        X0 = gt.sample_stationary(args.batch_size).to(device)
        X0_flat = [X0[:, i : i + 1] for i in range(N)]
        t = torch.randint(0, args.n_diff_steps + 1, (1,)).item()
        Xt, noise = ddpm_forward_n(X0_flat, torch.tensor(t, device=device), ac)
        eps_pred = score_net(Xt, t)
        loss = sum(F.mse_loss(p, n) for p, n in zip(eps_pred, noise))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return score_net, ac, betas, alphas


@torch.no_grad()
def sample_ddpm(args, score_net, ac, betas, alphas, device) -> torch.Tensor:
    N = args.N
    X = [torch.randn(args.n_samples, 1, device=device) for _ in range(N)]
    for t in reversed(range(args.n_diff_steps)):
        eps_pred = score_net(X, t)
        X = ddpm_reverse_step_n(X, eps_pred, t, betas, alphas, ac)
    return torch.cat(X, dim=-1)


def train_sdm(args, gt: GroundTruthCoupledOU, device) -> nn.Module:
    N = args.N
    score_net = FlatScoreNet(N, args.hidden_dim, args.n_layers, args.time_embed_dim).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=args.lr)
    dt_step = 1.0 / args.n_diff_steps

    for _ in range(args.n_train_iters):
        X0 = gt.sample_stationary(args.batch_size).to(device)
        X0_flat = [X0[:, i : i + 1] for i in range(N)]
        t_idx = torch.randint(1, args.n_diff_steps + 1, (1,)).item()
        t_cont = t_idx / args.n_diff_steps
        beta_t = vp_beta_t(torch.tensor(t_cont, device=device), 1.0)
        Xt, noise = vp_sde_forward_step_n(X0_flat, beta_t, dt_step)
        eps_pred = score_net(Xt, t_cont)
        loss = sum(F.mse_loss(p, n) for p, n in zip(eps_pred, noise))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return score_net


@torch.no_grad()
def sample_sdm(args, score_net, device) -> torch.Tensor:
    N = args.N
    dt_step = 1.0 / args.n_diff_steps
    X = [torch.randn(args.n_samples, 1, device=device) for _ in range(N)]
    for t_idx in reversed(range(1, args.n_diff_steps + 1)):
        t_cont = t_idx / args.n_diff_steps
        beta_t = vp_beta_t(torch.tensor(t_cont, device=device), 1.0)
        eps_pred = score_net(X, t_cont)
        g_t = torch.sqrt(beta_t)
        score = [-e / (g_t * math.sqrt(dt_step) + 1e-8) for e in eps_pred]
        X = vp_sde_reverse_step_n(X, score, beta_t, dt_step)
    return torch.cat(X, dim=-1)


def _measure_mixing_time(args, gt: GroundTruthCoupledOU, coupling, gamma, G, device) -> Tuple[float, float]:
    N = args.N
    K_self, K_global = _make_conditioning(N, 1, args.k_reference, device)
    X0 = gt.sample_stationary(1).to(device)
    X = [[X0[:, i : i + 1].clone()] for i in range(N)]
    V = [[torch.zeros_like(X0[:, i : i + 1])] for i in range(N)]
    t_fixed = args.mixing_t_fixed

    # Repeated application at a fixed t needs a stable (attracting) kernel to have
    # a meaningful mixing time at all -- em_step_n is deliberately the repelling
    # corruption direction (see its docstring/comment) and has no equilibrium under
    # indefinite repetition. reverse_step_n with a zero score gives the generative
    # (attracting) direction's own intrinsic mixing behavior instead.
    zero_score = [[torch.zeros_like(V[i][0])] for i in range(N)]
    trace = torch.zeros(args.mixing_n_steps)
    for step in range(args.mixing_n_steps):
        X, V = reverse_step_n(
            X, V, K_self, K_global, zero_score, t_fixed, args.n_diff_steps,
            args.alpha, args.beta, gamma, coupling, True, args.constant_k, args.dt, G,
        )
        trace[step] = torch.stack([X[i][0].mean() for i in range(N)]).mean()

    return lag_k_autocorrelation(trace, k=1), integrated_autocorrelation_time(trace)


def evaluate_sampling_quality(generated: torch.Tensor, gt: GroundTruthCoupledOU) -> Dict[str, float]:
    mean_gen, cov_gen = fit_gaussian(generated.cpu())
    mean_true = torch.zeros(gt.N, dtype=cov_gen.dtype)
    cov_true = gt.stationary_covariance()

    kl = gaussian_kl(mean_gen, cov_gen, mean_true, cov_true)
    w2 = gaussian_wasserstein2(mean_gen, cov_gen, mean_true, cov_true)

    mi_gen = gaussian_mutual_information_matrix(cov_gen)
    mi_true = gt.pairwise_mutual_information()
    off_diag = ~torch.eye(gt.N, dtype=torch.bool)
    mi_mae = (mi_gen - mi_true).abs()[off_diag].mean().item() if gt.N > 1 else 0.0

    return {"kl_divergence": float(kl.item()), "wasserstein2": float(w2.item()), "mi_mae": float(mi_mae)}


def train_one_seed(args: argparse.Namespace, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    gt_seed = args.gt_seed if args.gt_seed is not None else seed
    gt = make_ground_truth(args.N, args.coupling_strength, gt_seed, args.base_decay, args.gt_sigma_scale, device=device)

    if args.method in CSHO_METHODS:
        score_net, gamma, coupling, prior_std = train_csho(args, gt, device)
        generated = sample_csho(args, score_net, gamma, coupling, prior_std, device)
    elif args.method == "ddpm":
        score_net, ac, betas, alphas = train_ddpm(args, gt, device)
        generated = sample_ddpm(args, score_net, ac, betas, alphas, device)
        gamma, coupling = None, None
    elif args.method == "sdm":
        score_net = train_sdm(args, gt, device)
        generated = sample_sdm(args, score_net, device)
        gamma, coupling = None, None
    else:
        raise ValueError(f"Unknown method {args.method!r}")

    metrics = evaluate_sampling_quality(generated, gt)

    if args.method in CSHO_METHODS:
        _, gamma_full, G = _build_csho_state(args, device)
        autocorr_lag1, iac_time = _measure_mixing_time(args, gt, coupling, gamma_full, G, device)
        metrics["autocorr_lag1"] = autocorr_lag1
        metrics["integrated_autocorr_time"] = iac_time

        K_self_h, K_global_h = _make_hypo_conditioning(args.N, seed)
        hypo = hypoellipticity_check(
            N=args.N, K_self=K_self_h, K_global=K_global_h, t=torch.tensor(args.n_diff_steps / 2.0), T=args.n_diff_steps,
            alpha=args.alpha, beta=args.beta, gamma=gamma_full, coupling_matrix=coupling.cpu(),
            use_gamma=True, constant_k=args.constant_k, G=G.cpu(),
        )
        metrics["hypo_passed"] = float(hypo["passed"])
        metrics["hypo_min_eig"] = float(hypo["min_eig"])
    else:
        metrics["autocorr_lag1"] = float("nan")
        metrics["integrated_autocorr_time"] = float("nan")
        metrics["hypo_passed"] = float("nan")
        metrics["hypo_min_eig"] = float("nan")

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
        experiment_name=f"Experiment 3 (synthetic) -- {args.method}",
        summary_rows=summary_rows, significance_rows=[], figure_paths=[fig_path],
        out_path=os.path.join(args.out_dir, f"{args.method}_report.md"),
    )
    print(f"Done. Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
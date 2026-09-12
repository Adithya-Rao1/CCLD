from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
from tqdm import tqdm

import synthetic.additive_skew_pairwise as asp
from core.coupling import build_coupling_matrix
from core.reporting import write_csv
from synthetic.additive_skew_experiment import build_generic_skew_additive, build_structured_skew_additive
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas_spectral, calibrate_sigma_fdt_spectral
from synthetic.exact_dsm import (
    closed_form_propagator_skew_additive, elapsed_time_at_step, sample_and_analytic_score_target_skew,
    score_target_from_covariance,
)
from synthetic.ground_truth_sde import make_asymmetric_ground_truth
from synthetic.run_experiment import CoupledScoreNet

N_SWEEP = [2, 3, 4]
COUPLING_STRENGTH = 0.6
SKEW_SCALE = 1.0
SCALE_SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0, 10.0]
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_ITERS = 2000
EVAL_EVERY = 100
HELD_OUT_BATCH = 512
N_DIFF_STEPS = 32
OUT_DIR = "results/experiment_3_synthetic/additive_skew_learning_diagnostics"


def _skew_conditions(gt, coupling, device) -> List:
    return [
        ("symmetric_only", None),
        ("skew_generic", build_generic_skew_additive(gt.N, seed=0, device=device, scale=SKEW_SCALE)),
        ("skew_structured", build_structured_skew_additive(gt, coupling, device, scale=SKEW_SCALE)),
    ]


def _held_out_loss(N: int, gt, device, score_net, params_skew, t_idx: int, batch_size: int) -> float:
    X0 = gt.sample_stationary(batch_size).to(device)
    Z0 = torch.cat([X0, torch.zeros_like(X0)], dim=1)
    Phi_t, Sigma_t = params_skew[t_idx - 1]
    Zt, score_target = sample_and_analytic_score_target_skew(Z0, Phi_t, Sigma_t)
    X_t = [[Zt[:, i:i + 1]] for i in range(N)]
    V_t = [[Zt[:, N + i:N + i + 1]] for i in range(N)]
    with torch.no_grad():
        score_pred = score_net(X_t, V_t, t_idx)
        loss = sum(((score_pred[i][0] - score_target[:, i:i + 1]) ** 2).mean() for i in range(N)).item()
    return loss


def train_with_diagnostics(N: int, coupling: torch.Tensor, gt, device, skew_matrix, seed: int, label: str = "",
                            n_train_iters: int = N_TRAIN_ITERS, eval_every: int = EVAL_EVERY,
                            held_out_batch: int = HELD_OUT_BATCH):
    torch.manual_seed(seed)
    Gamma = calibrate_coupled_gammas_spectral(asp.ALPHA_V, asp.BETA, asp.K_REFERENCE, asp.K_REFERENCE, coupling, target_zeta=asp.TARGET_ZETA)
    G0 = calibrate_sigma_fdt_spectral(Gamma, coupling, target_variance=1.0)

    params_skew = [
        closed_form_propagator_skew_additive(
            N, None, None, [asp.ALPHA_V] * N, [asp.BETA] * N, asp.K_REFERENCE, coupling,
            tau_hat=elapsed_time_at_step(t_idx, N_DIFF_STEPS, asp.DT, asp.TIME_SCALE_FN),
            constant_k=True, G0=G0, damping_matrix=Gamma, skew_matrix=skew_matrix,
        )
        for t_idx in range(1, N_DIFF_STEPS + 1)
    ]

    score_net = CoupledScoreNet(N, 64, 3, 16).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-3)

    loss_curve_rows: List[Dict] = []
    pbar = tqdm(total=n_train_iters, desc=label or f"[N={N}] train+diag")
    for it in range(n_train_iters):
        X0 = gt.sample_stationary(asp.BATCH_SIZE).to(device)
        Z0 = torch.cat([X0, torch.zeros_like(X0)], dim=1)
        t_idx = torch.randint(1, N_DIFF_STEPS + 1, (1,)).item()
        Phi_t, Sigma_t = params_skew[t_idx - 1]
        Zt, score_v_target = sample_and_analytic_score_target_skew(Z0, Phi_t, Sigma_t)
        X_t = [[Zt[:, i:i + 1]] for i in range(N)]
        V_t = [[Zt[:, N + i:N + i + 1]] for i in range(N)]
        score_pred = score_net(X_t, V_t, t_idx)
        loss = sum(((score_pred[i][0] - score_v_target[:, i:i + 1]) ** 2).mean() for i in range(N))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
        optimizer.step()

        if (it + 1) % eval_every == 0 or it == n_train_iters - 1:
            held_out_t = torch.randint(1, N_DIFF_STEPS + 1, (1,)).item()
            ho_loss = _held_out_loss(N, gt, device, score_net, params_skew, held_out_t, held_out_batch)
            loss_curve_rows.append({"iteration": it + 1, "held_out_loss": ho_loss})
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)
    pbar.close()

    profile_rows: List[Dict] = []
    for t_idx in range(1, N_DIFF_STEPS + 1):
        ho_loss = _held_out_loss(N, gt, device, score_net, params_skew, t_idx, held_out_batch)
        profile_rows.append({"t_idx": t_idx, "held_out_loss": ho_loss})

    return score_net, loss_curve_rows, profile_rows


def score_target_complexity(N: int, coupling: torch.Tensor, device, skew_matrix, t_idx_list: List[int]) -> List[Dict]:
    Gamma = calibrate_coupled_gammas_spectral(asp.ALPHA_V, asp.BETA, asp.K_REFERENCE, asp.K_REFERENCE, coupling, target_zeta=asp.TARGET_ZETA)
    G0 = calibrate_sigma_fdt_spectral(Gamma, coupling, target_variance=1.0)
    rows = []
    for t_idx in t_idx_list:
        tau_hat = elapsed_time_at_step(t_idx, N_DIFF_STEPS, asp.DT, asp.TIME_SCALE_FN)
        _, Sigma_t = closed_form_propagator_skew_additive(
            N, None, None, [asp.ALPHA_V] * N, [asp.BETA] * N, asp.K_REFERENCE, coupling,
            tau_hat=tau_hat, constant_k=True, G0=G0, damping_matrix=Gamma, skew_matrix=skew_matrix,
        )
        precision, _ = score_target_from_covariance(Sigma_t, N)
        eigs = torch.linalg.eigvalsh(precision).clamp_min(1e-10)
        cond = (eigs.max() / eigs.min()).item()
        rows.append({"t_idx": t_idx, "cond_number": cond, "eig_min": eigs.min().item(), "eig_max": eigs.max().item()})
    return rows


def run(complexity_only: bool = False) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    asp.N_DIFF_STEPS = N_DIFF_STEPS
    asp.DT = 1.0 / N_DIFF_STEPS

    loss_curve_all: List[Dict] = []
    profile_all: List[Dict] = []
    complexity_all: List[Dict] = []

    t_idx_profile = list(range(1, N_DIFF_STEPS + 1))

    for N in N_SWEEP:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt = make_asymmetric_ground_truth(N, COUPLING_STRENGTH, seed=0, device=device)
        coupling = build_coupling_matrix(N, mode="mean_field", device=device)

        print(f"\nN={N}: score-target complexity sweep (no training, scale-swept)")
        for label, skew_builder in [("skew_generic", build_generic_skew_additive), ("skew_structured", build_structured_skew_additive)]:
            for scale in SCALE_SWEEP:
                if scale == 0.0:
                    skew_matrix = None
                elif skew_builder is build_generic_skew_additive:
                    skew_matrix = build_generic_skew_additive(N, seed=0, device=device, scale=scale)
                else:
                    skew_matrix = build_structured_skew_additive(gt, coupling, device, scale=scale)
                rows = score_target_complexity(N, coupling, device, skew_matrix, t_idx_profile)
                for row in rows:
                    complexity_all.append({"N": N, "condition": label, "scale": scale, **row})
            print(f"  {label}: done")

        if complexity_only:
            continue

        conditions = _skew_conditions(gt, coupling, device)
        for label, skew_matrix in conditions:
            for seed in SEEDS:
                tag = f"[N={N}/diag/{label}] seed={seed}"
                _, loss_curve_rows, profile_rows = train_with_diagnostics(
                    N, coupling, gt, device, skew_matrix, seed, label=tag,
                    n_train_iters=N_TRAIN_ITERS, eval_every=EVAL_EVERY, held_out_batch=HELD_OUT_BATCH,
                )
                for row in loss_curve_rows:
                    loss_curve_all.append({"N": N, "condition": label, "seed": seed, **row})
                for row in profile_rows:
                    profile_all.append({"N": N, "condition": label, "seed": seed, **row})

    write_csv(loss_curve_all, os.path.join(OUT_DIR, "held_out_loss_curves.csv"))
    write_csv(profile_all, os.path.join(OUT_DIR, "per_timestep_score_error_profile.csv"))
    write_csv(complexity_all, os.path.join(OUT_DIR, "score_target_complexity.csv"))
    print(f"\nWrote results to {OUT_DIR}/")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolate the effect of asymmetric (additive) skew coupling on score network learning.")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-train-iters", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--held-out-batch", type=int, default=512)
    p.add_argument("--n-sweep", default="2,3,4")
    p.add_argument("--n-diff-steps", type=int, default=32)
    p.add_argument("--skew-scale", type=float, default=1.0)
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/additive_skew_learning_diagnostics")
    p.add_argument("--complexity-only", action="store_true",)
    p.add_argument("--quick", action="store_true",)
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    N_TRAIN_ITERS = _args.n_train_iters
    EVAL_EVERY = _args.eval_every
    HELD_OUT_BATCH = _args.held_out_batch
    N_SWEEP = [int(n) for n in _args.n_sweep.split(",") if n.strip()]
    N_DIFF_STEPS = _args.n_diff_steps
    SKEW_SCALE = _args.skew_scale
    OUT_DIR = _args.out_dir
    if _args.quick:
        SEEDS = [0]
        N_TRAIN_ITERS = 20
        EVAL_EVERY = 5
        HELD_OUT_BATCH = 32
        N_SWEEP = [2, 3]
        N_DIFF_STEPS = 8
        SCALE_SWEEP = [0.0, 1.0]
    run(complexity_only=_args.complexity_only)
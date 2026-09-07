from __future__ import annotations

import argparse
import math
import sys
import traceback
from typing import Dict, List

import torch
import torch.nn.functional as F

from pde.fno_score_net import FNOScoreNetwork, FlatFNOScoreNetwork
from pde.model import FlatScoreNetwork, MultiPhysicsScoreNetwork, PhysicsModel, SpatialFieldModel
from pde.run_experiment import _build_score_fns, _encode_state, build_method_state
from pde.unet_model_score_net import FlatUNetModelScoreNetwork, UNetModelScoreNetwork
from synthetic.anderson_sde import anderson_reverse_step_coupled_gamma
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas, calibrate_sigma_fdt_coupled
from synthetic.exact_dsm import elapsed_time_at_step, precompute_transition_params, sample_and_analytic_score_target

SCORE_ARCHES = ["attention", "fno", "unet_model"]
PROBLEMS = ["TE_heat", "E_flow", "VA"]

PROBLEM_TASK_NAMES = {
    "TE_heat": ["T", "Re{Ez}", "Im{Ez}"],
    "E_flow": ["ec_V", "u_flow", "v_flow"],
    "VA": [
        "Re{p_t}", "Im{p_t}", "Re{Sxx}", "Im{Sxx}", "Re{Sxy}", "Im{Sxy}",
        "Re{Syy}", "Im{Syy}", "Re{x_u}", "Im{x_u}", "Re{x_v}", "Im{x_v}",
    ],
}
COND_IN_CH = 1
IMAGE_SIZE = 32
BATCH = 4

REAL_N_DIFF_STEPS = 20
REAL_DT = 1.0 / REAL_N_DIFF_STEPS 

ZERO_SCORE_RATIO_BOUND = 3.0
STEP_SWEEP_RATIO_BOUND = 3.0
STEP_SWEEP_N_DIFF_STEPS = [2, 4, 8, 16, 20]
STEP_SWEEP_N_TRAIN_STEPS = 8


def _make_args(score_arch: str, n_diff_steps: int, dt: float) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.score_arch = score_arch
    ns.method = "csho"
    ns.k_reference = 1.0
    ns.n_diff_steps = n_diff_steps
    ns.dt = dt
    ns.lam = 1.0
    ns.constant_k = True  # required for the closed-form kernel
    ns.damping_regime = "critically_damped"
    ns.target_zeta = None
    return ns


def _build_model_and_score_net(score_arch: str, task_names: List[str], n_diff_steps: int, device):
    N = len(task_names)
    is_spatial = score_arch in ("fno", "unet_model")
    if is_spatial:
        model = SpatialFieldModel(
            task_names, cond_in_ch=COND_IN_CH, out_hw=(IMAGE_SIZE, IMAGE_SIZE),
            backbone_ch=32, base_ch=8, n_downsample=1, init_ch=8,
        ).to(device)
    else:
        model = PhysicsModel(
            task_names, cond_in_ch=COND_IN_CH, out_hw=(IMAGE_SIZE, IMAGE_SIZE),
            latent_dim=16, backbone_ch=32, base_ch=8, n_downsample=1,
        ).to(device)

    if score_arch == "fno":
        score_net = FNOScoreNetwork(
            N, COND_IN_CH, n_diff_steps, n_modes=(8, 8), hidden_channels=16,
        ).to(device)
    elif score_arch == "unet_model":
        score_net = UNetModelScoreNetwork(
            N, COND_IN_CH, n_diff_steps, img_resolution=IMAGE_SIZE,
            model_channels=8, channel_mult=(1, 2), num_blocks=1, attn_resolutions=(),
        ).to(device)
    else:
        score_net = MultiPhysicsScoreNetwork(N, model.latent_dim, model.backbone.out_ch, n_blocks=1, n_heads=1).to(device)
    return model, score_net, is_spatial


def _calibrate_sigma(gamma_self, gamma_couple, N):
    return calibrate_sigma_fdt_coupled(gamma_self, gamma_couple, N)


def _realistic_targets(N: int, device) -> List[torch.Tensor]:
    scales = [1.0e5] + [1.0e2] * (N - 1)
    return [scale + 0.01 * scale * torch.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE, device=device) for scale in scales]


def _state_norm(X: List[List[torch.Tensor]]) -> float:
    return torch.cat([x[0].reshape(BATCH, -1) for x in X], dim=1).norm(dim=1).mean().item()


ATTENTION_LATENT_DIM = 16 

def _prior_state_shape(score_arch: str):
    if score_arch in ("fno", "unet_model"):
        return (BATCH, 1, IMAGE_SIZE, IMAGE_SIZE)
    return (BATCH, ATTENTION_LATENT_DIM)


def zero_score_round_trip(score_arch: str, problem: str, device) -> Dict[str, float]:
    task_names = PROBLEM_TASK_NAMES[problem]
    N = len(task_names)
    args = _make_args(score_arch, REAL_N_DIFF_STEPS, REAL_DT)
    args.alpha_list = [1.0] * N
    args.beta_list = [0.5] * N  # pde/run_experiment.py --beta default

    shape = _prior_state_shape(score_arch)
    X = [[torch.randn(shape, device=device)] for _ in range(N)]  # prior_std=1.0
    k_const = torch.full((BATCH, 1), args.k_reference, device=device)
    K_self = [[k_const] for _ in range(N)]
    K_global = [k_const for _ in range(N)]

    gamma_self, gamma_couple = calibrate_coupled_gammas(
        args.alpha_list[0], args.beta_list[0], args.k_reference, args.k_reference, N,
        regime=args.damping_regime, target_zeta=args.target_zeta,
    )
    state = build_method_state(args, N, device, sigma_ab=(0.0, 0.0))
    coupling, G = state["coupling"], state["g_matrix"]

    V = [[torch.zeros_like(x[0])] for x in X]
    X_cur, V_cur = X, V
    zero_score = [[torch.zeros_like(x[0])] for x in X]
    x0_norm = _state_norm(X)

    with torch.no_grad():
        for t_idx in reversed(range(1, REAL_N_DIFF_STEPS + 1)):
            X_cur, V_cur = anderson_reverse_step_coupled_gamma(
                X_cur, V_cur, K_self, K_global, zero_score, t_idx, REAL_N_DIFF_STEPS,
                args.alpha_list, args.beta_list, gamma_self, gamma_couple,
                coupling, args.constant_k, args.dt, G,
            )
    xf_norm = _state_norm(X_cur)
    ratio = xf_norm / max(x0_norm, 1e-8)
    passed = math.isfinite(ratio) and ratio < ZERO_SCORE_RATIO_BOUND
    return {"ratio": ratio, "passed": passed}


def _train_csho_briefly(score_arch: str, problem: str, n_diff_steps: int, dt: float, device, n_train_steps: int):
    task_names = PROBLEM_TASK_NAMES[problem]
    N = len(task_names)
    args = _make_args(score_arch, n_diff_steps, dt)
    args.alpha_list = [1.0] * N
    args.beta_list = [0.5] * N  # pde/run_experiment.py --beta default

    model, score_net, is_spatial = _build_model_and_score_net(score_arch, task_names, n_diff_steps, device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(score_net.parameters()), lr=1e-3)

    gamma_self, gamma_couple = calibrate_coupled_gammas(
        args.alpha_list[0], args.beta_list[0], args.k_reference, args.k_reference, N,
        regime=args.damping_regime, target_zeta=args.target_zeta,
    )

    conditioning = torch.randn(BATCH, COND_IN_CH, IMAGE_SIZE, IMAGE_SIZE, device=device)
    targets = _realistic_targets(N, device)
    target_mean = [t.mean() for t in targets]
    target_std = [t.std().clamp_min(1e-6) for t in targets]
    targets_norm = [(t - m) / s for t, m, s in zip(targets, target_mean, target_std)]

    sigma_ab = _calibrate_sigma(gamma_self, gamma_couple, N)
    state = build_method_state(args, N, device, sigma_ab=sigma_ab)
    coupling = state["coupling"]
    g_matrix = state["g_matrix"]
    params = precompute_transition_params(
        N, gamma_self, gamma_couple, args.alpha_list, args.beta_list, args.k_reference, coupling,
        n_diff_steps, dt, lambda t, T: g_matrix, args.constant_k, None,
    )

    model.train()
    score_net.train()
    for _ in range(n_train_steps):
        X, K_self, K_global, feat, y0 = _encode_state(args, model, task_names, conditioning, is_spatial)
        init_loss = torch.zeros((), device=device)
        for name, target_n in zip(task_names, targets_norm):
            init_loss = init_loss + F.l1_loss(y0[name], target_n)

        t_idx = torch.randint(1, n_diff_steps + 1, (1,)).item()
        orig_shape = X[0][0].shape
        X_flat = torch.cat([X[i][0].reshape(-1, 1) for i in range(N)], dim=-1)
        Z0 = torch.cat([X_flat, torch.zeros_like(X_flat)], dim=-1)
        Phi_t, _ = params[t_idx - 1]
        q = elapsed_time_at_step(t_idx, n_diff_steps, dt, None)
        Zt, score_target = sample_and_analytic_score_target(Z0, Phi_t, N, gamma_self, gamma_couple, q)
        X_t = [[Zt[:, i:i + 1].reshape(*orig_shape)] for i in range(N)]
        V_t = [[Zt[:, N + i:N + i + 1].reshape(*orig_shape)] for i in range(N)]

        score_fn, _ = _build_score_fns(args, score_net, feat)
        score_pred = score_fn(X_t, V_t, t_idx)
        dsm_loss = torch.zeros((), device=device)
        for i in range(N):
            target_i = score_target[:, i:i + 1].reshape(*orig_shape)
            dsm_loss = dsm_loss + F.mse_loss(score_pred[i][0], target_i)

        loss = init_loss + dsm_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "model": model, "score_net": score_net, "args": args, "task_names": task_names, "N": N,
        "gamma_self": gamma_self, "gamma_couple": gamma_couple, "state": state,
        "is_spatial": is_spatial, "conditioning": conditioning,
    }


def _rollout_ratio(trained: dict, n_diff_steps: int, dt: float, device) -> float:
    model, score_net, args = trained["model"], trained["score_net"], trained["args"]
    task_names, N = trained["task_names"], trained["N"]
    is_spatial, state = trained["is_spatial"], trained["state"]

    model.eval()
    score_net.eval()
    with torch.no_grad():
        conditioning = trained["conditioning"]
        X, K_self, K_global, feat, _ = _encode_state(args, model, task_names, conditioning, is_spatial)
        coupling, G = state["coupling"], state["g_matrix"]
        score_fn, _ = _build_score_fns(args, score_net, feat)

        V = [[torch.zeros_like(x[0])] for x in X]
        X_cur, V_cur = X, V
        for t_idx in reversed(range(1, n_diff_steps + 1)):
            score_outputs = score_fn(X_cur, V_cur, t_idx)
            X_cur, V_cur = anderson_reverse_step_coupled_gamma(
                X_cur, V_cur, K_self, K_global, score_outputs, t_idx, n_diff_steps,
                args.alpha_list, args.beta_list, trained["gamma_self"], trained["gamma_couple"],
                coupling, args.constant_k, dt, G,
            )
        xf_norm_trained = _state_norm(X_cur)

        V0 = [[torch.zeros_like(x[0])] for x in X]
        X0_cur, V0_cur = X, V0
        zero_score = [[torch.zeros_like(x[0])] for x in X]
        for t_idx in reversed(range(1, n_diff_steps + 1)):
            X0_cur, V0_cur = anderson_reverse_step_coupled_gamma(
                X0_cur, V0_cur, K_self, K_global, zero_score, t_idx, n_diff_steps,
                args.alpha_list, args.beta_list, trained["gamma_self"], trained["gamma_couple"],
                coupling, args.constant_k, dt, G,
            )
        xf_norm_zero = _state_norm(X0_cur)
    return xf_norm_trained / max(xf_norm_zero, 1e-8)


def step_count_sweep(score_arch: str, problem: str, device) -> Dict[str, dict]:
    results = {}
    for n_diff_steps in STEP_SWEEP_N_DIFF_STEPS:
        dt = 1.0 / n_diff_steps
        trained = _train_csho_briefly(score_arch, problem, n_diff_steps, dt, device, STEP_SWEEP_N_TRAIN_STEPS)
        ratio = _rollout_ratio(trained, n_diff_steps, dt, device)
        passed = math.isfinite(ratio) and ratio < STEP_SWEEP_RATIO_BOUND
        results[n_diff_steps] = {"ratio": ratio, "passed": passed}
    return results


def run_all(device) -> bool:
    all_passed = True
    for problem in PROBLEMS:
        for score_arch in SCORE_ARCHES:
            label = f"[{problem:8s} / {score_arch:9s}]"
            try:
                zs = zero_score_round_trip(score_arch, problem, device)
            except Exception:
                print(f"{label} zero-score-round-trip: ERROR\n{traceback.format_exc()}")
                all_passed = False
                zs = None
            else:
                status = "PASS" if zs["passed"] else "FAIL"
                print(f"{label} zero-score-round-trip: {status}  ratio={zs['ratio']:.3f}  bound<{ZERO_SCORE_RATIO_BOUND}")
                all_passed = all_passed and zs["passed"]

            try:
                sweep = step_count_sweep(score_arch, problem, device)
            except Exception:
                print(f"{label} step-count-sweep: ERROR\n{traceback.format_exc()}")
                all_passed = False
                continue
            for n_diff_steps, r in sweep.items():
                status = "PASS" if r["passed"] else "FAIL"
                print(f"{label} step-count-sweep n_diff_steps={n_diff_steps:2d}: {status}  "
                      f"ratio={r['ratio']:.3f}  bound<{STEP_SWEEP_RATIO_BOUND}")
                all_passed = all_passed and r["passed"]
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli_args = parser.parse_args()
    device = torch.device(cli_args.device)
    print(f"Running pde/ reverse-SDE stability harness on device={device} "
          f"({len(SCORE_ARCHES)} archs x {len(PROBLEMS)} problems = {len(SCORE_ARCHES) * len(PROBLEMS)} combinations)")
    ok = run_all(device)
    print("ALL COMBINATIONS PASSED." if ok else "AT LEAST ONE COMBINATION FAILED.")
    sys.exit(0 if ok else 1)

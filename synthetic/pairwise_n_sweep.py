from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from core.coupling import block_coupling, build_coupling_matrix, random_heterogeneous_coupling
from core.reporting import write_csv, write_json
from core.sde import build_g_matrix_n
from core.stats import aggregate_over_seeds, compare_configs
from synthetic.anderson_sde import anderson_em_step_coupled_gamma, anderson_reverse_step_coupled_gamma
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas_spectral, calibrate_sigma_fdt_spectral
from synthetic.exact_dsm import (
    _extract_Avx_Avv_coupled_gamma, closed_form_propagator_skew, elapsed_time_at_step,
    precompute_transition_params, sample_and_analytic_score_target_skew, sample_and_analytic_score_target_spectral,
)
from synthetic.ground_truth_sde import GroundTruthCoupledOU, make_ground_truth_from_coupling
from synthetic.run_experiment import CoupledScoreNet, _make_conditioning, evaluate_sampling_quality
from synthetic.skew_coupling import reference_stationary_covariance


def _constant_tau_time_scale(t, T) -> torch.Tensor:
    return torch.tensor(2.0)


ALPHA_V = 1.0
K_REFERENCE = 1.0
TARGET_ZETA = 1.0
N_DIFF_STEPS = 32
DT = 1.0 / N_DIFF_STEPS
BATCH_SIZE = 256
COUPLING_STRENGTH = 0.6
BETA = 1.0
TIME_SCALE_FN = _constant_tau_time_scale

N_TRAIN_ITERS = 2000
N_SAMPLES = 4000
SEEDS = [0, 1, 2, 3, 4]
N_SWEEP = [2, 3, 4, 5]
OUT_DIR = "results/experiment_3_synthetic/pairwise_n_sweep"


def build_coupling_conditions(N: int, seed: int, device) -> List[Tuple[str, torch.Tensor]]:
    conditions: List[Tuple[str, torch.Tensor]] = [
        ("mean_field", build_coupling_matrix(N, mode="mean_field", device=device)),
        ("heterogeneous_eps0.5", random_heterogeneous_coupling(N, epsilon=0.5, seed=1000 + seed, device=device)),
        ("heterogeneous_eps1.0", random_heterogeneous_coupling(N, epsilon=1.0, seed=1000 + seed, device=device)),
    ]
    if N >= 4:
        half = N // 2
        conditions.append(("block", block_coupling(N, [half, N - half], w_in=5.0, w_out=0.5, device=device)))
    return conditions


def _g_fn(N: int, sigma_ab: Tuple[float, float], coupling: torch.Tensor, device):
    a, b = sigma_ab
    g_matrix = build_g_matrix_n(torch.tensor(a, device=device), N, diffusion_mode="shared", coupling_matrix=b * coupling)

    def g_fn(t, T) -> torch.Tensor:
        return g_matrix

    return g_fn


def _estimate_prior_std_anderson(
    N: int, gt: GroundTruthCoupledOU, coupling, Gamma, g_fn, device,
    skew_matrix=None, skew_sigma_ref=None,
) -> Tuple[List[float], List[float]]:
    B = 512
    X0 = gt.sample_stationary(B).to(device)
    X = [[X0[:, i : i + 1].clone()] for i in range(N)]
    V = [[torch.zeros_like(X0[:, i : i + 1])] for i in range(N)]
    K_self_p, K_global_p = _make_conditioning(N, B, K_REFERENCE, device)
    for step in range(1, N_DIFF_STEPS + 1):
        X, V, _, _, _ = anderson_em_step_coupled_gamma(
            X, V, K_self_p, K_global_p, step, N_DIFF_STEPS,
            [ALPHA_V] * N, [BETA] * N, None, None, coupling, True, DT,
            g_fn(step, N_DIFF_STEPS), time_scale_fn=TIME_SCALE_FN, damping_matrix=Gamma,
            skew_matrix=skew_matrix, skew_sigma_ref=skew_sigma_ref,
        )
    prior_std_x = [max(X[i][0].std().item(), 1e-3) for i in range(N)]
    prior_std_v = [max(V[i][0].std().item(), 1e-3) for i in range(N)]
    return prior_std_x, prior_std_v


def train_ccld_pairwise(N: int, coupling: torch.Tensor, gt: GroundTruthCoupledOU, device, desc: str = "", skew_matrix=None):
    Gamma = calibrate_coupled_gammas_spectral(ALPHA_V, BETA, K_REFERENCE, K_REFERENCE, coupling, target_zeta=TARGET_ZETA)
    G0 = calibrate_sigma_fdt_spectral(Gamma, coupling, target_variance=1.0)
    g_fn = lambda t, T: G0  # matches _g_fn's constant time schedule

    skew_sigma_ref = None
    params = params_skew = None
    if skew_matrix is not None:
        A_vx0, _ = _extract_Avx_Avv_coupled_gamma(
            N, None, None, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, coupling,
            t=1, T=1, constant_k=True, time_scale_fn=lambda t, T: torch.tensor(1.0), damping_matrix=Gamma,
        )
        skew_sigma_ref = reference_stationary_covariance(-A_vx0.to(device), target_variance=1.0)
        params_skew = [
            closed_form_propagator_skew(
                N, None, None, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, coupling,
                tau_hat=elapsed_time_at_step(t_idx, N_DIFF_STEPS, DT, TIME_SCALE_FN),
                constant_k=True, G0=G0, damping_matrix=Gamma, skew_matrix=skew_matrix, target_variance=1.0,
            )
            for t_idx in range(1, N_DIFF_STEPS + 1)
        ]
    else:
        params = precompute_transition_params(
            N, None, None, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, coupling,
            N_DIFF_STEPS, DT, g_fn, True, TIME_SCALE_FN, damping_matrix=Gamma,
        )

    score_net = CoupledScoreNet(N, 64, 3, 16).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-3)

    pbar = tqdm(total=N_TRAIN_ITERS, desc=desc or f"[N={N}] train")
    ema_loss = None
    for it in range(N_TRAIN_ITERS):
        X0 = gt.sample_stationary(BATCH_SIZE).to(device)
        Z0 = torch.cat([X0, torch.zeros_like(X0)], dim=1)
        t_idx = torch.randint(1, N_DIFF_STEPS + 1, (1,)).item()

        if params_skew is not None:
            Phi_t, Sigma_t = params_skew[t_idx - 1]
            Zt, score_v_target = sample_and_analytic_score_target_skew(Z0, Phi_t, Sigma_t)
        else:
            Phi_t, _ = params[t_idx - 1]
            q = elapsed_time_at_step(t_idx, N_DIFF_STEPS, DT, TIME_SCALE_FN)
            Zt, score_v_target = sample_and_analytic_score_target_spectral(Z0, Phi_t, Gamma, coupling, q)
        X_t = [[Zt[:, i : i + 1]] for i in range(N)]
        V_t = [[Zt[:, N + i : N + i + 1]] for i in range(N)]

        score_pred = score_net(X_t, V_t, t_idx)
        loss = sum(((score_pred[i][0] - score_v_target[:, i : i + 1]) ** 2).mean() for i in range(N))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}", ema_loss=f"{ema_loss:.4f}", grad_norm=f"{grad_norm.item():.3f}", t_idx=t_idx)
        pbar.update(1)
    pbar.close()

    prior_std = _estimate_prior_std_anderson(
        N, gt, coupling, Gamma, g_fn, device, skew_matrix=skew_matrix, skew_sigma_ref=skew_sigma_ref,
    )
    return score_net, Gamma, G0, prior_std, skew_sigma_ref


@torch.no_grad()
def sample_ccld_anderson(
    N: int, coupling: torch.Tensor, score_net, Gamma, G0, prior_std, device, desc: str = "",
    skew_matrix=None, skew_sigma_ref=None,
):
    g_fn = lambda t, T: G0  # noqa: E731
    K_self, K_global = _make_conditioning(N, N_SAMPLES, K_REFERENCE, device)
    prior_std_x, prior_std_v = prior_std
    X = [[prior_std_x[i] * torch.randn(N_SAMPLES, 1, device=device)] for i in range(N)]
    V = [[prior_std_v[i] * torch.randn(N_SAMPLES, 1, device=device)] for i in range(N)]

    pbar = tqdm(total=N_DIFF_STEPS, desc=desc or f"[N={N}] sample")
    for t_idx in reversed(range(1, N_DIFF_STEPS + 1)):
        score_outputs = score_net(X, V, t_idx)
        X, V = anderson_reverse_step_coupled_gamma(
            X, V, K_self, K_global, score_outputs, t_idx, N_DIFF_STEPS,
            [ALPHA_V] * N, [BETA] * N, None, None, coupling, True, DT,
            g_fn(t_idx, N_DIFF_STEPS), time_scale_fn=TIME_SCALE_FN, damping_matrix=Gamma,
            skew_matrix=skew_matrix, skew_sigma_ref=skew_sigma_ref,
        )
        pbar.set_postfix(t_idx=t_idx, x_std=f"{X[0][0].std().item():.3f}")
        pbar.update(1)
    pbar.close()
    return torch.cat([X[i][0] for i in range(N)], dim=-1)


def train_one_seed(
    N: int, coupling: torch.Tensor, seed: int, label: str = "", skew_matrix=None,
    gt=None, return_samples: bool = False,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gt is None:
        gt = make_ground_truth_from_coupling(N, coupling.to(device), COUPLING_STRENGTH)
    tag = f"[N={N}/{label}] seed={seed}" if label else f"[N={N}] seed={seed}"
    skew_matrix_dev = skew_matrix.to(device) if skew_matrix is not None else None
    score_net, Gamma, G0, prior_std, skew_sigma_ref = train_ccld_pairwise(
        N, coupling, gt, device, desc=f"{tag} train", skew_matrix=skew_matrix_dev,
    )
    generated = sample_ccld_anderson(
        N, coupling, score_net, Gamma, G0, prior_std, device, desc=f"{tag} sample",
        skew_matrix=skew_matrix_dev, skew_sigma_ref=skew_sigma_ref,
    )
    metrics = evaluate_sampling_quality(generated, gt)
    if return_samples:
        return metrics, generated
    return metrics


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_rows, all_rows, per_seed_rows, sig_rows = [], [], [], []

    for N in N_SWEEP:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conditions = build_coupling_conditions(N, seed=0, device=device)
        per_seed_by_condition: Dict[str, Dict[int, Dict[str, float]]] = {}

        for label, coupling in conditions:
            print(f"\n=== N={N} condition={label} ===")
            per_seed = {}
            for seed in SEEDS:
                m = train_one_seed(N, coupling, seed, label=label)
                per_seed[seed] = m
                per_seed_rows.append({"N": N, "condition": label, "seed": seed, **m})
                print(f"  seed={seed}: kl={m['kl_divergence']:.4f} corr_gen={m['mean_pairwise_corr_gen']:.4f} corr_true={m['mean_pairwise_corr_true']:.4f}")
            per_seed_by_condition[label] = per_seed

            cond_summary = aggregate_over_seeds(per_seed)
            for metric, stats in cond_summary.items():
                all_rows.append({"N": N, "condition": label, "metric": metric, **stats})
            summary_rows.append({
                "N": N, "condition": label,
                "kl_mean": cond_summary["kl_divergence"]["mean"], "kl_std": cond_summary["kl_divergence"]["std"],
                "corr_gen_mean": cond_summary["mean_pairwise_corr_gen"]["mean"], "corr_gen_std": cond_summary["mean_pairwise_corr_gen"]["std"],
                "corr_true": cond_summary["mean_pairwise_corr_true"]["mean"],
                "corr_pct_of_true": cond_summary["mean_pairwise_corr_gen"]["mean"] / cond_summary["mean_pairwise_corr_true"]["mean"] * 100.0,
            })

        mean_field_per_seed = per_seed_by_condition["mean_field"]
        sig_metric_names = ["kl_divergence", "wasserstein2", "mi_mae", "mean_pairwise_corr_gen"]
        for label, per_seed in per_seed_by_condition.items():
            if label == "mean_field":
                continue
            sig = compare_configs(mean_field_per_seed, per_seed, metric_names=sig_metric_names)
            for metric, s in sig.items():
                sig_rows.append({"N": N, "comparison": f"{label}_vs_mean_field", "metric": metric, **s})

    write_csv(per_seed_rows, os.path.join(OUT_DIR, "pairwise_n_sweep_per_seed.csv"))
    write_csv(sig_rows, os.path.join(OUT_DIR, "pairwise_n_sweep_significance.csv"))
    write_csv(all_rows, os.path.join(OUT_DIR, "pairwise_n_sweep_full.csv"))
    write_csv(summary_rows, os.path.join(OUT_DIR, "pairwise_n_sweep_summary.csv"))
    write_json(
        {"beta": BETA, "n_sweep": N_SWEEP, "n_diff_steps": N_DIFF_STEPS, "dt": DT, "summary_rows": summary_rows},
        os.path.join(OUT_DIR, "pairwise_n_sweep_results.json"),
    )

    print("\n\n=== SUMMARY: mean_field vs heterogeneous(eps) vs block, matched ground truth per condition ===")
    print(f"{'N':>3} {'condition':>22} {'KL':>10} {'corr_gen':>10} {'corr_true':>10} {'%true':>8}")
    for row in summary_rows:
        print(f"{row['N']:>3} {row['condition']:>22} {row['kl_mean']:>10.4f} {row['corr_gen_mean']:>10.4f} {row['corr_true']:>10.4f} {row['corr_pct_of_true']:>8.1f}")
    print(f"\nWrote results to {OUT_DIR}/")
    return summary_rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCLD mean-field vs pairwise (heterogeneous/block) coupling, N=2..5")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-train-iters", type=int, default=2000)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-sweep", default="2,3,4,5")
    p.add_argument("--n-diff-steps", type=int, default=32)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/pairwise_n_sweep")
    p.add_argument("--quick", action="store_true", help="tiny scale for smoke-testing the pipeline end-to-end")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    N_TRAIN_ITERS = _args.n_train_iters
    N_SAMPLES = _args.n_samples
    N_SWEEP = [int(n) for n in _args.n_sweep.split(",") if n.strip()]
    OUT_DIR = _args.out_dir
    N_DIFF_STEPS = _args.n_diff_steps
    DT = _args.dt if _args.dt is not None else 1.0 / N_DIFF_STEPS
    if _args.quick:
        SEEDS = [0]
        N_TRAIN_ITERS = 20
        N_SAMPLES = 128
        N_SWEEP = [2, 4]
        N_DIFF_STEPS = 8
        DT = 1.0 / N_DIFF_STEPS
    run()

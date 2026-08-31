from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch

from core.coupling import build_coupling_matrix
from core.reporting import write_csv, write_json
from core.sde import build_g_matrix_n
from core.stats import aggregate_over_seeds, compare_configs
from synthetic.anderson_sde import anderson_em_step_coupled_gamma, anderson_reverse_step_coupled_gamma
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas
from synthetic.exact_dsm import calibrate_sigma_for_leak, precompute_transition_params, sample_and_tikhonov_score_target
from synthetic.ground_truth_sde import GroundTruthCoupledOU, make_ground_truth
from synthetic.run_experiment import CoupledScoreNet, _make_conditioning, _vp_linear_time_scale, evaluate_sampling_quality

ALPHA_V = 1.0
K_REFERENCE = 1.0
TARGET_ZETA = 1.0
LAM = 0.1
N_DIFF_STEPS = 20
DT = 0.05
BATCH_SIZE = 256
COUPLING_STRENGTH = 0.6
BETA = 1.0
TIME_SCALE_FN = _vp_linear_time_scale

LEAK_FRACTION = 0.01
_sigma_cache: Dict[Tuple[int, int], float] = {}


def _get_sigma_n(N: int) -> float:
    key = (N, N_DIFF_STEPS)
    if key not in _sigma_cache:
        gamma_self_dc, _ = calibrate_coupled_gammas(ALPHA_V, 0.0, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
        gt_ref = make_ground_truth(N, COUPLING_STRENGTH, seed=0, base_decay=1.0, sigma_scale=1.0)
        cov_data = gt_ref.stationary_covariance()
        _sigma_cache[key] = calibrate_sigma_for_leak(
            N, gamma_self_dc, [ALPHA_V] * N, K_REFERENCE, cov_data,
            N_DIFF_STEPS, DT, TIME_SCALE_FN, leak_fraction=LEAK_FRACTION,
        )
    return _sigma_cache[key]

N_TRAIN_ITERS = 2000
N_SAMPLES = 4000
SEEDS = [0, 1, 2, 3, 4]
N_SWEEP = [2, 3, 4, 5]
OUT_DIR = "results/experiment_3_synthetic/anderson_tikhonov_n_sweep"
BASELINE_DIR = "results/experiment_3_synthetic/exact_prior_std_sweep"


def _g_fn(N: int, sigma: float, device):
    def g_fn(t, T) -> torch.Tensor:
        return build_g_matrix_n(torch.tensor(sigma, device=device), N, diffusion_mode="shared")
    return g_fn


def _estimate_prior_std_anderson(N: int, sigma: float, gt: GroundTruthCoupledOU, coupling, gamma_self, gamma_couple, g_fn, device) -> Tuple[List[float], List[float]]:
    B = 512
    X0 = gt.sample_stationary(B).to(device)
    X = [[X0[:, i : i + 1].clone()] for i in range(N)]
    V = [[torch.zeros_like(X0[:, i : i + 1])] for i in range(N)]
    K_self_p, K_global_p = _make_conditioning(N, B, K_REFERENCE, device)
    for step in range(1, N_DIFF_STEPS + 1):
        X, V, _, _, _ = anderson_em_step_coupled_gamma(
            X, V, K_self_p, K_global_p, step, N_DIFF_STEPS,
            [ALPHA_V] * N, [BETA] * N, gamma_self, gamma_couple, coupling, False, DT,
            g_fn(step, N_DIFF_STEPS), time_scale_fn=TIME_SCALE_FN,
        )
    prior_std_x = [max(X[i][0].std().item(), 1e-3) for i in range(N)]
    prior_std_v = [max(V[i][0].std().item(), 1e-3) for i in range(N)]
    return prior_std_x, prior_std_v


def train_csho_tikhonov(N: int, sigma: float, gt: GroundTruthCoupledOU, device):
    coupling = build_coupling_matrix(N, mode="mean_field", device=device)
    gamma_self, gamma_couple = calibrate_coupled_gammas(ALPHA_V, BETA, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
    g_fn = _g_fn(N, sigma, device)

    params = precompute_transition_params(
        N, gamma_self, gamma_couple, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, coupling,
        N_DIFF_STEPS, DT, g_fn, False, TIME_SCALE_FN,
    )

    score_net = CoupledScoreNet(N, 64, 3, 16).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-3)

    for _ in range(N_TRAIN_ITERS):
        X0 = gt.sample_stationary(BATCH_SIZE).to(device)
        Z0 = torch.cat([X0, torch.zeros_like(X0)], dim=1)
        t_idx = torch.randint(1, N_DIFF_STEPS + 1, (1,)).item()
        Phi_t, Sigma_t = params[t_idx - 1]

        Zt, score_v_target = sample_and_tikhonov_score_target(Z0, Phi_t, Sigma_t, N, LAM)
        X_t = [[Zt[:, i : i + 1]] for i in range(N)]
        V_t = [[Zt[:, N + i : N + i + 1]] for i in range(N)]

        score_pred = score_net(X_t, V_t, t_idx)
        loss = sum(((score_pred[i][0] - score_v_target[:, i : i + 1]) ** 2).mean() for i in range(N))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
        optimizer.step()

    prior_std = _estimate_prior_std_anderson(N, sigma, gt, coupling, gamma_self, gamma_couple, g_fn, device)
    return score_net, gamma_self, gamma_couple, coupling, prior_std


@torch.no_grad()
def sample_csho_anderson(N: int, sigma: float, score_net, gamma_self, gamma_couple, coupling, prior_std, device):
    g_fn = _g_fn(N, sigma, device)
    K_self, K_global = _make_conditioning(N, N_SAMPLES, K_REFERENCE, device)
    prior_std_x, prior_std_v = prior_std
    X = [[prior_std_x[i] * torch.randn(N_SAMPLES, 1, device=device)] for i in range(N)]
    V = [[prior_std_v[i] * torch.randn(N_SAMPLES, 1, device=device)] for i in range(N)]

    for t_idx in reversed(range(1, N_DIFF_STEPS + 1)):
        score_outputs = score_net(X, V, t_idx)
        X, V = anderson_reverse_step_coupled_gamma(
            X, V, K_self, K_global, score_outputs, t_idx, N_DIFF_STEPS,
            [ALPHA_V] * N, [BETA] * N, gamma_self, gamma_couple, coupling, False, DT,
            g_fn(t_idx, N_DIFF_STEPS), time_scale_fn=TIME_SCALE_FN,
        )
    return torch.cat([X[i][0] for i in range(N)], dim=-1)


def train_one_seed(N: int, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma = _get_sigma_n(N)
    gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
    score_net, gamma_self, gamma_couple, coupling, prior_std = train_csho_tikhonov(N, sigma, gt, device)
    generated = sample_csho_anderson(N, sigma, score_net, gamma_self, gamma_couple, coupling, prior_std, device)
    return evaluate_sampling_quality(generated, gt)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_rows = []
    all_rows = []
    per_seed_rows = []
    sig_rows = []

    for N in N_SWEEP:
        with open(f"{BASELINE_DIR}/N{N}_ddpm/ddpm_results.json") as f:
            ddpm_full = json.load(f)
        with open(f"{BASELINE_DIR}/N{N}_sdm/sdm_results.json") as f:
            sdm_full = json.load(f)
        ddpm = ddpm_full["summary"]
        sdm = sdm_full["summary"]
        ddpm_per_seed = {int(s): m for s, m in ddpm_full["per_seed"].items()}
        sdm_per_seed = {int(s): m for s, m in sdm_full["per_seed"].items()}

        for method_name, baseline_per_seed in [("ddpm", ddpm_per_seed), ("sdm", sdm_per_seed)]:
            for seed, m in baseline_per_seed.items():
                per_seed_rows.append({"N": N, "method": method_name, "seed": seed, **m})

        print(f"\n=== N={N} (sigma={_get_sigma_n(N):.4f}, lam={LAM}, beta={BETA}, n_diff_steps={N_DIFF_STEPS}, dt={DT}) ===")
        per_seed = {}
        for seed in SEEDS:
            m = train_one_seed(N, seed)
            per_seed[seed] = m
            per_seed_rows.append({"N": N, "method": "csho_tikhonov", "seed": seed, **m})
            print(f"  seed={seed}: kl={m['kl_divergence']:.4f} corr_gen={m['mean_pairwise_corr_gen']:.4f} corr_true={m['mean_pairwise_corr_true']:.4f}")

        csho_summary = aggregate_over_seeds(per_seed)
        for metric, stats in csho_summary.items():
            all_rows.append({"N": N, "method": "csho_tikhonov", "metric": metric, **stats})

        for method_name, s in [("ddpm", ddpm), ("sdm", sdm), ("csho_tikhonov", csho_summary)]:
            summary_rows.append({
                "N": N, "method": method_name,
                "kl_mean": s["kl_divergence"]["mean"], "kl_std": s["kl_divergence"]["std"],
                "corr_gen_mean": s["mean_pairwise_corr_gen"]["mean"], "corr_gen_std": s["mean_pairwise_corr_gen"]["std"],
                "corr_true": s["mean_pairwise_corr_true"]["mean"],
                "corr_pct_of_true": s["mean_pairwise_corr_gen"]["mean"] / s["mean_pairwise_corr_true"]["mean"] * 100.0,
            })

        sig_metric_names = ["kl_divergence", "wasserstein2", "mi_mae", "mean_pairwise_corr_gen"]
        for baseline_name, baseline_per_seed in [("ddpm", ddpm_per_seed), ("sdm", sdm_per_seed)]:
            sig = compare_configs(baseline_per_seed, per_seed, metric_names=sig_metric_names)
            for metric, s in sig.items():
                sig_rows.append({"N": N, "comparison": f"csho_tikhonov_vs_{baseline_name}", "metric": metric, **s})

    write_csv(per_seed_rows, os.path.join(OUT_DIR, "tikhonov_n_sweep_per_seed.csv"))
    write_csv(sig_rows, os.path.join(OUT_DIR, "tikhonov_n_sweep_significance.csv"))
    write_csv(all_rows, os.path.join(OUT_DIR, "tikhonov_n_sweep_full.csv"))
    write_csv(summary_rows, os.path.join(OUT_DIR, "tikhonov_n_sweep_summary.csv"))
    write_json(
        {"lam": LAM, "beta": BETA, "sigma_by_n": {N: _get_sigma_n(N) for N in N_SWEEP}, "n_sweep": N_SWEEP,
         "n_diff_steps": N_DIFF_STEPS, "dt": DT, "summary_rows": summary_rows},
        os.path.join(OUT_DIR, "tikhonov_n_sweep_results.json"),
    )

    print("\n\n=== SUMMARY: N=2..5, DDPM vs SDM vs CSHO-Tikhonov (Anderson-corrected, lam=0.1) ===")
    print(f"{'N':>3} {'method':>16} {'KL':>10} {'corr_gen':>10} {'corr_true':>10} {'%true':>8}")
    for row in summary_rows:
        print(f"{row['N']:>3} {row['method']:>16} {row['kl_mean']:>10.4f} {row['corr_gen_mean']:>10.4f} {row['corr_true']:>10.4f} {row['corr_pct_of_true']:>8.1f}")

    print(f"\nWrote results to {OUT_DIR}/ (per-seed table for Wilcoxon: tikhonov_n_sweep_per_seed.csv; significance: tikhonov_n_sweep_significance.csv)")
    return summary_rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Champion Tikhonov-DSM CSHO vs. DDPM/SDM, N=2..5")
    p.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated seed list")
    p.add_argument("--n-train-iters", type=int, default=2000)
    p.add_argument("--n-samples", type=int, default=4000, help="evaluation sample count for KL/correlation")
    p.add_argument("--n-sweep", default="2,3,4,5", help="comma-separated N values")
    p.add_argument("--n-diff-steps", type=int, default=20,
                    help="reverse/forward diffusion step count; must match --n-diff-steps passed to "
                         "run_experiment.py for the DDPM/SDM baselines for a fair matched-budget comparison")
    p.add_argument("--dt", type=float, default=None,
                    help="per-step size; defaults to 1/n_diff_steps, holding the total diffusion "
                         "horizon n_diff_steps*dt fixed at 1.0 as n_diff_steps changes")
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/anderson_tikhonov_n_sweep")
    p.add_argument("--baseline-dir", default="results/experiment_3_synthetic/exact_prior_std_sweep",
                    help="directory containing N{n}_ddpm/ddpm_results.json and N{n}_sdm/sdm_results.json "
                         "at a MATCHING seed count and n_train_iters -- for a fair comparison, regenerate "
                         "these at the same budget as this run rather than reusing the original 5-seed/"
                         "2000-iter baselines (see run_final_synthetic_experiments.sh).")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    N_TRAIN_ITERS = _args.n_train_iters
    N_SAMPLES = _args.n_samples
    N_SWEEP = [int(n) for n in _args.n_sweep.split(",") if n.strip()]
    OUT_DIR = _args.out_dir
    BASELINE_DIR = _args.baseline_dir
    N_DIFF_STEPS = _args.n_diff_steps
    DT = _args.dt if _args.dt is not None else 1.0 / N_DIFF_STEPS
    run()

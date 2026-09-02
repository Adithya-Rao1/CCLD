from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import torch
from scipy.linalg import solve_continuous_lyapunov

import synthetic.anderson_tikhonov_n_sweep as sweep
from core.coupling import build_coupling_matrix
from synthetic.anderson_sde import anderson_em_step_coupled_gamma, anderson_reverse_step_coupled_gamma
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas, calibrate_sigma_fdt, calibrate_sigma_fdt_coupled
from synthetic.exact_dsm import _extract_Avx_Avv_coupled_gamma, precompute_transition_params
from synthetic.ground_truth_sde import make_ground_truth
from synthetic.run_experiment import _make_conditioning, evaluate_sampling_quality

N_SWEEP = [2, 3, 5]
N_TRAIN_ITERS_SMOKE = 500
SEEDS_SMOKE = [0, 1, 2]
N_SAMPLES_SMOKE = 2000
N_DIFF_STEPS_BASE = 32
K_REFERENCE = 1.0
TARGET_ZETA = 1.0
COUPLING_STRENGTH = 0.6

results: List[Dict] = []


def _record(group: str, label: str, N: int, corr_gen: float, corr_true: float, kl: float = None, extra: str = ""):
    pct = 100.0 * corr_gen / corr_true if corr_true else float("nan")
    row = {"group": group, "label": label, "N": N, "corr_gen": corr_gen, "corr_true": corr_true,
           "pct_of_true": pct, "kl": kl, "extra": extra}
    results.append(row)
    kl_str = f"kl={kl:.4f} " if kl is not None else ""
    print(f"  [{group}] {label:32s} N={N} corr_gen={corr_gen:.4f} corr_true={corr_true:.4f} "
          f"pct={pct:6.1f}% {kl_str}{extra}")


def _setup_sweep_module(device: str, n_diff_steps: int, dt: float, n_train_iters: int):
    sweep.N_DIFF_STEPS = n_diff_steps
    sweep.DT = dt
    sweep.N_TRAIN_ITERS = n_train_iters
    sweep._sigma_cache.clear()


def _exact_marginal_score(Zt, Phi_t, Sigma_gt, Sigma_t, N, jitter=1e-4):
    device = Zt.device
    Zaug = torch.zeros(2 * N, 2 * N, device=device)
    Zaug[:N, :N] = Sigma_gt
    Sigma_m = Phi_t @ Zaug @ Phi_t.T + Sigma_t
    Sxx, Sxv, Svx, Svv = Sigma_m[:N, :N], Sigma_m[:N, N:], Sigma_m[N:, :N], Sigma_m[N:, N:]
    scale = Sigma_m.diagonal().abs().max().clamp_min(1.0)
    Sxx_inv = torch.linalg.inv(Sxx + jitter * scale * torch.eye(N, device=device))
    mean_v_given_x = Zt[:, :N] @ (Sxx_inv @ Sxv)
    Var_V_given_X = Svv - Svx @ Sxx_inv @ Sxv
    resid = Zt[:, N:] - mean_v_given_x
    reg_precision = torch.linalg.inv(Var_V_given_X + jitter * scale * torch.eye(N, device=device))
    return -resid @ reg_precision.T


def _mean_offdiag_corr(cov: torch.Tensor) -> float:
    N = cov.shape[0]
    std = cov.diagonal().clamp_min(1e-12).sqrt()
    corr = cov / (std[:, None] * std[None, :])
    off = ~torch.eye(N, dtype=torch.bool)
    return corr[off].mean().item()


def test_A1_reproduce(device):
    print("\n=== A1: reproduce the regression via the REAL training pipeline ===")
    _setup_sweep_module(device, N_DIFF_STEPS_BASE, 1.0 / N_DIFF_STEPS_BASE, N_TRAIN_ITERS_SMOKE)
    for N in N_SWEEP:
        sigma_ab = sweep._get_sigma_n(N)
        corr_gens, corr_trues, kls = [], [], []
        for seed in SEEDS_SMOKE:
            torch.manual_seed(seed)
            gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
            score_net, gamma_self, gamma_couple, coupling, prior_std = sweep.train_csho_tikhonov(N, sigma_ab, gt, device)
            sweep.N_SAMPLES = N_SAMPLES_SMOKE
            generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gamma_self, gamma_couple, coupling, prior_std, device)
            m = evaluate_sampling_quality(generated, gt)
            corr_gens.append(m["mean_pairwise_corr_gen"])
            corr_trues.append(m["mean_pairwise_corr_true"])
            kls.append(m["kl_divergence"])
        cg = sum(corr_gens) / len(corr_gens)
        ct = sum(corr_trues) / len(corr_trues)
        kl = sum(kls) / len(kls)
        _record("A1", "current (coupled G, tau_hat=0.5)", N, cg, ct, kl)


def test_A2_exact_score_oracle(device):
    print("\n=== A2: exact-score-oracle bisection (no trained network) ===")
    for N in N_SWEEP:
        coupling = build_coupling_matrix(N, mode="mean_field", device=device)
        gamma_self, gamma_couple = calibrate_coupled_gammas(sweep.ALPHA_V, sweep.BETA, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
        a, b = calibrate_sigma_fdt_coupled(gamma_self, gamma_couple, N)
        g_matrix = sweep._g_fn(N, (a, b), coupling, device)(1, N_DIFF_STEPS_BASE)
        g_fn = lambda t, T: g_matrix
        dt = 1.0 / N_DIFF_STEPS_BASE

        params = precompute_transition_params(
            N, gamma_self, gamma_couple, [sweep.ALPHA_V] * N, [sweep.BETA] * N, K_REFERENCE, coupling,
            N_DIFF_STEPS_BASE, dt, g_fn, True, sweep.TIME_SCALE_FN,
        )
        gt = make_ground_truth(N, COUPLING_STRENGTH, seed=0, base_decay=1.0, sigma_scale=1.0, device=device)
        Sigma_gt = gt.stationary_covariance()

        B = 512
        X0 = gt.sample_stationary(B).to(device)
        X = [[X0[:, i:i + 1].clone()] for i in range(N)]
        V = [[torch.zeros_like(X0[:, i:i + 1])] for i in range(N)]
        K_self_p, K_global_p = _make_conditioning(N, B, K_REFERENCE, device)
        for step in range(1, N_DIFF_STEPS_BASE + 1):
            X, V, _, _, _ = anderson_em_step_coupled_gamma(
                X, V, K_self_p, K_global_p, step, N_DIFF_STEPS_BASE,
                [sweep.ALPHA_V] * N, [sweep.BETA] * N, gamma_self, gamma_couple, coupling, True, dt,
                g_fn(step, N_DIFF_STEPS_BASE), time_scale_fn=sweep.TIME_SCALE_FN,
            )
        prior_std_x = [max(X[i][0].std().item(), 1e-3) for i in range(N)]
        prior_std_v = [max(V[i][0].std().item(), 1e-3) for i in range(N)]

        K_self, K_global = _make_conditioning(N, N_SAMPLES_SMOKE, K_REFERENCE, device)
        Xg = [[prior_std_x[i] * torch.randn(N_SAMPLES_SMOKE, 1, device=device)] for i in range(N)]
        Vg = [[prior_std_v[i] * torch.randn(N_SAMPLES_SMOKE, 1, device=device)] for i in range(N)]
        for t_idx in reversed(range(1, N_DIFF_STEPS_BASE + 1)):
            Phi_t, Sigma_t = params[t_idx - 1]
            Zt = torch.cat([Xg[i][0] for i in range(N)] + [Vg[i][0] for i in range(N)], dim=-1)
            score_v = _exact_marginal_score(Zt, Phi_t, Sigma_gt, Sigma_t, N)
            score_outputs = [[score_v[:, i:i + 1]] for i in range(N)]
            Xg, Vg = anderson_reverse_step_coupled_gamma(
                Xg, Vg, K_self, K_global, score_outputs, t_idx, N_DIFF_STEPS_BASE,
                [sweep.ALPHA_V] * N, [sweep.BETA] * N, gamma_self, gamma_couple, coupling, True, dt,
                g_fn(t_idx, N_DIFF_STEPS_BASE), time_scale_fn=sweep.TIME_SCALE_FN,
            )
        generated = torch.cat([Xg[i][0] for i in range(N)], dim=-1)
        cg = _mean_offdiag_corr(torch.cov(generated.T))
        ct = _mean_offdiag_corr(Sigma_gt)
        _record("A2", "exact score oracle (ceiling)", N, cg, ct)


def test_A3_residual_corr_at_T(device):
    print("\n=== A3: residual correlation at t=T (exact recursion vs Monte Carlo) ===")
    for N in N_SWEEP:
        for beta_val, beta_label in [(1.0, "beta=1.0 (synthetic)"), (0.5, "beta=0.5 (pde)")]:
            coupling = build_coupling_matrix(N, mode="mean_field", device=device)
            gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, beta_val, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
            a, b = calibrate_sigma_fdt_coupled(gamma_self, gamma_couple, N)
            from core.sde import build_g_matrix_n
            g_matrix = build_g_matrix_n(torch.tensor(a, device=device), N, diffusion_mode="shared", coupling_matrix=b * coupling)
            g_fn = lambda t, T: g_matrix
            dt = 1.0 / N_DIFF_STEPS_BASE

            params = precompute_transition_params(
                N, gamma_self, gamma_couple, [1.0] * N, [beta_val] * N, K_REFERENCE, coupling,
                N_DIFF_STEPS_BASE, dt, g_fn, True, sweep.TIME_SCALE_FN,
            )
            Phi_T, Sigma_T = params[-1]
            gt = make_ground_truth(N, COUPLING_STRENGTH, seed=0, base_decay=1.0, sigma_scale=1.0, device=device)
            Sigma_gt = gt.stationary_covariance()
            Zaug = torch.zeros(2 * N, 2 * N, device=device)
            Zaug[:N, :N] = Sigma_gt
            Sigma_marginal_T = Phi_T @ Zaug @ Phi_T.T + Sigma_T
            exact_corr_T = _mean_offdiag_corr(Sigma_marginal_T[:N, :N])

            B = 2000
            X0 = gt.sample_stationary(B).to(device)
            X = [[X0[:, i:i + 1].clone()] for i in range(N)]
            V = [[torch.zeros_like(X0[:, i:i + 1])] for i in range(N)]
            K_self_p, K_global_p = _make_conditioning(N, B, K_REFERENCE, device)
            for step in range(1, N_DIFF_STEPS_BASE + 1):
                X, V, _, _, _ = anderson_em_step_coupled_gamma(
                    X, V, K_self_p, K_global_p, step, N_DIFF_STEPS_BASE,
                    [1.0] * N, [beta_val] * N, gamma_self, gamma_couple, coupling, True, dt,
                    g_fn(step, N_DIFF_STEPS_BASE), time_scale_fn=sweep.TIME_SCALE_FN,
                )
            Xmat = torch.cat([X[i][0] for i in range(N)], dim=1)
            mc_corr = _mean_offdiag_corr(torch.cov(Xmat.T))
            ct = _mean_offdiag_corr(Sigma_gt)
            _record("A3", f"residual@T, {beta_label}", N, exact_corr_T, ct, extra=f"MC_check={mc_corr:.4f}")


def test_A4_stationary_correlation(device):
    print("\n=== A4: stationary (tau_hat->infinity) correlation (continuous Lyapunov) ===")
    for N in N_SWEEP:
        for beta_val in [0.0, 0.5, 1.0]:
            coupling = build_coupling_matrix(N, mode="mean_field", device=device)
            gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, beta_val, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
            a, b = calibrate_sigma_fdt_coupled(gamma_self, gamma_couple, N)
            from core.sde import build_g_matrix_n
            G0 = build_g_matrix_n(torch.tensor(a, device=device), N, diffusion_mode="shared", coupling_matrix=b * coupling).double()
            const_one = lambda t, T: torch.tensor(1.0)
            A_vx0, A_vv0 = _extract_Avx_Avv_coupled_gamma(N, gamma_self, gamma_couple, [1.0] * N, [beta_val] * N, K_REFERENCE, coupling, t=1, T=1, constant_k=True, time_scale_fn=const_one)
            J0 = torch.zeros(2 * N, 2 * N, dtype=torch.float64)
            J0[:N, N:] = torch.eye(N)
            J0[N:, :N] = A_vx0.double()
            J0[N:, N:] = A_vv0.double()
            L0 = torch.zeros(2 * N, N, dtype=torch.float64)
            L0[N:, :] = G0
            Q = L0 @ L0.T
            Sigma_inf = torch.from_numpy(solve_continuous_lyapunov(J0.numpy(), -Q.numpy()))
            corr_inf = _mean_offdiag_corr(Sigma_inf[:N, :N])
            print(f"  [A4] N={N} beta={beta_val}: stationary correlation = {corr_inf:.6f}"
                  f"  {'(expect exactly 0)' if beta_val == 0.0 else ''}")


def test_A5_isolate_elapsed_time(device):
    print("\n=== A5: isolate elapsed-time effect (diagonal-only G at actual tau_hat) ===")
    rho_true = 0.6
    for N in N_SWEEP:
        gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, 1.0, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
        sigma_flat = calibrate_sigma_fdt(gamma_self)
        coupling = build_coupling_matrix(N, mode="mean_field", device=device)
        from synthetic.exact_dsm import closed_form_propagator, tau_hat_vp_linear_schedule
        tau_hat = tau_hat_vp_linear_schedule(1, 1.0)
        Phi, Sigma = closed_form_propagator(N, gamma_self, gamma_couple, [1.0] * N, [1.0] * N, K_REFERENCE, coupling, tau_hat, constant_k=True, sigma_ref=sigma_flat, dtype=torch.float64)
        Phi_x = Phi[:N, :N].to(device)
        gt = make_ground_truth(N, COUPLING_STRENGTH, seed=0, base_decay=1.0, sigma_scale=1.0, device=device)
        Sigma_gt = gt.stationary_covariance().double()
        Sigma_marginal = Phi_x @ Sigma_gt @ Phi_x.T + Sigma[:N, :N].to(device)
        residual_corr = _mean_offdiag_corr(Sigma_marginal)
        ct = _mean_offdiag_corr(Sigma_gt)
        print(f"  [A5] N={N}: diagonal-G residual corr at actual tau_hat={tau_hat:.4f} = {residual_corr:.4f}"
              f"  (true={ct:.4f}) -- {'barely decorrelated' if residual_corr > 0.5 * ct else 'meaningfully decorrelated'}")


def test_B1_diagonal_diffusion(device):
    print("\n=== B1: candidate fix -- diagonal-only FDT diffusion (b=0) ===")
    _setup_sweep_module(device, N_DIFF_STEPS_BASE, 1.0 / N_DIFF_STEPS_BASE, N_TRAIN_ITERS_SMOKE)
    for N in N_SWEEP:
        gamma_self, gamma_couple = calibrate_coupled_gammas(sweep.ALPHA_V, sweep.BETA, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
        sigma_ab = (calibrate_sigma_fdt(gamma_self), 0.0)
        corr_gens, corr_trues, kls = [], [], []
        for seed in SEEDS_SMOKE:
            torch.manual_seed(seed)
            gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
            score_net, gs, gc, coupling, prior_std = sweep.train_csho_tikhonov(N, sigma_ab, gt, device)
            sweep.N_SAMPLES = N_SAMPLES_SMOKE
            generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gs, gc, coupling, prior_std, device)
            m = evaluate_sampling_quality(generated, gt)
            corr_gens.append(m["mean_pairwise_corr_gen"]); corr_trues.append(m["mean_pairwise_corr_true"]); kls.append(m["kl_divergence"])
        _record("B1", "diagonal G (b=0), tau_hat=0.5", N, sum(corr_gens) / len(corr_gens), sum(corr_trues) / len(corr_trues), sum(kls) / len(kls))


def test_B2_increased_elapsed_time(device):
    print("\n=== B2: candidate fix -- increased elapsed time (dt fixed, more steps) ===")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    for duration_mult in [1, 2, 4]:
        n_diff_steps = N_DIFF_STEPS_BASE * duration_mult
        _setup_sweep_module(device, n_diff_steps, dt_fixed, N_TRAIN_ITERS_SMOKE)
        for N in N_SWEEP:
            sigma_ab = sweep._get_sigma_n(N)
            corr_gens, corr_trues, kls = [], [], []
            for seed in SEEDS_SMOKE:
                torch.manual_seed(seed)
                gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
                score_net, gs, gc, coupling, prior_std = sweep.train_csho_tikhonov(N, sigma_ab, gt, device)
                sweep.N_SAMPLES = N_SAMPLES_SMOKE
                generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gs, gc, coupling, prior_std, device)
                m = evaluate_sampling_quality(generated, gt)
                corr_gens.append(m["mean_pairwise_corr_gen"]); corr_trues.append(m["mean_pairwise_corr_true"]); kls.append(m["kl_divergence"])
            _record("B2", f"coupled G, tau_hat~{0.5 * duration_mult:.2f} ({duration_mult}x)", N,
                    sum(corr_gens) / len(corr_gens), sum(corr_trues) / len(corr_trues), sum(kls) / len(kls))


def test_B3_combined(device):
    print("\n=== B3: candidate fix -- combined (diagonal G + increased elapsed time) ===")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    for duration_mult in [1, 4]:
        n_diff_steps = N_DIFF_STEPS_BASE * duration_mult
        _setup_sweep_module(device, n_diff_steps, dt_fixed, N_TRAIN_ITERS_SMOKE)
        for N in N_SWEEP:
            gamma_self, gamma_couple = calibrate_coupled_gammas(sweep.ALPHA_V, sweep.BETA, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
            sigma_ab = (calibrate_sigma_fdt(gamma_self), 0.0)
            corr_gens, corr_trues, kls = [], [], []
            for seed in SEEDS_SMOKE:
                torch.manual_seed(seed)
                gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
                score_net, gs, gc, coupling, prior_std = sweep.train_csho_tikhonov(N, sigma_ab, gt, device)
                sweep.N_SAMPLES = N_SAMPLES_SMOKE
                generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gs, gc, coupling, prior_std, device)
                m = evaluate_sampling_quality(generated, gt)
                corr_gens.append(m["mean_pairwise_corr_gen"]); corr_trues.append(m["mean_pairwise_corr_true"]); kls.append(m["kl_divergence"])
            _record("B3", f"diagonal G, tau_hat~{0.5 * duration_mult:.2f} ({duration_mult}x)", N,
                    sum(corr_gens) / len(corr_gens), sum(corr_trues) / len(corr_trues), sum(kls) / len(kls))


def print_summary():
    print("\n" + "=" * 100)
    print("SUMMARY (all groups)")
    print("=" * 100)
    print(f"{'group':6s} {'label':38s} {'N':>3s} {'corr_gen':>9s} {'corr_true':>10s} {'pct':>7s} {'kl':>8s}")
    for r in results:
        kl_str = f"{r['kl']:.4f}" if r["kl"] is not None else "  --  "
        print(f"{r['group']:6s} {r['label']:38s} {r['N']:>3d} {r['corr_gen']:>9.4f} {r['corr_true']:>10.4f} {r['pct_of_true']:>6.1f}% {kl_str:>8s}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--skip-training", action="store_true",
                    help="skip the training-based tests (A1, B1, B2, B3) and run only the fast algebra-only checks (A2-A5)")
    args = p.parse_args()
    device = torch.device(args.device)

    t0 = time.time()
    test_A4_stationary_correlation(device)
    test_A3_residual_corr_at_T(device)
    test_A5_isolate_elapsed_time(device)
    test_A2_exact_score_oracle(device)
    if not args.skip_training:
        test_A1_reproduce(device)
        test_B1_diagonal_diffusion(device)
        test_B2_increased_elapsed_time(device)
        test_B3_combined(device)
    print_summary()
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

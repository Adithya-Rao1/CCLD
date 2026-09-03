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
from synthetic.metrics import fit_gaussian
from synthetic.run_experiment import CoupledScoreNet, _make_conditioning, evaluate_sampling_quality

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


def _setup_sweep_module(device: str, n_diff_steps: int, dt: float, n_train_iters: int, time_scale_fn=None):
    sweep.N_DIFF_STEPS = n_diff_steps
    sweep.DT = dt
    sweep.N_TRAIN_ITERS = n_train_iters
    sweep.TIME_SCALE_FN = time_scale_fn if time_scale_fn is not None else sweep._vp_linear_time_scale
    sweep._sigma_cache.clear()


def _constant_time_scale(c: float):
    def time_scale_fn(t, T):
        return torch.as_tensor(c, dtype=torch.float32)
    return time_scale_fn


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


def test_B4_free_dt_rescale(device):
    print("\n=== B4: candidate fix -- 'free' tau rescale (same steps, dt x4, zero extra compute) ===")
    dt_free = 4.0 / N_DIFF_STEPS_BASE 
    _setup_sweep_module(device, N_DIFF_STEPS_BASE, dt_free, N_TRAIN_ITERS_SMOKE)
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
        _record("B4", "coupled G, tau_hat~2.00 (free dt x4)", N,
                sum(corr_gens) / len(corr_gens), sum(corr_trues) / len(corr_trues), sum(kls) / len(kls))


def test_C1_constant_tau(device):
    print("\n=== C1: curiosity -- constant tau(t)=c schedule (vs vp_linear ramp) ===")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    for c in [3.0, 4.0]:
        _setup_sweep_module(device, N_DIFF_STEPS_BASE, dt_fixed, N_TRAIN_ITERS_SMOKE, time_scale_fn=_constant_time_scale(c))
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
            tau_hat_total = c * (N_DIFF_STEPS_BASE * dt_fixed)
            _record("C1", f"constant tau={c}, tau_hat_total={tau_hat_total:.2f}", N,
                    sum(corr_gens) / len(corr_gens), sum(corr_trues) / len(corr_trues), sum(kls) / len(kls))


def test_C2_full_scale_verification(device, tau_values=(1.5, 2.0)):
    print(f"\n=== C2: full-scale verification (production N_TRAIN_ITERS=2000, N_SAMPLES=4000, 5 seeds) ===")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    seeds_full = [0, 1, 2, 3, 4]
    n_train_iters_full = 2000
    n_samples_full = 4000
    for c in tau_values:
        _setup_sweep_module(device, N_DIFF_STEPS_BASE, dt_fixed, n_train_iters_full, time_scale_fn=_constant_time_scale(c))
        for N in N_SWEEP:
            sigma_ab = sweep._get_sigma_n(N)
            corr_gens, corr_trues, kls = [], [], []
            for seed in seeds_full:
                torch.manual_seed(seed)
                gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
                score_net, gs, gc, coupling, prior_std = sweep.train_csho_tikhonov(N, sigma_ab, gt, device)
                sweep.N_SAMPLES = n_samples_full
                generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gs, gc, coupling, prior_std, device)
                m = evaluate_sampling_quality(generated, gt)
                corr_gens.append(m["mean_pairwise_corr_gen"]); corr_trues.append(m["mean_pairwise_corr_true"]); kls.append(m["kl_divergence"])
            tau_hat_total = c * (N_DIFF_STEPS_BASE * dt_fixed)
            cg_mean = sum(corr_gens) / len(corr_gens)
            ct_mean = sum(corr_trues) / len(corr_trues)
            kl_mean = sum(kls) / len(kls)
            kl_std = (sum((k - kl_mean) ** 2 for k in kls) / len(kls)) ** 0.5
            _record("C2", f"[FULL SCALE] constant tau={c}, tau_hat={tau_hat_total:.2f}", N, cg_mean, ct_mean, kl_mean,
                    extra=f"kl_std={kl_std:.4f} (n=5 seeds)")


def test_D1_kl_decomposition(device, tau=2.0):
    print(f"\n=== D1: KL decomposition (constant tau={tau}) — is the residual KL variance-scale or correlation? ===")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    _setup_sweep_module(device, N_DIFF_STEPS_BASE, dt_fixed, N_TRAIN_ITERS_SMOKE, time_scale_fn=_constant_time_scale(tau))
    for N in N_SWEEP:
        sigma_ab = sweep._get_sigma_n(N)
        rows = []
        for seed in SEEDS_SMOKE:
            torch.manual_seed(seed)
            gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
            score_net, gs, gc, coupling, prior_std = sweep.train_csho_tikhonov(N, sigma_ab, gt, device)
            sweep.N_SAMPLES = N_SAMPLES_SMOKE
            generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gs, gc, coupling, prior_std, device)

            mean_gen, cov_gen = fit_gaussian(generated.cpu())
            mean_true = torch.zeros(N, dtype=cov_gen.dtype)
            cov_true = gt.stationary_covariance().cpu()

            eps = 1e-6
            I = torch.eye(N, dtype=cov_gen.dtype)
            cov_p = (cov_gen + cov_gen.T) / 2 + eps * I
            cov_q = (cov_true + cov_true.T) / 2 + eps * I
            cov_q_inv = torch.linalg.inv(cov_q)
            diff = (mean_true - mean_gen).reshape(-1, 1)
            trace_term = torch.trace(cov_q_inv @ cov_p).item()
            mean_term = (diff.T @ cov_q_inv @ diff).squeeze().item()
            logdet_q = torch.linalg.slogdet(cov_q)[1].item()
            logdet_p = torch.linalg.slogdet(cov_p)[1].item()
            kl = 0.5 * (trace_term + mean_term - N + logdet_q - logdet_p)

            std_gen = cov_gen.diagonal().clamp_min(1e-12).sqrt()
            std_true = cov_true.diagonal().clamp_min(1e-12).sqrt()
            var_ratio = (std_gen / std_true).mean().item()

            rows.append((trace_term, mean_term, logdet_q - logdet_p, kl, var_ratio))

        n_rows = len(rows)
        trace_m = sum(r[0] for r in rows) / n_rows
        mean_m = sum(r[1] for r in rows) / n_rows
        logdet_m = sum(r[2] for r in rows) / n_rows
        kl_m = sum(r[3] for r in rows) / n_rows
        var_ratio_m = sum(r[4] for r in rows) / n_rows
        print(f"  [D1] N={N}: kl={kl_m:.4f}  trace_term={trace_m:.4f} (k={N})  mean_term={mean_m:.4f}  "
              f"logdet(q)-logdet(p)={logdet_m:.4f}  std_gen/std_true={var_ratio_m:.4f}")
        _record("D1", f"KL decomposition (tau={tau})", N, 0.0, 0.0, kl_m,
                extra=f"trace={trace_m:.3f} mean_term={mean_m:.3f} logdet_diff={logdet_m:.3f} var_ratio={var_ratio_m:.3f}")


def test_D2_score_error_vs_t(device, tau=2.0, n_sweep=None):
    print(f"\n=== D2: trained score error vs. exact marginal score, as a function of diffusion step (tau={tau}) ===")
    print("    (tests whether error concentrates near t_idx->0, i.e. Tikhonov under-strength near the singularity)")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    _setup_sweep_module(device, N_DIFF_STEPS_BASE, dt_fixed, N_TRAIN_ITERS_SMOKE, time_scale_fn=_constant_time_scale(tau))
    for N in (n_sweep if n_sweep is not None else N_SWEEP):
        sigma_ab = sweep._get_sigma_n(N)
        coupling = build_coupling_matrix(N, mode="mean_field", device=device)
        gamma_self, gamma_couple = calibrate_coupled_gammas(sweep.ALPHA_V, sweep.BETA, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
        g_fn = sweep._g_fn(N, sigma_ab, coupling, device)
        params = precompute_transition_params(
            N, gamma_self, gamma_couple, [sweep.ALPHA_V] * N, [sweep.BETA] * N, K_REFERENCE, coupling,
            N_DIFF_STEPS_BASE, dt_fixed, g_fn, True, sweep.TIME_SCALE_FN,
        )
        gt = make_ground_truth(N, COUPLING_STRENGTH, seed=0, base_decay=1.0, sigma_scale=1.0, device=device)
        Sigma_gt = gt.stationary_covariance()

        torch.manual_seed(0)
        score_net, gs, gc, coupling_trained, prior_std = sweep.train_csho_tikhonov(N, sigma_ab, gt, device)

        B = 1024
        rows = []
        for t_idx in range(1, N_DIFF_STEPS_BASE + 1):
            Phi_t, Sigma_t = params[t_idx - 1]
            Zaug = torch.zeros(2 * N, 2 * N, device=device)
            Zaug[:N, :N] = Sigma_gt
            Sigma_m = Phi_t @ Zaug @ Phi_t.T + Sigma_t
            Sigma_m = (Sigma_m + Sigma_m.T) / 2 + 1e-8 * torch.eye(2 * N, device=device)
            L = torch.linalg.cholesky(Sigma_m)
            Zt = torch.randn(B, 2 * N, device=device) @ L.T

            X_t = [[Zt[:, i:i + 1]] for i in range(N)]
            V_t = [[Zt[:, N + i:N + i + 1]] for i in range(N)]
            with torch.no_grad():
                score_pred_raw = score_net(X_t, V_t, t_idx)
            score_pred = torch.cat([score_pred_raw[i][0] for i in range(N)], dim=-1)
            score_exact = _exact_marginal_score(Zt, Phi_t, Sigma_gt, Sigma_t, N)

            rel_err = ((score_pred - score_exact).norm(dim=-1) / score_exact.norm(dim=-1).clamp_min(1e-8)).mean().item()
            mag_ratio = (score_pred.norm(dim=-1).mean() / score_exact.norm(dim=-1).mean().clamp_min(1e-8)).item()
            tau_hat_t = tau * t_idx * dt_fixed
            rows.append((t_idx, tau_hat_t, rel_err, mag_ratio))

        print(f"  N={N}:")
        for t_idx, tau_hat_t, rel_err, mag_ratio in rows:
            if t_idx in (1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 31, 32):
                print(f"    t_idx={t_idx:3d}  tau_hat_t={tau_hat_t:.3f}  rel_err={rel_err:.3f}  "
                      f"|pred|/|exact|={mag_ratio:.3f}")
        near_zero = [r for r in rows if r[0] <= 4]
        far = [r for r in rows if r[0] > N_DIFF_STEPS_BASE - 4]
        mag_near0 = sum(r[3] for r in near_zero) / len(near_zero)
        mag_far = sum(r[3] for r in far) / len(far)
        _record("D2", f"score mag ratio (tau={tau})", N, mag_near0, mag_far,
                extra=f"|pred|/|exact| near t_idx=0 (last few steps): {mag_near0:.3f}, near t_idx=T (first few steps): {mag_far:.3f}")


def _sample_and_score_target_custom_lam(Z0, Phi_t, Sigma_t, N, lam_fn, jitter=1e-6):
    B = Z0.shape[0]
    device = Z0.device
    mean = Z0 @ Phi_t.T
    scale = Sigma_t.diagonal().abs().max().clamp_min(1.0)
    Sigma_reg = Sigma_t + jitter * scale * torch.eye(2 * N, device=device)
    L = torch.linalg.cholesky(Sigma_reg)
    eps = torch.randn(B, 2 * N, device=device)
    Zt = mean + eps @ L.T

    Sxx, Sxv, Svx, Svv = Sigma_t[:N, :N], Sigma_t[:N, N:], Sigma_t[N:, :N], Sigma_t[N:, N:]
    Sxx_inv = torch.linalg.inv(Sxx + jitter * scale * torch.eye(N, device=device))
    Var_V_given_X = Svv - Svx @ Sxx_inv @ Sxv

    diff_x = Zt[:, :N] - mean[:, :N]
    diff_v = Zt[:, N:] - mean[:, N:]
    E_V_given_X_dev = diff_x @ (Sxx_inv @ Sxv).clone()
    resid = diff_v - E_V_given_X_dev

    lam = lam_fn(Var_V_given_X)
    reg_precision = torch.linalg.inv(Var_V_given_X + lam * torch.eye(N, device=device))
    score_v_reg = -resid @ reg_precision.T
    return Zt, score_v_reg


def _fixed_lam(v):
    return lambda Var_V_given_X: v


def _relative_lam(eps_rel, floor=1e-4):
    def f(Var_V_given_X):
        N = Var_V_given_X.shape[0]
        return max(eps_rel * (Var_V_given_X.trace().item() / N), floor)
    return f


LAM_SCHEMES = {
    "fixed_0.1 (current)": _fixed_lam(0.1),
    "fixed_0.03": _fixed_lam(0.03),
    "fixed_0.01": _fixed_lam(0.01),
    "relative_eps0.05": _relative_lam(0.05),
    "relative_eps0.1": _relative_lam(0.1),
}


def _train_csho_custom_lam(N, sigma_ab, gt, device, lam_fn, n_train_iters):
    coupling = build_coupling_matrix(N, mode="mean_field", device=device)
    gamma_self, gamma_couple = calibrate_coupled_gammas(sweep.ALPHA_V, sweep.BETA, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)
    g_fn = sweep._g_fn(N, sigma_ab, coupling, device)
    params = precompute_transition_params(
        N, gamma_self, gamma_couple, [sweep.ALPHA_V] * N, [sweep.BETA] * N, K_REFERENCE, coupling,
        N_DIFF_STEPS_BASE, 1.0 / N_DIFF_STEPS_BASE, g_fn, True, sweep.TIME_SCALE_FN,
    )

    score_net = CoupledScoreNet(N, 64, 3, 16).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-3)

    for _ in range(n_train_iters):
        X0 = gt.sample_stationary(sweep.BATCH_SIZE).to(device)
        Z0 = torch.cat([X0, torch.zeros_like(X0)], dim=1)
        t_idx = torch.randint(1, N_DIFF_STEPS_BASE + 1, (1,)).item()
        Phi_t, Sigma_t = params[t_idx - 1]

        Zt, score_v_target = _sample_and_score_target_custom_lam(Z0, Phi_t, Sigma_t, N, lam_fn)
        X_t = [[Zt[:, i:i + 1]] for i in range(N)]
        V_t = [[Zt[:, N + i:N + i + 1]] for i in range(N)]

        score_pred = score_net(X_t, V_t, t_idx)
        loss = sum(((score_pred[i][0] - score_v_target[:, i:i + 1]) ** 2).mean() for i in range(N))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
        optimizer.step()

    prior_std = sweep._estimate_prior_std_anderson(N, gt, coupling, gamma_self, gamma_couple, g_fn, device)
    return score_net, gamma_self, gamma_couple, coupling, prior_std


def test_E1_lam_sweep(device, tau=2.0):
    print(f"\n=== E1: Tikhonov regularization scheme sweep (constant tau={tau}) ===")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    _setup_sweep_module(device, N_DIFF_STEPS_BASE, dt_fixed, N_TRAIN_ITERS_SMOKE, time_scale_fn=_constant_time_scale(tau))
    for scheme_name, lam_fn in LAM_SCHEMES.items():
        for N in N_SWEEP:
            sigma_ab = sweep._get_sigma_n(N)
            corr_gens, corr_trues, kls, var_ratios = [], [], [], []
            n_nonfinite = 0
            for seed in SEEDS_SMOKE:
                torch.manual_seed(seed)
                gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
                score_net, gs, gc, coupling, prior_std = _train_csho_custom_lam(N, sigma_ab, gt, device, lam_fn, N_TRAIN_ITERS_SMOKE)
                sweep.N_SAMPLES = N_SAMPLES_SMOKE
                generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gs, gc, coupling, prior_std, device)
                if not torch.isfinite(generated).all():
                    n_nonfinite += 1
                    continue
                m = evaluate_sampling_quality(generated, gt)
                corr_gens.append(m["mean_pairwise_corr_gen"])
                corr_trues.append(m["mean_pairwise_corr_true"])
                kls.append(m["kl_divergence"])
                mean_gen, cov_gen = fit_gaussian(generated.cpu())
                cov_true = gt.stationary_covariance().cpu()
                std_gen = cov_gen.diagonal().clamp_min(1e-12).sqrt()
                std_true = cov_true.diagonal().clamp_min(1e-12).sqrt()
                var_ratios.append((std_gen / std_true).mean().item())

            if not corr_gens:
                print(f"  [E1] {scheme_name:20s} N={N}: ALL {len(SEEDS_SMOKE)} SEEDS NON-FINITE (unstable)")
                continue

            cg = sum(corr_gens) / len(corr_gens)
            ct = sum(corr_trues) / len(corr_trues)
            kl = sum(kls) / len(kls)
            vr = sum(var_ratios) / len(var_ratios)
            unstable_note = f" [{n_nonfinite}/{len(SEEDS_SMOKE)} seeds non-finite]" if n_nonfinite else ""
            _record("E1", f"{scheme_name}", N, cg, ct, kl, extra=f"var_ratio={vr:.3f}{unstable_note}")


def test_E1b_lam_lower_sweep(device, tau=2.0, lam_values=(0.01, 0.003, 0.001, 0.0003, 0.0001)):
    print(f"\n=== E1b: pushing fixed lam lower to find the instability floor (constant tau={tau}) ===")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    _setup_sweep_module(device, N_DIFF_STEPS_BASE, dt_fixed, N_TRAIN_ITERS_SMOKE, time_scale_fn=_constant_time_scale(tau))
    for lam_value in lam_values:
        lam_fn = _fixed_lam(lam_value)
        scheme_name = f"fixed_{lam_value}"
        for N in N_SWEEP:
            sigma_ab = sweep._get_sigma_n(N)
            corr_gens, corr_trues, kls, var_ratios = [], [], [], []
            n_nonfinite = 0
            for seed in SEEDS_SMOKE:
                torch.manual_seed(seed)
                gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
                score_net, gs, gc, coupling, prior_std = _train_csho_custom_lam(N, sigma_ab, gt, device, lam_fn, N_TRAIN_ITERS_SMOKE)
                sweep.N_SAMPLES = N_SAMPLES_SMOKE
                generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gs, gc, coupling, prior_std, device)
                if not torch.isfinite(generated).all():
                    n_nonfinite += 1
                    continue
                m = evaluate_sampling_quality(generated, gt)
                corr_gens.append(m["mean_pairwise_corr_gen"])
                corr_trues.append(m["mean_pairwise_corr_true"])
                kls.append(m["kl_divergence"])
                mean_gen, cov_gen = fit_gaussian(generated.cpu())
                cov_true = gt.stationary_covariance().cpu()
                std_gen = cov_gen.diagonal().clamp_min(1e-12).sqrt()
                std_true = cov_true.diagonal().clamp_min(1e-12).sqrt()
                var_ratios.append((std_gen / std_true).mean().item())

            if not corr_gens:
                print(f"  [E1b] {scheme_name:20s} N={N}: ALL {len(SEEDS_SMOKE)} SEEDS NON-FINITE (unstable)")
                _record("E1b", scheme_name, N, 0.0, 0.0, None, extra="ALL SEEDS NON-FINITE (unstable)")
                continue

            cg = sum(corr_gens) / len(corr_gens)
            ct = sum(corr_trues) / len(corr_trues)
            kl = sum(kls) / len(kls)
            vr = sum(var_ratios) / len(var_ratios)
            unstable_note = f" [{n_nonfinite}/{len(SEEDS_SMOKE)} seeds non-finite]" if n_nonfinite else ""
            _record("E1b", scheme_name, N, cg, ct, kl, extra=f"var_ratio={vr:.3f}{unstable_note}")


def test_E2_full_scale_lam_verification(device, lam_values=(0.01,), tau=2.0):
    print(f"\n=== E2: full-scale verification of winning lam (production N_TRAIN_ITERS=2000, N_SAMPLES=4000, 5 seeds, tau={tau}) ===")
    dt_fixed = 1.0 / N_DIFF_STEPS_BASE
    seeds_full = [0, 1, 2, 3, 4]
    n_train_iters_full = 2000
    n_samples_full = 4000
    _setup_sweep_module(device, N_DIFF_STEPS_BASE, dt_fixed, n_train_iters_full, time_scale_fn=_constant_time_scale(tau))
    for lam_value in lam_values:
        lam_fn = _fixed_lam(lam_value)
        scheme_name = f"[FULL SCALE] fixed_{lam_value}"
        for N in N_SWEEP:
            sigma_ab = sweep._get_sigma_n(N)
            corr_gens, corr_trues, kls, var_ratios = [], [], [], []
            n_nonfinite = 0
            for seed in seeds_full:
                torch.manual_seed(seed)
                gt = make_ground_truth(N, COUPLING_STRENGTH, seed, 1.0, 1.0, device=device)
                score_net, gs, gc, coupling, prior_std = _train_csho_custom_lam(N, sigma_ab, gt, device, lam_fn, n_train_iters_full)
                sweep.N_SAMPLES = n_samples_full
                generated = sweep.sample_csho_anderson(N, sigma_ab, score_net, gs, gc, coupling, prior_std, device)
                if not torch.isfinite(generated).all():
                    n_nonfinite += 1
                    continue
                m = evaluate_sampling_quality(generated, gt)
                corr_gens.append(m["mean_pairwise_corr_gen"])
                corr_trues.append(m["mean_pairwise_corr_true"])
                kls.append(m["kl_divergence"])
                mean_gen, cov_gen = fit_gaussian(generated.cpu())
                cov_true = gt.stationary_covariance().cpu()
                std_gen = cov_gen.diagonal().clamp_min(1e-12).sqrt()
                std_true = cov_true.diagonal().clamp_min(1e-12).sqrt()
                var_ratios.append((std_gen / std_true).mean().item())

            if not corr_gens:
                print(f"  [E2] {scheme_name:30s} N={N}: ALL {len(seeds_full)} SEEDS NON-FINITE (unstable)")
                _record("E2", scheme_name, N, 0.0, 0.0, None, extra="ALL SEEDS NON-FINITE (unstable)")
                continue

            cg = sum(corr_gens) / len(corr_gens)
            ct = sum(corr_trues) / len(corr_trues)
            kl_mean = sum(kls) / len(kls)
            kl_std = (sum((k - kl_mean) ** 2 for k in kls) / len(kls)) ** 0.5
            vr = sum(var_ratios) / len(var_ratios)
            unstable_note = f" [{n_nonfinite}/{len(seeds_full)} seeds non-finite]" if n_nonfinite else ""
            _record("E2", scheme_name, N, cg, ct, kl_mean,
                    extra=f"kl_std={kl_std:.4f} var_ratio={vr:.3f} (n={len(seeds_full)} seeds){unstable_note}")


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
                    help="skip the training-based tests (A1, B1-B4, C1) and run only the fast algebra-only checks (A2-A5)")
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
        test_B4_free_dt_rescale(device)
        test_C1_constant_tau(device)
    print_summary()
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

from __future__ import annotations

import math

import torch

from core.coupling import build_coupling_matrix
from core.sde import build_g_matrix_n
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas
from synthetic.exact_dsm import (
    closed_form_propagator,
    precompute_transition_params,
    tau_hat_default_schedule,
    tau_hat_vp_linear_schedule,
)

torch.set_default_dtype(torch.float64)


def _vp_linear_time_scale(t, T):
    return torch.as_tensor(t, dtype=torch.float32) / float(T)


def _reference_fine_dt_rollout(N, A_vx0, A_vv0, G0, tau_fn, T_total, n_steps):
    dt = T_total / n_steps
    Phi = torch.eye(2 * N)
    Sigma = torch.zeros(2 * N, 2 * N)
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        tau = float(tau_fn(t_mid, T_total))
        Jt = torch.zeros(2 * N, 2 * N)
        Jt[:N, N:] = tau * torch.eye(N)
        Jt[N:, :N] = tau * A_vx0
        Jt[N:, N:] = tau * A_vv0
        M = torch.eye(2 * N) + dt * Jt
        L = torch.zeros(2 * N, N)
        L[N:, :] = math.sqrt(max(tau, 0.0)) * G0
        Nmat = math.sqrt(dt) * L
        Phi = M @ Phi
        Sigma = M @ Sigma @ M.T + Nmat @ Nmat.T
    return Phi, Sigma


def test_closed_form_matches_fine_dt_reference():
    for N in (1, 2, 3):
        coupling = build_coupling_matrix(N, mode="mean_field") if N > 1 else None
        alpha = [1.0] * N
        beta = [0.5] * N
        k_reference = 1.0
        gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, 0.5, k_reference, k_reference, N)
        T_total = 20.0
        for constant_k in (True,):
            for tau_fn, tau_hat_fn, label in (
                (lambda t, T: (T - t) / (t + T), tau_hat_default_schedule, "default"),
                (_vp_linear_time_scale, tau_hat_vp_linear_schedule, "vp_linear"),
            ):
                A_vx0, A_vv0 = _extract_A0(N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, constant_k)
                Phi_ref, Sigma_ref = _reference_fine_dt_rollout(N, A_vx0, A_vv0, torch.eye(N), tau_fn, T_total, n_steps=40000)
                tau_hat = tau_hat_fn(T_total, 1.0)
                Phi_cf, Sigma_cf = closed_form_propagator(
                    N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, tau_hat,
                    constant_k=constant_k, sigma_ref=1.0, dtype=torch.float64,
                )
                phi_err = ((Phi_cf - Phi_ref).norm() / Phi_ref.norm().clamp_min(1e-8)).item()
                sigma_err = ((Sigma_cf - Sigma_ref).norm() / Sigma_ref.norm().clamp_min(1e-8)).item()
                assert phi_err < 5e-3, f"N={N} constant_k={constant_k} {label}: Phi rel_err={phi_err:.2e}"
                assert sigma_err < 5e-3, f"N={N} constant_k={constant_k} {label}: Sigma rel_err={sigma_err:.2e}"


def _extract_A0(N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, constant_k):
    from synthetic.exact_dsm import _extract_Avx_Avv_coupled_gamma
    const_one = lambda t, T: torch.tensor(1.0)
    return _extract_Avx_Avv_coupled_gamma(
        N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling,
        t=1, T=1, constant_k=constant_k, time_scale_fn=const_one,
    )


def test_closed_form_matches_production_discrete_composition_in_fine_dt_limit():
    N = 2
    coupling = build_coupling_matrix(N, mode="mean_field")
    alpha = [1.0] * N
    beta = [0.5] * N
    k_reference = 1.0
    gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, 0.5, k_reference, k_reference, N)

    for time_scale_fn, tau_hat_fn, label in (
        (None, tau_hat_default_schedule, "default"),
        (_vp_linear_time_scale, tau_hat_vp_linear_schedule, "vp_linear"),
    ):
        n_diff_steps = 5000
        dt = 1.0 / n_diff_steps
        g_fn = lambda t, T: build_g_matrix_n(torch.tensor(1.0), N, diffusion_mode="shared").double()
        params = precompute_transition_params(
            N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling,
            n_diff_steps, dt, g_fn, constant_k=True, time_scale_fn=time_scale_fn,
        )
        Phi_disc, Sigma_disc = params[-1]
        tau_hat = tau_hat_fn(n_diff_steps, dt)
        Phi_cf, Sigma_cf = closed_form_propagator(
            N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, tau_hat,
            constant_k=True, sigma_ref=1.0, dtype=torch.float64,
        )
        phi_err = ((Phi_cf - Phi_disc).norm() / Phi_disc.norm().clamp_min(1e-8)).item()
        sigma_err = ((Sigma_cf - Sigma_disc).norm() / Sigma_disc.norm().clamp_min(1e-8)).item()
        assert phi_err < 1e-2, f"{label}: Phi rel_err={phi_err:.2e} vs production discrete composition"
        assert sigma_err < 1e-2, f"{label}: Sigma rel_err={sigma_err:.2e} vs production discrete composition"


def test_closed_form_matches_production_discrete_composition_with_coupled_G0():
    from synthetic.drift_coupled_gamma import calibrate_sigma_fdt_coupled

    N = 2
    coupling = build_coupling_matrix(N, mode="mean_field")
    alpha = [1.0] * N
    beta = [0.5] * N
    k_reference = 1.0
    gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, 0.5, k_reference, k_reference, N)
    a, b = calibrate_sigma_fdt_coupled(gamma_self, gamma_couple, N)
    G0 = build_g_matrix_n(torch.tensor(a), N, diffusion_mode="shared", coupling_matrix=b * coupling).double()

    for time_scale_fn, tau_hat_fn, label in (
        (None, tau_hat_default_schedule, "default"),
        (_vp_linear_time_scale, tau_hat_vp_linear_schedule, "vp_linear"),
    ):
        n_diff_steps = 5000
        dt = 1.0 / n_diff_steps
        g_fn = lambda t, T: G0
        params = precompute_transition_params(
            N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling,
            n_diff_steps, dt, g_fn, constant_k=True, time_scale_fn=time_scale_fn,
        )
        Phi_disc, Sigma_disc = params[-1]
        tau_hat = tau_hat_fn(n_diff_steps, dt)
        Phi_cf, Sigma_cf = closed_form_propagator(
            N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, tau_hat,
            constant_k=True, dtype=torch.float64, G0=G0,
        )
        phi_err = ((Phi_cf - Phi_disc).norm() / Phi_disc.norm().clamp_min(1e-8)).item()
        sigma_err = ((Sigma_cf - Sigma_disc).norm() / Sigma_disc.norm().clamp_min(1e-8)).item()
        assert phi_err < 1e-2, f"{label} (coupled G0): Phi rel_err={phi_err:.2e}"
        assert sigma_err < 1e-2, f"{label} (coupled G0): Sigma rel_err={sigma_err:.2e}"

        Phi_diag, Sigma_diag = closed_form_propagator(
            N, gamma_self, gamma_couple, alpha, beta, k_reference, coupling, tau_hat,
            constant_k=True, sigma_ref=1.0, dtype=torch.float64,
        )
        assert (Sigma_cf - Sigma_diag).norm() > 1e-3, "coupled G0 and diagonal sigma_ref=1.0 gave suspiciously identical Sigma"


def test_constant_k_false_raises():
    N = 2
    coupling = build_coupling_matrix(N, mode="mean_field")
    try:
        closed_form_propagator(N, 1.0, 0.1, [1.0] * N, [0.5] * N, 1.0, coupling, tau_hat=0.5, constant_k=False)
        raise AssertionError("closed_form_propagator should reject constant_k=False")
    except ValueError:
        pass


if __name__ == "__main__":
    test_closed_form_matches_fine_dt_reference()
    test_closed_form_matches_production_discrete_composition_in_fine_dt_limit()
    test_closed_form_matches_production_discrete_composition_with_coupled_G0()
    test_constant_k_false_raises()
    print("OK")

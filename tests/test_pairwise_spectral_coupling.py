from __future__ import annotations

import torch

from core.coupling import (
    block_coupling,
    build_coupling_matrix,
    random_heterogeneous_coupling,
    sinkhorn_symmetric_doubly_stochastic,
)
from core.hypoellipticity import hypoellipticity_check_coupled_gamma
from synthetic.drift_coupled_gamma import (
    calibrate_coupled_gammas,
    calibrate_coupled_gammas_spectral,
    calibrate_sigma_fdt_coupled,
    calibrate_sigma_fdt_spectral,
    drift_fn_coupled_gamma,
)
from synthetic.exact_dsm import (
    _extract_Avx_Avv_coupled_gamma,
    analytic_conditional_covariance_n,
    analytic_conditional_covariance_n_spectral,
    analytic_score_precision_n,
    analytic_score_precision_n_spectral,
    closed_form_propagator,
    sample_and_analytic_score_target,
    sample_and_analytic_score_target_spectral,
)

def test_sinkhorn_is_symmetric_doubly_stochastic():
    torch.manual_seed(0)
    for N in (3, 5, 8):
        raw = torch.rand(N, N)
        W = (raw + raw.T) / 2.0
        W.fill_diagonal_(0.0)
        C = sinkhorn_symmetric_doubly_stochastic(W)
        assert torch.allclose(C, C.T, atol=1e-6), f"N={N}: Sinkhorn output not symmetric"
        assert torch.allclose(C.sum(dim=-1), torch.ones(N), atol=1e-5), f"N={N}: rows don't sum to 1"
        assert torch.allclose(C.diagonal(), torch.zeros(N), atol=1e-8), f"N={N}: nonzero diagonal"


def test_mean_field_is_sinkhorn_fixed_point():
    for N in (2, 3, 5):
        C_mf = build_coupling_matrix(N, mode="mean_field")
        C_scaled = sinkhorn_symmetric_doubly_stochastic(C_mf)
        assert torch.allclose(C_mf, C_scaled, atol=1e-6), "mean-field should be an exact Sinkhorn fixed point"


def test_random_heterogeneous_epsilon_zero_is_exactly_mean_field():
    for N in (2, 3, 4, 6):
        C0 = random_heterogeneous_coupling(N, epsilon=0.0, seed=1)
        C_mf = build_coupling_matrix(N, mode="mean_field")
        assert torch.allclose(C0, C_mf, atol=1e-5), f"N={N}: epsilon=0 should reduce exactly to mean-field"


def test_random_heterogeneous_and_block_are_valid_coupling_matrices():
    for N in (3, 4, 6):
        for eps in (0.25, 0.5, 1.0):
            C = random_heterogeneous_coupling(N, epsilon=eps, seed=7)
            assert torch.allclose(C, C.T, atol=1e-5)
            assert torch.allclose(C.sum(dim=-1), torch.ones(N, dtype=C.dtype), atol=1e-4)
            assert torch.allclose(C.diagonal(), torch.zeros(N, dtype=C.dtype), atol=1e-6)
    C = block_coupling(6, [2, 4], w_in=5.0, w_out=0.5)
    assert torch.allclose(C, C.T, atol=1e-5)
    assert torch.allclose(C.sum(dim=-1), torch.ones(6, dtype=C.dtype), atol=1e-4)


def test_calibrate_coupled_gammas_spectral_reduces_to_affine_at_mean_field():
    for N in (2, 3, 4, 5, 8):
        for alpha, beta, regime in [(1.0, 0.6, "critically_damped"), (1.2, 0.3, "underdamped"), (0.8, 0.9, "overdamped")]:
            C = build_coupling_matrix(N, mode="mean_field")
            gamma_self, gamma_couple = calibrate_coupled_gammas(alpha, beta, 1.0, 1.0, N, regime)
            Gamma_spectral = calibrate_coupled_gammas_spectral(alpha, beta, 1.0, 1.0, C, regime)
            eye = torch.eye(N)
            Gamma_affine = gamma_self * eye + gamma_couple * (eye - C)
            assert torch.allclose(Gamma_spectral, Gamma_affine, atol=1e-6), (
                f"N={N} regime={regime}: spectral Gamma doesn't match affine gamma_self/gamma_couple formula"
            )


def test_calibrate_sigma_fdt_spectral_reduces_to_affine_at_mean_field():
    for N in (2, 3, 5):
        C = build_coupling_matrix(N, mode="mean_field")
        gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, 0.6, 1.0, 1.0, N, "critically_damped")
        a, b = calibrate_sigma_fdt_coupled(gamma_self, gamma_couple, N)
        Gamma_spectral = calibrate_coupled_gammas_spectral(1.0, 0.6, 1.0, 1.0, C, "critically_damped")
        G_spectral = calibrate_sigma_fdt_spectral(Gamma_spectral, C, target_variance=1.0)
        eye = torch.eye(N)
        G_affine = a * eye + b * C
        assert torch.allclose(G_spectral, G_affine, atol=1e-6), f"N={N}: spectral G doesn't match a*I+b*C"


def test_drift_fn_coupled_gamma_damping_matrix_reduces_to_scalar_path_at_mean_field():
    torch.manual_seed(0)
    for N in (2, 3, 4):
        B = 5
        X = [[torch.randn(B, 1)] for _ in range(N)]
        V = [[torch.randn(B, 1)] for _ in range(N)]
        K_self = [[torch.rand(B, 1) + 0.1] for _ in range(N)]
        K_global = [torch.rand(B, 1) + 0.1 for _ in range(N)]
        alpha, beta = [1.0] * N, [0.6] * N
        t, T = torch.tensor(4.0), 10
        C = build_coupling_matrix(N, mode="mean_field")
        gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, 0.6, 1.0, 1.0, N, "critically_damped")
        Gamma = calibrate_coupled_gammas_spectral(1.0, 0.6, 1.0, 1.0, C, "critically_damped")

        dV_scalar = drift_fn_coupled_gamma(X, V, K_self, K_global, t, T, alpha, beta, gamma_self, gamma_couple,
                                            coupling_matrix=C, constant_k=False)
        dV_matrix = drift_fn_coupled_gamma(X, V, K_self, K_global, t, T, alpha, beta,
                                            coupling_matrix=C, constant_k=False, damping_matrix=Gamma)
        for i in range(N):
            assert torch.allclose(dV_scalar[i][0], dV_matrix[i][0], atol=1e-6), (
                f"N={N} population {i}: damping_matrix path diverges from scalar (gamma_self, gamma_couple) path"
            )


def test_analytic_score_machinery_reduces_to_mean_field_at_mean_field():
    for N in (2, 3, 5):
        C = build_coupling_matrix(N, mode="mean_field")
        gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, 0.6, 1.0, 1.0, N, "critically_damped")
        Gamma = calibrate_coupled_gammas_spectral(1.0, 0.6, 1.0, 1.0, C, "critically_damped")
        q = torch.tensor(0.7)

        prec_old, coeff_old = analytic_score_precision_n(N, gamma_self, gamma_couple, q, target_variance=1.0)
        prec_new, coeff_new = analytic_score_precision_n_spectral(Gamma, C, q, target_variance=1.0)
        assert torch.allclose(prec_old, prec_new, atol=1e-4), f"N={N}: precision mismatch"
        assert torch.allclose(coeff_old, coeff_new, atol=1e-4), f"N={N}: regression coeff mismatch"

        cov_old = analytic_conditional_covariance_n(N, gamma_self, gamma_couple, q, target_variance=1.0)
        cov_new = analytic_conditional_covariance_n_spectral(Gamma, C, q, target_variance=1.0)
        assert torch.allclose(cov_old, cov_new, atol=1e-4), f"N={N}: conditional covariance mismatch"

        B = 8
        Z0 = torch.randn(B, 2 * N)
        Phi_t = torch.eye(2 * N) * 0.5
        torch.manual_seed(123)
        Zt_old, score_old = sample_and_analytic_score_target(Z0, Phi_t, N, gamma_self, gamma_couple, q, target_variance=1.0)
        torch.manual_seed(123)
        Zt_new, score_new = sample_and_analytic_score_target_spectral(Z0, Phi_t, Gamma, C, q, target_variance=1.0)
        assert torch.allclose(Zt_old, Zt_new, atol=1e-4), f"N={N}: sampled Zt mismatch"
        assert torch.allclose(score_old, score_new, atol=1e-4), f"N={N}: analytic score target mismatch"


def test_closed_form_propagator_damping_matrix_reduces_to_scalar_path_at_mean_field():
    for N in (2, 3, 4):
        C = build_coupling_matrix(N, mode="mean_field")
        gamma_self, gamma_couple = calibrate_coupled_gammas(1.0, 0.5, 1.0, 1.0, N, "critically_damped")
        Gamma = calibrate_coupled_gammas_spectral(1.0, 0.5, 1.0, 1.0, C, "critically_damped")
        Phi_scalar, Sigma_scalar = closed_form_propagator(N, gamma_self, gamma_couple, [1.0] * N, [0.5] * N,
                                                            1.0, C, tau_hat=0.7, constant_k=True)
        Phi_matrix, Sigma_matrix = closed_form_propagator(N, None, None, [1.0] * N, [0.5] * N,
                                                            1.0, C, tau_hat=0.7, constant_k=True, damping_matrix=Gamma)
        assert torch.allclose(Phi_scalar, Phi_matrix, atol=1e-5), f"N={N}: Phi mismatch"
        assert torch.allclose(Sigma_scalar, Sigma_matrix, atol=1e-5), f"N={N}: Sigma mismatch"

def _per_mode_damping_ratios(N, C, Gamma, alpha=1.0, beta=0.6, k_reference=1.0):
    const_one = lambda t, T: torch.tensor(1.0)
    A_vx, A_vv = _extract_Avx_Avv_coupled_gamma(
        N, None, None, [alpha] * N, [beta] * N, k_reference, C, t=1, T=1, constant_k=True,
        time_scale_fn=const_one, damping_matrix=Gamma,
    )
    L = torch.eye(N, dtype=C.dtype) - C
    _, U = torch.linalg.eigh(L)
    A_vx_modes = torch.diagonal(U.T @ A_vx @ U)
    A_vv_modes = torch.diagonal(U.T @ A_vv @ U)
    off_diag_vx = (U.T @ A_vx @ U) - torch.diag(A_vx_modes)
    off_diag_vv = (U.T @ A_vv @ U) - torch.diag(A_vv_modes)
    Omega_k = torch.sqrt((-A_vx_modes).clamp_min(0.0))
    Gamma_k = -A_vv_modes
    zeta_k = Gamma_k / (2.0 * Omega_k.clamp_min(1e-12))
    return zeta_k, off_diag_vx.abs().max().item(), off_diag_vv.abs().max().item()


def test_spectral_calibration_achieves_exact_per_mode_critical_damping_on_heterogeneous_C():
    for N in (3, 4, 5):
        C = random_heterogeneous_coupling(N, epsilon=1.0, seed=11)
        Gamma = calibrate_coupled_gammas_spectral(1.0, 0.6, 1.0, 1.0, C, "critically_damped")
        zeta_k, off_vx, off_vv = _per_mode_damping_ratios(N, C, Gamma)
        assert off_vx < 1e-4 and off_vv < 1e-4, f"N={N}: linearized Jacobian isn't diagonal in L's eigenbasis"
        assert torch.allclose(zeta_k, torch.ones(N, dtype=zeta_k.dtype), atol=1e-4), (
            f"N={N}: spectral calibration should achieve zeta_k=1 exactly for every mode, got {zeta_k}"
        )


def test_naive_affine_formula_misapplied_to_heterogeneous_C_is_not_exact():
    N, alpha, beta, k_ref = 5, 0.1, 5.0, 0.1
    C = block_coupling(N, [2, 3], w_in=20.0, w_out=0.01)
    gamma_self, gamma_couple = calibrate_coupled_gammas(alpha, beta, k_ref, k_ref, N, "critically_damped")
    eye = torch.eye(N, dtype=C.dtype)
    Gamma_naive = gamma_self * eye + gamma_couple * (eye - C)
    zeta_k, _, _ = _per_mode_damping_ratios(N, C, Gamma_naive, alpha=alpha, beta=beta, k_reference=k_ref)
    max_deviation = (zeta_k - 1.0).abs().max().item()
    assert max_deviation > 1e-1, (
        f"expected the naive affine formula to visibly miss critical damping on a heterogeneous C, "
        f"deviation was only {max_deviation:.2e} -- test fixture may be too close to mean-field"
    )


def test_hypoellipticity_holds_for_heterogeneous_and_block_coupling():
    def make_K(N, k_ref):
        K_self = [[torch.tensor([[k_ref]])] for _ in range(N)]
        K_global = [torch.tensor([[k_ref]]) for _ in range(N)]
        return K_self, K_global

    for N in (3, 5):
        for C in (
            build_coupling_matrix(N, mode="mean_field"),
            random_heterogeneous_coupling(N, epsilon=1.0, seed=11),
            block_coupling(N, [2, N - 2], w_in=5.0, w_out=0.5),
        ):
            Gamma = calibrate_coupled_gammas_spectral(1.0, 0.6, 1.0, 1.0, C, "critically_damped")
            G = calibrate_sigma_fdt_spectral(Gamma, C, target_variance=1.0)
            K_self, K_global = make_K(N, 1.0)
            t, T = torch.tensor(5.0), 10
            res = hypoellipticity_check_coupled_gamma(
                N, K_self, K_global, t, T, [1.0] * N, [0.6] * N, C, True, G, damping_matrix=Gamma,
            )
            assert res["passed"], f"N={N}: hypoellipticity/controllability check failed, min_eig={res['min_eig']}"


if __name__ == "__main__":
    test_sinkhorn_is_symmetric_doubly_stochastic()
    test_mean_field_is_sinkhorn_fixed_point()
    test_random_heterogeneous_epsilon_zero_is_exactly_mean_field()
    test_random_heterogeneous_and_block_are_valid_coupling_matrices()
    test_calibrate_coupled_gammas_spectral_reduces_to_affine_at_mean_field()
    test_calibrate_sigma_fdt_spectral_reduces_to_affine_at_mean_field()
    test_drift_fn_coupled_gamma_damping_matrix_reduces_to_scalar_path_at_mean_field()
    test_analytic_score_machinery_reduces_to_mean_field_at_mean_field()
    test_closed_form_propagator_damping_matrix_reduces_to_scalar_path_at_mean_field()
    test_spectral_calibration_achieves_exact_per_mode_critical_damping_on_heterogeneous_C()
    test_naive_affine_formula_misapplied_to_heterogeneous_C_is_not_exact()
    test_hypoellipticity_holds_for_heterogeneous_and_block_coupling()
    print("OK")

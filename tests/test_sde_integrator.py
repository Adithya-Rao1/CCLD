from __future__ import annotations

import torch

from core.coupling import build_coupling_matrix
from core.drift import drift_fn_n
from core.sde import build_g_matrix_n, em_step_n, ndsm_loss_n, reverse_step_n


def _make_state(N: int, seed: int, shape=(4, 6)):
    g = torch.Generator().manual_seed(seed)
    X = [[torch.randn(shape, generator=g)] for _ in range(N)]
    V = [[torch.zeros(shape) for _ in pop] for pop in X]
    K_self = [[torch.rand(shape, generator=g) + 0.1] for _ in range(N)]
    K_global = [torch.rand(shape, generator=g) + 0.1 for _ in range(N)]
    return X, V, K_self, K_global


def test_em_step_n_shapes_and_finite():
    N = 4
    X, V, K_self, K_global = _make_state(N, seed=0)
    coupling = build_coupling_matrix(N, mode="mean_field")
    G = build_g_matrix_n(torch.tensor(0.1), N, diffusion_mode="shared")

    X_next, V_next, mu, z_list, sigma_list = em_step_n(
        X, V, K_self, K_global, t=torch.tensor(2.0), T=10,
        alpha=[0.5] * N, beta=[0.5] * N, gamma=[1.0] * N,
        coupling_matrix_drift=coupling, use_gamma=True, constant_k=False,
        dt=1e-3, G=G,
    )

    assert len(X_next) == N and len(V_next) == N
    for i in range(N):
        assert X_next[i][0].shape == X[i][0].shape
        assert V_next[i][0].shape == V[i][0].shape
        assert torch.isfinite(X_next[i][0]).all()
        assert torch.isfinite(V_next[i][0]).all()
    assert len(z_list) == N and len(sigma_list) == N


def test_em_step_n_independent_diffusion_mode():
    N = 3
    X, V, K_self, K_global = _make_state(N, seed=1)
    coupling = build_coupling_matrix(N, mode="pairwise", weights=torch.rand(N, N) + 0.1)
    G = build_g_matrix_n(torch.tensor(0.1), N, diffusion_mode="independent", g_per_task=[0.05, 0.1, 0.2])

    X_next, V_next, _, _, _ = em_step_n(
        X, V, K_self, K_global, t=torch.tensor(5.0), T=10,
        alpha=[0.5] * N, beta=[0.3] * N, gamma=[1.0] * N,
        coupling_matrix_drift=coupling, use_gamma=True, constant_k=True,
        dt=1e-3, G=G,
    )
    for i in range(N):
        assert torch.isfinite(X_next[i][0]).all()
        assert torch.isfinite(V_next[i][0]).all()


def test_reverse_step_n_shapes_and_finite():
    N = 3
    X, V, K_self, K_global = _make_state(N, seed=2)
    coupling = build_coupling_matrix(N, mode="mean_field")
    G = build_g_matrix_n(torch.tensor(0.1), N, diffusion_mode="shared")
    score_outputs = [[torch.randn_like(x) for x in pop] for pop in X]

    X_new, V_new = reverse_step_n(
        X, V, K_self, K_global, score_outputs, t=4, T=10,
        alpha=[0.5] * N, beta=[0.5] * N, gamma=[1.0] * N,
        coupling_matrix_drift=coupling, use_gamma=True, constant_k=False,
        dt=1e-3, G=G,
    )
    for i in range(N):
        assert X_new[i][0].shape == X[i][0].shape
        assert torch.isfinite(X_new[i][0]).all()
        assert torch.isfinite(V_new[i][0]).all()


def test_em_step_n_determinism_with_seeded_generator():
    torch.manual_seed(42)
    N = 2
    X, V, K_self, K_global = _make_state(N, seed=3)
    coupling = build_coupling_matrix(N, mode="mean_field")
    G = build_g_matrix_n(torch.tensor(0.1), N, diffusion_mode="shared")

    torch.manual_seed(123)
    X1, V1, _, _, _ = em_step_n(
        X, V, K_self, K_global, t=torch.tensor(1.0), T=10,
        alpha=[0.5] * N, beta=[0.5] * N, gamma=[1.0] * N,
        coupling_matrix_drift=coupling, use_gamma=True, constant_k=False, dt=1e-3, G=G,
    )
    torch.manual_seed(123)
    X2, V2, _, _, _ = em_step_n(
        X, V, K_self, K_global, t=torch.tensor(1.0), T=10,
        alpha=[0.5] * N, beta=[0.5] * N, gamma=[1.0] * N,
        coupling_matrix_drift=coupling, use_gamma=True, constant_k=False, dt=1e-3, G=G,
    )
    for i in range(N):
        assert torch.allclose(X1[i][0], X2[i][0])
        assert torch.allclose(V1[i][0], V2[i][0])


def test_ndsm_loss_n_runs_and_scalar():
    N = 2
    X, V, K_self, K_global = _make_state(N, seed=4)
    coupling = build_coupling_matrix(N, mode="mean_field")
    G = build_g_matrix_n(torch.tensor(0.1), N, diffusion_mode="shared")

    X_next, V_next, mu, z_list, sigma_list = em_step_n(
        X, V, K_self, K_global, t=torch.tensor(2.0), T=10,
        alpha=[0.5] * N, beta=[0.5] * N, gamma=[1.0] * N,
        coupling_matrix_drift=coupling, use_gamma=True, constant_k=False, dt=1e-3, G=G,
    )

    def dummy_score_fn(X, V_query, t_n):
        return [[torch.randn_like(v) for v in pop] for pop in V_query]

    loss = ndsm_loss_n(X, dummy_score_fn, V_next, mu, z_list, sigma_list, t_n=2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_em_step_n_forward_drift_sign_matches_production():
    """Regression test for the 2026-08-15 em_step_n sign bug: em_step_n used `mu =
    v + dv*dt` instead of the required `mu = v - dv*dt`. drift_fn_n is an attracting
    restoring force; the forward/noising process must run it in reverse (push away
    from equilibrium) to actually corrupt data into noise -- reverse_step_n runs the
    same drift forward (+dv) to pull noise back toward data, and already matched this
    convention. Confirmed against the production reference this repo generalizes from
    (Metis_V1/src/diffusion/loss_fn.py::em_mean_multi_state's `v - dv*dt`, vs.
    reverse_sampler.py::reverse_step's `v + dv*dt`) and against the NDSM paper's own
    forward-step definition (eq. 6: mu = yn - f(yn,T-tn)*dtn).

    This bug passed every shape/finiteness check and even every KL/sample-quality
    check under `--quick`'s tiny n_diff_steps=5 (the corruption barely needed to
    diverge from near-equilibrium at that scale) -- it only produces a dramatic
    quality regression at real (n_diff_steps~20+) settings, so a purely statistical
    round-trip probe on a single toy state is unreliable at this dt/T scale (both
    sign conventions show O(1) mean bias in that setup, likely due to the NDSM
    paper's own noted small-dt discretization bias in the raw conditional score,
    which is exactly why their training loss adds a specific bias-correcting term
    rather than relying on the raw discrete score directly). This test instead pins
    the sign deterministically -- no statistics, no flakiness -- by checking the mean
    a single em_step_n call actually returns against the drift's known sign, for a
    state where the drift is unambiguously nonzero.
    """
    N = 2
    shape = (2, 3)
    g = torch.Generator().manual_seed(11)
    X = [[torch.randn(shape, generator=g) + 2.0] for _ in range(N)]  # away from equilibrium (0)
    V = [[torch.randn(shape, generator=g)] for _ in range(N)]
    K_self = [[torch.rand(shape, generator=g) + 0.5] for _ in range(N)]
    K_global = [torch.rand(shape, generator=g) + 0.5 for _ in range(N)]
    alpha, beta, gamma = [1.0] * N, [0.5] * N, [1.0] * N
    coupling = build_coupling_matrix(N, mode="mean_field")
    G = build_g_matrix_n(torch.tensor(0.1), N, diffusion_mode="shared")
    t, T, dt = torch.tensor(5.0), 10, 0.02

    dV = drift_fn_n(X, V, K_self, K_global, t, T, alpha, beta, gamma,
                     coupling_matrix=coupling, use_gamma=True, constant_k=False)
    X_next, V_next, mu, z_list, sigma_list = em_step_n(
        X, V, K_self, K_global, t, T, alpha, beta, gamma, coupling, True, False, dt, G,
    )
    for i in range(N):
        expected_mu = V[i][0] - dV[i][0] * dt
        assert torch.allclose(mu[i][0], expected_mu, atol=1e-6), (
            f"population {i}: em_step_n's mean is {mu[i][0]}, expected V - dV*dt = "
            f"{expected_mu} (i.e. the forward step must subtract the drift, not add it)"
        )


if __name__ == "__main__":
    test_em_step_n_shapes_and_finite()
    test_em_step_n_independent_diffusion_mode()
    test_reverse_step_n_shapes_and_finite()
    test_em_step_n_determinism_with_seeded_generator()
    test_ndsm_loss_n_runs_and_scalar()
    test_em_step_n_forward_drift_sign_matches_production()
    print("OK: em_step_n / reverse_step_n / ndsm_loss_n pass shape, finiteness, and determinism checks.")

from __future__ import annotations

import torch

from core.coupling import build_coupling_matrix
from core.drift import drift_fn_n


def _make_state(N: int, seed: int, shape=(3, 4)):
    g = torch.Generator().manual_seed(seed)
    X = [[torch.randn(shape, generator=g)] for _ in range(N)]
    V = [[torch.randn(shape, generator=g)] for _ in range(N)]
    K_self = [[torch.rand(shape, generator=g) + 0.1] for _ in range(N)]
    K_global = [torch.rand(shape, generator=g) + 0.1 for _ in range(N)]
    return X, V, K_self, K_global


def _common_kwargs():
    return dict(t=torch.tensor(3.0), T=10, use_gamma=True, constant_k=False)


def test_mean_field_permutation_equivariance():
    N = 4
    X, V, K_self, K_global = _make_state(N, seed=0)
    alpha, beta, gamma = [0.5, 1.0, 1.5, 2.0], [0.3, 0.4, 0.5, 0.6], [1.0, 1.2, 0.8, 1.1]

    perm = [2, 0, 3, 1]
    X_p = [X[i] for i in perm]
    V_p = [V[i] for i in perm]
    K_self_p = [K_self[i] for i in perm]
    K_global_p = [K_global[i] for i in perm]
    alpha_p = [alpha[i] for i in perm]
    beta_p = [beta[i] for i in perm]
    gamma_p = [gamma[i] for i in perm]

    dV = drift_fn_n(X, V, K_self, K_global, alpha=alpha, beta=beta, gamma=gamma, **_common_kwargs())
    dV_p = drift_fn_n(X_p, V_p, K_self_p, K_global_p, alpha=alpha_p, beta=beta_p, gamma=gamma_p, **_common_kwargs())

    for new_pos, old_pos in enumerate(perm):
        assert torch.allclose(dV_p[new_pos][0], dV[old_pos][0], atol=1e-6), (
            f"mean-field drift is not permutation-equivariant at index {old_pos}->{new_pos}"
        )


def test_independent_coupling_is_separable():
    N = 4
    X, V, K_self, K_global = _make_state(N, seed=1)
    alpha, beta, gamma = [1.0] * N, [0.5] * N, [1.0] * N
    coupling = build_coupling_matrix(N, mode="independent")

    dV_before = drift_fn_n(X, V, K_self, K_global, alpha=alpha, beta=beta, gamma=gamma,
                            coupling_matrix=coupling, **_common_kwargs())

    j = 1
    X_perturbed = [pop if i != j else [x + 5.0 for x in pop] for i, pop in enumerate(X)]
    dV_after = drift_fn_n(X_perturbed, V, K_self, K_global, alpha=alpha, beta=beta, gamma=gamma,
                           coupling_matrix=coupling, **_common_kwargs())

    for i in range(N):
        if i == j:
            continue
        assert torch.allclose(dV_before[i][0], dV_after[i][0], atol=1e-6), (
            f"independent coupling: population {i}'s drift changed after perturbing population {j}"
        )


def test_mean_field_coupling_vanishes_at_consensus():
    # The coupling target for population i is the *scalar mean* of every other
    # population's tensor (m[j]), not an elementwise value. Consensus therefore
    # means every population is a constant tensor holding the same scalar: then
    # each m[j] equals that constant, target[i]-x is elementwise zero, and the
    # coupling term beta*(target-x) must vanish, reducing to the single-population
    # SHO term.
    N = 2
    shape = (3, 4)
    g = torch.Generator().manual_seed(2)
    c = 1.75
    X = [[torch.full(shape, c)], [torch.full(shape, c)]]
    V = [[torch.zeros(shape)], [torch.zeros(shape)]]
    K_self = [[torch.rand(shape, generator=g) + 0.1] for _ in range(N)]
    K_global = [torch.rand(shape, generator=g) + 0.1 for _ in range(N)]
    alpha, beta, gamma = [1.0, 1.0], [0.7, 0.7], [1.0, 1.0]

    dV_coupled = drift_fn_n(X, V, K_self, K_global, alpha=alpha, beta=beta, gamma=gamma, **_common_kwargs())
    dV_no_coupling = drift_fn_n(X, V, K_self, K_global, alpha=alpha, beta=[0.0, 0.0], gamma=gamma, **_common_kwargs())

    for i in range(N):
        assert torch.allclose(dV_coupled[i][0], dV_no_coupling[i][0], atol=1e-6), (
            "coupling term should vanish when populations are already at consensus"
        )


def test_use_gamma_toggle_isolates_damping_term():
    N = 3
    X, V, K_self, K_global = _make_state(N, seed=3)
    alpha, beta, gamma = [1.0] * N, [0.5] * N, [0.9, 1.3, 0.6]
    t, T = torch.tensor(4.0), 10

    dV_damped = drift_fn_n(X, V, K_self, K_global, t=t, T=T, alpha=alpha, beta=beta, gamma=gamma,
                            use_gamma=True, constant_k=False)
    dV_undamped = drift_fn_n(X, V, K_self, K_global, t=t, T=T, alpha=alpha, beta=beta, gamma=gamma,
                              use_gamma=False, constant_k=False)

    time_scale = (T - t) / (t + T)
    for i in range(N):
        expected_diff = -time_scale * gamma[i] * V[i][0]
        actual_diff = dV_damped[i][0] - dV_undamped[i][0]
        assert torch.allclose(actual_diff, expected_diff, atol=1e-6), (
            f"use_gamma toggle should isolate exactly the -gamma*v damping term for population {i}"
        )


def test_constant_k_makes_drift_invariant_to_k_global_magnitude():
    # constant_k freezes norm_f_k_global to 1.0 regardless of K_global's contents
    # (it does NOT freeze K_self -- omega_sq still depends on K_self's magnitude).
    N = 2
    X, V, K_self, K_global = _make_state(N, seed=4)
    K_global_scaled = [k * 50.0 for k in K_global]
    alpha, beta, gamma = [1.0] * N, [0.5] * N, [1.0] * N

    dV_a = drift_fn_n(X, V, K_self, K_global, alpha=alpha, beta=beta, gamma=gamma,
                       t=torch.tensor(4.0), T=10, use_gamma=True, constant_k=True)
    dV_b = drift_fn_n(X, V, K_self, K_global_scaled, alpha=alpha, beta=beta, gamma=gamma,
                       t=torch.tensor(4.0), T=10, use_gamma=True, constant_k=True)

    for i in range(N):
        assert torch.allclose(dV_a[i][0], dV_b[i][0], atol=1e-6), (
            "constant_k=True should make the drift invariant to K_global's magnitude"
        )


def test_invalid_inputs_raise():
    N = 1
    X, V, K_self, K_global = _make_state(N, seed=5)
    try:
        drift_fn_n(X, V, K_self, K_global, alpha=[1.0], beta=[0.5], gamma=[1.0], **_common_kwargs())
        raise AssertionError("drift_fn_n should reject N < 2")
    except ValueError:
        pass

    N = 3
    X, V, K_self, K_global = _make_state(N, seed=6)
    try:
        drift_fn_n(X, V, K_self, K_global, alpha=[1.0, 1.0], beta=[0.5] * N, gamma=[1.0] * N, **_common_kwargs())
        raise AssertionError("drift_fn_n should reject ragged-length alpha")
    except ValueError:
        pass


if __name__ == "__main__":
    test_mean_field_permutation_equivariance()
    test_independent_coupling_is_separable()
    test_mean_field_coupling_vanishes_at_consensus()
    test_use_gamma_toggle_isolates_damping_term()
    test_constant_k_makes_drift_invariant_to_k_global_magnitude()
    test_invalid_inputs_raise()
    print("OK: drift_fn_n permutation equivariance, coupling separability/consensus, "
          "use_gamma/constant_k toggles, and input validation all hold.")

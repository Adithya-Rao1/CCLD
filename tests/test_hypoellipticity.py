from __future__ import annotations

import itertools

import torch

from core.coupling import build_coupling_matrix
from core.damping import calibrate_gammas_for_regime
from core.hypoellipticity import hypoellipticity_check
from core.sde import build_g_matrix_n


def _make_conditioning(N: int, seed: int, shape=(2, 2)):
    g = torch.Generator().manual_seed(seed)
    K_self = [[torch.rand(shape, generator=g) + 0.1] for _ in range(N)]
    K_global = [torch.rand(shape, generator=g) + 0.1 for _ in range(N)]
    return K_self, K_global


def _run_one_config(N, coupling_mode, diffusion_mode, regime, use_gamma, constant_k, seed):
    K_self, K_global = _make_conditioning(N, seed)
    t = torch.tensor(3.0)
    T = 10
    alpha = [0.5] * N

    k_reference = [1.0] * N
    if use_gamma:
        gamma = calibrate_gammas_for_regime(alpha, k_reference, regime)
    else:
        gamma = [0.0] * N

    beta = [0.5] * N
    coupling = build_coupling_matrix(N, mode=coupling_mode)

    sigma_t = torch.tensor(0.1)
    g_per_task = [0.1 + 0.05 * i for i in range(N)] if diffusion_mode == "independent" else None
    G = build_g_matrix_n(sigma_t, N, diffusion_mode=diffusion_mode, g_per_task=g_per_task)

    result = hypoellipticity_check(
        N=N, K_self=K_self, K_global=K_global, t=t, T=T,
        alpha=alpha, beta=beta, gamma=gamma,
        coupling_matrix=coupling, use_gamma=use_gamma, constant_k=constant_k,
        G=G,
    )
    return result


def test_hypoellipticity_grid():
    Ns = [2, 3, 5]
    coupling_modes = ["mean_field", "independent"]
    diffusion_modes = ["shared", "independent"]
    regimes = ["underdamped", "critically_damped", "overdamped"]

    results = {}
    for N, coupling_mode, diffusion_mode, regime in itertools.product(Ns, coupling_modes, diffusion_modes, regimes):
        key = (N, coupling_mode, diffusion_mode, regime)
        result = _run_one_config(N, coupling_mode, diffusion_mode, regime, use_gamma=True, constant_k=False, seed=0)
        results[key] = result
        assert result["passed"], f"expected controllable for config={key}, got min_eig={result['min_eig']:.3g}"

    return results


def test_hypoellipticity_no_damping_no_coupling_degenerate():
    result = _run_one_config(
        N=3, coupling_mode="independent", diffusion_mode="shared",
        regime="critically_damped", use_gamma=False, constant_k=True, seed=1,
    )
    assert result["passed"], f"expected controllable even without damping, got min_eig={result['min_eig']:.3g}"


if __name__ == "__main__":
    results = test_hypoellipticity_grid()
    test_hypoellipticity_no_damping_no_coupling_degenerate()
    n_pass = sum(1 for r in results.values() if r["passed"])
    print(f"OK: {n_pass}/{len(results)} hypoellipticity-grid configs passed the controllability-Gramian check.")
    for key, r in results.items():
        print(f"  N={key[0]} coupling={key[1]:10s} diffusion={key[2]:11s} regime={key[3]:17s} "
              f"min_eig={r['min_eig']:.4g} passed={r['passed']}")
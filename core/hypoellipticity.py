from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import expm, solve_continuous_lyapunov

from core.drift import _time_scale, drift_fn_n


def _flatten_state(X: List[List[torch.Tensor]], V: List[List[torch.Tensor]]) -> torch.Tensor:
    N = len(X)
    x_flat = torch.cat([X[i][0].reshape(-1) for i in range(N)])
    v_flat = torch.cat([V[i][0].reshape(-1) for i in range(N)])
    return torch.cat([x_flat, v_flat])


def _unflatten_state(z: torch.Tensor, N: int, shape: Tuple[int, ...], n: int):
    x_flat, v_flat = z[: N * n], z[N * n :]
    X = [[x_flat[i * n : (i + 1) * n].reshape(shape)] for i in range(N)]
    V = [[v_flat[i * n : (i + 1) * n].reshape(shape)] for i in range(N)]
    return X, V


def linearize_drift(
    N: int,
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma: List[float],
    coupling_matrix: Optional[torch.Tensor],
    use_gamma: bool,
    constant_k: bool,
    shape: Tuple[int, ...] = (2, 2),
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    scale_kinematics_with_time: bool = True,
) -> Tuple[np.ndarray, int]:
    n = int(torch.zeros(shape).numel())
    time_scale = _time_scale(t, T, time_scale_fn)
    kin_scale = time_scale if scale_kinematics_with_time else torch.as_tensor(1.0)

    def f(z: torch.Tensor) -> torch.Tensor:
        X, V = _unflatten_state(z, N, shape, n)
        dV = drift_fn_n(
            X, V, K_self, K_global, t, T, alpha, beta, gamma,
            coupling_matrix=coupling_matrix, use_gamma=use_gamma, constant_k=constant_k,
            scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
        )
        dX = [kin_scale * V[i][0].reshape(-1) for i in range(N)]
        dV_flat = [dV[i][0].reshape(-1) for i in range(N)]
        return torch.cat(dX + dV_flat)

    z0 = torch.zeros(2 * N * n)
    A = torch.autograd.functional.jacobian(f, z0)
    return A.detach().cpu().numpy(), n


def linearize_coupled_gamma_drift(
    N: int,
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    coupling_matrix: Optional[torch.Tensor],
    constant_k: bool,
    gamma_self: Optional[float] = None,
    gamma_couple: Optional[float] = None,
    damping_matrix: Optional[torch.Tensor] = None,
    skew_matrix: Optional[torch.Tensor] = None,
    skew_sigma_ref: Optional[torch.Tensor] = None,
    shape: Tuple[int, ...] = (2, 2),
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    scale_kinematics_with_time: bool = True,
) -> Tuple[np.ndarray, int]:
    from synthetic.drift_coupled_gamma import drift_fn_coupled_gamma
    from synthetic.skew_coupling import skew_drift_correction

    n = int(torch.zeros(shape).numel())
    time_scale = _time_scale(t, T, time_scale_fn)
    kin_scale = time_scale if scale_kinematics_with_time else torch.as_tensor(1.0)

    def f(z: torch.Tensor) -> torch.Tensor:
        X, V = _unflatten_state(z, N, shape, n)
        dV = drift_fn_coupled_gamma(
            X, V, K_self, K_global, t, T, alpha, beta, gamma_self, gamma_couple,
            coupling_matrix=coupling_matrix, constant_k=constant_k,
            scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
            damping_matrix=damping_matrix,
        )
        if skew_matrix is not None:
            dX_skew, dV_skew = skew_drift_correction(X, V, skew_matrix, skew_sigma_ref)
            dX = [kin_scale * V[i][0].reshape(-1) + dX_skew[i][0].reshape(-1) for i in range(N)]
            dV_flat = [dV[i][0].reshape(-1) + dV_skew[i][0].reshape(-1) for i in range(N)]
        else:
            dX = [kin_scale * V[i][0].reshape(-1) for i in range(N)]
            dV_flat = [dV[i][0].reshape(-1) for i in range(N)]
        return torch.cat(dX + dV_flat)

    z0 = torch.zeros(2 * N * n)
    A = torch.autograd.functional.jacobian(f, z0)
    return A.detach().cpu().numpy(), n


def hypoellipticity_check_coupled_gamma(
    N: int,
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    coupling_matrix: Optional[torch.Tensor],
    constant_k: bool,
    G: torch.Tensor,
    gamma_self: Optional[float] = None,
    gamma_couple: Optional[float] = None,
    damping_matrix: Optional[torch.Tensor] = None,
    skew_matrix: Optional[torch.Tensor] = None,
    skew_sigma_ref: Optional[torch.Tensor] = None,
    shape: Tuple[int, ...] = (2, 2),
    tol: float = 1e-8,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    scale_kinematics_with_time: bool = True,
    scale_diffusion_with_time: bool = True,
) -> Dict:
    A, n = linearize_coupled_gamma_drift(
        N, K_self, K_global, t, T, alpha, beta, coupling_matrix, constant_k,
        gamma_self=gamma_self, gamma_couple=gamma_couple, damping_matrix=damping_matrix,
        skew_matrix=skew_matrix, skew_sigma_ref=skew_sigma_ref, shape=shape,
        scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
        scale_kinematics_with_time=scale_kinematics_with_time,
    )
    if scale_diffusion_with_time:
        time_scale = _time_scale(t, T, time_scale_fn)
        G = G * time_scale.clamp_min(0).sqrt()
    B = build_B_matrix(G, N, n)
    passed, min_eig, eigvals = controllability_check(A, B, tol)
    return {"passed": passed, "min_eig": min_eig, "eigvals": eigvals, "A": A, "B": B, "N": N, "n": n}


def build_B_matrix(G: torch.Tensor, N: int, n: int) -> np.ndarray:
    top = np.zeros((N * n, N * n))
    bottom = np.kron(G.detach().cpu().numpy(), np.eye(n))
    return np.concatenate([top, bottom], axis=0)


def finite_horizon_gramian(A: np.ndarray, B: np.ndarray, tau: float = 5.0, n_steps: int = 200) -> np.ndarray:
    BBt = B @ B.T
    ss = np.linspace(0.0, tau, n_steps)
    integrand = np.empty((n_steps, A.shape[0], A.shape[0]))
    for i, s in enumerate(ss):
        eAs = expm(A * s)
        integrand[i] = eAs @ BBt @ eAs.T
    return np.trapz(integrand, ss, axis=0)


def _is_hurwitz(A: np.ndarray, margin: float = 1e-6) -> bool:
    return bool(np.all(np.linalg.eigvals(A).real < -margin))


def controllability_check(
    A: np.ndarray, B: np.ndarray, tol: float = 1e-8, tau: float = 5.0, n_steps: int = 200,
) -> Tuple[bool, float, np.ndarray]:
    if _is_hurwitz(A):
        W = solve_continuous_lyapunov(A, -B @ B.T)
    else:
        W = finite_horizon_gramian(A, B, tau=tau, n_steps=n_steps)
    eigvals = np.linalg.eigvalsh((W + W.T) / 2)
    min_eig = float(eigvals.min())
    return min_eig > tol, min_eig, eigvals


def hypoellipticity_check(
    N: int,
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    t: torch.Tensor,
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma: List[float],
    coupling_matrix: Optional[torch.Tensor],
    use_gamma: bool,
    constant_k: bool,
    G: torch.Tensor,
    shape: Tuple[int, ...] = (2, 2),
    tol: float = 1e-8,
    scale_damping_with_time: bool = True,
    time_scale_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    scale_kinematics_with_time: bool = True,
    scale_diffusion_with_time: bool = True,
) -> Dict:
    A, n = linearize_drift(
        N, K_self, K_global, t, T, alpha, beta, gamma, coupling_matrix, use_gamma, constant_k, shape,
        scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
        scale_kinematics_with_time=scale_kinematics_with_time,
    )
    if scale_diffusion_with_time:
        time_scale = _time_scale(t, T, time_scale_fn)
        G = G * time_scale.clamp_min(0).sqrt()
    B = build_B_matrix(G, N, n)
    passed, min_eig, eigvals = controllability_check(A, B, tol)
    return {"passed": passed, "min_eig": min_eig, "eigvals": eigvals, "A": A, "B": B, "N": N, "n": n}


def small_time_variance_scaling(
    X0: List[torch.Tensor],
    K_self: List[List[torch.Tensor]],
    K_global: List[torch.Tensor],
    T: int,
    alpha: List[float],
    beta: List[float],
    gamma: List[float],
    coupling_matrix: Optional[torch.Tensor],
    use_gamma: bool,
    constant_k: bool,
    G: torch.Tensor,
    taus: List[float],
    n_samples: int = 2000,
    substeps_per_tau: int = 5,
) -> Dict[float, float]:
    from core.sde import em_step_n

    N = len(X0)
    results: Dict[float, float] = {}
    for tau in taus:
        dt = tau / substeps_per_tau
        finals = []
        for _ in range(n_samples):
            X = [[x.clone()] for x in X0]
            V = [[torch.zeros_like(x)] for x in X0]
            t = torch.tensor(0.0)
            for _ in range(substeps_per_tau):
                X, V, _, _, _ = em_step_n(
                    X, V, K_self, K_global, t, T, alpha, beta, gamma,
                    coupling_matrix, use_gamma, constant_k, dt, G,
                )
            finals.append(torch.cat([X[i][0].reshape(-1) for i in range(N)]))
        stacked = torch.stack(finals)
        cov = torch.cov(stacked.T)
        eigvals = torch.linalg.eigvalsh(cov)
        results[tau] = float(eigvals.min().item())
    return results


def fit_scaling_exponent(results: Dict[float, float]) -> float:
    taus = np.array(sorted(results.keys()))
    min_eigs = np.array([results[t] for t in taus])
    mask = min_eigs > 0
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(taus[mask]), np.log(min_eigs[mask]), 1)
    return float(slope)
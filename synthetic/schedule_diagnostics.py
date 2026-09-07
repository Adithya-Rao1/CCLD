from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import torch

from core.coupling import build_coupling_matrix
from core.damping import calibrate_gammas_for_regime
from core.drift import _time_scale, drift_fn_n
from core.sde import build_g_matrix_n, reverse_step_n

VARIANTS: Dict[str, Dict[str, bool]] = {
    "baseline": {"scale_damping_with_time": True, "constant_k": False},
    "direction1_damping_undecayed": {"scale_damping_with_time": False, "constant_k": False},
    "direction2_no_double_decay": {"scale_damping_with_time": True, "constant_k": True},
    "direction1and2_combined": {"scale_damping_with_time": False, "constant_k": True},
}


def _conditioning(N: int, B: int, k_reference: float, dtype=torch.float32):
    K_self = [[torch.full((B, 1), k_reference, dtype=dtype)] for _ in range(N)]
    K_global = [torch.full((B, 1), k_reference, dtype=dtype) for _ in range(N)]
    return K_self, K_global


def build_state(N: int, alpha: List[float], damping_regime: str, k_reference: float, sigma: float):
    coupling = build_coupling_matrix(N, mode="mean_field")
    gamma = calibrate_gammas_for_regime(alpha, [k_reference] * N, damping_regime)
    G = build_g_matrix_n(torch.tensor(sigma), N, diffusion_mode="shared")
    return coupling, gamma, G


def monte_carlo_probe(
    N: int, coupling, gamma, alpha, beta, k_reference: float, T: int, dt: float, G,
    constant_k: bool, scale_damping_with_time: bool, prior_std: float = 1.0,
    batch: int = 20000, seed: int = 0,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    K_self, K_global = _conditioning(N, batch, k_reference)
    X = [[prior_std * torch.randn(batch, 1)] for _ in range(N)]
    V = [[prior_std * torch.randn(batch, 1)] for _ in range(N)]
    zero_score = [[torch.zeros(batch, 1)] for _ in range(N)]
    for t in reversed(range(1, T + 1)):
        X, V = reverse_step_n(
            X, V, K_self, K_global, zero_score, t, T, alpha, beta, gamma, coupling, True, constant_k, dt, G,
            scale_damping_with_time=scale_damping_with_time,
        )
    Xmat = torch.cat([X[i][0] for i in range(N)], dim=1)
    Vmat = torch.cat([V[i][0] for i in range(N)], dim=1)
    corr = torch.corrcoef(Xmat.T)
    off_diag = ~torch.eye(N, dtype=torch.bool)
    return {
        "mc_mean_x_corr": corr[off_diag].mean().item(),
        "mc_x_var_mean": Xmat.var(dim=0).mean().item(),
        "mc_v_var_mean": Vmat.var(dim=0).mean().item(),
        "mc_v_var_max": Vmat.var(dim=0).max().item(),
        "mc_any_nan": bool(torch.isnan(Xmat).any().item() or torch.isnan(Vmat).any().item()),
    }


def _extract_Avx_Avv(
    N: int, coupling, gamma, alpha, beta, k_reference: float, t: int, T: int,
    constant_k: bool, scale_damping_with_time: bool, time_scale_fn=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    K_self, K_global = _conditioning(N, 1, k_reference)
    t_tensor = torch.tensor(float(t))

    def eval_dV(x_vals: List[float], v_vals: List[float]) -> torch.Tensor:
        X = [[torch.tensor([[x_vals[i]]])] for i in range(N)]
        V = [[torch.tensor([[v_vals[i]]])] for i in range(N)]
        dV = drift_fn_n(
            X, V, K_self, K_global, t_tensor, T, alpha, beta, gamma,
            coupling_matrix=coupling, use_gamma=True, constant_k=constant_k,
            scale_damping_with_time=scale_damping_with_time, time_scale_fn=time_scale_fn,
        )
        return torch.tensor([dV[i][0].item() for i in range(N)])

    A_vx = torch.zeros(N, N)
    A_vv = torch.zeros(N, N)
    zeros = [0.0] * N
    for k in range(N):
        e = list(zeros)
        e[k] = 1.0
        A_vx[:, k] = eval_dV(e, zeros)
        A_vv[:, k] = eval_dV(zeros, e)
    return A_vx, A_vv


def _generative_transition(A_vx: torch.Tensor, A_vv: torch.Tensor, dt: float, time_scale=1.0) -> torch.Tensor:
    N = A_vx.shape[0]
    I = torch.eye(N)
    M = torch.zeros(2 * N, 2 * N)
    M[:N, :N] = I + time_scale * dt**2 * A_vx
    M[:N, N:] = time_scale * dt * (I + dt * A_vv)
    M[N:, :N] = dt * A_vx
    M[N:, N:] = I + dt * A_vv
    return M


def _corruption_transition(A_vx: torch.Tensor, A_vv: torch.Tensor, dt: float, time_scale=1.0) -> torch.Tensor:
    N = A_vx.shape[0]
    I = torch.eye(N)
    M = torch.zeros(2 * N, 2 * N)
    M[:N, :N] = I - time_scale * dt**2 * A_vx
    M[:N, N:] = time_scale * dt * (I - dt * A_vv)
    M[N:, :N] = -dt * A_vx
    M[N:, N:] = I - dt * A_vv
    return M


def corruption_spectral_radius(
    N: int, coupling, gamma, alpha, beta, k_reference: float, T: int, dt: float,
    constant_k: bool, scale_damping_with_time: bool, time_scale_fn=None,
    scale_kinematics_with_time: bool = True,
) -> Dict[str, float]:
    max_radius = 0.0
    n_unstable_steps = 0
    for t in range(1, T + 1):
        A_vx, A_vv = _extract_Avx_Avv(N, coupling, gamma, alpha, beta, k_reference, t, T, constant_k, scale_damping_with_time, time_scale_fn)
        time_scale = _time_scale(torch.tensor(float(t)), T, time_scale_fn)
        kin_scale = time_scale if scale_kinematics_with_time else 1.0
        M = _corruption_transition(A_vx, A_vv, dt, time_scale=kin_scale)
        radius = torch.linalg.eigvals(M).abs().max().item()
        max_radius = max(max_radius, radius)
        if radius > 1.0 + 1e-9:
            n_unstable_steps += 1
    return {"corrupt_max_spectral_radius": max_radius, "corrupt_n_unstable_steps": n_unstable_steps}


def _noise_injection(G: torch.Tensor, dt: float, time_scale=1.0) -> torch.Tensor:
    N = G.shape[0]
    sqrt_dt = dt**0.5
    Nmat = torch.zeros(2 * N, N)
    Nmat[:N, :] = time_scale * dt * sqrt_dt * G
    Nmat[N:, :] = sqrt_dt * G
    return Nmat


def exact_corruption_covariance(
    N: int, coupling, gamma, alpha, beta, k_reference: float, T: int, dt: float, G,
    cov0: torch.Tensor, constant_k: bool = False, scale_damping_with_time: bool = True,
    time_scale_fn=None, scale_kinematics_with_time: bool = True, scale_diffusion_with_time: bool = True,
) -> torch.Tensor:
    Sigma = cov0.clone()
    for t in range(1, T + 1):
        A_vx, A_vv = _extract_Avx_Avv(N, coupling, gamma, alpha, beta, k_reference, t, T, constant_k, scale_damping_with_time, time_scale_fn)
        time_scale = _time_scale(torch.tensor(float(t)), T, time_scale_fn)
        kin_scale = time_scale if scale_kinematics_with_time else 1.0
        G_t = G * time_scale.clamp_min(0).sqrt() if scale_diffusion_with_time else G
        M = _corruption_transition(A_vx, A_vv, dt, time_scale=kin_scale)
        Nmat = _noise_injection(G_t, dt, time_scale=kin_scale)
        Sigma = M @ Sigma @ M.T + Nmat @ Nmat.T
    return Sigma


def corruption_snr_timeseries(
    N: int, coupling, gamma, alpha, beta, k_reference: float, T: int, dt: float, G,
    constant_k: bool, scale_damping_with_time: bool, time_scale_fn=None,
    signal_cov0_x: float = 1.0, G_fn=None,
    scale_kinematics_with_time: bool = True, scale_diffusion_with_time: bool = True,
) -> List[Dict[str, float]]:
    Signal = torch.zeros(2 * N, 2 * N)
    Signal[:N, :N] = signal_cov0_x * torch.eye(N)
    Noise = torch.zeros(2 * N, 2 * N)

    rows = []
    for t in range(1, T + 1):
        A_vx, A_vv = _extract_Avx_Avv(N, coupling, gamma, alpha, beta, k_reference, t, T, constant_k, scale_damping_with_time, time_scale_fn)
        time_scale = _time_scale(torch.tensor(float(t)), T, time_scale_fn)
        kin_scale = time_scale if scale_kinematics_with_time else 1.0
        M = _corruption_transition(A_vx, A_vv, dt, time_scale=kin_scale)
        G_t = G_fn(t, T) if G_fn is not None else G
        if scale_diffusion_with_time:
            G_t = G_t * time_scale.clamp_min(0).sqrt()
        Nmat = _noise_injection(G_t, dt, time_scale=kin_scale)
        Q = Nmat @ Nmat.T

        Signal = M @ Signal @ M.T
        Noise = M @ Noise @ M.T + Q

        t_tensor = torch.tensor(float(t))
        ts_val = time_scale_fn(t_tensor, T).item() if time_scale_fn is not None else ((T - t) / (t + T))
        sigma_val = float(G_t.diagonal().mean().item())

        sig_x = Signal[:N, :N].diagonal().mean().item()
        noise_x = Noise[:N, :N].diagonal().mean().item()
        sig_v = Signal[N:, N:].diagonal().mean().item()
        noise_v = Noise[N:, N:].diagonal().mean().item()
        eps = 1e-30
        rows.append({
            "t": t, "T": T, "time_scale": ts_val, "sigma_t": sigma_val,
            "signal_var_x": sig_x, "noise_var_x": noise_x, "snr_x": sig_x / (noise_x + eps),
            "signal_var_v": sig_v, "noise_var_v": noise_v, "snr_v": sig_v / (noise_v + eps),
            "total_var_x": sig_x + noise_x, "total_var_v": sig_v + noise_v,
        })
    return rows


def propagate_exact(
    N: int, coupling, gamma, alpha, beta, k_reference: float, T: int, dt: float, G,
    constant_k: bool, scale_damping_with_time: bool, prior_std: float = 1.0,
    scale_kinematics_with_time: bool = True, scale_diffusion_with_time: bool = True,
) -> Dict[str, float]:
    Sigma = prior_std**2 * torch.eye(2 * N)
    for t in reversed(range(1, T + 1)):
        A_vx, A_vv = _extract_Avx_Avv(N, coupling, gamma, alpha, beta, k_reference, t, T, constant_k, scale_damping_with_time)
        time_scale = _time_scale(torch.tensor(float(t)), T, None)
        kin_scale = time_scale if scale_kinematics_with_time else 1.0
        G_t = G * time_scale.clamp_min(0).sqrt() if scale_diffusion_with_time else G
        M = _generative_transition(A_vx, A_vv, dt, time_scale=kin_scale)
        Nmat = _noise_injection(G_t, dt, time_scale=kin_scale)
        Sigma = M @ Sigma @ M.T + Nmat @ Nmat.T

    x_cov = Sigma[:N, :N]
    v_cov = Sigma[N:, N:]
    x_std = x_cov.diagonal().clamp_min(1e-12).sqrt()
    corr = x_cov / (x_std.unsqueeze(0) * x_std.unsqueeze(1))
    off_diag = ~torch.eye(N, dtype=torch.bool)
    return {
        "exact_mean_x_corr": corr[off_diag].mean().item(),
        "exact_x_var_mean": x_cov.diagonal().mean().item(),
        "exact_v_var_mean": v_cov.diagonal().mean().item(),
        "exact_v_var_max": v_cov.diagonal().max().item(),
        "exact_any_nonfinite": bool(not torch.isfinite(Sigma).all().item()),
    }


def run_grid(
    N: int = 3, alpha=(1.0, 1.0, 1.0), beta=(0.5, 0.5, 0.5), k_reference: float = 1.0,
    damping_regime: str = "critically_damped", sigma: float = 0.3, dt: float = 0.05,
    step_counts: List[int] = (5, 20, 50, 100, 200), variants: Dict[str, Dict[str, bool]] = VARIANTS,
    mc_batch: int = 20000, prior_std: float = 1.0,
) -> List[Dict]:
    alpha, beta = list(alpha), list(beta)
    rows = []
    for T in step_counts:
        for name, cfg in variants.items():
            coupling, gamma, G = build_state(N, alpha, damping_regime, k_reference, sigma)
            row = {"n_diff_steps": T, "variant": name}
            row.update(monte_carlo_probe(
                N, coupling, gamma, alpha, beta, k_reference, T, dt, G, batch=mc_batch, prior_std=prior_std, **cfg,
            ))
            row.update(propagate_exact(
                N, coupling, gamma, alpha, beta, k_reference, T, dt, G, prior_std=prior_std, **cfg,
            ))
            row.update(corruption_spectral_radius(
                N, coupling, gamma, alpha, beta, k_reference, T, dt, **cfg,
            ))
            rows.append(row)
    return rows


def print_table(rows: List[Dict]) -> None:
    cols = [
        "n_diff_steps", "variant",
        "mc_mean_x_corr", "exact_mean_x_corr",
        "mc_v_var_mean", "exact_v_var_mean", "mc_v_var_max", "exact_v_var_max",
        "corrupt_max_spectral_radius", "corrupt_n_unstable_steps",
        "mc_any_nan", "exact_any_nonfinite",
    ]
    widths = {c: max(len(c), 10) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4g}".ljust(widths[c]))
            else:
                cells.append(str(v).ljust(widths[c]))
        print(" | ".join(cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step-counts", default="5,20,50,100,200")
    p.add_argument("--mc-batch", type=int, default=20000)
    p.add_argument("--prior-std", type=float, default=1.0)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    step_counts = [int(s) for s in args.step_counts.split(",") if s.strip()]
    rows = run_grid(step_counts=step_counts, mc_batch=args.mc_batch, prior_std=args.prior_std)
    print_table(rows)

    if args.out_csv:
        import csv
        import os
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()

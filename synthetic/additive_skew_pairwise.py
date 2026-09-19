from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from core.drift import _time_scale
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas_spectral, calibrate_sigma_fdt_spectral
from synthetic.exact_dsm import (
    _extract_Avx_Avv_coupled_gamma, closed_form_propagator_skew_additive, elapsed_time_at_step,
    exact_reverse_step_matrices_additive, sample_and_analytic_score_target_skew,
)
from synthetic.ground_truth_sde import GroundTruthCoupledOU
from synthetic.run_experiment import CoupledScoreNet, evaluate_sampling_quality
from synthetic.skew_coupling import inject_skew_coupling_additive, stationary_covariance_from_drift

ALPHA_V = 1.0
K_REFERENCE = 1.0
TARGET_ZETA = 1.0
N_DIFF_STEPS = 32
DT = 1.0 / N_DIFF_STEPS
BATCH_SIZE = 256
BETA = 1.0
TIME_SCALE_FN = lambda t, T: torch.tensor(2.0)  # noqa: E731

N_TRAIN_ITERS = 2000
N_SAMPLES = 4000


def base_A0(N: int, coupling: torch.Tensor, Gamma: torch.Tensor, device) -> torch.Tensor:
    A_vx0, A_vv0 = _extract_Avx_Avv_coupled_gamma(
        N, None, None, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, coupling,
        t=1, T=1, constant_k=True, time_scale_fn=lambda t, T: torch.tensor(1.0), damping_matrix=Gamma,
    )
    A0 = torch.zeros(2 * N, 2 * N, device=device)
    A0[:N, N:] = torch.eye(N, device=device)
    A0[N:, :N] = A_vx0.to(device)
    A0[N:, N:] = A_vv0.to(device)
    return A0


def actual_stationary_covariance(N: int, coupling: torch.Tensor, skew_matrix: Optional[torch.Tensor], device) -> torch.Tensor:
    Gamma = calibrate_coupled_gammas_spectral(ALPHA_V, BETA, K_REFERENCE, K_REFERENCE, coupling, target_zeta=TARGET_ZETA)
    G0 = calibrate_sigma_fdt_spectral(Gamma, coupling, target_variance=1.0)
    A0 = base_A0(N, coupling, Gamma, device)
    if skew_matrix is not None:
        A0 = inject_skew_coupling_additive(A0, skew_matrix.to(device))
    L = torch.zeros(2 * N, N, device=device)
    L[N:, :] = G0.to(device)
    return stationary_covariance_from_drift(A0, L @ L.T)


def _prior_std_analytic(N: int, gt: GroundTruthCoupledOU, coupling: torch.Tensor, Gamma: torch.Tensor, G0: torch.Tensor,
                         skew_matrix: Optional[torch.Tensor], device) -> Tuple[List[float], List[float]]:
    T_total = elapsed_time_at_step(N_DIFF_STEPS, N_DIFF_STEPS, DT, TIME_SCALE_FN)
    Phi_T, Sigma_T = closed_form_propagator_skew_additive(
        N, None, None, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, coupling, tau_hat=T_total,
        constant_k=True, G0=G0, damping_matrix=Gamma, skew_matrix=skew_matrix,
    )
    Cov_X0 = gt.stationary_covariance().to(device)
    Cov_Z0 = torch.zeros(2 * N, 2 * N, device=device)
    Cov_Z0[:N, :N] = Cov_X0
    Sigma_total = Sigma_T + Phi_T @ Cov_Z0 @ Phi_T.T
    prior_std_x = [max(Sigma_total[i, i].clamp_min(1e-8).sqrt().item(), 1e-3) for i in range(N)]
    prior_std_v = [max(Sigma_total[N + i, N + i].clamp_min(1e-8).sqrt().item(), 1e-3) for i in range(N)]
    return prior_std_x, prior_std_v


def train_ccld_pairwise_additive(N: int, coupling: torch.Tensor, gt: GroundTruthCoupledOU, device, desc: str = "",
                                  skew_matrix: Optional[torch.Tensor] = None):
    Gamma = calibrate_coupled_gammas_spectral(ALPHA_V, BETA, K_REFERENCE, K_REFERENCE, coupling, target_zeta=TARGET_ZETA)
    G0 = calibrate_sigma_fdt_spectral(Gamma, coupling, target_variance=1.0)

    params_skew = [
        closed_form_propagator_skew_additive(
            N, None, None, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, coupling,
            tau_hat=elapsed_time_at_step(t_idx, N_DIFF_STEPS, DT, TIME_SCALE_FN),
            constant_k=True, G0=G0, damping_matrix=Gamma, skew_matrix=skew_matrix,
        )
        for t_idx in range(1, N_DIFF_STEPS + 1)
    ]

    score_net = CoupledScoreNet(N, 64, 3, 16).to(device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-3)

    pbar = tqdm(total=N_TRAIN_ITERS, desc=desc or f"[N={N}] train")
    ema_loss = None
    for it in range(N_TRAIN_ITERS):
        X0 = gt.sample_stationary(BATCH_SIZE).to(device)
        Z0 = torch.cat([X0, torch.zeros_like(X0)], dim=1)
        t_idx = torch.randint(1, N_DIFF_STEPS + 1, (1,)).item()
        Phi_t, Sigma_t = params_skew[t_idx - 1]
        Zt, score_v_target = sample_and_analytic_score_target_skew(Z0, Phi_t, Sigma_t)
        X_t = [[Zt[:, i:i + 1]] for i in range(N)]
        V_t = [[Zt[:, N + i:N + i + 1]] for i in range(N)]

        score_pred = score_net(X_t, V_t, t_idx)
        loss = sum(((score_pred[i][0] - score_v_target[:, i:i + 1]) ** 2).mean() for i in range(N))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}", ema_loss=f"{ema_loss:.4f}", grad_norm=f"{grad_norm.item():.3f}", t_idx=t_idx)
        pbar.update(1)
    pbar.close()

    prior_std = _prior_std_analytic(N, gt, coupling, Gamma, G0, skew_matrix, device)
    return score_net, Gamma, G0, prior_std


@torch.no_grad()
def sample_ccld_anderson_exact_additive(N: int, coupling: torch.Tensor, score_net, Gamma, G0, prior_std, device,
                                         desc: str = "", skew_matrix: Optional[torch.Tensor] = None):
    prior_std_x, prior_std_v = prior_std
    X = [[prior_std_x[i] * torch.randn(N_SAMPLES, 1, device=device)] for i in range(N)]
    V = [[prior_std_v[i] * torch.randn(N_SAMPLES, 1, device=device)] for i in range(N)]
    Sigma_diff = G0 @ G0.T

    pbar = tqdm(total=N_DIFF_STEPS, desc=desc or f"[N={N}] sample-exact-additive")
    for t_idx in reversed(range(1, N_DIFF_STEPS + 1)):
        score_outputs = score_net(X, V, t_idx)
        time_scale = _time_scale(torch.as_tensor(float(t_idx)), N_DIFF_STEPS, TIME_SCALE_FN)
        tau_hat_step = float(time_scale.clamp_min(0).item()) * DT

        Phi_rev, Sigma_rev, Psi_rev = exact_reverse_step_matrices_additive(
            N, None, None, [ALPHA_V] * N, [BETA] * N, K_REFERENCE, coupling,
            tau_hat=tau_hat_step, constant_k=True, G0=G0, damping_matrix=Gamma, skew_matrix=skew_matrix,
        )

        Z = torch.cat([torch.cat([X[i][0] for i in range(N)], dim=-1),
                       torch.cat([V[i][0] for i in range(N)], dim=-1)], dim=-1)
        score_vec = torch.cat([score_outputs[i][0] for i in range(N)], dim=-1)
        b_v = score_vec @ Sigma_diff.T
        b = torch.cat([torch.zeros(N_SAMPLES, N, device=device), b_v], dim=-1)

        mean_new = Z @ Phi_rev.T + b @ Psi_rev.T
        scale = Sigma_rev.diagonal().abs().max().clamp_min(1.0)
        Sigma_reg = Sigma_rev + 1e-6 * scale * torch.eye(2 * N, device=device, dtype=Sigma_rev.dtype)
        L_chol = torch.linalg.cholesky(Sigma_reg)
        eps = torch.randn(N_SAMPLES, 2 * N, device=device)
        Z_new = mean_new + eps @ L_chol.T

        X = [[Z_new[:, i:i + 1]] for i in range(N)]
        V = [[Z_new[:, N + i:N + i + 1]] for i in range(N)]
        pbar.set_postfix(t_idx=t_idx, x_std=f"{X[0][0].std().item():.3f}")
        pbar.update(1)
    pbar.close()
    return torch.cat([X[i][0] for i in range(N)], dim=-1)


def train_one_seed_additive(N: int, coupling: torch.Tensor, seed: int, label: str = "",
                            skew_matrix: Optional[torch.Tensor] = None, gt: Optional[GroundTruthCoupledOU] = None,
                            return_samples: bool = False) -> Dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"[N={N}/{label}] seed={seed}" if label else f"[N={N}] seed={seed}"
    skew_matrix_dev = skew_matrix.to(device) if skew_matrix is not None else None
    score_net, Gamma, G0, prior_std = train_ccld_pairwise_additive(
        N, coupling, gt, device, desc=f"{tag} train", skew_matrix=skew_matrix_dev,
    )
    generated = sample_ccld_anderson_exact_additive(
        N, coupling, score_net, Gamma, G0, prior_std, device, desc=f"{tag} sample-exact", skew_matrix=skew_matrix_dev,
    )
    metrics = evaluate_sampling_quality(generated, gt)
    if return_samples:
        return metrics, generated
    return metrics

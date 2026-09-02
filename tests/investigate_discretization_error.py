from __future__ import annotations

from typing import List

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

N_SWEEP: List[int] = [2, 3, 4, 5]
STEP_COUNT_SWEEP: List[int] = [8, 16, 32, 64, 128, 20]  
ALPHA_V = 1.0
BETA = 0.5
K_REFERENCE = 1.0
TARGET_ZETA = 1.0
FLAG_THRESHOLD = 0.05 


def _vp_linear_time_scale(t, T):
    return torch.as_tensor(t, dtype=torch.float64) / float(T)


SCHEDULES = {
    "default": (None, tau_hat_default_schedule),
    "vp_linear": (_vp_linear_time_scale, tau_hat_vp_linear_schedule),
}


def run_one(N: int, n_diff_steps: int, schedule_name: str) -> dict:
    time_scale_fn, tau_hat_fn = SCHEDULES[schedule_name]
    coupling = build_coupling_matrix(N, mode="mean_field") if N > 1 else None
    alpha = [ALPHA_V] * N
    beta = [BETA] * N
    gamma_self, gamma_couple = calibrate_coupled_gammas(ALPHA_V, BETA, K_REFERENCE, K_REFERENCE, N, target_zeta=TARGET_ZETA)

    dt = 1.0 / n_diff_steps  
    g_fn = lambda t, T: build_g_matrix_n(torch.tensor(1.0), N, diffusion_mode="shared").double()
    params = precompute_transition_params(
        N, gamma_self, gamma_couple, alpha, beta, K_REFERENCE, coupling,
        n_diff_steps, dt, g_fn, constant_k=True, time_scale_fn=time_scale_fn,
    )
    Phi_disc, Sigma_disc = params[-1]

    tau_hat = tau_hat_fn(n_diff_steps, dt)
    Phi_cf, Sigma_cf = closed_form_propagator(
        N, gamma_self, gamma_couple, alpha, beta, K_REFERENCE, coupling, tau_hat,
        constant_k=True, sigma_ref=1.0, dtype=torch.float64,
    )

    phi_err = ((Phi_cf - Phi_disc).norm() / Phi_disc.norm().clamp_min(1e-12)).item()
    sigma_err = ((Sigma_cf - Sigma_disc).norm() / Sigma_disc.norm().clamp_min(1e-12)).item()
    return {
        "N": N, "n_diff_steps": n_diff_steps, "schedule": schedule_name,
        "phi_rel_err": phi_err, "sigma_rel_err": sigma_err,
        "flagged": phi_err > FLAG_THRESHOLD or sigma_err > FLAG_THRESHOLD,
    }


def run_all() -> List[dict]:
    rows = []
    for N in N_SWEEP:
        for schedule_name in SCHEDULES:
            for n_diff_steps in STEP_COUNT_SWEEP:
                rows.append(run_one(N, n_diff_steps, schedule_name))
    return rows


def print_table(rows: List[dict]) -> None:
    header = f"{'N':>3} {'schedule':10} {'n_diff_steps':>12} {'phi_rel_err':>12} {'sigma_rel_err':>14} {'flag':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        flag = "FLAG" if r["flagged"] else ""
        print(f"{r['N']:>3} {r['schedule']:10} {r['n_diff_steps']:>12} "
              f"{r['phi_rel_err']:>12.4e} {r['sigma_rel_err']:>14.4e} {flag:>6}")


if __name__ == "__main__":
    rows = run_all()
    print_table(rows)
    n_flagged = sum(r["flagged"] for r in rows)
    prod_rows = [r for r in rows if r["n_diff_steps"] == 20]
    prod_flagged = sum(r["flagged"] for r in prod_rows)
    print(f"\n{n_flagged}/{len(rows)} configs flagged (rel_err > {FLAG_THRESHOLD}) overall; "
          f"{prod_flagged}/{len(prod_rows)} at the actual production n_diff_steps=20.")
    if n_flagged:
        import sys
        sys.exit(1)

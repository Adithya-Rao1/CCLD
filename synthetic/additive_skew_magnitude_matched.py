from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch

import synthetic.additive_skew_learning_diagnostics as asld
import synthetic.additive_skew_pairwise as asp
from core.coupling import build_coupling_matrix
from core.reporting import write_csv
from synthetic.additive_skew_experiment import build_generic_skew_additive, build_structured_skew_additive
from synthetic.additive_skew_learning_diagnostics import score_target_complexity, train_with_diagnostics
from synthetic.drift_coupled_gamma import calibrate_coupled_gammas_spectral, calibrate_sigma_fdt_spectral
from synthetic.ground_truth_sde import make_asymmetric_ground_truth
from synthetic.run_experiment import evaluate_sampling_quality

N_SWEEP = [2, 3, 4]
COUPLING_STRENGTH = 0.6
STRUCTURED_SCALE = 10.0
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_ITERS = 2000
EVAL_EVERY = 100
HELD_OUT_BATCH = 512
N_DIFF_STEPS = 32
OUT_DIR = "results/experiment_3_synthetic/additive_skew_magnitude_matched"


def match_frobenius_norm(J: torch.Tensor, target_norm: float) -> torch.Tensor:
    current_norm = torch.linalg.norm(J)
    if current_norm < 1e-12:
        raise ValueError
    return J * (target_norm / current_norm)


def build_conditions(N: int, gt, coupling: torch.Tensor, device) -> Tuple[List[Tuple[str, torch.Tensor]], float, float]:
    J_structured = build_structured_skew_additive(gt, coupling, device, scale=STRUCTURED_SCALE)
    target_norm = torch.linalg.norm(J_structured).item()
    J_generic_raw = build_generic_skew_additive(N, seed=0, device=device, scale=1.0)
    J_generic_matched = match_frobenius_norm(J_generic_raw, target_norm)
    generic_norm = torch.linalg.norm(J_generic_matched).item()
    conditions = [
        ("symmetric_only", None),
        ("skew_structured", J_structured),
        ("skew_generic_matched", J_generic_matched),
    ]
    return conditions, target_norm, generic_norm


def run() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    asp.N_DIFF_STEPS = N_DIFF_STEPS
    asp.DT = 1.0 / N_DIFF_STEPS
    asld.N_DIFF_STEPS = N_DIFF_STEPS

    norm_rows: List[Dict] = []
    loss_curve_all: List[Dict] = []
    profile_all: List[Dict] = []
    complexity_all: List[Dict] = []
    quality_all: List[Dict] = []

    t_idx_profile = list(range(1, N_DIFF_STEPS + 1))

    for N in N_SWEEP:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt = make_asymmetric_ground_truth(N, COUPLING_STRENGTH, seed=0, device=device)
        coupling = build_coupling_matrix(N, mode="mean_field", device=device)

        conditions, target_norm, generic_norm = build_conditions(N, gt, coupling, device)
        norm_rows.append({"N": N, "structured_norm": target_norm, "generic_matched_norm": generic_norm})
        print(f"N={N}: ||J_structured||_F={target_norm:.4f}  ||J_generic_matched||_F={generic_norm:.4f}")

        for label, skew_matrix in conditions:
            rows = score_target_complexity(N, coupling, device, skew_matrix, t_idx_profile)
            for row in rows:
                complexity_all.append({"N": N, "condition": label, **row})

            for seed in SEEDS:
                tag = f"[N={N}/matched/{label}] seed={seed}"
                score_net, loss_curve_rows, profile_rows = train_with_diagnostics(
                    N, coupling, gt, device, skew_matrix, seed, label=tag,
                    n_train_iters=N_TRAIN_ITERS, eval_every=EVAL_EVERY, held_out_batch=HELD_OUT_BATCH,
                )
                for row in loss_curve_rows:
                    loss_curve_all.append({"N": N, "condition": label, "seed": seed, **row})
                for row in profile_rows:
                    profile_all.append({"N": N, "condition": label, "seed": seed, **row})

                Gamma = calibrate_coupled_gammas_spectral(
                    asp.ALPHA_V, asp.BETA, asp.K_REFERENCE, asp.K_REFERENCE, coupling, target_zeta=asp.TARGET_ZETA,
                )
                G0 = calibrate_sigma_fdt_spectral(Gamma, coupling, target_variance=1.0)
                prior_std = asp._prior_std_analytic(N, gt, coupling, Gamma, G0, skew_matrix, device)
                generated = asp.sample_ccld_anderson_exact_additive(
                    N, coupling, score_net, Gamma, G0, prior_std, device,
                    desc=f"{tag} sample-exact", skew_matrix=skew_matrix,
                )
                metrics = evaluate_sampling_quality(generated, gt)
                quality_all.append({"N": N, "condition": label, "seed": seed, **metrics})

    write_csv(norm_rows, os.path.join(OUT_DIR, "matched_norms.csv"))
    write_csv(loss_curve_all, os.path.join(OUT_DIR, "held_out_loss_curves.csv"))
    write_csv(profile_all, os.path.join(OUT_DIR, "per_timestep_score_error_profile.csv"))
    write_csv(complexity_all, os.path.join(OUT_DIR, "score_target_complexity.csv"))
    write_csv(quality_all, os.path.join(OUT_DIR, "final_sample_quality.csv"))
    print(f"\nWrote results to {OUT_DIR}/")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structure tests for same magnitude skew_structured vs skew_generic")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-train-iters", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--held-out-batch", type=int, default=512)
    p.add_argument("--n-sweep", default="2,3,4")
    p.add_argument("--n-diff-steps", type=int, default=32)
    p.add_argument("--structured-scale", type=float, default=10.0)
    p.add_argument("--out-dir", default="results/experiment_3_synthetic/additive_skew_magnitude_matched")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    SEEDS = [int(s) for s in _args.seeds.split(",") if s.strip()]
    N_TRAIN_ITERS = _args.n_train_iters
    EVAL_EVERY = _args.eval_every
    HELD_OUT_BATCH = _args.held_out_batch
    N_SWEEP = [int(n) for n in _args.n_sweep.split(",") if n.strip()]
    N_DIFF_STEPS = _args.n_diff_steps
    STRUCTURED_SCALE = _args.structured_scale
    OUT_DIR = _args.out_dir
    if _args.quick:
        SEEDS = [0]
        N_TRAIN_ITERS = 20
        EVAL_EVERY = 5
        HELD_OUT_BATCH = 32
        N_SWEEP = [2, 3]
        N_DIFF_STEPS = 8
        STRUCTURED_SCALE = 1.0
    run()
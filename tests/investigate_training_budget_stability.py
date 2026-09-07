from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import numpy as np
import scipy.io as sio

from pde.run_experiment import parse_args, train_one_seed

SCORE_ARCHES = ["attention", "fno", "unet_model"]
PROBLEMS = ["TE_heat", "E_flow", "VA"]
N_EPOCHS_GRID = [5, 20, 50, 100]  # 50 matches production (A2)

RESOLUTION = 32  # A2
N_SAMPLES = 64  # A2
BATCH_SIZE = 16  # A2
SEED = 0
DIVERGENCE_THRESHOLD = 5.0  # A4, tests/test_experiments_smoke.py bound

_ARCH_ARGS = {  # A3, moderate configs
    "attention": ["--backbone-channels", "64", "--base-channels", "16", "--n-downsample", "2",
                  "--latent-dim", "32", "--score-blocks", "2", "--score-heads", "2"],
    "fno": ["--score-arch", "fno", "--fno-modes", "8,8", "--fno-hidden-channels", "64",
            "--fno-init-channels", "16", "--base-channels", "16", "--n-downsample", "2"],
    "unet_model": ["--score-arch", "unet_model", "--unet-model-channels", "16",
                 "--unet-channel-mult", "1,2", "--unet-num-blocks", "2",
                 "--fno-init-channels", "16", "--base-channels", "16", "--n-downsample", "2"],
}


def _write_te_heat_fixture(root: str) -> None:
    rng = np.random.default_rng(SEED)
    problem_root = os.path.join(root, "training", "TE_heat")
    for field in ("mater", "T", "Ez", "ellipticcsv"):
        os.makedirs(os.path.join(problem_root, field), exist_ok=True)
    for sid in range(1, N_SAMPLES + 1):
        mater = rng.random((RESOLUTION, RESOLUTION)) + 1.0
        sio.savemat(os.path.join(problem_root, "mater", f"{sid}.mat"), {"mater": mater})
        T = rng.random((RESOLUTION, RESOLUTION)) * 3.7 + 300.9  # real range 300.9-304.6K
        sio.savemat(os.path.join(problem_root, "T", f"{sid}.mat"), {"T": T})
        Ez = (2.7e5 + 1.1e5 * rng.standard_normal((RESOLUTION, RESOLUTION))
              + 1j * (2.7e5 + 1.1e5 * rng.standard_normal((RESOLUTION, RESOLUTION))))
        sio.savemat(os.path.join(problem_root, "Ez", f"{sid}.mat"), {"Ez": Ez})
        np.savetxt(os.path.join(problem_root, "ellipticcsv", f"{sid}.csv"),
                    np.array([[4.0, 3.0, 30.0]]), delimiter=",")


def _write_e_flow_fixture(root: str) -> None:
    rng = np.random.default_rng(SEED)
    problem_root = os.path.join(root, "training", "E_flow")
    for field in ("kappa", "ec_V", "u_flow", "v_flow"):
        os.makedirs(os.path.join(problem_root, field), exist_ok=True)
    for sid in range(1, N_SAMPLES + 1):
        kappa = rng.random((RESOLUTION, RESOLUTION)) + 0.5
        sio.savemat(os.path.join(problem_root, "kappa", f"{sid}.mat"), {"kappa": kappa})
        ec_V = 25.8 + 5.0 * rng.standard_normal((RESOLUTION, RESOLUTION))
        sio.savemat(os.path.join(problem_root, "ec_V", f"{sid}.mat"), {"ec_V": ec_V})
        for name in ("u_flow", "v_flow"):
            field_vals = 0.005 + 0.002 * rng.standard_normal((RESOLUTION, RESOLUTION))
            sio.savemat(os.path.join(problem_root, name, f"{sid}.mat"), {name: field_vals})


def _write_va_fixture(root: str) -> None:
    rng = np.random.default_rng(SEED)
    problem_root = os.path.join(root, "training", "VA")
    small_fields = ("p_t", "Sxx", "Sxy", "Syy")
    large_fields = ("x_u", "x_v")
    for field in ("rho_water",) + small_fields + large_fields:
        os.makedirs(os.path.join(problem_root, field), exist_ok=True)
    for sid in range(1, N_SAMPLES + 1):
        rho = rng.random((RESOLUTION, RESOLUTION)) * 200 + 900.0
        sio.savemat(os.path.join(problem_root, "rho_water", f"{sid}.mat"), {"rho_water": rho})
        for name in small_fields:
            arr = (0.06 + 0.02 * rng.standard_normal((RESOLUTION, RESOLUTION))
                   + 1j * (0.06 + 0.02 * rng.standard_normal((RESOLUTION, RESOLUTION))))
            sio.savemat(os.path.join(problem_root, name, f"{sid}.mat"), {name: arr})
        for name in large_fields:
            arr = (18.0 + 4.0 * rng.standard_normal((RESOLUTION, RESOLUTION))
                   + 1j * (18.0 + 4.0 * rng.standard_normal((RESOLUTION, RESOLUTION))))
            sio.savemat(os.path.join(problem_root, name, f"{sid}.mat"), {name: arr})


_FIXTURE_WRITERS = {"TE_heat": _write_te_heat_fixture, "E_flow": _write_e_flow_fixture, "VA": _write_va_fixture}


def run_combo(problem: str, score_arch: str, n_epochs: int, data_root: str, device: str) -> dict:
    argv = [
        "--data-root", data_root, "--problem", problem, "--method", "csho",
        "--n-diff-steps", "20", 
        "--batch-size", str(BATCH_SIZE), "--n-epochs", str(n_epochs), "--seeds", str(SEED),
        "--max-samples", str(N_SAMPLES), "--image-size", str(RESOLUTION),
        "--num-workers", "0", "--device", device,
        "--out-dir", os.path.join(data_root, "out"),
        *_ARCH_ARGS[score_arch],
    ]
    args = parse_args(argv)
    t0 = time.time()
    metrics = train_one_seed(args, seed=SEED)
    elapsed = time.time() - t0
    rel_l2 = {k: v for k, v in metrics.items() if k.endswith("_rel_l2")}
    max_rel_l2 = max(rel_l2.values()) if rel_l2 else float("nan")
    diverged = max_rel_l2 > DIVERGENCE_THRESHOLD or metrics.get("nan_events", 0) > 0
    return {
        "max_rel_l2": max_rel_l2, "nan_events": metrics.get("nan_events", 0.0),
        "explosion_events": metrics.get("explosion_events", 0.0), "diverged": diverged,
        "elapsed_s": elapsed,
    }


def run_all(device: str) -> None:
    tmp = tempfile.mkdtemp(prefix="csho_training_budget_")
    try:
        for problem in PROBLEMS:
            _FIXTURE_WRITERS[problem](tmp)

        print(f"{'problem':8s} {'arch':9s} {'n_epochs':8s} {'max_rel_l2':11s} "
              f"{'nan_events':10s} {'explosions':10s} {'status':10s} {'time_s':8s}")
        for problem in PROBLEMS:
            for score_arch in SCORE_ARCHES:
                for n_epochs in N_EPOCHS_GRID:
                    try:
                        r = run_combo(problem, score_arch, n_epochs, tmp, device)
                    except Exception as e:
                        print(f"{problem:8s} {score_arch:9s} {n_epochs:<8d} ERROR: {e}")
                        continue
                    status = "DIVERGED" if r["diverged"] else "stable"
                    print(f"{problem:8s} {score_arch:9s} {n_epochs:<8d} {r['max_rel_l2']:<11.4f} "
                          f"{r['nan_events']:<10.0f} {r['explosion_events']:<10.0f} {status:10s} "
                          f"{r['elapsed_s']:<8.1f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    cli_args = parser.parse_args()
    run_all(cli_args.device)

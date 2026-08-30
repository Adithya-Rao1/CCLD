from __future__ import annotations

import io
import os
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _png_bytes(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _jpg_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG")
    return buf.getvalue()


def test_experiment_1_vision_smoke():
    from image.run_experiment import parse_args, train_one_seed

    tmp = tempfile.mkdtemp(prefix="csho_smoke_exp1_")
    try:
        rng = np.random.default_rng(0)
        n = 4
        rgb_cells, depth_cells, ids = [], [], []
        for i in range(n):
            rgb_cells.append({"bytes": _jpg_bytes((rng.random((16, 16, 3)) * 255).astype(np.uint8)), "path": None})
            depth_cells.append({"bytes": _png_bytes((rng.random((16, 16)) * 255).astype(np.uint8), "L"), "path": None})
            ids.append(f"id{i}")
        table = pa.table({
            "rgb": pa.StructArray.from_arrays(
                [pa.array([c["bytes"] for c in rgb_cells], type=pa.binary()), pa.array([c["path"] for c in rgb_cells], type=pa.string())],
                ["bytes", "path"],
            ),
            "depth": pa.StructArray.from_arrays(
                [pa.array([c["bytes"] for c in depth_cells], type=pa.binary()), pa.array([c["path"] for c in depth_cells], type=pa.string())],
                ["bytes", "path"],
            ),
            "id": pa.array(ids, type=pa.string()),
        })
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir, exist_ok=True)
        pq.write_table(table, os.path.join(data_dir, "train-00000-of-00001.parquet"))
        pq.write_table(table, os.path.join(data_dir, "validation-00000-of-00001.parquet"))

        argv = [
            "--data-root", tmp, "--tasks", "depth,normals", "--source", "nyudv2", "--method", "csho",
            "--n-diff-steps", "2", "--batch-size", "2", "--n-epochs", "1", "--seeds", "0",
            "--image-size", "16", "--latent-dim", "8", "--backbone-channels", "16",
            "--no-pretrained-backbone", "--score-blocks", "1", "--score-heads", "1",
            "--score-spatial-stride", "2", "--num-workers", "0", "--device", "cpu",
            "--out-dir", os.path.join(tmp, "out"),
        ]
        args = parse_args(argv)
        metrics = train_one_seed(args, seed=0)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_experiment_2_physics_smoke():
    from pde.run_experiment import parse_args, train_one_seed

    tmp = tempfile.mkdtemp(prefix="csho_smoke_exp2_")
    try:
        rng = np.random.default_rng(0)
        # pde/dataset.py's NS_heat spec expects a directory tree of per-field CSVs:
        # {data_root}/{split}/NS_heat/{field_dir}/{sample_idx}.csv (first-match field
        # dir names, from PROBLEM_SPECS["NS_heat"]: input "Q_heat", outputs "u_u",
        # "u_v", and "u_T" (first candidate of ["u_T", "T"])).
        problem_root = os.path.join(tmp, "training", "NS_heat")
        for field in ["Q_heat", "u_u", "u_v", "u_T"]:
            field_dir = os.path.join(problem_root, field)
            os.makedirs(field_dir, exist_ok=True)
            for sid in ["1", "2", "3", "4"]:
                np.savetxt(os.path.join(field_dir, f"{sid}.csv"), rng.random((8, 8)), delimiter=",")

        argv = [
            "--data-root", tmp, "--problem", "NS_heat", "--method", "csho", "--n-diff-steps", "2",
            "--batch-size", "2", "--n-epochs", "1", "--seeds", "0", "--max-samples", "10", "--image-size", "8",
            "--latent-dim", "8", "--backbone-channels", "16", "--base-channels", "8", "--n-downsample", "1",
            "--score-blocks", "1", "--score-heads", "1", "--num-workers", "0", "--device", "cpu",
            "--out-dir", os.path.join(tmp, "out"),
        ]
        args = parse_args(argv)
        metrics = train_one_seed(args, seed=0)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_experiment_2_physics_smoke_e_flow():
    # Exercises the new PDE-residual metric (pde/pde_residuals.py::e_flow_residual), wired into
    # evaluate() for E_flow specifically. Field dirs per PROBLEM_SPECS["E_flow"]: input "kappa",
    # outputs "ec_V", "u_flow", "v_flow" (all real-valued .mat).
    import scipy.io as sio

    from pde.run_experiment import parse_args, train_one_seed

    tmp = tempfile.mkdtemp(prefix="csho_smoke_exp2_eflow_")
    try:
        rng = np.random.default_rng(0)
        problem_root = os.path.join(tmp, "training", "E_flow")
        for field in ["kappa", "ec_V", "u_flow", "v_flow"]:
            field_dir = os.path.join(problem_root, field)
            os.makedirs(field_dir, exist_ok=True)
            for sid in ["1", "2", "3", "4"]:
                sio.savemat(os.path.join(field_dir, f"{sid}.mat"), {field: rng.random((8, 8))})

        argv = [
            "--data-root", tmp, "--problem", "E_flow", "--method", "csho", "--n-diff-steps", "2",
            "--batch-size", "2", "--n-epochs", "1", "--seeds", "0", "--max-samples", "10", "--image-size", "8",
            "--latent-dim", "8", "--backbone-channels", "16", "--base-channels", "8", "--n-downsample", "1",
            "--score-blocks", "1", "--score-heads", "1", "--num-workers", "0", "--device", "cpu",
            "--out-dir", os.path.join(tmp, "out"),
        ]
        args = parse_args(argv)
        metrics = train_one_seed(args, seed=0)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
        assert "flow_continuity_pde_residual" in metrics and "current_continuity_pde_residual" in metrics, metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_experiment_2_physics_smoke_te_heat():
    # Exercises pde_residuals.py::te_heat_residual (piecewise material-mask reconstruction from
    # elliptic_params + the complex Helmholtz/heat residual). Field dirs per
    # PROBLEM_SPECS["TE_heat"]: input "mater" (real), outputs "Ez" (complex), "T" (real); plus
    # the ellipticcsv/{idx}.csv geometry files pde/dataset.py now loads for this problem only.
    import scipy.io as sio

    from pde.run_experiment import parse_args, train_one_seed

    tmp = tempfile.mkdtemp(prefix="csho_smoke_exp2_teheat_")
    try:
        rng = np.random.default_rng(0)
        problem_root = os.path.join(tmp, "training", "TE_heat")
        mater_dir = os.path.join(problem_root, "mater")
        os.makedirs(mater_dir, exist_ok=True)
        T_dir = os.path.join(problem_root, "T")
        os.makedirs(T_dir, exist_ok=True)
        Ez_dir = os.path.join(problem_root, "Ez")
        os.makedirs(Ez_dir, exist_ok=True)
        elliptic_dir = os.path.join(problem_root, "ellipticcsv")
        os.makedirs(elliptic_dir, exist_ok=True)
        for sid in ["1", "2", "3", "4"]:
            sio.savemat(os.path.join(mater_dir, f"{sid}.mat"), {"mater": rng.random((8, 8)) + 1.0})
            sio.savemat(os.path.join(T_dir, f"{sid}.mat"), {"T": rng.random((8, 8)) * 50 + 300})  # Kelvin-ish, keeps exp(-Eg/(kB*T)) well-behaved
            arr = rng.random((8, 8)) + 1j * rng.random((8, 8))
            sio.savemat(os.path.join(Ez_dir, f"{sid}.mat"), {"Ez": arr})
            # elliptic_params = [cx, cy, r] in meters, matching TE_heat's coord convention
            # ((arange(8)-3.5)*1e-3 spans about +-3.5mm); r chosen to intersect some pixels.
            np.savetxt(os.path.join(elliptic_dir, f"{sid}.csv"), np.array([[0.0, 0.0, 0.002]]), delimiter=",")

        argv = [
            "--data-root", tmp, "--problem", "TE_heat", "--method", "csho", "--n-diff-steps", "2",
            "--batch-size", "2", "--n-epochs", "1", "--seeds", "0", "--max-samples", "10", "--image-size", "8",
            "--latent-dim", "8", "--backbone-channels", "16", "--base-channels", "8", "--n-downsample", "1",
            "--score-blocks", "1", "--score-heads", "1", "--num-workers", "0", "--device", "cpu",
            "--out-dir", os.path.join(tmp, "out"),
        ]
        args = parse_args(argv)
        metrics = train_one_seed(args, seed=0)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
        assert "e_field_pde_residual" in metrics and "heat_pde_residual" in metrics, metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_experiment_2_physics_smoke_va():
    # Exercises pde_residuals.py::va_residual (complex-field reconstruction + the 6-equation
    # acoustic-structure residual). Field dirs per PROBLEM_SPECS["VA"]: input "rho_water" (real),
    # outputs "p_t", "Sxx", "Sxy", "Syy", "x_u", "x_v" (all complex .mat).
    import scipy.io as sio

    from pde.run_experiment import parse_args, train_one_seed

    tmp = tempfile.mkdtemp(prefix="csho_smoke_exp2_va_")
    try:
        rng = np.random.default_rng(0)
        problem_root = os.path.join(tmp, "training", "VA")
        rho_dir = os.path.join(problem_root, "rho_water")
        os.makedirs(rho_dir, exist_ok=True)
        for sid in ["1", "2", "3", "4"]:
            # rho_water must be nonzero (it's a divisor in the acoustic residual)
            sio.savemat(os.path.join(rho_dir, f"{sid}.mat"), {"rho_water": rng.random((8, 8)) + 500.0})
        for field in ["p_t", "Sxx", "Sxy", "Syy", "x_u", "x_v"]:
            field_dir = os.path.join(problem_root, field)
            os.makedirs(field_dir, exist_ok=True)
            for sid in ["1", "2", "3", "4"]:
                arr = rng.random((8, 8)) + 1j * rng.random((8, 8))
                sio.savemat(os.path.join(field_dir, f"{sid}.mat"), {field: arr})

        argv = [
            "--data-root", tmp, "--problem", "VA", "--method", "csho", "--n-diff-steps", "2",
            "--batch-size", "2", "--n-epochs", "1", "--seeds", "0", "--max-samples", "10", "--image-size", "8",
            "--latent-dim", "8", "--backbone-channels", "16", "--base-channels", "8", "--n-downsample", "1",
            "--score-blocks", "1", "--score-heads", "1", "--num-workers", "0", "--device", "cpu",
            "--out-dir", os.path.join(tmp, "out"),
        ]
        args = parse_args(argv)
        metrics = train_one_seed(args, seed=0)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
        for eq in ["acoustic_real", "acoustic_imag", "structure_x_real", "structure_x_imag", "structure_y_real", "structure_y_imag"]:
            assert f"{eq}_pde_residual" in metrics, metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_experiment_3_synthetic_smoke():
    from synthetic.run_experiment import parse_args, train_one_seed

    tmp = tempfile.mkdtemp(prefix="csho_smoke_exp3_")
    try:
        # Coupled-OU ground truth self-generates data -- no fixture needed, unlike exp1/exp2.
        argv = [
            "--N", "3", "--coupling-strength", "0.6", "--method", "csho",
            "--quick", "--device", "cpu", "--out-dir", os.path.join(tmp, "out"),
        ]
        args = parse_args(argv)
        metrics = train_one_seed(args, seed=0)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_experiment_1_vision_smoke()
    print("OK: experiment_1_vision smoke test passed.")
    test_experiment_2_physics_smoke()
    print("OK: experiment_2_physics smoke test passed.")
    test_experiment_2_physics_smoke_e_flow()
    print("OK: experiment_2_physics (E_flow, pde_residual) smoke test passed.")
    test_experiment_2_physics_smoke_te_heat()
    print("OK: experiment_2_physics (TE_heat, pde_residual) smoke test passed.")
    test_experiment_2_physics_smoke_va()
    print("OK: experiment_2_physics (VA, pde_residual) smoke test passed.")
    test_experiment_3_synthetic_smoke()
    print("OK: experiment_3_synthetic smoke test passed.")
    print("ALL EXPERIMENT SMOKE TESTS PASSED.")
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import scipy.io as sio


def test_te_heat():
    from pde.dataset import _held_out_split_path, MultiPhysicsFieldDataset

    tmp = tempfile.mkdtemp(prefix="csho_held_out_split_")
    cache_path = _held_out_split_path("TE_heat")
    cache_existed_before = os.path.isfile(cache_path)
    cache_backup = cache_path + ".test_backup" if cache_existed_before else None
    if cache_existed_before:
        shutil.move(cache_path, cache_backup)
    try:
        rng = np.random.default_rng(0)

        # training/: full, working set of 10 samples
        train_root = os.path.join(tmp, "training", "TE_heat")
        for field in ["mater", "T", "Ez"]:
            os.makedirs(os.path.join(train_root, field), exist_ok=True)
        os.makedirs(os.path.join(train_root, "ellipticcsv"), exist_ok=True)
        for sid in range(1, 11):
            sio.savemat(os.path.join(train_root, "mater", f"{sid}.mat"), {"mater": rng.random((8, 8)) + 1.0})
            sio.savemat(os.path.join(train_root, "T", f"{sid}.mat"), {"T": rng.random((8, 8)) * 50 + 300})
            arr = rng.random((8, 8)) + 1j * rng.random((8, 8))
            sio.savemat(os.path.join(train_root, "Ez", f"{sid}.mat"), {"Ez": arr})
            np.savetxt(os.path.join(train_root, "ellipticcsv", f"{sid}.csv"), np.array([[2.0, 1.5, 30.0]]), delimiter=",")

        # testing/: mater/ exists but is EMPTY; Ez/, T/ are populated -- the real-world scenario
        test_root = os.path.join(tmp, "testing", "TE_heat")
        os.makedirs(os.path.join(test_root, "mater"), exist_ok=True)
        os.makedirs(os.path.join(test_root, "T"), exist_ok=True)
        os.makedirs(os.path.join(test_root, "Ez"), exist_ok=True)
        for sid in range(100001, 100004):
            sio.savemat(os.path.join(test_root, "T", f"{sid}.mat"), {"T": rng.random((8, 8)) * 50 + 300})
            arr = rng.random((8, 8)) + 1j * rng.random((8, 8))
            sio.savemat(os.path.join(test_root, "Ez", f"{sid}.mat"), {"Ez": arr})

        train_ds = MultiPhysicsFieldDataset(tmp, "TE_heat", split="training", image_size=(8, 8))
        test_ds = MultiPhysicsFieldDataset(tmp, "TE_heat", split="testing", image_size=(8, 8))

        assert train_ds._use_held_out_split, "expected the held-out fallback to trigger"
        assert test_ds._use_held_out_split, "expected the held-out fallback to trigger"

        train_idx = set(train_ds._sample_indices)
        test_idx = set(test_ds._sample_indices)
        assert train_idx, "held-out training slice must be non-empty"
        assert test_idx, "held-out testing slice must be non-empty"
        assert train_idx.isdisjoint(test_idx), f"held-out split must not overlap: {train_idx} vs {test_idx}"
        assert train_idx | test_idx == set(range(1, 11)), "held-out split must partition all 10 training-dir samples"
        # 90/10 split of 10 samples -> 9 train / 1 test
        assert len(train_idx) == 9 and len(test_idx) == 1, (len(train_idx), len(test_idx))

        _ = train_ds[0]
        _ = test_ds[0]

        assert os.path.isfile(cache_path), "held-out split must be persisted to disk"

        # reproducibility: re-constructing must reuse the exact same persisted split
        train_ds_2 = MultiPhysicsFieldDataset(tmp, "TE_heat", split="training", image_size=(8, 8))
        test_ds_2 = MultiPhysicsFieldDataset(tmp, "TE_heat", split="testing", image_size=(8, 8))
        assert train_ds_2._sample_indices == train_ds._sample_indices
        assert test_ds_2._sample_indices == test_ds._sample_indices
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.isfile(cache_path):
            os.remove(cache_path)
        if cache_backup and os.path.isfile(cache_backup):
            shutil.move(cache_backup, cache_path)


def missing_test_dir():
    from pde.dataset import MultiPhysicsFieldDataset

    tmp = tempfile.mkdtemp(prefix="csho_no_testing_dir_")
    try:
        rng = np.random.default_rng(0)
        train_root = os.path.join(tmp, "training", "E_flow")
        for field in ["kappa", "ec_V", "u_flow", "v_flow"]:
            os.makedirs(os.path.join(train_root, field), exist_ok=True)
            for sid in range(1, 5):
                sio.savemat(os.path.join(train_root, field, f"{sid}.mat"), {field: rng.random((8, 8))})

        train_ds = MultiPhysicsFieldDataset(tmp, "E_flow", split="training", image_size=(8, 8))
        assert not train_ds._use_held_out_split
        assert train_ds._sample_indices == [1, 2, 3, 4]

        try:
            MultiPhysicsFieldDataset(tmp, "E_flow", split="testing", image_size=(8, 8))
            assert False, "expected FileNotFoundError for a wholly-absent testing directory"
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_te_heat()
    print("OK: TE_heat broken-testing-split held-out fallback test passed.")
    missing_test_dir()
    print("OK: missing-testing-dir-entirely (no held-out fallback) test passed.")
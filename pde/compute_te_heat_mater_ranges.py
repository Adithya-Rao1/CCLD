from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import scipy.io as sio
import torch

from pde.pde_residuals import _te_heat_mater_iden


def _load_mat_scalar_field(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]
    return np.asarray(mat[keys[0]] if len(keys) == 1 else mat["mater"])


def main():
    p = argparse.ArgumentParser(description="Compute TE_heat's real inside/outside mater value ranges from the full dataset")
    p.add_argument("--data-root", required=True, help="Multiphysics-Bench root (contains training/, testing/)")
    p.add_argument("--splits", default="training,testing")
    args = p.parse_args()

    inside_min, inside_max = float("inf"), float("-inf")
    outside_min, outside_max = float("inf"), float("-inf")
    n_samples, n_inside_px, n_outside_px, n_between_px = 0, 0, 0, 0

    for split in args.splits.split(","):
        mater_dir = os.path.join(args.data_root, split, "TE_heat", "mater")
        elliptic_dir = os.path.join(args.data_root, split, "TE_heat", "ellipticcsv")
        if not os.path.isdir(mater_dir):
            print(f"skipping {split}: {mater_dir} not found")
            continue
        mat_files = sorted(glob.glob(os.path.join(mater_dir, "*.mat")))
        print(f"{split}: {len(mat_files)} samples found")

        for path in mat_files:
            idx = os.path.splitext(os.path.basename(path))[0]
            elliptic_path = os.path.join(elliptic_dir, f"{idx}.csv")
            if not os.path.isfile(elliptic_path):
                continue

            mater = _load_mat_scalar_field(path)
            H, W = mater.shape
            elliptic_params = torch.as_tensor(
                np.genfromtxt(elliptic_path, delimiter=",").reshape(-1, 3)[:1], dtype=torch.float64
            )
            mater_t = torch.as_tensor(mater, dtype=torch.float64).unsqueeze(0)
            mater_iden = _te_heat_mater_iden(elliptic_params, H, W, mater_t.device, mater_t.dtype)

            inside_vals = mater_t[mater_iden > 1e-5]
            outside_vals = mater_t[mater_iden <= 1e-5]
            if inside_vals.numel():
                inside_min = min(inside_min, inside_vals.min().item())
                inside_max = max(inside_max, inside_vals.max().item())
            if outside_vals.numel():
                outside_min = min(outside_min, outside_vals.min().item())
                outside_max = max(outside_max, outside_vals.max().item())
                
            n_inside_px += inside_vals.numel()
            n_outside_px += outside_vals.numel()
            n_samples += 1

        print(f"  running: inside=[{inside_min:.6g}, {inside_max:.6g}]  outside=[{outside_min:.6g}, {outside_max:.6g}]")

    print(f"\n=== {n_samples} samples, {n_inside_px} inside px, {n_outside_px} outside px ===")
    print(f"TE_HEAT_MATER_INSIDE_RANGE = ({inside_min!r}, {inside_max!r})")
    print(f"TE_HEAT_MATER_OUTSIDE_RANGE = ({outside_min!r}, {outside_max!r})")
    overlap = max(inside_min, outside_min) <= min(inside_max, outside_max)
    if overlap:
        print(
            f"\nWARNING: inside range [{inside_min:.6g}, {inside_max:.6g}] and outside range "
            f"[{outside_min:.6g}, {outside_max:.6g}] overlap across the dataset"
        )

if __name__ == "__main__":
    main()

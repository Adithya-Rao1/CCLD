from __future__ import annotations

import argparse
import os
import tarfile

from huggingface_hub import hf_hub_download

# NOTE on total size: this downloads and extracts BOTH training.tar.gz (~28.9GB) and
# testing.tar.gz (~2.9GB) by default -- roughly 32GB total on disk (plus the compressed
# archives themselves while they sit alongside the extraction, unless removed). This is NOT
# something to run in a smoke-test / CI context; it is a real, multi-hour download intended
# for an actual training run against real data.

REPO_ID = "Indulge-Bai/Multiphysics_Bench"
REVISION = "main"

# Verified live (this session): the dataset ships on the `main` branch as two WebDataset
# tar.gz archives -- NOT the `refs/convert/parquet` HF auto-mirror (confirmed partial/
# unreliable) and NOT loose files. Each tar member's internal path already matches the
# directory-tree layout PROBLEM_SPECS-driven loading in dataset.py expects:
#   {split}/{problem}/{field}/{sample_idx}.mat
# e.g. a verified live sample key: "training/NS_heat/u_u/555" ("testing/..." for the test
# split). For Elder specifically there's an extra nesting level for timestep:
#   {split}/Elder/{field}/{sample_idx}/{timestep}.mat
# matching _get_elder's expected path exactly. So extracting the tar directly into --out-dir
# reproduces the exact directory tree the dataset loader needs -- no path remapping required.
SPLIT_FILENAMES = {
    "train": "training.tar.gz",
    "test": "testing.tar.gz",
}


def _safe_extract(tar_path: str, out_dir: str) -> None:
    """Extract a tar.gz, guarding against path traversal (CVE-2007-4559-style) members."""
    out_dir_real = os.path.realpath(out_dir)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = os.path.realpath(os.path.join(out_dir, member.name))
            if not (member_path == out_dir_real or member_path.startswith(out_dir_real + os.sep)):
                raise ValueError(
                    f"Refusing to extract {member.name!r} from {tar_path}: resolves outside {out_dir}"
                )
        tf.extractall(out_dir)  # safe: every member was validated above


def download(out_dir: str, force: bool = False, splits=("train", "test")) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not force and any(os.scandir(out_dir)):
        print(f"{out_dir} is non-empty; skipping download (use --force to re-download).")
        return

    token = os.environ.get("HF_TOKEN")
    for split in splits:
        filename = SPLIT_FILENAMES[split]
        print(f"Fetching {filename} ({REPO_ID}@{REVISION}) ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=REVISION,
            local_dir=out_dir,
            token=token,
        )
        print(f"Extracting {local_path} -> {out_dir} ...")
        _safe_extract(local_path, out_dir)
    print(f"Done. Multiphysics_Bench data extracted under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and extract Multiphysics_Bench (6 coupled-PDE systems) from Hugging Face. "
            "Total size ~32GB (training.tar.gz ~28.9GB + testing.tar.gz ~2.9GB) -- not for smoke tests."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--splits", default="train,test",
        help="comma-separated subset of {train,test} to download (default: both)",
    )
    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SPLIT_FILENAMES]
    if unknown:
        raise ValueError(f"Unknown split(s) {unknown}; choose from {sorted(SPLIT_FILENAMES)}")
    download(args.out_dir, args.force, splits)


if __name__ == "__main__":
    main()

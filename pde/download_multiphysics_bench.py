from __future__ import annotations

import argparse
import os
import tarfile

from huggingface_hub import hf_hub_download

REPO_ID = "Indulge-Bai/Multiphysics_Bench"
REVISION = "main"

SPLIT_FILENAMES = {
    "train": "training.tar.gz",
    "test": "testing.tar.gz",
}


def _safe_extract(tar_path: str, out_dir: str) -> None:
    out_dir_real = os.path.realpath(out_dir)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = os.path.realpath(os.path.join(out_dir, member.name))
            if not (member_path == out_dir_real or member_path.startswith(out_dir_real + os.sep)):
                raise ValueError
        tf.extractall(out_dir)


def download(out_dir: str, force: bool = False, splits=("train", "test")) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not force and any(os.scandir(out_dir)):
        print(f"{out_dir} is non-empty")
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Download and extract Multiphysics_Bench (6 coupled-PDE systems) from Hugging Face")
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
        raise ValueError
    download(args.out_dir, args.force, splits)


if __name__ == "__main__":
    main()

# Downloads PASCAL VOC from the same mirror Ultralytics' VOC.yaml uses (github.com/ultralytics/
# assets releases -- a direct re-host of the official host.robots.ox.ac.uk archives, no login
# required). Extracting VOCtrainval_11-May-2012.zip yields VOCdevkit/VOC2012/{JPEGImages,
# SegmentationClass,ImageSets,Annotations}, the layout dataset.py's PASCAL loader expects.
# See https://docs.ultralytics.com/datasets/detect/voc for the reference recipe (that recipe only
# converts boxes to YOLO format; the segmentation masks we need come along in the raw archive).
from __future__ import annotations

import argparse
import os
import zipfile

import requests

ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"

PARTS = {
    "trainval2007": "VOCtrainval_06-Nov-2007.zip",
    "test2007": "VOCtest_06-Nov-2007.zip",
    "trainval2012": "VOCtrainval_11-May-2012.zip",
}

DEFAULT_PARTS = ["trainval2012"]


def download_part(name: str, out_dir: str, force: bool) -> str:
    filename = PARTS[name]
    out_path = os.path.join(out_dir, filename)
    if os.path.isfile(out_path) and not force:
        print(f"{out_path} already exists; skipping (use --force to re-download).")
        return out_path
    resp = requests.get(f"{ASSETS_URL}/{filename}", stream=True)
    resp.raise_for_status()
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return out_path


def extract(zip_path: str, out_dir: str) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PASCAL VOC from the Ultralytics asset mirror.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--parts", default=",".join(DEFAULT_PARTS), help=f"comma-separated subset of {list(PARTS)}")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    names = [n.strip() for n in args.parts.split(",") if n.strip()]
    unknown = [n for n in names if n not in PARTS]
    if unknown:
        raise ValueError(f"Unknown part(s) {unknown}; choose from {list(PARTS)}")

    os.makedirs(args.out_dir, exist_ok=True)
    for name in names:
        zip_path = download_part(name, args.out_dir, args.force)
        if not args.no_extract:
            extract(zip_path, args.out_dir)


if __name__ == "__main__":
    main()

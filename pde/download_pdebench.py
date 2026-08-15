from __future__ import annotations

import argparse
import hashlib
import os

import requests

# PDEBench's 2D diffusion-reaction dataset is a single HDF5 file hosted on DaRUS (not Hugging
# Face). File is a few GB; a streaming download is used to avoid holding it all in memory.
URL = "https://darus.uni-stuttgart.de/api/access/datafile/133017"
FILENAME = "2D_diff-react_NA_NA.h5"
MD5 = "b8d0b86064193195ddc30c33be5dc949"

CHUNK_SIZE = 1 << 20  # 1MB


def _md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out_dir: str, force: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, FILENAME)

    if os.path.exists(out_path) and not force:
        existing_md5 = _md5sum(out_path)
        if existing_md5 == MD5:
            print(f"{out_path} already present with matching md5; skipping download (use --force to re-download).")
            return out_path
        print(f"{out_path} exists but md5 mismatch ({existing_md5} != {MD5}); re-downloading.")

    print(f"Downloading {URL} -> {out_path} ...")
    with requests.get(URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        tmp_path = out_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, out_path)

    downloaded_md5 = _md5sum(out_path)
    if downloaded_md5 != MD5:
        raise RuntimeError(
            f"md5 mismatch for {out_path}: expected {MD5}, got {downloaded_md5}. "
            "The download may be corrupt or the upstream file may have changed."
        )
    print(f"Done. Verified md5 {downloaded_md5} for {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PDEBench's 2D diffusion-reaction dataset (single .h5 file, from DaRUS)."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.out_dir, args.force)


if __name__ == "__main__":
    main()

# Downloads jagennath-hari/nyuv2 (https://huggingface.co/datasets/jagennath-hari/nyuv2), an NYUv2
# mirror with rgb/depth/semantic/instance columns, MIT licensed, no login required for anonymous
# read access. Fetches the `refs/convert/parquet` revision -- the HF-generated parquet mirror every
# dataset-viewer-enabled repo has -- so we only need `huggingface_hub` (already a dependency) and
# never need to install the heavier `datasets` package. Optional HF_TOKEN env var is honored for
# gated/rate-limited access but is not required for this public dataset.
from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "jagennath-hari/nyuv2"
DEFAULT_REVISION = "refs/convert/parquet"


def download(out_dir: str, force: bool = False, revision: str = DEFAULT_REVISION) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not force and any(os.scandir(out_dir)):
        print(f"{out_dir} is non-empty; skipping download (use --force to re-download).")
        return
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=out_dir,
        token=os.environ.get("HF_TOKEN"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NYUv2 (rgb/depth/semantic) from Hugging Face.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    download(args.out_dir, args.force, args.revision)


if __name__ == "__main__":
    main()

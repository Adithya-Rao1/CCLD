# Login-based download from cityscapes-dataset.com, following the request pattern used by the
# official cityscapesScripts `csDownload` tool: POST credentials to /login/, then GET each package
# by packageID using the authenticated session cookie.
# Requires CITYSCAPES_USERNAME / CITYSCAPES_PASSWORD env vars (register at
# https://www.cityscapes-dataset.com/register/). Credentials are never accepted as CLI args --
# that would leak them into shell history -- and this script never hardcodes or persists them.
from __future__ import annotations

import argparse
import os
import zipfile

import requests

LOGIN_URL = "https://www.cityscapes-dataset.com/login/"
DOWNLOAD_URL = "https://www.cityscapes-dataset.com/file-handling/"

PACKAGE_IDS = {
    "gtFine_trainvaltest": 1,
    "gtCoarse": 2,
    "leftImg8bit_trainvaltest": 3,
    "leftImg8bit_trainextra": 4,
}

DEFAULT_PACKAGES = ["gtFine_trainvaltest", "leftImg8bit_trainvaltest"]


def make_session() -> requests.Session:
    username = os.environ.get("CITYSCAPES_USERNAME")
    password = os.environ.get("CITYSCAPES_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD environment variables "
            "(register at https://www.cityscapes-dataset.com/register/)."
        )
    session = requests.Session()
    resp = session.post(LOGIN_URL, data={"username": username, "password": password, "submit": "Login"})
    resp.raise_for_status()
    if "login" in resp.url:
        raise RuntimeError("Cityscapes login failed -- check CITYSCAPES_USERNAME/CITYSCAPES_PASSWORD.")
    return session


def download_package(session: requests.Session, name: str, out_dir: str, force: bool) -> str:
    package_id = PACKAGE_IDS[name]
    out_path = os.path.join(out_dir, f"{name}.zip")
    if os.path.isfile(out_path) and not force:
        print(f"{out_path} already exists; skipping (use --force to re-download).")
        return out_path
    resp = session.get(DOWNLOAD_URL, params={"packageID": package_id}, stream=True)
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
    parser = argparse.ArgumentParser(description="Download Cityscapes packages (login-based).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--packages", default=",".join(DEFAULT_PACKAGES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    names = [n.strip() for n in args.packages.split(",") if n.strip()]
    unknown = [n for n in names if n not in PACKAGE_IDS]
    if unknown:
        raise ValueError(f"Unknown package(s) {unknown}; choose from {list(PACKAGE_IDS)}")

    session = make_session()
    os.makedirs(args.out_dir, exist_ok=True)
    for name in names:
        zip_path = download_package(session, name, args.out_dir, args.force)
        if not args.no_extract:
            extract(zip_path, args.out_dir)


if __name__ == "__main__":
    main()

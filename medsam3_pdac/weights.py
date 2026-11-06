"""Download the fine-tuned LoRA weights from this repository's GitHub Release (cached after the first time).

The repository is private, so the download needs a GitHub token with read access:
  - set GITHUB_TOKEN, or
  - be logged in with the GitHub CLI (`gh auth login`); its token is used automatically.
Or skip the download and pass a local file: predict.py --weights path/to/medsam3_pdac_lora_epoch5.pt
"""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import requests

REPO = "JacobLiu124/MedSAM3-PDAC"
TAG = "v1.0"
ASSET = "medsam3_pdac_lora_epoch5.pt"
SHA256 = "448d28a0f6588083a9db114d128a17bef54ef0f08e7f3d47079f372ff7bd00bc"
CACHE = Path(os.environ.get("MEDSAM3_PDAC_CACHE", Path.home() / ".cache" / "medsam3_pdac"))


def _token():
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    if shutil.which("gh"):
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_weights() -> Path:
    """Return the local path of the fine-tuned weights, downloading them once if needed."""
    dest = CACHE / ASSET
    if dest.exists() and _sha256(dest) == SHA256:
        return dest
    headers = {"Accept": "application/vnd.github+json"}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    r = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}", headers=headers, timeout=30)
    if r.status_code == 404:
        raise SystemExit(f"Release {TAG} of {REPO} not found. The repository is private: set GITHUB_TOKEN "
                         "(or run `gh auth login`), or pass --weights with a local copy of the file.")
    r.raise_for_status()
    asset = next((x for x in r.json()["assets"] if x["name"] == ASSET), None)
    if asset is None:
        raise SystemExit(f"{ASSET} is not attached to release {TAG} of {REPO}.")
    headers["Accept"] = "application/octet-stream"
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {ASSET} ({asset['size'] / 1e6:.0f} MB) ...", flush=True)
    with requests.get(asset["url"], headers=headers, stream=True, timeout=60) as d:
        d.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in d.iter_content(1 << 20):
                f.write(chunk)
    if _sha256(tmp) != SHA256:
        tmp.unlink()
        raise SystemExit("downloaded file failed its checksum; try again")
    tmp.replace(dest)
    return dest


if __name__ == "__main__":
    print(get_weights())

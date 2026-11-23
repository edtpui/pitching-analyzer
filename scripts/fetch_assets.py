"""Download the external assets the analyzer needs, verify them, and record their provenance.

Assets:
  * Driveline OpenBiomechanics Project (OBP) pitching data, CC BY-NC-SA 4.0 (non-commercial; see
    https://openbiomechanics.org/#terms): metadata.csv, poi_metrics.csv, data_dictionary.csv and the
    full-signal landmarks table (GitHub release `dataset-v1`, SHA-256 checked against the repository's
    committed release_checksums.sha256).
  * MediaPipe pose landmarker models (lite / full / heavy), Apache-2.0.

Re-running is safe: files that are already present with the expected size/checksum are skipped.
Usage:  .venv\\Scripts\\python.exe scripts\\fetch_assets.py [--models lite,full,heavy]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchlab.config import DRIVELINE_DIR, MODELS_DIR  # noqa: E402

OBP_RAW = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main"
OBP_RELEASE = "https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1"
OBP_FILES = {
    "metadata.csv": f"{OBP_RAW}/baseball_pitching/data/metadata.csv",
    "poi_metrics.csv": f"{OBP_RAW}/baseball_pitching/data/poi/poi_metrics.csv",
    "data_dictionary.csv": f"{OBP_RAW}/baseball_pitching/data/data_dictionary.csv",
}
OBP_CHECKSUMS = f"{OBP_RAW}/scripts/release_checksums.sha256"
OBP_LANDMARKS = "pitching_landmarks.zip"

MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)

USER_AGENT = "pitching-analyzer-asset-fetch/1.0"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` through a .part file; verify the byte count against Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        expected = resp.headers.get("Content-Length")
        written = 0
        with tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
    if expected is not None and int(expected) != written:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{url}: got {written} bytes, expected {expected}")
    tmp.replace(dest)


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def expected_release_checksums() -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in fetch_text(OBP_CHECKSUMS).splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="lite,full,heavy", help="comma-separated MediaPipe variants")
    args = ap.parse_args()
    variants = [v.strip() for v in args.models.split(",") if v.strip()]
    bad = set(variants) - {"lite", "full", "heavy"}
    if bad:
        ap.error(f"unknown model variant(s): {', '.join(sorted(bad))}")

    manifest_path = ROOT / "assets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for name, url in OBP_FILES.items():
        dest = DRIVELINE_DIR / name
        if dest.exists() and manifest.get(f"driveline/{name}", {}).get("sha256") == sha256_of(dest):
            print(f"ok      {dest.relative_to(ROOT)}")
            continue
        print(f"fetch   {url}")
        download(url, dest)
        manifest[f"driveline/{name}"] = {"url": url, "bytes": dest.stat().st_size,
                                         "sha256": sha256_of(dest), "fetched_utc": now}

    zip_dest = DRIVELINE_DIR / OBP_LANDMARKS
    expected = expected_release_checksums().get(OBP_LANDMARKS)
    if expected is None:
        raise SystemExit(f"no committed checksum for {OBP_LANDMARKS} in {OBP_CHECKSUMS}")
    if not (zip_dest.exists() and sha256_of(zip_dest) == expected):
        url = f"{OBP_RELEASE}/{OBP_LANDMARKS}"
        print(f"fetch   {url}")
        download(url, zip_dest)
        actual = sha256_of(zip_dest)
        if actual != expected:
            zip_dest.unlink()
            raise SystemExit(f"checksum mismatch for {OBP_LANDMARKS}: {actual} != {expected}")
        manifest[f"driveline/{OBP_LANDMARKS}"] = {"url": url, "bytes": zip_dest.stat().st_size,
                                                  "sha256": actual, "fetched_utc": now,
                                                  "checksum_source": OBP_CHECKSUMS}
    else:
        print(f"ok      {zip_dest.relative_to(ROOT)} (sha256 verified)")

    with zipfile.ZipFile(zip_dest) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv") and not m.startswith("__MACOSX")]
        if not members:
            raise SystemExit(f"{OBP_LANDMARKS} contains no CSV")
        for m in members:
            target = DRIVELINE_DIR / Path(m).name
            info = zf.getinfo(m)
            if target.exists() and target.stat().st_size == info.file_size:
                print(f"ok      {target.relative_to(ROOT)}")
                continue
            print(f"extract {m} -> {target.relative_to(ROOT)}")
            with zf.open(m) as src, target.open("wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)

    for v in variants:
        dest = MODELS_DIR / f"pose_landmarker_{v}.task"
        key = f"models/{dest.name}"
        if dest.exists() and manifest.get(key, {}).get("sha256") == sha256_of(dest):
            print(f"ok      {dest.relative_to(ROOT)}")
            continue
        url = MP_MODEL_URL.format(v=v)
        print(f"fetch   {url}")
        download(url, dest)
        manifest[key] = {"url": url, "bytes": dest.stat().st_size, "sha256": sha256_of(dest), "fetched_utc": now}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

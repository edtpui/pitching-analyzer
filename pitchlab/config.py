"""Paths and runtime settings. Every setting can be overridden with an environment variable (or `.env`)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader so a project-local .env works without an extra dependency.
    Existing environment variables win over the file."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

ASSETS_DIR = Path(os.environ.get("PITCHLAB_ASSETS_DIR", ROOT / "assets"))
DRIVELINE_DIR = ASSETS_DIR / "driveline"
MODELS_DIR = ASSETS_DIR / "models"
REFERENCE_DIR = ASSETS_DIR / "reference"          # built by scripts/build_reference.py
DATA_DIR = Path(os.environ.get("PITCHLAB_DATA_DIR", ROOT / "data"))
ANALYSES_DIR = DATA_DIR / "analyses"

HOST = os.environ.get("PITCHLAB_HOST", "127.0.0.1")
PASSWORD = os.environ.get("PITCHLAB_PASSWORD", "")      # set -> HTTP Basic auth on every route but /api/health
PORT = int(os.environ.get("PITCHLAB_PORT", "8091"))

MAX_UPLOAD_MB = int(os.environ.get("PITCHLAB_MAX_UPLOAD_MB", "600"))
MAX_FRAMES = int(os.environ.get("PITCHLAB_MAX_FRAMES", "4000"))
ANALYSIS_MAX_SIDE = int(os.environ.get("PITCHLAB_ANALYSIS_MAX_SIDE", "1280"))   # longest side fed to the pose model
RENDER_MAX_SIDE = int(os.environ.get("PITCHLAB_RENDER_MAX_SIDE", "960"))       # longest side of the overlay video

ANTHROPIC_MODEL = os.environ.get("PITCHLAB_LLM_MODEL", "claude-opus-5-5")

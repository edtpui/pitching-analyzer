"""Swing-analysis paths. Runtime settings (pose models, limits, the Claude model) are the app's own, in
pitchlab.config; the hitting data and its reference bundle live apart from the pitching ones."""
from __future__ import annotations

from ..config import ANALYSIS_MAX_SIDE, ANTHROPIC_MODEL, ASSETS_DIR, DATA_DIR, MAX_FRAMES, MODELS_DIR, RENDER_MAX_SIDE

SWING_ASSETS_DIR = ASSETS_DIR / "swing"
DRIVELINE_DIR = SWING_ASSETS_DIR / "driveline"      # Driveline OpenBiomechanics hitting data
REFERENCE_DIR = SWING_ASSETS_DIR / "reference"      # built by scripts/swing_build_reference.py
ANALYSES_DIR = DATA_DIR / "swings"

__all__ = ["ANALYSES_DIR", "ANALYSIS_MAX_SIDE", "ANTHROPIC_MODEL", "DRIVELINE_DIR", "MAX_FRAMES", "MODELS_DIR",
           "REFERENCE_DIR", "RENDER_MAX_SIDE", "SWING_ASSETS_DIR"]

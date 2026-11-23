"""Loads the Driveline reference bundle and places a pitcher's metrics within it."""
from __future__ import annotations

import json
import threading
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import joblib

from .config import REFERENCE_DIR
from .events import DEFAULT_CALIBRATION
from .metrics import METRIC_BY_KEY, METRICS

VELO_BANDS = ("0-80", "80-85", "85-90", "90-200")


class ReferenceMissing(RuntimeError):
    pass


def fps_bucket(fps: float) -> str:
    if fps >= 180:
        return "240"
    if fps >= 90:
        return "120"
    return "60"


def velocity_band(mph: float | None) -> str | None:
    if mph is None:
        return None
    for b in VELO_BANDS:
        lo, hi = (float(x) for x in b.split("-"))
        if lo <= mph < hi:
            return b
    return None


def band_label(band: str) -> str:
    lo, hi = band.split("-")
    if lo == "0":
        return f"under {hi} mph"
    if hi == "200":
        return f"{lo}+ mph"
    return f"{lo}-{hi} mph"


@dataclass
class Reference:
    bundle: dict
    models: dict | None

    @classmethod
    def load(cls, directory: Path = REFERENCE_DIR) -> "Reference":
        path = directory / "reference.json"
        if not path.exists():
            raise ReferenceMissing(f"{path} missing - run scripts/build_reference.py")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        mpath = directory / "models.joblib"
        models = joblib.load(mpath) if mpath.exists() else None
        return cls(bundle=bundle, models=models)

    @cached_property
    def calibration(self) -> dict:
        return {**DEFAULT_CALIBRATION, **self.bundle.get("calibration", {})}

    @property
    def height_per_leg_length(self) -> float:
        return float(self.calibration["height_per_leg_length"])

    def display(self, key: str, raw: float | None) -> float | None:
        if raw is None:
            return None
        m = self.bundle["mapping"].get(key)
        return raw if m is None else m["offset"] + m["scale"] * raw

    def display_scale(self, key: str) -> float:
        m = self.bundle["mapping"].get(key)
        return 1.0 if m is None else m["scale"]

    def percentile(self, key: str, raw: float | None) -> float | None:
        d = self.bundle["distributions"].get(key)
        if raw is None or d is None or not d["values"]:
            return None
        vals = d["values"]
        lo, hi = bisect_left(vals, raw), bisect_right(vals, raw)
        return 100.0 * (lo + 0.5 * (hi - lo)) / len(vals)

    def rating(self, key: str, fps: float) -> str:
        return self.bundle["ratings"][fps_bucket(fps)].get(key, "low")

    def compare(self, raw: dict[str, float | None], fps: float, *, unavailable: dict[str, str],
                velo_mph: float | None = None, level: str | None = None) -> list[dict]:
        """One entry per metric with display value, percentile, typical ranges and confidence."""
        bucket = fps_bucket(fps)
        band = velocity_band(velo_mph)
        out = []
        for md in METRICS:
            key = md.key
            v = raw.get(key)
            dist = self.bundle["distributions"].get(key, {})
            pct = dist.get("percentiles", {})
            typical = {p: self.display(key, pct[p]) for p in ("10", "25", "50", "75", "90")} if pct else None
            band_stats = None
            if band and dist.get("by_velocity", {}).get(band):
                b = dist["by_velocity"][band]
                band_stats = {"band": band, "label": band_label(band), "n": b["n"],
                              **{p: self.display(key, b[p]) for p in ("25", "50", "75")}}
            level_stats = None
            if level and dist.get("by_level", {}).get(level):
                b = dist["by_level"][level]
                level_stats = {"level": level, "n": b["n"], **{p: self.display(key, b[p]) for p in ("25", "50", "75")}}
            rel = self.bundle["reliability"][bucket].get(key, {})
            mae = rel.get("mae")
            out.append({
                "key": key, "label": md.label, "unit": md.unit, "group": md.group,
                "description": md.description, "decimals": md.decimals,
                "raw": v, "value": self.display(key, v), "calibrated": key in self.bundle["mapping"],
                "percentile": self.percentile(key, v),
                "typical": typical, "band": band_stats, "level": level_stats,
                "rating": self.rating(key, fps) if v is not None else None,
                "reliability_r": rel.get("r"), "validity_r": self.bundle["validity"].get(key, {}).get("r"),
                "error_margin": None if mae is None else abs(mae * self.display_scale(key)),
                "velocity_correlation": self.bundle.get("velocity_correlation", {}).get(key),
                "unavailable": unavailable.get(key) if v is None else None,
            })
        return out

    def summary(self) -> dict:
        """What the About page shows: source, calibration, validation tables, model scores."""
        b = self.bundle
        return {k: b[k] for k in ("version", "built_utc", "source", "simulation", "calibration", "validity",
                                  "reliability", "ratings", "event_error", "models", "mapping")} | {
            "metrics": b["metrics"],
            "distributions": {k: {kk: v[kk] for kk in ("n", "mean", "sd", "percentiles", "by_velocity", "by_level")}
                              for k, v in b["distributions"].items()},
            "velocity_correlation": b.get("velocity_correlation", {}),
        }


_lock = threading.Lock()
_cached: Reference | None = None


def get_reference() -> Reference:
    global _cached
    with _lock:
        if _cached is None:
            _cached = Reference.load()
        return _cached


def metric_label(key: str) -> str:
    return METRIC_BY_KEY[key].label

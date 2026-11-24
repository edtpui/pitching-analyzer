"""End-to-end analysis of one uploaded clip, and re-computation with manually corrected events."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from ..pipeline import AnalysisError as _AppAnalysisError
from .coaching import findings, rules_report
from .config import ANALYSIS_MAX_SIDE, MAX_FRAMES, RENDER_MAX_SIDE
from .events import EVENT_NAMES, Events, calib_for_view, detect_events, validate_order
from .kinematics import kinematics
from .metrics import METRIC_BY_KEY, METRIC_KEYS, MPS_TO_MPH, compute_metrics, hands_untracked
from .model import predict
from ..pose import MODEL_VARIANTS, PoseTrack, estimate
from .preprocess import Prepared, PreprocessError, prepare
from .reference import Reference, bat_speed_band, get_reference
from .render import render_keyframes, render_overlay
from ..video_io import VideoError, iter_frames, open_readable

log = logging.getLogger(__name__)

LEVELS = ("high_school", "college", "independent", "milb", "pro", "youth", "other")
REFERENCE_LEVELS = ("high_school", "college", "independent", "milb")
SERIES_KEYS = ("pelvis_rot_velo", "torso_rot_velo", "lead_arm_speed", "hand_speed", "hip_shoulder_sep", "pelvis_rot",
               "torso_rot", "lead_knee_flexion", "trunk_flexion", "trunk_lateral", "rear_elbow_flexion",
               "lead_elbow_flexion", "rear_shoulder_abduction", "foot_speed")
MAX_SERIES_POINTS = 600


class AnalysisError(_AppAnalysisError):
    """A user-facing failure (bad video, no hitter found...). The app's job queue reports these as failures
    of the clip, not crashes."""


@dataclass
class Options:
    bats: str = "auto"                   # auto | R | L
    alignment: str = "square"            # square | angled
    capture_fps: float | None = None     # real capture rate of slow-motion exports
    model: str = "full"                  # lite | full | heavy
    trim_start_s: float | None = None
    trim_end_s: float | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    bat_speed_mph: float | None = None   # bat-sensor reading, optional (picks the comparison band)
    level: str | None = None
    label: str = ""

    def validate(self) -> "Options":
        if self.bats not in ("auto", "R", "L"):
            raise ValueError("batting side must be auto, R or L")
        if self.alignment not in ("square", "angled"):
            raise ValueError("camera alignment must be square or angled")
        if self.model not in MODEL_VARIANTS:
            raise ValueError(f"model must be one of {', '.join(MODEL_VARIANTS)}")
        checks = {"capture_fps": (15, 2000), "trim_start_s": (0, 36000), "trim_end_s": (0, 36000),
                  "height_cm": (120, 230), "weight_kg": (30, 200), "bat_speed_mph": (30, 110)}
        for name, (lo, hi) in checks.items():
            v = getattr(self, name)
            if v is not None and not (lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi}")
        if self.trim_start_s is not None and self.trim_end_s is not None and self.trim_end_s <= self.trim_start_s:
            raise ValueError("trim end must be after trim start")
        if self.level is not None and self.level not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}")
        self.label = (self.label or "")[:120]
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "Options":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known}).validate()

    @property
    def height_m(self) -> float | None:
        return self.height_cm / 100.0 if self.height_cm else None


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False, default=_json_default), encoding="utf-8")
    # On Windows a replace fails while another thread has the target open for reading; retry briefly.
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _finite(o):
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_json(path: Path, data: dict) -> None:
    _atomic_write_json(path, _finite(data))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ assembly
def _unavailable_reasons(prep: Prepared, ev: Events, has_height: bool) -> dict[str, str]:
    out = {}
    for k in METRIC_KEYS:
        md = METRIC_BY_KEY[k]
        if md.needs_heading and not prep.heading_known:
            out[k] = "Needs a camera squared to the pitcher-plate line."
        elif md.needs_translation and prep.motion.track_forward is None:
            out[k] = "Needs a static side-on camera."
        elif md.needs_height and not has_height:
            out[k] = "Enter the hitter's height to convert this to mph."
        elif md.needs_load and ev.load is None:
            out[k] = "The load (maximum hip coil) is not in the clip."
        elif hands_untracked(prep.motion, ev, k):
            out[k] = "The hands were blurred or untracked at this moment; film at 120-240 fps."
        else:
            out[k] = "Landmarks were missing at the event."
    return out


def _series(prep: Prepared, kin: dict, ev: Events, hand_mph_per_ll: float | None) -> dict:
    t = prep.motion.t
    first = next((k for k in (ev.lift, ev.load) if k is not None), None)
    start_t = t[first] - 0.25 if first is not None else t[ev.fp] - 0.6
    lo = int(np.searchsorted(t, start_t))
    hi = int(np.searchsorted(t, t[ev.contact] + 0.3))
    lo, hi = max(0, lo), min(len(t), max(hi, lo + 2))
    idx = np.arange(lo, hi)
    if idx.size > MAX_SERIES_POINTS:
        idx = np.unique(np.linspace(lo, hi - 1, MAX_SERIES_POINTS).round().astype(int))
    data = {"sample": idx.tolist(), "t_ms": ((t[idx] - t[ev.contact]) * 1000.0).round(2).tolist()}
    for k in SERIES_KEYS:
        data[k] = [None if not np.isfinite(x) else round(float(x), 2) for x in kin[k][idx]]
    if hand_mph_per_ll:
        data["hand_speed_mph"] = [None if v is None else round(v * hand_mph_per_ll, 2) for v in data["hand_speed"]]
    return data


def _quality_warnings(prep: Prepared, ev: Events, ref: Reference) -> list[str]:
    q = prep.quality
    w = list(prep.notes) + list(ev.notes)
    t = prep.motion.t
    fps = q["fps_effective"]
    if fps < 100:
        w.append(f"The clip runs at {fps:.0f} fps. Hand and arm speeds and contact timing need 120-240 fps; their "
                 "ratings were lowered accordingly.")
    cal = ref.calibration
    dur = t[ev.contact] - t[ev.fp]
    hi = cal.get("fp_to_contact_p95_s", 0.153)
    lo = cal.get("fp_to_contact_p05_s", 0.058)
    if dur > 3 * hi:
        w.append(f"Foot plant to contact took {dur * 1000:.0f} ms (typical {lo * 1000:.0f}-{hi * 1000:.0f} ms). "
                 "If this is a slow-motion clip, re-run it with the capture frame rate (e.g. 240).")
    elif dur < 0.5 * lo:
        w.append(f"Foot plant to contact took only {dur * 1000:.0f} ms; check the capture frame rate setting.")
    if q["pose_detected_fraction_window"] < 0.9:
        w.append(f"The pose model lost the hitter in {100 * (1 - q['pose_detected_fraction_window']):.0f}% of "
                 "frames around the swing; gaps were interpolated.")
    if q["key_joint_visibility_window"] < 0.5:
        w.append("The hands or the front foot are hard to see during the swing (low landmark visibility); film "
                 "from the open side (facing the hitter's chest) and avoid netting or people in front of the hitter.")
    swaps = sum(q["swaps_repaired"].values())
    if swaps > 0.1 * q["frames"]:
        w.append(f"The pose model confused left and right in {swaps} frames (repaired automatically).")
    return w


def assemble(prep: Prepared, ev: Events, ref: Reference, opts: Options, video: dict) -> dict:
    kin = kinematics(prep.motion)
    raw = compute_metrics(prep.motion, kin, ev, ref.height_per_leg_length, opts.height_m, ref.height_per_leg_length_3d)
    t = prep.motion.t
    fps = prep.quality["fps_effective"]

    body = None
    if opts.height_cm and opts.weight_kg:
        body = {"height_m": opts.height_cm / 100.0, "mass_kg": opts.weight_kg}
    predictions = predict(ref.models, raw, body) if ref.models else []

    band_source = None
    speed_for_band = None
    if opts.bat_speed_mph:
        speed_for_band, band_source = opts.bat_speed_mph, "sensor"
    else:
        bp = next((p for p in predictions if p["target"] == "bat_speed_mph" and p["reportable"]), None)
        if bp:
            speed_for_band, band_source = bp["value"], "predicted"
    level = opts.level if opts.level in REFERENCE_LEVELS else None
    metrics = ref.compare(raw, fps, unavailable=_unavailable_reasons(prep, ev, opts.height_m is not None),
                          bat_speed_mph=speed_for_band, level=level)

    warnings = _quality_warnings(prep, ev, ref)
    pelvis_ct = kin["pelvis_rot"][ev.contact]
    if np.isfinite(pelvis_ct) and prep.heading_known and not (20.0 <= pelvis_ct <= 200.0):
        warnings.append("At contact the hips do not face the pitcher in the reconstructed 3D pose; rotation metrics "
                        "may be invalid (check the batting-side setting).")

    events = {}
    for name, k in ev.as_dict().items():
        events[name] = None if k is None else {
            "sample": int(k), "frame": int(prep.frame_index[k]), "t_s": float(t[k]),
            "ms_from_contact": float((t[k] - t[ev.contact]) * 1000.0), "source": ev.source.get(name, "auto")}

    hand_mph_per_ll = (opts.height_m / ref.height_per_leg_length_3d * MPS_TO_MPH) if opts.height_m else None
    src = ref.bundle.get("source", {})
    result = {
        "analysis_version": __version__,
        "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "athlete": {"bats": prep.bats, "bats_detected_in_video": prep.bats_detected,
                    "height_cm": opts.height_cm, "weight_kg": opts.weight_kg, "level": opts.level,
                    "sensor_bat_speed_mph": opts.bat_speed_mph},
        "reference_group": {"n_swings": src.get("n_swings"), "n_hitters": src.get("n_athletes")},
        "video": video | {"fps_effective": fps, "samples": int(len(t)), "capture_fps_override": opts.capture_fps},
        "quality": prep.quality | {"warnings": warnings, "heading_known": prep.heading_known},
        "events": events,
        "event_summary": {k: round(v["ms_from_contact"]) for k, v in events.items() if v and k != "contact"},
        "metrics": metrics,
        "comparison_band": None if speed_for_band is None else {
            "bat_speed_mph": speed_for_band, "source": band_source, "band": bat_speed_band(speed_for_band)},
        "predictions": predictions,
        "series": _series(prep, kin, ev, hand_mph_per_ll),
    }
    result["findings"] = findings(metrics)
    result["report"] = {"source": "rules", "text": rules_report(result)}
    return result


# ------------------------------------------------------------------------------------------ running
def run(analysis_dir: Path, opts: Options, input_path: Path,
        progress: Callable[[float, str], None] = lambda f, s: None) -> dict:
    ref = get_reference()
    started = time.time()
    progress(0.01, "Reading video")
    try:
        readable, info = open_readable(input_path, analysis_dir)
    except VideoError as exc:
        raise AnalysisError(f"Could not read the video: {exc}") from exc

    est_frames = info.frame_count
    if opts.trim_start_s is not None or opts.trim_end_s is not None:
        a = opts.trim_start_s or 0.0
        b = opts.trim_end_s or info.duration_s
        est_frames = int(max(0.0, b - a) * info.fps) if info.duration_s else info.frame_count
    if est_frames > MAX_FRAMES:
        raise AnalysisError(f"The selected range has about {est_frames} frames (limit {MAX_FRAMES}). Trim the "
                            "clip to the swing with the start/end fields.")

    progress(0.03, "Estimating pose")
    total = max(est_frames, 1)

    def pose_progress(k: int) -> None:
        if k % 15 == 0:
            progress(0.03 + 0.67 * min(k / total, 1.0), f"Estimating pose ({k}/{total} frames)")

    frames = iter_frames(readable, max_side=ANALYSIS_MAX_SIDE, start_s=opts.trim_start_s, end_s=opts.trim_end_s)
    try:
        track = estimate(frames, opts.model, expected_frames=total, progress=pose_progress)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from exc
    if len(track.t) > MAX_FRAMES:
        raise AnalysisError(f"The clip has {len(track.t)} frames (limit {MAX_FRAMES}); trim it to the swing.")
    track.save(analysis_dir / "pose.npz")

    progress(0.72, "Detecting events and computing metrics")
    try:
        prep = prepare(track, opts.bats, opts.alignment, opts.capture_fps, ref.calibration)
        kin = kinematics(prep.motion)
        ev = detect_events(prep.motion.t, kin, calib_for_view(ref.calibration, prep.camera_view))
    except PreprocessError as exc:
        raise AnalysisError(str(exc)) from exc
    except ValueError as exc:
        raise AnalysisError(f"Could not analyse the motion: {exc}") from exc

    video = info.as_dict() | {"decoded_path": readable.name, "analysis_size": [track.width, track.height]}
    result = assemble(prep, ev, ref, opts, video)

    progress(0.8, "Rendering key frames")
    result["keyframes"] = render_keyframes(readable, analysis_dir / "keyframes", prep.frame_index, prep.image_smooth,
                                           (track.width, track.height), prep.bats, ev.as_dict(), RENDER_MAX_SIDE,
                                           opts.trim_start_s, opts.trim_end_s)
    progress(0.83, "Rendering overlay video")
    size = render_overlay(readable, analysis_dir / "overlay.mp4", prep.frame_index, prep.image_smooth,
                          (track.width, track.height), prep.bats, info.fps, RENDER_MAX_SIDE,
                          opts.trim_start_s, opts.trim_end_s,
                          progress=lambda f: progress(0.83 + 0.16 * f, "Rendering overlay video"))
    result["overlay"] = {"file": "overlay.mp4", "fps": info.fps, "size": list(size), "frames": int(len(prep.motion.t))}
    result["processing_s"] = round(time.time() - started, 1)
    write_json(analysis_dir / "result.json", result)
    progress(1.0, "Done")
    return result


def recompute_with_events(analysis_dir: Path, opts: Options, input_path: Path, manual: dict[str, int | None]) -> dict:
    """Re-derive every metric with user-corrected event samples (no new pose estimation)."""
    ref = get_reference()
    prev = read_json(analysis_dir / "result.json")
    track = PoseTrack.load(analysis_dir / "pose.npz")
    prep = prepare(track, opts.bats, opts.alignment, opts.capture_fps, ref.calibration)
    n = len(prep.motion.t)
    old = {k: (v["sample"] if v else None) for k, v in prev["events"].items()}
    merged = {k: (manual.get(k) if manual.get(k) is not None else old.get(k)) for k in EVENT_NAMES}
    if merged["fp"] is None or merged["contact"] is None:
        raise ValueError("foot plant and contact are required")
    source = {k: ("manual" if manual.get(k) is not None and manual.get(k) != old.get(k)
                  else prev["events"][k]["source"] if prev["events"].get(k) else "missing")
              for k in merged}
    ev = Events(fp=int(merged["fp"]), contact=int(merged["contact"]),
                load=None if merged["load"] is None else int(merged["load"]),
                lift=None if merged["lift"] is None else int(merged["lift"]), source=source)
    validate_order(ev, n)
    readable = analysis_dir / prev["video"]["decoded_path"]
    if not readable.exists():
        readable = input_path
    result = assemble(prep, ev, ref, opts, prev["video"])
    result["keyframes"] = render_keyframes(readable, analysis_dir / "keyframes", prep.frame_index, prep.image_smooth,
                                           (track.width, track.height), prep.bats, ev.as_dict(), RENDER_MAX_SIDE,
                                           opts.trim_start_s, opts.trim_end_s)
    for k in ("overlay", "processing_s"):
        if k in prev:
            result[k] = prev[k]
    if prev.get("llm_report"):
        result["llm_report"] = prev["llm_report"] | {"stale": True}
    write_json(analysis_dir / "result.json", result)
    return result


def options_dict(opts: Options) -> dict:
    return asdict(opts)

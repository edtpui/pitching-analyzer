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
from .coaching import findings, rules_report
from .config import ANALYSIS_MAX_SIDE, MAX_FRAMES, RENDER_MAX_SIDE
from .events import Events, calib_for_view, detect_events, validate_order
from .kinematics import kinematics
from .metrics import METRIC_BY_KEY, METRIC_KEYS, arm_untracked, compute_metrics, normalise_er
from .model import predict
from .pose import MODEL_VARIANTS, PoseTrack, estimate
from .preprocess import Prepared, PreprocessError, prepare
from .reference import Reference, get_reference, velocity_band
from .render import render_keyframes, render_overlay
from .video_io import VideoError, iter_frames, open_readable

log = logging.getLogger(__name__)

LEVELS = ("high_school", "college", "independent", "milb", "pro", "youth", "other")
SERIES_KEYS = ("pelvis_rot_velo", "torso_rot_velo", "elbow_ext_velo", "hip_shoulder_sep", "pelvis_rot",
               "torso_rot", "lead_knee_flexion", "trunk_flexion", "trunk_lateral", "shoulder_er",
               "shoulder_abduction", "elbow_flexion", "wrist_speed", "foot_speed")
MAX_SERIES_POINTS = 600


class AnalysisError(RuntimeError):
    """A user-facing failure (bad video, no pitcher found...)."""


@dataclass
class Options:
    throws: str = "auto"                 # auto | R | L
    alignment: str = "square"            # square | angled
    capture_fps: float | None = None     # real capture rate of slow-motion exports
    model: str = "full"                  # lite | full | heavy
    trim_start_s: float | None = None
    trim_end_s: float | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    velocity_mph: float | None = None    # radar gun reading, optional
    level: str | None = None
    label: str = ""

    def validate(self) -> "Options":
        if self.throws not in ("auto", "R", "L"):
            raise ValueError("throws must be auto, R or L")
        if self.alignment not in ("square", "angled"):
            raise ValueError("camera alignment must be square or angled")
        if self.model not in MODEL_VARIANTS:
            raise ValueError(f"model must be one of {', '.join(MODEL_VARIANTS)}")
        checks = {"capture_fps": (15, 2000), "trim_start_s": (0, 36000), "trim_end_s": (0, 36000),
                  "height_cm": (120, 230), "weight_kg": (30, 200), "velocity_mph": (30, 110)}
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
def _unavailable_reasons(prep: Prepared, ev: Events) -> dict[str, str]:
    out = {}
    for k in METRIC_KEYS:
        md = METRIC_BY_KEY[k]
        if md.needs_heading and not prep.heading_known:
            out[k] = "Needs a camera squared to the mound-home line."
        elif md.needs_translation and prep.motion.track_forward is None:
            out[k] = "Needs a static side-on camera."
        elif arm_untracked(prep.motion, ev, k):
            out[k] = "The throwing arm was blurred or untracked at this moment; film at 120-240 fps."
        elif k == "stride_length" and ev.pkh is None:
            out[k] = "Peak knee height is not in the clip."
        else:
            out[k] = "Landmarks were missing at the event."
    return out


def _series(prep: Prepared, kin: dict, ev: Events) -> dict:
    t = prep.motion.t
    start_t = t[ev.pkh] - 0.25 if ev.pkh is not None else t[ev.fp] - 0.6
    lo = int(np.searchsorted(t, start_t))
    hi = int(np.searchsorted(t, t[ev.br] + 0.3))
    lo, hi = max(0, lo), min(len(t), max(hi, lo + 2))
    idx = np.arange(lo, hi)
    if idx.size > MAX_SERIES_POINTS:
        idx = np.unique(np.linspace(lo, hi - 1, MAX_SERIES_POINTS).round().astype(int))
    er = kin["shoulder_er"]
    data = {"sample": idx.tolist(), "t_ms": ((t[idx] - t[ev.br]) * 1000.0).round(2).tolist()}
    for k in SERIES_KEYS:
        v = normalise_er(er, ev.fp) if k == "shoulder_er" else kin[k]
        data[k] = [None if not np.isfinite(x) else round(float(x), 2) for x in v[idx]]
    return data


def _quality_warnings(prep: Prepared, ev: Events, ref: Reference) -> list[str]:
    q = prep.quality
    w = list(prep.notes) + list(ev.notes)
    t = prep.motion.t
    fps = q["fps_effective"]
    if fps < 100:
        w.append(f"The clip runs at {fps:.0f} fps. Release-phase arm metrics need 120-240 fps; their ratings "
                 "were lowered accordingly.")
    cal = ref.calibration
    dur = t[ev.br] - t[ev.fp]
    hi = cal.get("fp_to_br_p95_s", 0.162)
    lo = cal.get("fp_to_br_p05_s", 0.108)
    if dur > 3 * hi:
        w.append(f"Foot plant to release took {dur * 1000:.0f} ms (typical {lo * 1000:.0f}-{hi * 1000:.0f} ms). "
                 "If this is a slow-motion clip, re-run it with the capture frame rate (e.g. 240).")
    elif dur < 0.5 * lo:
        w.append(f"Foot plant to release took only {dur * 1000:.0f} ms; check the capture frame rate setting.")
    if q["pose_detected_fraction_window"] < 0.9:
        w.append(f"The pose model lost the pitcher in {100 * (1 - q['pose_detected_fraction_window']):.0f}% of "
                 "frames around the pitch; gaps were interpolated.")
    if q["key_joint_visibility_window"] < 0.5:
        w.append("The throwing arm or the lead foot is hard to see during the delivery (low landmark "
                 "visibility); film with the arm side toward the camera and avoid occlusions.")
    swaps = sum(q["swaps_repaired"].values())
    if swaps > 0.1 * q["frames"]:
        w.append(f"The pose model confused left and right in {swaps} frames (repaired automatically).")
    return w


def assemble(prep: Prepared, ev: Events, ref: Reference, opts: Options, video: dict) -> dict:
    kin = kinematics(prep.motion)
    raw = compute_metrics(prep.motion, kin, ev, ref.height_per_leg_length)
    t = prep.motion.t
    fps = prep.quality["fps_effective"]

    body = None
    if opts.height_cm and opts.weight_kg:
        body = {"height_m": opts.height_cm / 100.0, "mass_kg": opts.weight_kg}
    predictions = predict(ref.models, raw, body) if ref.models else []

    band_source = None
    velo_for_band = None
    if opts.velocity_mph:
        velo_for_band, band_source = opts.velocity_mph, "radar"
    else:
        vp = next((p for p in predictions if p["target"] == "pitch_speed_mph" and p["reportable"]), None)
        if vp:
            velo_for_band, band_source = vp["value"], "predicted"
    level = opts.level if opts.level in ("high_school", "college", "independent", "milb") else None
    metrics = ref.compare(raw, fps, unavailable=_unavailable_reasons(prep, ev), velo_mph=velo_for_band, level=level)

    warnings = _quality_warnings(prep, ev, ref)
    torso_br = kin["torso_rot"][ev.br]
    if np.isfinite(torso_br) and not (20.0 <= torso_br <= 220.0):
        warnings.append("At release the trunk does not face home in the reconstructed 3D pose; trunk "
                        "rotation metrics may be invalid (check the throwing-hand setting).")

    events = {}
    for name, k in ev.as_dict().items():
        events[name] = None if k is None else {
            "sample": int(k), "frame": int(prep.frame_index[k]), "t_s": float(t[k]),
            "ms_from_release": float((t[k] - t[ev.br]) * 1000.0), "source": ev.source.get(name, "auto")}

    result = {
        "analysis_version": __version__,
        "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "athlete": {"throws": prep.throws, "throws_detected_in_video": prep.throws_detected,
                    "height_cm": opts.height_cm, "weight_kg": opts.weight_kg, "level": opts.level,
                    "radar_velocity_mph": opts.velocity_mph},
        "video": video | {"fps_effective": fps, "samples": int(len(t)), "capture_fps_override": opts.capture_fps},
        "quality": prep.quality | {"warnings": warnings, "heading_known": prep.heading_known},
        "events": events,
        "event_summary": {k: round(v["ms_from_release"]) for k, v in events.items() if v and k != "br"},
        "metrics": metrics,
        "comparison_band": None if velo_for_band is None else {
            "velocity_mph": velo_for_band, "source": band_source, "band": velocity_band(velo_for_band)},
        "predictions": predictions,
        "series": _series(prep, kin, ev),
    }
    result["findings"] = findings(metrics)
    result["report"] = {"source": "rules", "text": rules_report(result)}
    return result


# ------------------------------------------------------------------------------------------ running
def _analysis_size(track: PoseTrack) -> tuple[int, int]:
    return track.width, track.height


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
                            "clip to the pitch with the start/end fields.")

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
        raise AnalysisError(f"The clip has {len(track.t)} frames (limit {MAX_FRAMES}); trim it to the pitch.")
    track.save(analysis_dir / "pose.npz")

    progress(0.72, "Detecting events and computing metrics")
    try:
        prep = prepare(track, opts.throws, opts.alignment, opts.capture_fps, ref.calibration)
        kin = kinematics(prep.motion)
        ev = detect_events(prep.motion.t, kin, calib_for_view(ref.calibration, prep.camera_view))
    except PreprocessError as exc:
        raise AnalysisError(str(exc)) from exc
    except ValueError as exc:
        raise AnalysisError(f"Could not analyse the motion: {exc}") from exc

    video = info.as_dict() | {"decoded_path": readable.name, "analysis_size": list(_analysis_size(track))}
    result = assemble(prep, ev, ref, opts, video)

    progress(0.8, "Rendering key frames")
    result["keyframes"] = render_keyframes(readable, analysis_dir / "keyframes", prep.frame_index, prep.image_smooth,
                                           _analysis_size(track), prep.throws, ev.as_dict(), RENDER_MAX_SIDE,
                                           opts.trim_start_s, opts.trim_end_s)
    progress(0.83, "Rendering overlay video")
    size = render_overlay(readable, analysis_dir / "overlay.mp4", prep.frame_index, prep.image_smooth,
                          _analysis_size(track), prep.throws, info.fps, RENDER_MAX_SIDE,
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
    prep = prepare(track, opts.throws, opts.alignment, opts.capture_fps, ref.calibration)
    n = len(prep.motion.t)
    old = {k: (v["sample"] if v else None) for k, v in prev["events"].items()}
    merged = {k: (manual.get(k) if manual.get(k) is not None else old.get(k)) for k in ("pkh", "fp", "mer", "br")}
    if merged["fp"] is None or merged["mer"] is None or merged["br"] is None:
        raise ValueError("foot plant, max external rotation and release are required")
    source = {k: ("manual" if manual.get(k) is not None and manual.get(k) != old.get(k)
                  else prev["events"][k]["source"] if prev["events"].get(k) else "missing")
              for k in merged}
    ev = Events(fp=int(merged["fp"]), mer=int(merged["mer"]), br=int(merged["br"]),
                pkh=None if merged["pkh"] is None else int(merged["pkh"]), source=source)
    validate_order(ev, n)
    readable = analysis_dir / prev["video"]["decoded_path"]
    if not readable.exists():
        readable = input_path
    result = assemble(prep, ev, ref, opts, prev["video"])
    result["keyframes"] = render_keyframes(readable, analysis_dir / "keyframes", prep.frame_index, prep.image_smooth,
                                           (track.width, track.height), prep.throws, ev.as_dict(), RENDER_MAX_SIDE,
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

"""One ball-flight analysis: prepare (background for marking the field), calibrate (the user's marks), analyze
(track, fit, spin, tracer video), and re-fit with a release frame the user picked."""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .. import __version__
from ..config import RENDER_MAX_SIDE
from ..pipeline import AnalysisError, read_json, write_json
from ..video_io import VideoError, open_readable
from .detect import Corridor, compute_background, detect, frame_times
from .geometry import (FT, IN, PITCHING_DISTANCES_FT, CalibrationError, Calibration, Camera, calibrate,
                       camera_view, plate_points, refine_with_image, rubber_points, zone_corners)
from .measure import remeasure
from .render import calibration_preview, render_tracer, seam_strip
from .spin import MIN_FPS as SPIN_MIN_FPS
from .spin import estimate_spin
from .track import build_tracklets, select_track
from .trajectory import (LIFT_MODEL, PitchFit, Setup, Track, Trajectory, clock_from_angle, fit_pitch,
                         metric_uncertainty, pitch_metrics)

log = logging.getLogger(__name__)
MAX_FLIGHT_FRAMES = 2400           # 10 s at 240 fps
TRACER_PRE_S = 0.35
TRACER_POST_S = 0.15


class FlightError(AnalysisError):
    """A user-facing failure of a ball-flight analysis."""


@dataclass
class FlightOptions:
    throws: str = "R"                       # R | L
    distance_ft: float = 60.5
    mound_in: float | None = None           # default by pitching distance
    zone_bottom_in: float = 18.0
    zone_top_in: float = 42.0
    capture_fps: float | None = None
    trim_start_s: float | None = None
    trim_end_s: float | None = None
    pitcher_height_in: float | None = None
    altitude_ft: float = 0.0
    label: str = ""

    def validate(self) -> "FlightOptions":
        if self.throws not in ("R", "L"):
            raise ValueError("throws must be R or L")
        if not (30.0 <= self.distance_ft <= 70.0):
            raise ValueError("pitching distance must be between 30 and 70 ft")
        if self.mound_in is None:
            self.mound_in = PITCHING_DISTANCES_FT.get(self.distance_ft, 10.0)
        checks = {"mound_in": (0, 18), "zone_bottom_in": (6, 48), "zone_top_in": (18, 72), "capture_fps": (15, 2000),
                  "trim_start_s": (0, 36000), "trim_end_s": (0, 36000), "pitcher_height_in": (40, 90),
                  "altitude_ft": (-500, 12000)}
        for name, (lo, hi) in checks.items():
            v = getattr(self, name)
            if v is not None and not (lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi}")
        if self.zone_top_in <= self.zone_bottom_in + 6:
            raise ValueError("the top of the zone must be at least 6 in above the bottom")
        if self.trim_start_s is not None and self.trim_end_s is not None and self.trim_end_s <= self.trim_start_s:
            raise ValueError("trim end must be after trim start")
        self.label = (self.label or "")[:120]
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "FlightOptions":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known}).validate()

    def setup(self) -> Setup:
        return Setup(distance_ft=self.distance_ft, mound_in=float(self.mound_in), throws=self.throws,
                     zone_bottom_in=self.zone_bottom_in, zone_top_in=self.zone_top_in, altitude_ft=self.altitude_ft,
                     pitcher_height_in=self.pitcher_height_in)


def options_dict(opts: FlightOptions) -> dict:
    return asdict(opts)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _readable(d: Path, meta: dict) -> Path:
    p = d / meta.get("video", {}).get("decoded_path", meta["input_file"])
    return p if p.exists() else d / meta["input_file"]


# ------------------------------------------------------------------------------------------ prepare
def prepare(d: Path, opts: FlightOptions, input_path: Path, progress: Callable[[float, str], None]) -> dict:
    """Probe and (if needed) transcode the clip, and build the people-removed background used for marking
    the field and for detection."""
    progress(0.02, "Reading video")
    try:
        readable, info = open_readable(input_path, d)
    except VideoError as exc:
        raise FlightError(f"Could not read the video: {exc}") from exc
    est = info.frame_count
    if opts.trim_start_s is not None or opts.trim_end_s is not None:
        a = opts.trim_start_s or 0.0
        b = opts.trim_end_s or info.duration_s
        est = int(max(0.0, b - a) * info.fps) if info.duration_s else info.frame_count
    if est > MAX_FLIGHT_FRAMES:
        raise FlightError(f"The selected range has about {est} frames (limit {MAX_FLIGHT_FRAMES}). Trim the clip to "
                          "the pitch (about 1 s before release to 0.5 s after the catch).")
    progress(0.05, "Building the background")
    try:
        bg, n = compute_background(readable, est, opts.trim_start_s, opts.trim_end_s,
                                   progress=lambda f: progress(0.05 + 0.9 * f, "Building the background"))
    except ValueError as exc:
        raise FlightError(str(exc)) from exc
    if n < 20:
        raise FlightError(f"The selected range has only {n} frames; a pitch needs at least 20.")
    cv2.imwrite(str(d / "background.png"), bg)
    cv2.imwrite(str(d / "background.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
    first = next(iter(frame_times(readable, opts.trim_start_s, opts.trim_end_s, opts.capture_fps)))
    fps_eff = float(opts.capture_fps or info.fps)
    return {"video": info.as_dict() | {"decoded_path": readable.name, "frames_in_range": n,
                                       "first_frame": int(first[0]), "fps_effective": fps_eff}}


def frame_image(d: Path, meta: dict, index: int) -> np.ndarray:
    """Decoded frame `index` of the clip (frame numbers of the file, trim ignored)."""
    path = _readable(d, meta)
    for i, _, frame in frame_times(path, None, None, None):
        if i == index:
            return frame
        if i > index:
            break
    raise FlightError(f"frame {index} is not in the video")


# ------------------------------------------------------------------------------------------ calibrate
def _calibration_image(d: Path, meta: dict, source) -> np.ndarray:
    if source in (None, "background"):
        img = cv2.imread(str(d / "background.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FlightError("the background image is missing; upload the clip again")
        return img
    return frame_image(d, meta, int(source))


def solve_calibration(d: Path, meta: dict, body: dict, n_samples: int = 32) -> tuple[Calibration, dict]:
    """Camera from the user's marks, refined against the picture they marked on. Returns the calibration
    and the record stored in calibration.json."""
    opts = FlightOptions.from_dict(meta["options"])
    image = _calibration_image(d, meta, body.get("image", "background"))
    h, w = image.shape[:2]
    distance = float(body.get("distance_ft") or opts.distance_ft)
    mound = float(body["mound_in"]) if body.get("mound_in") is not None else float(opts.mound_in)
    height = body.get("camera_height_ft")
    height = None if height in (None, "") else float(height)
    try:
        calib = calibrate(body["plate_px"], body["rubber_px"], (w, h), distance, mound, camera_height_ft=height,
                          n_samples=n_samples)
    except (KeyError, TypeError) as exc:
        raise CalibrationError("mark all five corners of home plate and both front corners of the rubber") from exc
    refined, rinfo = refine_with_image(calib, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), n_samples=n_samples)
    cam = refined.camera
    record = {
        "plate_px": np.asarray(body["plate_px"], float).tolist(), "rubber_px": np.asarray(body["rubber_px"], float).tolist(),
        "image": body.get("image", "background"), "image_size": [w, h], "distance_ft": distance, "mound_in": mound,
        "camera_height_ft": height, "camera": cam.as_dict(), "rms_px": calib.rms_px,
        "residuals_px": calib.residuals_px.tolist(), "refinement": {k: v for k, v in rinfo.items()},
        "view": camera_view(cam, distance), "camera_position_ft": (cam.center / FT).tolist(), "hfov_deg": cam.hfov_deg,
        "computed_utc": _now(),
    }
    return refined, record


def overlay_lines(cam: Camera, distance_ft: float, mound_in: float, zone_bottom_in: float, zone_top_in: float) -> dict:
    """Projected outlines (image pixels) for the browser to draw over the marking image."""
    def proj(pts):
        uv, z = cam.project(pts)
        return None if np.any(z <= 0.05) else np.round(uv, 2).tolist()
    rb = rubber_points(distance_ft, mound_in)
    return {"plate": proj(plate_points()), "rubber": proj(np.vstack([rb, rb[::-1] + [0.0, 6 * IN, 0.0]])),
            "zone": proj(zone_corners(zone_bottom_in, zone_top_in)),
            "line": proj(np.array([[0.0, 17 * IN, 0.0], [0.0, distance_ft * FT, mound_in * IN]]))}


def calibrate_flight(d: Path, meta: dict, body: dict) -> dict:
    opts = FlightOptions.from_dict(meta["options"])
    refined, record = solve_calibration(d, meta, body, n_samples=0)
    write_json(d / "calibration.json", record)
    record["overlay"] = overlay_lines(refined.camera, record["distance_ft"], record["mound_in"], opts.zone_bottom_in,
                                      opts.zone_top_in)
    return record


def load_calibration(d: Path, meta: dict, n_samples: int = 32) -> Calibration:
    rec = read_json(d / "calibration.json")
    calib, _ = solve_calibration(d, meta, rec, n_samples=n_samples)
    return calib


# ------------------------------------------------------------------------------------------ analyze
def _warnings(opts: FlightOptions, fps: float, pf: PitchFit, m: dict, calib_rec: dict, track: Track,
              spin_res, notes: list[str]) -> list[str]:
    w = list(notes)
    if fps < 100:
        w.append(f"The clip runs at {fps:.0f} fps; 240 fps slow motion tracks the ball far better.")
    if fps < SPIN_MIN_FPS:
        w.append("Spin from the seams needs 240 fps; the spin axis and active spin shown come from the ball's movement.")
    if not pf.release_observed:
        w.append("The ball was not tracked out of the pitcher's hand, so the release distance could not anchor the "
                 "speed: it rests on the ball's apparent size (wider error). If you can see the release in the video, "
                 "step to it and use “Set release here”.")
    if calib_rec.get("view") == "behind_pitcher":
        w.append("Filmed from behind the pitcher: the plate end of the flight is far and small, so velocity and "
                 "movement are less certain than from behind home plate.")
    elif calib_rec.get("view") == "side":
        w.append("The camera is not along the pitching line; the method is built for behind-the-plate or "
                 "behind-the-pitcher views, so treat the numbers with caution.")
    if not calib_rec.get("refinement", {}).get("used"):
        w.append("The plate and rubber outlines could not be matched to the picture, so the marks were used exactly "
                 "as clicked; zoom in and re-mark them for a more precise calibration.")
    k = pf.fit.size_scale
    if abs(np.log(k)) > 0.08:
        w.append(f"The ball's apparent size disagrees with the calibration by {100 * (k - 1):+.0f}% (lighting, blur "
                 "or a calibration error); the result relies on the release distance.")
    t_last = float(track.t[-1])
    y_last = pf.fit.traj.pos(t_last)[1] / FT
    if y_last > 8.0:
        w.append(f"The ball was last seen {y_last:.0f} ft in front of the plate; the plate location is extrapolated.")
    if spin_res is not None and not spin_res.ok and fps >= SPIN_MIN_FPS:
        w.append(f"Spin was not measured from the seams: {spin_res.reason}.")
    elif spin_res is not None and spin_res.ok and spin_res.confidence == "low":
        w.append("The seam fit is weak (blurry or small ball), so the seam-based spin is not shown.")
    return w


def _summary_lines(m: dict, spin_d: dict | None) -> list[str]:
    out = [f"{m['release_speed_mph']:.1f} mph"]
    if spin_d and spin_d.get("rpm"):
        out.append(f"{spin_d['rpm']:.0f} rpm, {spin_d['tilt_clock']}")
    if m.get("ivb_in") is not None:
        out.append(f"IVB {m['ivb_in']:.1f} in, HB {m['hb_arm_side_in']:.1f} in arm side")
    return out


def _spin_dict(sr) -> dict:
    d = {"ok": bool(sr.ok and sr.confidence in ("high", "medium")), "reason": sr.reason}
    if sr.ok:
        d.update(rpm=sr.rpm, axis_world=sr.axis_world, efficiency=sr.efficiency, gyro_deg=sr.gyro_deg,
                 active_rpm=sr.active_rpm, tilt_deg=sr.tilt_deg, tilt_clock=clock_from_angle(sr.tilt_deg),
                 axis_deg=(sr.tilt_deg + 180.0) % 360.0, frames=sr.frames, mean_score=sr.mean_score, margin=sr.margin,
                 confidence=sr.confidence, alternative_rpm=sr.alternative_rpm)
        if sr.confidence == "low":
            d["reason"] = "the seam fit is too weak to trust"
    return d


def _result(d: Path, meta: dict, opts: FlightOptions, calib: Calibration, calib_rec: dict, track: Track,
            pf: PitchFit, notes: list[str], spin_d: dict, spin_res, extra: dict) -> dict:
    setup = opts.setup()
    traj = pf.fit.traj
    t_rel = pf.release_time(traj)
    m = pitch_metrics(traj, setup, t_rel)
    if m.get("t_plate") is None:
        raise FlightError("The fitted flight never reaches the plate; check the calibration marks.")
    sd = metric_uncertainty(pf, track, calib.samples, setup)
    fps = float(meta["video"]["fps_effective"])
    release_frame = int(round(meta["video"]["first_frame"] + (t_rel - extra["t0"]) * fps)) if extra.get("t0") is not None else None
    res = {
        "analysis_version": __version__, "kind": "flight", "computed_utc": _now(),
        "setup": asdict(setup), "video": meta["video"],
        "calibration": {k: calib_rec.get(k) for k in ("view", "camera_position_ft", "hfov_deg", "rms_px", "residuals_px",
                                                      "refinement", "image", "camera_height_ft", "distance_ft",
                                                      "mound_in")},
        "pitch": m, "uncertainty": sd,
        "release": {"observed": pf.release_observed, "source": pf.release_source, "t": t_rel, "frame": release_frame},
        "spin": spin_d,
        "movement_spin": {"tilt_clock": m.get("movement_tilt_clock"), "tilt_deg": m.get("movement_tilt_deg"),
                          "axis_deg": m.get("movement_axis_deg"), "active_rpm_est": m.get("active_spin_est_rpm"),
                          "lift_model": LIFT_MODEL},
        "trajectory": traj.as_dict(),
        "fit": {"rms_px": pf.fit.rms_px, "points": int(len(track)), "inliers": int(pf.fit.inliers.sum()),
                "size_scale": pf.fit.size_scale, "first_frame": int(track.frame[0]), "last_frame": int(track.frame[-1]),
                "extension_prior_used": pf.fit.extension_prior_used},
        "track": {"frame": [int(f) for f in track.frame], "u": np.round(track.uv[:, 0], 2).tolist(),
                  "v": np.round(track.uv[:, 1], 2).tolist(), "d": np.round(track.d, 2).tolist()},
    }
    res["quality"] = {"warnings": _warnings(opts, fps, pf, m, calib_rec, track, spin_res, notes)}
    return res


def _save_track(d: Path, track: Track, t0: float) -> None:
    np.savez_compressed(d / "track.npz", frame=track.frame, t=track.t, uv=track.uv, d=track.d, sig_uv=track.sig_uv,
                        d_kind=track.d_kind, t0=np.array([t0]))


def _load_track(d: Path) -> tuple[Track, float]:
    z = np.load(d / "track.npz")
    return Track(frame=z["frame"], t=z["t"], uv=z["uv"], d=z["d"], sig_uv=z["sig_uv"], d_kind=z["d_kind"]), float(z["t0"][0])


def _render(d: Path, meta: dict, opts: FlightOptions, calib: Calibration, res: dict, track: Track, t0: float,
            progress: Callable[[float], None] | None = None) -> None:
    traj = Trajectory.from_dict(res["trajectory"])
    fps = float(meta["video"]["fps_effective"])
    first0 = int(meta["video"]["first_frame"])
    t_rel, t_pl = res["release"]["t"], res["pitch"]["t_plate"]
    # anchored to the first tracked frame, not the release, so frame numbers stay put when the release is edited
    first = max(first0, int(np.floor(first0 + (float(track.t[0]) - TRACER_PRE_S - t0) * fps)))
    last = int(np.ceil(first0 + (t_pl + TRACER_POST_S - t0) * fps))
    measured = {int(f): (float(u), float(v)) for f, (u, v) in zip(track.frame, track.uv)}
    size, f0, n = render_tracer(_readable(d, meta), d / "tracer.mp4", calib.camera, traj, t_rel, t_pl, measured,
                                opts.setup(), float(meta["video"]["fps"]), RENDER_MAX_SIDE, opts.trim_start_s,
                                opts.trim_end_s, opts.capture_fps, first, last,
                                _summary_lines(res["pitch"], res["spin"]), progress)
    res["tracer"] = {"file": "tracer.mp4", "fps": float(meta["video"]["fps"]), "size": list(size), "first_frame": f0,
                     "frames": n}


def analyze(d: Path, meta: dict, progress: Callable[[float, str], None] = lambda f, s: None) -> dict:
    started = time.time()
    opts = FlightOptions.from_dict(meta["options"])
    setup = opts.setup()
    if not (d / "calibration.json").exists():
        raise FlightError("Mark home plate and the rubber first.")
    progress(0.02, "Loading the calibration")
    calib_rec = read_json(d / "calibration.json")
    try:
        calib = load_calibration(d, meta)
    except CalibrationError as exc:
        raise FlightError(f"The calibration no longer solves: {exc}") from exc
    cam = calib.camera
    bg = cv2.imread(str(d / "background.png"), cv2.IMREAD_COLOR)
    if bg is None or (bg.shape[1], bg.shape[0]) != (cam.width, cam.height):
        raise FlightError("The background image is missing or has a different size; upload the clip again.")
    path = _readable(d, meta)
    n_est = int(meta["video"].get("frames_in_range") or meta["video"]["frame_count"])

    progress(0.05, "Finding the ball")
    corridor = Corridor(cam, setup.distance_ft)
    try:
        frames = detect(path, corridor, bg, opts.trim_start_s, opts.trim_end_s, opts.capture_fps, n_est,
                        progress=lambda f: progress(0.05 + 0.45 * f, "Finding the ball"))
    except ValueError as exc:
        raise FlightError(str(exc)) from exc
    if not frames:
        raise FlightError("The selected range contains no frames.")
    t0 = frames[0].t
    progress(0.52, "Tracking the ball")
    sel = select_track(frames, build_tracklets(frames, cam.width, cam.height), cam, setup)
    if sel is None:
        raise FlightError("No ball flight from the mound to the plate was found. Check that the clip contains the whole "
                          "pitch, that the ball is visible against the background, and that home plate and the rubber "
                          "were marked on the right corners.")
    track0, fit0, notes = sel
    progress(0.58, "Measuring the ball in every frame")
    meas = remeasure(path, bg, fit0.traj, cam, track0, opts.capture_fps, opts.trim_start_s, opts.trim_end_s)
    track = meas.track if len(meas.track) >= 8 else track0
    progress(0.64, "Fitting the trajectory")
    try:
        pf = fit_pitch(track, cam, setup, init=fit0.traj)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise FlightError(f"The ball track could not be fitted: {exc}") from exc
    fps = float(meta["video"]["fps_effective"])
    progress(0.70, "Reading the spin from the seams")
    spin_res = estimate_spin(meas.patches, cam, pf.fit.traj, fps)
    spin_d = _spin_dict(spin_res)
    if spin_res.ok:
        try:
            spin_d["strip"] = "seams.jpg"
            spin_d["strip_frames"] = seam_strip(meas.patches, spin_res.orientations, d / "seams.jpg")
        except Exception as exc:  # noqa: BLE001 - the picture is a diagnostic, not a result
            log.warning("seam strip failed: %s", exc)
            spin_d.pop("strip", None)
    progress(0.80, "Computing uncertainty")
    res = _result(d, meta, opts, calib, calib_rec, track, pf, notes, spin_d, spin_res, {"t0": t0})
    _save_track(d, track, t0)
    progress(0.84, "Rendering the tracer video")
    _render(d, meta, opts, calib, res, track, t0, progress=lambda f: progress(0.84 + 0.15 * f, "Rendering the tracer video"))
    res["processing_s"] = round(time.time() - started, 1)
    write_json(d / "result.json", res)
    progress(1.0, "Done")
    return res


def recompute_release(d: Path, meta: dict, frame: int | None) -> dict:
    """Re-fit with the release at a frame the user picked (None: back to automatic)."""
    opts = FlightOptions.from_dict(meta["options"])
    setup = opts.setup()
    prev = read_json(d / "result.json")
    calib_rec = read_json(d / "calibration.json")
    calib = load_calibration(d, meta)
    track, t0 = _load_track(d)
    fps = float(meta["video"]["fps_effective"])
    t_manual = None
    if frame is not None:
        t_manual = t0 + (int(frame) - int(meta["video"]["first_frame"])) / fps
        if t_manual >= track.t[0] + 0.5 * (track.t[-1] - track.t[0]):
            raise ValueError("the release must be before the ball's flight")
    pf = fit_pitch(track, calib.camera, setup, init=Trajectory.from_dict(prev["trajectory"]), t_release_manual=t_manual)
    res = _result(d, meta, opts, calib, calib_rec, track, pf, [], prev.get("spin") or {}, None, {"t0": t0})
    _render(d, meta, opts, calib, res, track, t0)
    res["processing_s"] = prev.get("processing_s")
    write_json(d / "result.json", res)
    return res

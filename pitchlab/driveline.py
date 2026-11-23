"""Driveline OpenBiomechanics (OBP) pitching data: loading, and simulation of phone-video conditions.

The OBP landmarks table holds marker-derived joint centres at 360 Hz in the lab frame (+x toward home,
+y toward first base, +z up; left-handers are NOT mirrored in the raw table). Data licence:
CC BY-NC-SA 4.0 with additional exclusions - see https://openbiomechanics.org/#terms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DRIVELINE_DIR
from .kinematics import Motion
from .signals import fill_gaps, lowpass, max_gap_s, resample, smooth_skeleton
from .skeleton import DRIVELINE_COLUMNS, J, JOINTS, leg_length, leg_length_projected, mirror_y

EVENT_COLUMNS = {"pkh": "pkh_time", "fc10": "fp_10_time", "fp": "fp_100_time",
                 "mer": "MER_time", "br": "BR_time"}

# Per-joint noise (metres, 1 sigma) for the video simulation at noise_scale = 1. Wrists and elbows move
# fastest and blur most, so they get the largest error.
NOISE_SIGMA_M = {
    "t_shoulder": 0.015, "g_shoulder": 0.015, "r_hip": 0.015, "l_hip": 0.015,
    "r_knee": 0.020, "l_knee": 0.020, "r_ankle": 0.020, "l_ankle": 0.020,
    "t_elbow": 0.025, "g_elbow": 0.025, "t_wrist": 0.030, "g_wrist": 0.030,
}
LOW_FREQ_FRACTION = 0.6     # slow (drifting) error component, as a fraction of the per-frame sigma
LOW_FREQ_HZ = 2.0


@dataclass
class Trial:
    session_pitch: str
    user: int
    throws: str
    t: np.ndarray
    pos: np.ndarray                 # (T, 12, 3), right-hander normalised
    events_s: dict[str, float]      # Driveline event times (s); NaN when missing
    pitch_speed_mph: float
    height_m: float
    mass_kg: float
    age_yrs: float
    playing_level: str
    poi: dict[str, float]


def load_trials(directory=DRIVELINE_DIR) -> list[Trial]:
    lm_path = directory / "landmarks.csv"
    if not lm_path.exists():
        raise FileNotFoundError(f"{lm_path} missing - run scripts/fetch_assets.py first")
    lm = pd.read_csv(lm_path)
    poi = pd.read_csv(directory / "poi_metrics.csv").set_index("session_pitch")
    meta = pd.read_csv(directory / "metadata.csv").set_index("session_pitch")
    cols = [[f"{DRIVELINE_COLUMNS[j]}_{a}" for a in "xyz"] for j in JOINTS]
    trials: list[Trial] = []
    for sp, g in lm.groupby("session_pitch", sort=False):
        if sp not in poi.index or sp not in meta.index:
            continue
        g = g.sort_values("time")
        pos = np.stack([g[c].to_numpy(dtype=float) for c in cols], axis=1)
        p = poi.loc[sp]
        throws = str(p["p_throws"])
        if throws == "L":
            pos = mirror_y(pos)
        first = g.iloc[0]
        events = {k: float(first[c]) for k, c in EVENT_COLUMNS.items()}
        m = meta.loc[sp]
        trials.append(Trial(
            session_pitch=str(sp), user=int(m["user"]), throws=throws,
            t=g["time"].to_numpy(dtype=float), pos=pos, events_s=events,
            pitch_speed_mph=float(p["pitch_speed_mph"]), height_m=float(m["session_height_m"]),
            mass_kg=float(m["session_mass_kg"]), age_yrs=float(m["age_yrs"]),
            playing_level=str(m["playing_level"]),
            poi={k: float(v) for k, v in p.items() if isinstance(v, (int, float, np.floating, np.integer))},
        ))
    return trials


def _lab_track(pos: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "l_ankle": pos[:, J["l_ankle"]].copy(),
        "r_ankle": pos[:, J["r_ankle"]].copy(),
        "l_knee": pos[:, J["l_knee"]].copy(),
        "mid_hip": (pos[:, J["r_hip"]] + pos[:, J["l_hip"]]) / 2.0,
        "t_wrist": pos[:, J["t_wrist"]].copy(),
        "t_shoulder": pos[:, J["t_shoulder"]].copy(),
    }


UP = np.array([0.0, 0.0, 1.0])
FORWARD = np.array([1.0, 0.0, 0.0])
VIEWS = ("side", "behind", "3d")


@dataclass(frozen=True)
class Camera:
    """A static pinhole camera aimed at the middle of the delivery (right-hander-normalised frame).
    side: on the arm side (-y) looking across the mound; behind: behind the rubber looking at home."""
    view: str
    distance: float   # metres from the middle of the delivery
    height: float     # lens height above the ground, metres
    offset: float     # metres along the image-horizontal axis


NOMINAL_CAMERAS = {"side": Camera("side", 6.0, 1.1, 0.0), "behind": Camera("behind", 7.0, 1.3, 0.0)}


def random_camera(view: str, rng: np.random.Generator) -> Camera:
    """Phone-on-a-tripod geometry varied over the range people actually film from."""
    if view == "side":
        return Camera("side", rng.uniform(4.0, 9.0), rng.uniform(0.7, 1.6), rng.uniform(-0.6, 0.6))
    return Camera("behind", rng.uniform(5.0, 10.0), rng.uniform(0.9, 1.7), rng.uniform(-0.4, 0.4))


def _project(points: np.ndarray, cam: Camera, centre_x: float) -> np.ndarray:
    """Normalised pinhole image coordinates (x right, y down) of (..., 3) lab points."""
    if cam.view == "side":
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        C = np.array([centre_x + cam.offset, -cam.distance, cam.height])
    else:
        R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        C = np.array([centre_x - cam.distance, cam.offset, cam.height])
    pc = (points - C) @ R.T
    return pc[..., :2] / pc[..., 2:3]


def view_track(pos: np.ndarray, view: str, camera: Camera | None = None) -> tuple[dict, np.ndarray, np.ndarray | None, float]:
    """Absolute tracks as a static camera would record them: (track, up, forward-or-None, leg-length
    scale). The video pipeline measures foot plant, stride and momentum in the image, so the reference
    does too - including perspective, which turns motion toward or away from the lens into image motion."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    raw = _lab_track(pos)
    if view == "3d":
        return raw, UP, FORWARD, leg_length(pos)
    cam = camera or NOMINAL_CAMERAS[view]
    if cam.view != view:
        raise ValueError("camera view does not match")
    centre_x = float(np.nanmean(raw["mid_hip"][:, 0]))
    proj = {k: _project(v, cam, centre_x) for k, v in raw.items()}
    legs = (_project(pos[:, J[n]], cam, centre_x) for n in ("r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"))
    scale = leg_length_projected(*legs)
    forward = np.array([1.0, 0.0]) if view == "side" else None
    return proj, np.array([0.0, -1.0]), forward, scale


def side_leg_length_m(pos: np.ndarray) -> float:
    """Least-foreshortened leg length seen side-on, in metres (orthographic x-z projection) - the same
    definition the image track uses, so height / this ratio converts image strides to % height."""
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return leg_length_projected(*(pos[:, J[n]] @ P.T for n in ("r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle")))


def clean_motion(tr: Trial, view: str = "side") -> Motion:
    """The lab data as-is (already 20 Hz filtered by Driveline), tracks seen by the nominal camera."""
    track, up, fwd, scale = view_track(tr.pos, view)
    return Motion(t=tr.t, pos=tr.pos, track=track, track_up=up, track_forward=fwd, track_scale=scale,
                  heading_known=True, fps=1.0 / float(np.median(np.diff(tr.t))))


# Chance that the throwing arm is blurred out of tracking for 1-4 frames around release, by frame rate
# (on a real 60 fps clip MediaPipe lost the arm for 3 frames and mislabelled a 4th).
ARM_DROPOUT_P = {240: 0.1, 120: 0.35, 60: 0.7}


def arm_dropout_p(fps: float) -> float:
    return ARM_DROPOUT_P[min(ARM_DROPOUT_P, key=lambda f: abs(f - fps))]


def simulate_video(tr: Trial, fps: float, noise_scale: float, rng: np.random.Generator,
                   heading_known: bool = True, view: str = "side", camera: Camera | str = "random",
                   dropout_p: float = 0.0) -> Motion:
    """Resample to `fps`, add per-frame and slowly drifting joint noise, then smooth exactly like the
    video pipeline does; tracks are what a camera at `view` records ("random" draws a realistic
    tripod position, "nominal" uses NOMINAL_CAMERAS). With probability `dropout_p` the throwing elbow
    and wrist go missing for 1-4 frames between layback and just after release and are bridged by
    interpolation, as blurred frames are in the video pipeline."""
    t_new = np.arange(tr.t[0], tr.t[-1], 1.0 / fps)
    pos = resample(tr.pos, tr.t, t_new)
    sig = np.array([NOISE_SIGMA_M[j] for j in JOINTS]) * noise_scale
    hf = rng.standard_normal(pos.shape) * sig[None, :, None]
    lf_raw = lowpass(rng.standard_normal(pos.shape), fps, LOW_FREQ_HZ)
    lf_raw /= max(float(np.nanstd(lf_raw)), 1e-9)
    noisy = pos + hf + lf_raw * (sig * LOW_FREQ_FRACTION)[None, :, None]
    arm_valid = np.ones(len(t_new), dtype=bool)
    mer_t, br_t = tr.events_s.get("mer", np.nan), tr.events_s.get("br", np.nan)
    if dropout_p > 0 and np.isfinite(mer_t) and np.isfinite(br_t) and rng.random() < dropout_p:
        i_mer = int(np.argmin(np.abs(t_new - mer_t)))
        i_br = int(np.argmin(np.abs(t_new - br_t)))
        length = int(rng.integers(1, 5))
        start = int(rng.integers(max(0, i_mer - 1), max(i_mer, i_br + 2 - length) + 1))
        arm_valid[start:start + length] = False
        noisy[np.ix_(~arm_valid, [J["t_elbow"], J["t_wrist"]])] = np.nan
        noisy = fill_gaps(noisy, t_new, max_gap_s(fps))
    # MediaPipe's 3D landmarks are pelvis-centred per frame before the app filters them; with different
    # arm/body cut-offs that order matters (release timing), so the simulation centres first as well.
    mid_hip = (noisy[:, J["r_hip"]] + noisy[:, J["l_hip"]]) / 2.0
    centred = smooth_skeleton(noisy - mid_hip[:, None, :], fps)
    smooth = smooth_skeleton(noisy, fps)          # absolute positions -> camera tracks
    if view == "3d":
        cam = None
    elif camera == "random":
        cam = random_camera(view, rng)
    elif camera == "nominal":
        cam = NOMINAL_CAMERAS[view]
    else:
        cam = camera
    track, up, fwd, scale = view_track(smooth, view, cam)
    return Motion(t=t_new, pos=centred, track=track, track_up=up, track_forward=fwd, track_scale=scale,
                  heading_known=heading_known, fps=fps, arm_valid=arm_valid)


def true_event_indices(tr: Trial, t: np.ndarray) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for k, v in tr.events_s.items():
        out[k] = None if not np.isfinite(v) else int(np.argmin(np.abs(t - v)))
    return out

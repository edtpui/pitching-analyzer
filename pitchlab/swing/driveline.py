"""Driveline OpenBiomechanics (OBP) hitting data: loading, and simulation of phone-video conditions.

The OBP hitting landmarks table holds marker-derived joint centres at 360 Hz in the lab frame (+x from
home plate toward the pitcher, +y toward the right-handed batter's box, +z up; left-handed hitters are NOT
mirrored in the raw table). Data licence: CC BY-NC-SA 4.0 with additional exclusions - see
https://openbiomechanics.org/#terms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DRIVELINE_DIR
from .kinematics import Motion
from .signals import fill_gaps, lowpass, max_gap_s, resample, smooth_skeleton
from .skeleton import J, JOINTS, driveline_columns, leg_length, leg_length_projected, mirror_y

# fc = front-foot contact (10% body weight on the front plate), fp = foot plant (100%), contact = ball contact
EVENT_COLUMNS = {"fc": "fp_10_time", "fp": "fp_100_time", "contact": "contact_time"}
LB_TO_KG = 0.45359237
IN_TO_M = 0.0254

# Per-joint noise (metres, 1 sigma) for the video simulation at noise_scale = 1. Wrists and elbows move
# fastest and blur most, so they get the largest error.
NOISE_SIGMA_M = {
    "rear_shoulder": 0.015, "lead_shoulder": 0.015, "rear_hip": 0.015, "lead_hip": 0.015,
    "rear_knee": 0.020, "lead_knee": 0.020, "rear_ankle": 0.020, "lead_ankle": 0.020,
    "rear_elbow": 0.025, "lead_elbow": 0.025, "rear_wrist": 0.030, "lead_wrist": 0.030,
}
LOW_FREQ_FRACTION = 0.6     # slow (drifting) error component, as a fraction of the per-frame sigma
LOW_FREQ_HZ = 2.0
HAND_JOINTS = ("rear_elbow", "rear_wrist", "lead_elbow", "lead_wrist")


@dataclass
class Trial:
    session_swing: str
    user: int
    bats: str
    t: np.ndarray
    pos: np.ndarray                 # (T, 12, 3), right-hander normalised
    events_s: dict[str, float]      # Driveline event times (s); NaN when missing
    bat_speed_mph: float            # resultant sweet-spot speed at contact
    exit_velo_mph: float            # NaN when not recorded
    height_m: float
    mass_kg: float
    age_yrs: float
    playing_level: str
    poi: dict[str, float]


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


SAMPLE_RATE_HZ = 360.0


def _exact_times(t: np.ndarray) -> np.ndarray:
    """The table stores time rounded to 0.1 ms, so steps alternate 2.7 / 2.8 ms; that jitter is ~2% noise on
    every derivative. Restore the exact 360 Hz grid when the stamps are consistent with it."""
    k = np.round(t * SAMPLE_RATE_HZ)
    if len(k) > 1 and np.all(np.diff(k) == 1) and np.max(np.abs(t * SAMPLE_RATE_HZ - k)) < 0.1:
        return k / SAMPLE_RATE_HZ
    return t


def load_trials(directory=DRIVELINE_DIR) -> list[Trial]:
    lm_path = directory / "landmarks.csv"
    if not lm_path.exists():
        raise FileNotFoundError(f"{lm_path} missing - run scripts/fetch_assets.py first")
    lm = pd.read_csv(lm_path)
    poi = pd.read_csv(directory / "poi_metrics.csv").set_index("session_swing")
    meta = pd.read_csv(directory / "metadata.csv").set_index("session_swing")
    trials: list[Trial] = []
    for ss, g in lm.groupby("session_swing", sort=False):
        if ss not in poi.index or ss not in meta.index:
            continue
        m = meta.loc[ss]
        bats = str(m["hitter_side"])
        if bats not in ("R", "L"):
            continue
        g = g.sort_values("time")
        cols = driveline_columns(bats)
        pos = np.stack([g[[f"{cols[j]}_{a}" for a in "xyz"]].to_numpy(dtype=float) for j in JOINTS], axis=1)
        if bats == "L":
            pos = mirror_y(pos)
        first = g.iloc[0]
        events = {k: _num(first[c]) for k, c in EVENT_COLUMNS.items()}
        p = poi.loc[ss]
        trials.append(Trial(
            session_swing=str(ss), user=int(m["user"]), bats=bats,
            t=_exact_times(g["time"].to_numpy(dtype=float)), pos=pos, events_s=events,
            bat_speed_mph=_num(p["bat_speed_mph_contact_x"]), exit_velo_mph=_num(p["exit_velo_mph_x"]),
            height_m=_num(m["session_height_in"]) * IN_TO_M, mass_kg=_num(m["session_mass_lbs"]) * LB_TO_KG,
            age_yrs=_num(m["athlete_age"]), playing_level=str(m["highest_playing_level"]),
            poi={k: float(v) for k, v in p.items() if isinstance(v, (int, float, np.floating, np.integer))},
        ))
    return trials


def _lab_track(pos: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "lead_ankle": pos[:, J["lead_ankle"]].copy(),
        "rear_ankle": pos[:, J["rear_ankle"]].copy(),
        "lead_knee": pos[:, J["lead_knee"]].copy(),
        "mid_hip": (pos[:, J["rear_hip"]] + pos[:, J["lead_hip"]]) / 2.0,
        "hands": (pos[:, J["rear_wrist"]] + pos[:, J["lead_wrist"]]) / 2.0,
    }


UP = np.array([0.0, 0.0, 1.0])
FORWARD = np.array([1.0, 0.0, 0.0])
VIEWS = ("side", "behind", "3d")


@dataclass(frozen=True)
class Camera:
    """A static pinhole camera aimed at the hitter (right-hander-normalised frame).
    side: across the plate, facing the hitter's chest (open side), image right = toward the pitcher;
    behind: behind the catcher, looking toward the pitcher."""
    view: str
    distance: float   # metres from the hitter
    height: float     # lens height above the ground, metres
    offset: float     # metres along the image-horizontal axis


NOMINAL_CAMERAS = {"side": Camera("side", 5.0, 1.0, 0.0), "behind": Camera("behind", 5.0, 1.2, 0.0)}


def random_camera(view: str, rng: np.random.Generator) -> Camera:
    """Phone-on-a-tripod geometry varied over the range people actually film swings from."""
    if view == "side":
        return Camera("side", rng.uniform(3.5, 8.0), rng.uniform(0.6, 1.5), rng.uniform(-0.5, 0.5))
    return Camera("behind", rng.uniform(3.5, 8.0), rng.uniform(0.8, 1.6), rng.uniform(-0.5, 0.5))


def _project(points: np.ndarray, cam: Camera, centre: np.ndarray) -> np.ndarray:
    """Normalised pinhole image coordinates (x right, y down) of (..., 3) lab points."""
    cx, cy = float(centre[0]), float(centre[1])
    if cam.view == "side":
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        C = np.array([cx + cam.offset, cy - cam.distance, cam.height])
    else:
        R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        C = np.array([cx - cam.distance, cy + cam.offset, cam.height])
    pc = (points - C) @ R.T
    return pc[..., :2] / pc[..., 2:3]


def view_track(pos: np.ndarray, view: str, camera: Camera | None = None) -> tuple[dict, np.ndarray, np.ndarray | None, float]:
    """Absolute tracks as a static camera would record them: (track, up, forward-or-None, leg-length
    scale). The video pipeline measures foot plant, stride and the forward move in the image, so the
    reference does too - including perspective, which turns motion toward or away from the lens into
    image motion."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    raw = _lab_track(pos)
    if view == "3d":
        return raw, UP, FORWARD, leg_length(pos)
    cam = camera or NOMINAL_CAMERAS[view]
    if cam.view != view:
        raise ValueError("camera view does not match")
    centre = np.nanmean(raw["mid_hip"], axis=0)
    proj = {k: _project(v, cam, centre) for k, v in raw.items()}
    legs = (_project(pos[:, J[n]], cam, centre)
            for n in ("rear_hip", "rear_knee", "rear_ankle", "lead_hip", "lead_knee", "lead_ankle"))
    scale = leg_length_projected(*legs)
    forward = np.array([1.0, 0.0]) if view == "side" else None
    return proj, np.array([0.0, -1.0]), forward, scale


def side_leg_length_m(pos: np.ndarray) -> float:
    """Least-foreshortened leg length seen side-on, in metres (orthographic x-z projection) - the same
    definition the image track uses, so height / this ratio converts image distances to % height."""
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return leg_length_projected(*(pos[:, J[n]] @ P.T for n in ("rear_hip", "rear_knee", "rear_ankle",
                                                               "lead_hip", "lead_knee", "lead_ankle")))


def clean_motion(tr: Trial, view: str = "side") -> Motion:
    """The lab data as-is, tracks seen by the nominal camera."""
    track, up, fwd, scale = view_track(tr.pos, view)
    return Motion(t=tr.t, pos=tr.pos, track=track, track_up=up, track_forward=fwd, track_scale=scale,
                  heading_known=True, fps=1.0 / float(np.median(np.diff(tr.t))))


# Chance that the hands (both wrists and elbows) are blurred out of tracking for 1-3 frames around
# contact, by frame rate. Assumed values: hitters' hands move about a third as fast as a pitcher's at
# release, so these sit below the pitching analyzer's throwing-arm figures.
HANDS_DROPOUT_P = {240: 0.05, 120: 0.2, 60: 0.45}


def hands_dropout_p(fps: float) -> float:
    return HANDS_DROPOUT_P[min(HANDS_DROPOUT_P, key=lambda f: abs(f - fps))]


def simulate_video(tr: Trial, fps: float, noise_scale: float, rng: np.random.Generator,
                   heading_known: bool = True, view: str = "side", camera: Camera | str = "random",
                   dropout_p: float = 0.0) -> Motion:
    """Resample to `fps`, add per-frame and slowly drifting joint noise, then smooth exactly like the
    video pipeline does; tracks are what a camera at `view` records ("random" draws a realistic
    tripod position, "nominal" uses NOMINAL_CAMERAS). With probability `dropout_p` both wrists and
    elbows go missing for 1-3 frames within 40 ms before to 20 ms after contact and are bridged by
    interpolation, as blurred frames are in the video pipeline."""
    t_new = np.arange(tr.t[0], tr.t[-1], 1.0 / fps)
    pos = resample(tr.pos, tr.t, t_new)
    sig = np.array([NOISE_SIGMA_M[j] for j in JOINTS]) * noise_scale
    hf = rng.standard_normal(pos.shape) * sig[None, :, None]
    lf_raw = lowpass(rng.standard_normal(pos.shape), fps, LOW_FREQ_HZ)
    lf_raw /= max(float(np.nanstd(lf_raw)), 1e-9)
    noisy = pos + hf + lf_raw * (sig * LOW_FREQ_FRACTION)[None, :, None]
    hands_valid = np.ones(len(t_new), dtype=bool)
    ct = tr.events_s.get("contact", np.nan)
    if dropout_p > 0 and np.isfinite(ct) and rng.random() < dropout_p:
        lo = int(np.argmin(np.abs(t_new - (ct - 0.04))))
        hi = int(np.argmin(np.abs(t_new - (ct + 0.02))))
        length = int(rng.integers(1, 4))
        start = int(rng.integers(lo, max(lo, hi - length + 1) + 1))
        hands_valid[start:start + length] = False
        noisy[np.ix_(~hands_valid, [J[n] for n in HAND_JOINTS])] = np.nan
        noisy = fill_gaps(noisy, t_new, max_gap_s(fps))
    # MediaPipe's 3D landmarks are pelvis-centred per frame before the app filters them; with different
    # arm/body cut-offs that order matters, so the simulation centres first as well.
    mid_hip = (noisy[:, J["rear_hip"]] + noisy[:, J["lead_hip"]]) / 2.0
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
                  heading_known=heading_known, fps=fps, hands_valid=hands_valid)


def true_event_indices(tr: Trial, t: np.ndarray) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for k, v in tr.events_s.items():
        out[k] = None if not np.isfinite(v) else int(np.argmin(np.abs(t - v)))
    return out


FOOT_FLAT_HEIGHT_M = 0.005


def kinematic_foot_plant(t: np.ndarray, pos: np.ndarray, contact: int) -> int | None:
    """The reference definition of foot plant on lab data: after the lead ankle's peak speed in the 0.6 s
    before contact, the first sample at which it is within FOOT_FLAT_HEIGHT_M of its planted height (median
    over the 50 ms before contact) - the front foot is down flat."""
    la = pos[:, J["lead_ankle"]]
    v = np.linalg.norm(np.gradient(la, t, axis=0), axis=1)
    w = np.flatnonzero((t >= t[contact] - 0.6) & (t <= t[contact] - 0.03))
    planted = np.flatnonzero((t >= t[contact] - 0.05) & (t <= t[contact]))
    if w.size == 0 or planted.size == 0 or not np.isfinite(v[w]).any():
        return None
    p = int(w[np.nanargmax(v[w])])
    level = float(np.nanmedian(la[planted, 2]))
    for i in range(p, contact):
        if la[i, 2] - level < FOOT_FLAT_HEIGHT_M:
            return i
    return None

"""Detection of the swing events used for every discrete metric.

  LIFT     peak lead-knee height in the stride (leg lift); missing when the knee barely rises (toe tap)
  LOAD     maximum hip coil: the most closed pelvis rotation before foot plant. Driveline's hitting
           point-of-interest metrics labelled "hs" are taken at this instant (pelvis rotation there matches
           their value with r = 0.99 on the lab data).
  FP       foot plant: the front foot is down flat, i.e. the lead ankle is back at the height where it sits
           planted at contact (on the lab data: within 5 mm, a median 11 ms before Driveline's fp_100
           force-plate event). The force-plate instant itself cannot be seen on video: hitters who land and
           shift their weight later keep the foot still while the force changes.
  CONTACT  ball contact

Order of detection:
  CONTACT  side-on: peak speed of the hands relative to the pelvis as the camera sees it, plus a calibrated
           offset (the hands peak about 60 ms before contact and slow down while the bat whips through).
           MediaPipe's depth estimate for fast-moving hands is noisy, so side-on timing stays in the image.
           Behind / in front: the hands move toward the pitcher along the camera axis, so the image cannot
           time them; contact is the mean of the 3D hand-speed peak and the shoulder-rotation peak, each plus
           its own calibrated offset.
  FP       after the lead foot's peak stride speed, the first frame at which it is slow - below a fraction of
           that peak - and side-on: back within a small height of its planted level; behind: within a small
           radius of its planted position (from behind, perspective turns travel toward the pitcher into
           image height).
  LOAD     minimum pelvis rotation between 0.8 s before contact and foot plant (needs the lab frame, so it is
           found after the provisional events have fixed the direction to the pitcher).
  LIFT     highest lead-knee point in the second before foot plant, reported when the knee drops at least
           `min_leg_lift` leg lengths from it before landing.
The constants are calibrated on simulated phone video by scripts/build_reference.py (separately for side-on
and behind views) and stored in the reference bundle; the defaults below are only used before it exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_CALIBRATION = {
    "contact_method": "image",            # side-on; "3d" behind / in front of the hitter
    "contact_offset_s": 0.061,            # image: CONTACT = time of peak image hand speed + offset
    "contact_offset_hand3d_s": 0.060,     # 3d: mean of (peak 3D hand speed + this) ...
    "contact_offset_torso_s": 0.040,      # ... and (peak shoulder rotation speed + this)
    "fp_method": "height",                # side-on; "radius" behind / in front of the hitter
    "fp_speed_fraction": 0.3,             # FP: lead-foot speed below fraction x peak stride speed
    "fp_height": 0.015,                   # FP (height): ankle within this many leg lengths of its planted height
    "fp_plant_radius": 0.04,              # FP (radius): ankle within this many leg lengths of its planted position
    "fp_speed_fraction_behind": 0.3,
    "fp_plant_radius_behind": 0.04,
    "fp_to_contact_median_s": 0.117,      # Driveline medians, used only when an event cannot be found
    "load_to_contact_median_s": 0.267,
    "min_leg_lift": 0.06,                 # LIFT: knee must drop this many leg lengths from its peak to landing
}
EVENT_NAMES = ("lift", "load", "fp", "contact")
LOAD_WINDOW_S = 0.8
LIFT_WINDOW_S = 1.0
TORSO_SEARCH_S = 0.25        # 3d contact: shoulder-rotation peak within this of the 3D hand-speed peak


def calib_for_view(calib: dict, view: str) -> dict:
    """The camera sees the hands and the foot in a different image plane from behind, so contact and foot
    plant use their own signals and constants."""
    c = {**DEFAULT_CALIBRATION, **calib}
    if view in ("behind", "front"):
        c["contact_method"] = "3d"
        c["fp_method"] = "radius"
        c["fp_speed_fraction"] = c["fp_speed_fraction_behind"]
        c["fp_plant_radius"] = c["fp_plant_radius_behind"]
    else:
        c["contact_method"] = "image"
        c["fp_method"] = "height"
    return c


@dataclass
class Events:
    fp: int
    contact: int
    load: int | None = None
    lift: int | None = None
    source: dict[str, str] = field(default_factory=dict)   # event -> auto | manual | estimated | missing
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int | None]:
        return {"lift": self.lift, "load": self.load, "fp": self.fp, "contact": self.contact}


def _nearest(t: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(t - value)))


def _window(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.flatnonzero((t >= lo) & (t <= hi))


def detect_contact(t: np.ndarray, kin: dict[str, np.ndarray], calib: dict) -> tuple[int, int]:
    """Return (contact_index, peak_hand_speed_index) with the view's method (see the module docstring)."""
    if calib.get("contact_method", "image") == "image":
        hand = kin["hand_speed_img"]
        if not np.isfinite(hand).any():
            raise ValueError("the hands were never tracked")
        peak = int(np.nanargmax(hand))
        return _nearest(t, t[peak] + calib["contact_offset_s"]), peak
    hand = kin["hand_speed"]
    if not np.isfinite(hand).any():
        raise ValueError("the hands were never tracked")
    peak = int(np.nanargmax(hand))
    estimates = [t[peak] + calib["contact_offset_hand3d_s"]]
    near = _window(t, t[peak] - TORSO_SEARCH_S, t[peak] + TORSO_SEARCH_S)
    # the turn direction is unknown before the lab frame is set (left-handers turn clockwise), so use |speed|
    torso = np.abs(kin["torso_rot_velo"][near])
    if np.isfinite(torso).any():
        estimates.append(t[near[np.nanargmax(torso)]] + calib["contact_offset_torso_s"])
    return _nearest(t, float(np.mean(estimates))), peak


def detect_fp(t: np.ndarray, foot_speed: np.ndarray, foot_pos: np.ndarray, foot_height: np.ndarray, contact: int,
              calib: dict) -> int | None:
    """First sample after the lead foot's peak stride speed at which it is slower than `fp_speed_fraction` of
    that peak and back at its planted spot (median over the 50 ms before contact): within `fp_height` leg
    lengths of its planted height ("height" method) or within `fp_plant_radius` leg lengths of its planted
    position ("radius" method). `foot_pos` is the absolute ankle track divided by leg length and `foot_height`
    its upward component. None when no such sample exists."""
    planted_idx = _window(t, t[contact] - 0.05, t[contact])
    pos_ok = np.isfinite(foot_pos[planted_idx]).all(axis=1) if planted_idx.size else np.array([], bool)
    if not pos_ok.any():
        return None
    stride = _window(t, t[contact] - 0.6, t[contact] - 0.03)
    if stride.size == 0 or not np.isfinite(foot_speed[stride]).any():
        return None
    p = int(stride[np.nanargmax(foot_speed[stride])])
    if not np.isfinite(foot_speed[p]) or foot_speed[p] <= 0:
        return None
    thr = calib["fp_speed_fraction"] * foot_speed[p]
    if calib.get("fp_method", "radius") == "height":
        level = float(np.median(foot_height[planted_idx[pos_ok]]))
        for i in range(p + 1, contact):
            if np.isfinite(foot_speed[i]) and foot_speed[i] < thr and foot_height[i] - level < calib["fp_height"]:
                return i
        return None
    planted = np.median(foot_pos[planted_idx[pos_ok]], axis=0)
    dist = np.linalg.norm(foot_pos - planted, axis=1)
    radius = calib["fp_plant_radius"]
    for i in range(p + 1, contact):
        if not (np.isfinite(foot_speed[i]) and foot_speed[i] < thr and np.isfinite(dist[i])):
            continue
        if dist[i] < radius:
            return i
        nxt = dist[i + 1]
        if dist[i] < 2 * radius and np.isfinite(nxt) and nxt >= dist[i]:
            # closest approach to the planted spot: real pose jitter can keep a planted ankle a few
            # pixels outside the radius until long after it landed
            return i
    return None


def detect_load(t: np.ndarray, pelvis_rot: np.ndarray, contact: int, fp: int) -> int | None:
    """Most closed pelvis rotation between LOAD_WINDOW_S before contact and foot plant. A minimum on the
    first samples of the clip (the clip starts after the load) is not reported."""
    win = _window(t, t[contact] - LOAD_WINDOW_S, t[fp])
    if win.size < 3 or not np.isfinite(pelvis_rot[win]).any():
        return None
    i = int(win[np.nanargmin(pelvis_rot[win])])
    if i <= 2:
        return None
    if i == win[0] and np.isfinite(pelvis_rot[i - 1]) and pelvis_rot[i - 1] <= pelvis_rot[i]:
        return None
    return i


def detect_lift(t: np.ndarray, knee_height: np.ndarray, fp: int, min_lift: float) -> int | None:
    """Highest lead-knee position in the second before foot plant, when the knee then drops at least
    `min_lift` leg lengths before landing (otherwise there was no real leg lift). A maximum on the first
    samples of the clip is not reported."""
    win = _window(t, t[fp] - LIFT_WINDOW_S, t[fp] - 0.05)
    if win.size < 3 or not np.isfinite(knee_height[win]).any():
        return None
    i = int(win[np.nanargmax(knee_height[win])])
    if i <= 2:
        return None
    if i == win[0] and np.isfinite(knee_height[i - 1]) and knee_height[i - 1] >= knee_height[i]:
        return None
    after = knee_height[i:fp + 1]
    if not np.isfinite(after).any() or knee_height[i] - np.nanmin(after) < min_lift:
        return None
    return i


def _hands_tracked(hands_valid: np.ndarray, t: np.ndarray, lo_s: float, hi_s: float) -> bool:
    idx = _window(t, lo_s, hi_s)
    return bool(idx.size) and bool(hands_valid[idx].all())


def detect_events(t: np.ndarray, kin: dict[str, np.ndarray], calib: dict | None = None) -> Events:
    """All four events. `kin` needs the hand speeds (and shoulder rotation speed for the 3d contact method),
    foot_speed, foot_pos, foot_height and knee_height; the load is only searched for when pelvis_rot is present
    (lab-frame kinematics)."""
    c = {**DEFAULT_CALIBRATION, **(calib or {})}
    notes: list[str] = []
    source = {k: "auto" for k in EVENT_NAMES}

    contact, peak = detect_contact(t, kin, c)
    if t[peak] - t[0] < 0.4:
        notes.append("Peak hand speed is within 0.4 s of the clip start; the clip may begin too late to show the "
                     "load and stride.")
    if t[-1] - t[contact] < 0.03:
        notes.append("Contact is at the very end of the clip; the swing may be cut off.")
    hands_valid = kin.get("hands_valid")
    if hands_valid is not None and not _hands_tracked(hands_valid, t, t[peak] - 0.03, t[peak] + 0.03):
        notes.append("The hands were blurred or untracked around peak hand speed, so contact timing is less "
                     "certain. Filming at 120-240 fps fixes this.")

    fp = detect_fp(t, kin["foot_speed"], kin["foot_pos"], kin["foot_height"], contact, c)
    if fp is None:
        fp = _nearest(t, t[contact] - c["fp_to_contact_median_s"])
        source["fp"] = "estimated"
        notes.append("Lead-foot plant could not be detected; placed at the Driveline median "
                     f"{c['fp_to_contact_median_s'] * 1000:.0f} ms before contact.")
    if fp >= contact:
        fp = max(0, contact - 1)
        source["fp"] = "estimated"

    load = None
    if "pelvis_rot" in kin:
        load = detect_load(t, kin["pelvis_rot"], contact, fp)
        if load is None:
            source["load"] = "missing"
            notes.append("The load (maximum hip coil) is not in the clip; the clip seems to start after it, so "
                         "coil metrics are not reported.")
        elif load >= fp:
            load = None
            source["load"] = "missing"
    else:
        source["load"] = "missing"

    lift = detect_lift(t, kin["knee_height"], fp, c["min_leg_lift"])
    if lift is None:
        source["lift"] = "missing"
        notes.append("No leg lift was found (toe-tap or no-stride swing, or the clip starts after it).")
    return Events(fp=fp, contact=contact, load=load, lift=lift, source=source, notes=notes)


def validate_order(ev: Events, n: int) -> None:
    """Raise ValueError if manually supplied events are out of range or out of order."""
    for name, v in ev.as_dict().items():
        if v is not None and not (0 <= v < n):
            raise ValueError(f"{name} frame {v} is outside 0..{n - 1}")
    if not (ev.fp < ev.contact):
        raise ValueError("foot plant must come before contact")
    if ev.load is not None and not (ev.load < ev.fp):
        raise ValueError("the load must come before foot plant")
    if ev.lift is not None and not (ev.lift < ev.fp):
        raise ValueError("the leg lift must come before foot plant")

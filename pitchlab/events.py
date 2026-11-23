"""Detection of the pitching events used for every discrete metric.

  PKH  peak lead-knee height
  FP   lead-foot plant (Driveline's fp_100: full body weight on the lead force plate; Driveline's
       "_fp" point-of-interest metrics are taken at this instant)
  MER  maximum shoulder external rotation (layback)
  BR   ball release

Order of detection:
  BR0  peak throwing-wrist speed relative to the pelvis in the image (a first, rough release time).
  PKH  highest lead-knee position 1.5-0.25 s before release.
  FP   first frame after PKH at which the lead ankle (as the camera sees it) is both slow - below a
       fraction of its peak stride speed - and within a small radius of where it sits planted at release.
       Speed alone fails for pitchers whose foot stops travelling toward home and then drops, and from
       behind the plate, where the stride runs along the camera axis; the planted-position test fixes both.
  BR   side-on: the moment the throwing hand passes a calibrated distance in front of the throwing
       shoulder toward home (on Driveline data that times release to ~4 ms), interpolated across frames
       where the arm is blurred. The speed peak cannot do this: at 60 fps the arm is untracked or mislabelled
       for several frames around release and the "peak" lands on a tracking glitch. Other views: BR0 plus
       a calibrated offset. PKH and FP are then re-detected against the final release time.
  MER  maximum shoulder external rotation between FP and BR, when the arm was really tracked through
       layback; otherwise the Driveline median time before release, marked "estimated".
The constants are calibrated against Driveline's force-plate/ball events on simulated phone video by
scripts/build_reference.py (separately for side-on and behind views) and stored in the reference
bundle; the defaults below are only used before that bundle exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_CALIBRATION = {
    "br_offset_s": 0.0,              # BR = time of peak wrist speed + offset
    "fp_speed_fraction": 0.2,        # FP: lead-foot speed below fraction x peak stride speed (side-on)
    "fp_plant_radius": 0.03,         # FP: ankle within this many leg lengths of its planted position
    "fp_speed_fraction_behind": 0.2,  # the same two for a camera behind / in front of the pitcher
    "fp_plant_radius_behind": 0.03,
    "fp_to_br_median_s": 0.136,      # Driveline median, used only when FP cannot be found
    "mer_to_br_median_s": 0.031,
    "br_hand_ahead": 0.41,           # BR: hand this many leg lengths in front of the shoulder (side-on)
    "br_hand_offset_s": 0.0,         # BR = hand-crossing time + offset
}
EVENT_NAMES = ("pkh", "fp", "mer", "br")


def calib_for_view(calib: dict, view: str) -> dict:
    """The camera sees the foot in a different image plane from behind, so foot plant has its own thresholds."""
    c = {**DEFAULT_CALIBRATION, **calib}
    if view in ("behind", "front"):
        c["fp_speed_fraction"] = c["fp_speed_fraction_behind"]
        c["fp_plant_radius"] = c["fp_plant_radius_behind"]
    return c


@dataclass
class Events:
    fp: int
    mer: int
    br: int
    pkh: int | None = None
    source: dict[str, str] = field(default_factory=dict)   # event -> auto | manual | estimated | missing
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int | None]:
        return {"pkh": self.pkh, "fp": self.fp, "mer": self.mer, "br": self.br}


def _nearest(t: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(t - value)))


def _window(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.flatnonzero((t >= lo) & (t <= hi))


def _argmax_in(x: np.ndarray, idx: np.ndarray) -> int | None:
    if idx.size == 0:
        return None
    vals = x[idx]
    if not np.isfinite(vals).any():
        return None
    return int(idx[np.nanargmax(vals)])


def detect_br(t: np.ndarray, wrist_speed: np.ndarray, calib: dict) -> tuple[int, int]:
    """Return (br_index, peak_wrist_speed_index)."""
    if not np.isfinite(wrist_speed).any():
        raise ValueError("throwing wrist was never tracked")
    peak = int(np.nanargmax(wrist_speed))
    br = _nearest(t, t[peak] + calib.get("br_offset_s", 0.0))
    return br, peak


def detect_pkh(t: np.ndarray, knee_height: np.ndarray, br: int) -> int | None:
    """Highest lead-knee position 1.5-0.25 s before release. It must be a real peak: a maximum on the
    first samples of the clip (clip started after the leg lift), or on the window edge while the knee
    was still higher before it, is not reported."""
    win = _window(t, t[br] - 1.5, t[br] - 0.25)
    i = _argmax_in(knee_height, win)
    if i is None or i <= 2:
        return None
    if i == win[0] and np.isfinite(knee_height[i - 1]) and knee_height[i - 1] >= knee_height[i]:
        return None
    return i


def detect_fp(t: np.ndarray, foot_speed: np.ndarray, foot_pos: np.ndarray, br: int, pkh: int | None,
              calib: dict) -> int | None:
    """First sample after the lead foot's peak stride speed at which it is slow and within
    `fp_plant_radius` leg lengths of its planted position (median over the 50 ms before release).
    `foot_pos` is the absolute ankle track divided by leg length. None when no such sample exists."""
    planted_idx = _window(t, t[br] - 0.05, t[br])
    pos_ok = np.isfinite(foot_pos[planted_idx]).all(axis=1) if planted_idx.size else np.array([], bool)
    if not pos_ok.any():
        return None
    planted = np.median(foot_pos[planted_idx[pos_ok]], axis=0)
    dist = np.linalg.norm(foot_pos - planted, axis=1)
    lo = t[br] - 0.6 if pkh is None else max(t[br] - 0.6, t[pkh] + 0.05)
    stride = _window(t, lo, t[br] - 0.06)
    p = _argmax_in(foot_speed, stride)
    if p is None or not np.isfinite(foot_speed[p]) or foot_speed[p] <= 0:
        return None
    thr = calib["fp_speed_fraction"] * foot_speed[p]
    radius = calib["fp_plant_radius"]
    for i in range(p + 1, br):
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


def detect_mer(t: np.ndarray, shoulder_er: np.ndarray, fp: int, br: int) -> int | None:
    return _argmax_in(shoulder_er, np.arange(fp, br + 1))


def detect_br_hand(t: np.ndarray, hand_ahead: np.ndarray, fp: int, calib: dict) -> int | None:
    """Release as the throwing hand crosses `br_hand_ahead` leg lengths in front of the shoulder, between
    0.05 s before and 0.45 s after foot plant. Of several upward crossings (arm swing, follow-through) the
    release one is the fastest forward movement of the hand, so the steepest crossing wins. Interpolated
    between the bracketing samples (bridged blurred frames are linear in between)."""
    win = _window(t, t[fp] - 0.05, t[fp] + 0.45)
    if win.size < 2:
        return None
    c = calib["br_hand_ahead"]
    best: tuple[float, float] | None = None       # (slope, crossing time)
    for i in win[:-1]:
        a, b = hand_ahead[i], hand_ahead[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a < c <= b:
            slope = (b - a) / (t[i + 1] - t[i])
            t_cross = t[i] + (c - a) / (b - a) * (t[i + 1] - t[i])
            if best is None or slope > best[0]:
                best = (slope, t_cross)
    if best is None:
        return None
    return _nearest(t, best[1] + calib.get("br_hand_offset_s", 0.0))


def _arm_tracked(arm_valid: np.ndarray, t: np.ndarray, lo_s: float, hi_s: float) -> bool:
    idx = _window(t, lo_s, hi_s)
    return bool(idx.size) and bool(arm_valid[idx].all())


def detect_events(t: np.ndarray, kin: dict[str, np.ndarray], calib: dict | None = None) -> Events:
    c = {**DEFAULT_CALIBRATION, **(calib or {})}
    notes: list[str] = []
    source = {k: "auto" for k in EVENT_NAMES}

    br, peak = detect_br(t, kin["wrist_speed"], c)
    if t[peak] - t[0] < 0.25:
        notes.append("Peak arm speed is within 0.25 s of the clip start; the clip may begin too late.")
    if t[-1] - t[peak] < 0.05:
        notes.append("Peak arm speed is at the very end of the clip; release may be cut off.")

    def lower_body(br: int) -> tuple[int | None, int | None]:
        pkh = detect_pkh(t, kin["knee_height"], br)
        return pkh, detect_fp(t, kin["foot_speed"], kin["foot_pos"], br, pkh, c)

    pkh, fp = lower_body(br)
    if fp is not None and "hand_ahead" in kin:
        br_hand = detect_br_hand(t, kin["hand_ahead"], fp, c)
        if br_hand is not None and br_hand > fp:
            br = br_hand
            pkh2, fp2 = lower_body(br)          # windows follow the final release time
            if fp2 is not None and fp2 < br:
                pkh, fp = pkh2, fp2

    if pkh is None:
        source["pkh"] = "missing"
        notes.append("Peak knee height is not in the clip (the clip starts after the leg lift).")
    if fp is None:
        fp = _nearest(t, t[br] - c["fp_to_br_median_s"])
        source["fp"] = "estimated"
        notes.append("Lead-foot plant could not be detected; placed at the Driveline median "
                     f"{c['fp_to_br_median_s'] * 1000:.0f} ms before release.")
    if pkh is not None and pkh >= fp:
        pkh = None
        source["pkh"] = "missing"

    arm_valid = kin.get("arm_valid")
    tracked = arm_valid is None or _arm_tracked(arm_valid, t, t[br] - 0.07, t[br] + 0.02)
    mer = detect_mer(t, kin["shoulder_er"], fp, br) if tracked else None
    if mer is not None and mer == fp and br - fp > 2:
        # A "maximum" on the first sample is the window edge, not a layback peak.
        mer = None
    if mer is None:
        mer = _nearest(t, t[br] - c["mer_to_br_median_s"])
        source["mer"] = "estimated"
        if not tracked:
            notes.append("The throwing arm was blurred or untracked around release, so max layback is placed "
                         f"{c['mer_to_br_median_s'] * 1000:.0f} ms before release (Driveline median) and "
                         "layback angles are not reported. Filming at 120-240 fps fixes this.")
        else:
            notes.append("Max external rotation could not be found between foot plant and release.")
    return Events(fp=fp, mer=mer, br=br, pkh=pkh, source=source, notes=notes)


def validate_order(ev: Events, n: int) -> None:
    """Raise ValueError if manually supplied events are out of range or out of order."""
    for name, v in ev.as_dict().items():
        if v is not None and not (0 <= v < n):
            raise ValueError(f"{name} frame {v} is outside 0..{n - 1}")
    if not (ev.fp < ev.br):
        raise ValueError("foot plant must come before ball release")
    if not (ev.fp <= ev.mer <= ev.br):
        raise ValueError("max external rotation must lie between foot plant and ball release")
    if ev.pkh is not None and not (ev.pkh < ev.fp):
        raise ValueError("peak knee height must come before foot plant")

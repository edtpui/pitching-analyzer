"""Detection of the pitching events used for every discrete metric.

  PKH  peak lead-knee height
  FP   lead-foot plant (Driveline's fp_100: full body weight on the lead force plate; Driveline's
       "_fp" point-of-interest metrics are taken at this instant)
  MER  maximum shoulder external rotation (layback)
  BR   ball release

Order of detection:
  BR0  peak throwing-wrist speed relative to the pelvis in the image, untracked stretches of up to
       0.25 s bridged (a first, rough release time). If no foot plant is found from it, the hand-based
       release below is searched where foot plant usually is relative to BR0.
  PKH  highest lead-knee position 1.5-0.25 s before release.
  FP   first frame after PKH at which the lead ankle (as the camera sees it) is both slow - below a
       fraction of its peak stride speed - and within a small radius of where it sits planted at release.
       Speed alone fails for pitchers whose foot stops travelling toward home and then drops, and from
       behind the plate, where the stride runs along the camera axis; the planted-position test fixes both.
       Side-on video also has the lead heel and toe: the same test on them finds the touch-down, and
       foot plant (full weight) is Driveline's median contact-to-full-weight time later, if that comes
       before the ankle rule fires. MediaPipe's ankle point sits on the lower shin and keeps creeping
       forward while the shin rolls over the planted foot, which made foot plant read late on a real clip.
  BR   side-on: the moment the throwing hand passes a calibrated distance in front of the throwing
       shoulder toward home (on Driveline data that times release to ~4 ms), interpolated across frames
       where the arm is blurred. The speed peak cannot do this: at 60 fps the arm is untracked or mislabelled
       for several frames around release and the "peak" lands on a tracking glitch. When the arm is lost
       for too long to bridge (dim light: blurred from layback until well after release), release is
       extrapolated from the hand's last tracked position with Driveline's median arm path and marked
       "estimated". Other views: BR0 plus a calibrated offset. PKH and FP are then re-detected against the
       final release time.
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
    "fc_to_fp_median_s": 0.019,      # Driveline median from first foot contact (10% body weight) to FP
    "mer_to_br_median_s": 0.031,
    "br_hand_ahead": 0.41,           # BR: hand this many leg lengths in front of the shoulder (side-on)
    "br_hand_offset_s": 0.0,         # BR = hand-crossing time + offset
    # Driveline median time from the hand's last pass of each hand-ahead level to release (side-on), from
    # the end of the layback (the first level, below which pitchers hold the layback for very different
    # times) up to the release level
    "br_hand_path": {"levels": [-0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                     "dt_s": [0.047, 0.040, 0.035, 0.031, 0.027, 0.024, 0.021, 0.018, 0.015, 0.012, 0.009,
                              0.006, 0.003, 0.0]},
}
EVENT_NAMES = ("pkh", "fp", "mer", "br")
ER_MIN_ELBOW_FLEXION = 30.0     # degrees; shoulder ER is undefined with a straighter elbow


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


def detect_contact(t: np.ndarray, kin: dict[str, np.ndarray], br: int, pkh: int | None, calib: dict) -> int | None:
    """First touch-down of the lead foot in side-on video: the earlier of heel and toe passing the
    foot-plant test (slow, and at the spot where that point sits planted at release), so a toe-first
    landing counts too. None without foot landmarks or when neither point settles before release."""
    hits = [detect_fp(t, kin[f"{p}_speed"], kin[f"{p}_pos"], br, pkh, calib)
            for p in ("heel", "toe") if f"{p}_pos" in kin]
    hits = [h for h in hits if h is not None]
    return min(hits) if hits else None


def detect_mer(t: np.ndarray, shoulder_er: np.ndarray, fp: int, br: int,
               elbow_flexion: np.ndarray | None = None) -> int | None:
    """Sample of maximum shoulder external rotation between foot plant and release. ER is measured from the
    forearm's direction around the upper arm, which is undefined while the elbow is nearly straight - as it
    is at release - and there swings by up to 180 degrees on noise alone (on simulated video such a swing
    at release passed for a 350-degree layback). Samples with the elbow bent less than
    ER_MIN_ELBOW_FLEXION are skipped; at real layback it is bent about 90 degrees."""
    idx = np.arange(fp, br + 1)
    if elbow_flexion is not None:
        idx = idx[~(elbow_flexion[idx] < ER_MIN_ELBOW_FLEXION)]
    return _argmax_in(shoulder_er, idx)


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


def release_from_last_tracked(t: np.ndarray, hand_ahead: np.ndarray, fp: int, calib: dict) -> int | None:
    """Release when the throwing arm was untracked from before release until after it (a blur gap too
    long to bridge, so `hand_ahead` is NaN there and shows no crossing). For each untracked stretch that
    starts within 0.45 s after foot plant and has tracking after it, release is extrapolated from the last
    tracked sample before it with Driveline's median time from that hand position to release. The table
    describes the hand on its way forward from the end of the layback; a hand last seen still laid back
    (behind the table's first level, where pitchers hold the layback for very different times) or still
    moving back is taken to have turned forward when the blur began - fast motion is what blurs it - and
    is timed from the table's first level. The last tracked sample is looked for in the same window as
    the hand crossing, 0.05 s before to 0.45 s after foot plant.

    A stretch is rejected if the extrapolated release falls after its end, or if within 0.1 s after that
    release the hand is tracked still laid back (behind the table's first level) or further back than its
    last position: the arm was still cocking, as in a mirrored-body glitch while the arm swings back. Of
    the rest, the stretch after the deepest layback wins."""
    path = calib.get("br_hand_path") or DEFAULT_CALIBRATION["br_hand_path"]
    levels, dt = np.asarray(path["levels"], dtype=float), np.asarray(path["dt_s"], dtype=float)
    finite = np.isfinite(hand_ahead)
    n = len(t)
    best: tuple[float, float] | None = None       # (hand-ahead at the last tracked sample, release time)
    i = int(np.searchsorted(t, t[fp] - 0.05))
    while i < n - 1 and t[i] <= t[fp] + 0.45:
        if not (finite[i] and not finite[i + 1]):
            i += 1
            continue
        j = i + 1
        while j < n and not finite[j]:
            j += 1
        a = float(hand_ahead[i])
        earlier = np.flatnonzero(finite[:i] & (t[:i] >= t[i] - 0.03))
        k = earlier[np.argmin(np.abs(t[earlier] - (t[i] - 0.02)))] if earlier.size else None
        moving_back = k is not None and a < hand_ahead[k] - 0.02
        if j < n and a < calib["br_hand_ahead"]:
            t_br = t[i] + float(np.interp(levels[0] if moving_back else max(a, levels[0]), levels, dt))
            after = finite & (t > t_br) & (t <= t_br + 0.1)
            still_cocking = (hand_ahead[after] < max(a, levels[0])).any()
            if t_br < t[j] and not still_cocking and (best is None or a < best[0]):
                best = (a, t_br)
        i = j
    return None if best is None else _nearest(t, best[1])


def release_hand_speed(t: np.ndarray, wrist_speed: np.ndarray, br: int) -> float:
    """Peak throwing-hand speed (leg lengths/s, as the camera sees it) within 0.1 s of release: a real pitch
    is far faster than an arm swing before release or a slow-motion clip played at its playback rate."""
    sel = (t >= t[br] - 0.1) & (t <= t[br] + 0.1)
    vals = wrist_speed[sel]
    return float(np.nanmax(vals)) if sel.any() and np.isfinite(vals).any() else float("nan")


def _arm_tracked(arm_valid: np.ndarray, t: np.ndarray, lo_s: float, hi_s: float) -> bool:
    idx = _window(t, lo_s, hi_s)
    return bool(idx.size) and bool(arm_valid[idx].all())


def detect_events(t: np.ndarray, kin: dict[str, np.ndarray], calib: dict | None = None) -> Events:
    c = {**DEFAULT_CALIBRATION, **(calib or {})}
    notes: list[str] = []
    source = {k: "auto" for k in EVENT_NAMES}

    br, peak = detect_br(t, kin.get("wrist_speed_bridged", kin["wrist_speed"]), c)
    if t[peak] - t[0] < 0.25:
        notes.append("Peak arm speed is within 0.25 s of the clip start; the clip may begin too late.")
    if t[-1] - t[peak] < 0.05:
        notes.append("Peak arm speed is at the very end of the clip; release may be cut off.")

    def lower_body(br: int) -> tuple[int | None, int | None]:
        pkh = detect_pkh(t, kin["knee_height"], br)
        fp = detect_fp(t, kin["foot_speed"], kin["foot_pos"], br, pkh, c)
        contact = detect_contact(t, kin, br, pkh, c)
        if contact is not None:
            fp_contact = _nearest(t, t[contact] + c["fc_to_fp_median_s"])
            if fp_contact < br and (fp is None or fp_contact < fp):
                fp = fp_contact
        return pkh, fp

    pkh, fp = lower_body(br)
    if "hand_ahead" in kin:
        # Without a foot plant the rough release is far off (with the arm lost through release, the fastest
        # movement left in the clip can be anywhere): search where foot plant usually is relative to it.
        anchor = fp if fp is not None else _nearest(t, t[br] - c["fp_to_br_median_s"])
        br_hand = detect_br_hand(t, kin["hand_ahead"], anchor, c)
        extrapolated = br_hand is None
        if extrapolated:
            br_hand = release_from_last_tracked(t, kin["hand_ahead"], anchor, c)
        if br_hand is not None and br_hand > anchor:
            br = br_hand
            pkh2, fp2 = lower_body(br)          # windows follow the final release time
            if fp2 is not None and fp2 < br:
                pkh, fp = pkh2, fp2
            if extrapolated:
                source["br"] = "estimated"
                notes.append("The throwing arm was blurred out of tracking from layback until after release, "
                             "so release is placed from the hand's last tracked position using Driveline's "
                             "typical arm path (on simulated Driveline pitches usually within 2 frames at 60 fps). "
                             "Check it on the video and correct it if needed.")

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
    mer = detect_mer(t, kin["shoulder_er"], fp, br, kin.get("elbow_flexion")) if tracked else None
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

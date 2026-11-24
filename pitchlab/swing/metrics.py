"""Discrete swing metrics computed at the detected events. One definition serves both the video pipeline
and the Driveline reference, so a hitter's numbers and the reference distributions are directly comparable.

A Driveline point-of-interest (POI) column is attached only where Driveline measures the same quantity at
the same instant. Their "hs" angles are taken at the maximum hip coil (our load event; pelvis rotation
agrees with r = 0.99) and their "swing_max"/"stride_max" values are maxima over the same phases, so those
are validated. Their foot-plant ("fp" / "launchpos") values are taken at an instant that matches neither the
fp_100 force-plate event published with the data (pelvis rotation r = 0.35) nor our foot-flat definition
(r = 0.69), so foot-plant metrics carry no validity check and are rated on reliability alone."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .events import Events
from .kinematics import Motion
from .skeleton import J

DEFAULT_HEIGHT_PER_LEG_LENGTH = 1.98      # height / side-on image leg length; calibrated in the reference bundle
DEFAULT_HEIGHT_PER_LEG_LENGTH_3D = 2.05   # height / 3D thigh + shank length; calibrated in the reference bundle
MPS_TO_MPH = 2.2369363


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    unit: str
    group: str
    description: str
    driveline: str | None = None       # Driveline POI column measuring the same thing at the same instant
    driveline_scale: float = 1.0       # factor applied to the POI column before comparison
    needs_heading: bool = False        # needs a camera squared to the pitcher-plate line
    needs_translation: bool = False    # needs absolute movement in the image (static side-on camera)
    needs_height: bool = False         # needs the hitter's height to convert to real units
    needs_load: bool = False           # taken at, or measured from, the load
    decimals: int = 0


GROUPS = ("Lower half", "Rotation", "Posture & arms", "Speed & sequencing")

METRICS: tuple[MetricDef, ...] = (
    MetricDef("stride_length", "Stride length", "% height", "Lower half",
              "Distance between the ankles toward the pitcher at foot plant, as a percentage of estimated "
              "standing height.", needs_translation=True),
    MetricDef("stride_angle", "Stride direction", "deg", "Lower half",
              "Direction of the stride relative to the line to the pitcher; + = closed (toward the plate), "
              "- = open (stepping 'in the bucket').", needs_heading=True),
    MetricDef("max_lead_knee_flexion", "Peak lead knee bend in the stride", "deg", "Lower half",
              "Most the front knee bends during the leg lift and stride, before foot plant.",
              "lead_knee_stride_max_x"),
    MetricDef("lead_knee_flexion_fp", "Lead knee flexion at foot plant", "deg", "Lower half",
              "Front knee bend when the front foot is planted (0 = straight)."),
    MetricDef("lead_knee_flexion_contact", "Lead knee flexion at contact", "deg", "Lower half",
              "Front knee bend at contact (0 = straight)."),
    MetricDef("lead_knee_extension", "Lead knee extension, foot plant to contact", "deg", "Lower half",
              "How much the front knee straightens between foot plant and contact (bracing the front side); "
              "negative = it keeps bending."),
    MetricDef("max_pelvis_fwd_velo", "Forward move: peak pelvis speed", "leg lengths/s", "Lower half",
              "Fastest the pelvis travels toward the pitcher during the stride.",
              "max_cog_velo_x", needs_translation=True, decimals=2),
    MetricDef("pelvis_rotation_load", "Hip coil at load", "deg", "Rotation",
              "Pelvis rotation at maximum coil (0 = square to the plate, - = turned toward the catcher).",
              "pelvis_angle_hs_z", needs_heading=True, needs_load=True),
    MetricDef("torso_rotation_load", "Shoulder coil at load", "deg", "Rotation",
              "Shoulder-line rotation at maximum hip coil (0 = square to the plate, - = turned toward the catcher).",
              "torso_angle_hs_z", needs_heading=True, needs_load=True),
    MetricDef("pelvis_rotation_fp", "Pelvis rotation at foot plant", "deg", "Rotation",
              "How far the hips have opened toward the pitcher when the front foot is planted.",
              needs_heading=True),
    MetricDef("torso_rotation_fp", "Shoulder rotation at foot plant", "deg", "Rotation",
              "How far the shoulders have opened toward the pitcher at foot plant (high = opening early).",
              needs_heading=True),
    MetricDef("hip_shoulder_sep_fp", "Hip-shoulder separation at foot plant", "deg", "Rotation",
              "Pelvis rotation minus shoulder rotation at foot plant (+ = hips ahead of the shoulders)."),
    MetricDef("max_hip_shoulder_sep", "Peak hip-shoulder separation", "deg", "Rotation",
              "Largest pelvis-minus-shoulder rotation from the load to contact.",
              "torso_pelvis_swing_max_x", -1.0),
    MetricDef("pelvis_rotation_contact", "Pelvis rotation at contact", "deg", "Rotation",
              "How open the hips are at contact (90 = facing the pitcher).", needs_heading=True),
    MetricDef("trunk_flexion_load", "Forward tilt at load", "deg", "Posture & arms",
              "Forward bend of the trunk toward the plate at maximum hip coil.", "torso_angle_hs_x",
              needs_load=True),
    MetricDef("max_trunk_lateral", "Peak side bend", "deg", "Posture & arms",
              "Largest side bend of the trunk toward the rear leg (back shoulder below the front shoulder) "
              "from the load to contact.", "torso_swing_max_y"),
    MetricDef("rear_elbow_flexion_fp", "Rear elbow flexion at foot plant", "deg", "Posture & arms",
              "Back-arm elbow bend at foot plant."),
    MetricDef("rear_shoulder_abduction_fp", "Rear elbow height at foot plant", "deg", "Posture & arms",
              "Back upper arm angle from the trunk at foot plant: 90 = elbow at shoulder height, lower = tucked."),
    MetricDef("lead_elbow_flexion_contact", "Lead elbow flexion at contact", "deg", "Posture & arms",
              "Front-arm elbow bend at contact (0 = fully extended)."),
    MetricDef("max_pelvis_rot_velo", "Peak pelvis rotation speed", "deg/s", "Speed & sequencing",
              "Fastest hip rotation toward the pitcher.", "pelvis_angular_velocity_swing_max_x"),
    MetricDef("max_torso_rot_velo", "Peak shoulder rotation speed", "deg/s", "Speed & sequencing",
              "Fastest rotation of the shoulder line toward the pitcher (the shoulder joints move with the "
              "shoulder blades, so this runs faster than the thorax itself).",
              "torso_angular_velocity_swing_max_x"),
    MetricDef("max_lead_arm_speed", "Peak lead arm speed", "deg/s", "Speed & sequencing",
              "Fastest angular speed of the front upper arm.", "upper_arm_speed_mag_swing_max_velo_x"),
    MetricDef("max_hand_speed", "Peak hand speed", "mph", "Speed & sequencing",
              "Fastest speed of the hands relative to the pelvis (needs the hitter's height for the scale).",
              "hand_speed_blast_bat_mph_max_x", needs_height=True, decimals=1),
    MetricDef("pelvis_torso_timing", "Pelvis-to-shoulder peak timing", "ms", "Speed & sequencing",
              "Time from peak pelvis rotation speed to peak shoulder rotation speed (+ = hips first)."),
    MetricDef("fp_to_contact_time", "Foot plant to contact", "ms", "Speed & sequencing",
              "Time from the front foot being down flat to contact."),
)
METRIC_KEYS = tuple(m.key for m in METRICS)
METRIC_BY_KEY = {m.key: m for m in METRICS}


# Hand/arm metrics and the stretch of time the hands must really be tracked for them to be reported:
# (from event, seconds before it, to event, seconds after it).
HANDS_WINDOWS = {
    "rear_elbow_flexion_fp": ("fp", 0.02, "fp", 0.02),
    "rear_shoulder_abduction_fp": ("fp", 0.02, "fp", 0.02),
    "lead_elbow_flexion_contact": ("contact", 0.02, "contact", 0.02),
    "max_lead_arm_speed": ("contact", 0.12, "contact", 0.0),
    "max_hand_speed": ("contact", 0.12, "contact", 0.0),
}
PEAK_WINDOW_S = (0.3, 0.02)        # rotation / arm / hand speed peaks: 300 ms before to 20 ms after contact


def hands_untracked(m: Motion, ev: Events, key: str) -> bool:
    """True when `key` depends on hand/arm samples that were bridged (blurred / untracked)."""
    if key not in HANDS_WINDOWS or m.hands_valid is None:
        return False
    e0, before, e1, after = HANDS_WINDOWS[key]
    i0, i1 = getattr(ev, e0), getattr(ev, e1)
    sel = (m.t >= m.t[i0] - before) & (m.t <= m.t[i1] + after)
    return bool((~m.hands_valid[sel]).any())


def metric_definitions() -> list[dict]:
    return [asdict(m) for m in METRICS]


def _window(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.flatnonzero((t >= lo) & (t <= hi))


def _peak(x: np.ndarray, idx: np.ndarray) -> tuple[float, int | None]:
    if idx.size == 0 or not np.isfinite(x[idx]).any():
        return float("nan"), None
    i = int(idx[np.nanargmax(x[idx])])
    return float(x[i]), i


def compute_metrics(m: Motion, kin: dict[str, np.ndarray], ev: Events,
                    height_per_leg_length: float = DEFAULT_HEIGHT_PER_LEG_LENGTH,
                    height_m: float | None = None,
                    height_per_leg_length_3d: float = DEFAULT_HEIGHT_PER_LEG_LENGTH_3D) -> dict[str, float | None]:
    t = m.t
    fp, ct, load = ev.fp, ev.contact, ev.load
    nan = float("nan")
    at_load = (lambda x: float(x[load])) if load is not None else (lambda x: nan)

    out: dict[str, float] = {}
    if m.track_forward is not None:
        step = m.track["lead_ankle"][fp] - m.track["rear_ankle"][fp]
        out["stride_length"] = float((step @ m.track_forward) / m.track_scale / height_per_leg_length * 100.0)
    else:
        out["stride_length"] = nan
    d = m.pos[fp, J["lead_ankle"]] - m.pos[fp, J["rear_ankle"]]
    out["stride_angle"] = float(np.degrees(np.arctan2(-d[1], d[0])))
    out["max_lead_knee_flexion"], _ = _peak(kin["lead_knee_flexion"], _window(t, t[fp] - 0.8, t[fp]))
    out["lead_knee_flexion_fp"] = float(kin["lead_knee_flexion"][fp])
    out["lead_knee_flexion_contact"] = float(kin["lead_knee_flexion"][ct])
    out["lead_knee_extension"] = out["lead_knee_flexion_fp"] - out["lead_knee_flexion_contact"]
    if "pelvis_fwd_velo" in kin:
        out["max_pelvis_fwd_velo"], _ = _peak(kin["pelvis_fwd_velo"], _window(t, t[fp] - 0.8, t[fp] + 0.05))
    else:
        out["max_pelvis_fwd_velo"] = nan

    out["pelvis_rotation_load"] = at_load(kin["pelvis_rot"])
    out["torso_rotation_load"] = at_load(kin["torso_rot"])
    out["pelvis_rotation_fp"] = float(kin["pelvis_rot"][fp])
    out["torso_rotation_fp"] = float(kin["torso_rot"][fp])
    out["hip_shoulder_sep_fp"] = float(kin["hip_shoulder_sep"][fp])
    swing_start = t[load] if load is not None else t[fp] - 0.15
    swing = _window(t, swing_start, t[ct])
    out["max_hip_shoulder_sep"], _ = _peak(kin["hip_shoulder_sep"], swing)
    out["pelvis_rotation_contact"] = float(kin["pelvis_rot"][ct])

    out["trunk_flexion_load"] = at_load(kin["trunk_flexion"])
    out["max_trunk_lateral"], _ = _peak(kin["trunk_lateral"], swing)
    out["rear_elbow_flexion_fp"] = float(kin["rear_elbow_flexion"][fp])
    out["rear_shoulder_abduction_fp"] = float(kin["rear_shoulder_abduction"][fp])
    out["lead_elbow_flexion_contact"] = float(kin["lead_elbow_flexion"][ct])

    peak_win = _window(t, t[ct] - PEAK_WINDOW_S[0], t[ct] + PEAK_WINDOW_S[1])
    out["max_pelvis_rot_velo"], ip = _peak(kin["pelvis_rot_velo"], peak_win)
    out["max_torso_rot_velo"], it = _peak(kin["torso_rot_velo"], peak_win)
    out["max_lead_arm_speed"], _ = _peak(kin["lead_arm_speed"], peak_win)
    hand_ll, _ = _peak(kin["hand_speed"], peak_win)
    if height_m:
        out["max_hand_speed"] = hand_ll * height_m / height_per_leg_length_3d * MPS_TO_MPH
    else:
        out["max_hand_speed"] = nan
    out["pelvis_torso_timing"] = float((t[it] - t[ip]) * 1000.0) if ip is not None and it is not None else nan
    out["fp_to_contact_time"] = float((t[ct] - t[fp]) * 1000.0)

    if not m.heading_known:
        for k in METRIC_KEYS:
            if METRIC_BY_KEY[k].needs_heading:
                out[k] = nan
    for k in HANDS_WINDOWS:
        if hands_untracked(m, ev, k):
            out[k] = nan
    return {k: (None if not np.isfinite(out[k]) else round(out[k], 4)) for k in METRIC_KEYS}

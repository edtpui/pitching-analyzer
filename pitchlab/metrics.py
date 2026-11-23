"""Discrete pitching metrics computed at the detected events. One definition serves both the video
pipeline and the Driveline reference, so a user's numbers and the reference distributions are
directly comparable."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .events import Events
from .kinematics import Motion
from .skeleton import J

DEFAULT_HEIGHT_PER_LEG_LENGTH = 1.98   # replaced by the Driveline-calibrated value in the reference bundle


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    unit: str
    group: str
    description: str
    driveline: str | None = None       # Driveline POI column measuring the same thing (validation only)
    driveline_scale: float = 1.0       # factor applied to the POI column before comparison
    needs_heading: bool = False        # needs a camera squared to the mound-home line
    needs_translation: bool = False    # needs absolute pelvis travel (static side-on camera)
    decimals: int = 0


METRICS: tuple[MetricDef, ...] = (
    MetricDef("stride_length", "Stride length", "% height", "Lower half",
              "Distance toward home from the rear ankle at peak knee height to the lead ankle at foot "
              "plant, as a percentage of estimated standing height.",
              "stride_length", 100.0, needs_translation=True),
    MetricDef("stride_angle", "Stride direction", "deg", "Lower half",
              "Direction of the stride relative to the line to home; + = closed / cross-body.",
              "stride_angle", needs_heading=True),
    MetricDef("lead_knee_flexion_fp", "Lead knee flexion at foot plant", "deg", "Lower half",
              "Lead knee bend when the foot lands (0 = straight)."),
    MetricDef("lead_knee_flexion_br", "Lead knee flexion at release", "deg", "Lower half",
              "Lead knee bend at ball release (0 = straight)."),
    MetricDef("lead_knee_extension", "Lead knee extension, contact to release", "deg", "Lower half",
              "How much the lead knee straightens between foot plant and release; negative = it collapses.",
              "lead_knee_extension_from_fp_to_br"),
    MetricDef("max_pelvis_fwd_velo", "Peak pelvis speed toward home", "leg lengths/s", "Lower half",
              "Peak forward speed of the pelvis during the stride (momentum down the mound).",
              "max_cog_velo_x", needs_translation=True, decimals=2),
    MetricDef("pelvis_rotation_fp", "Pelvis rotation at foot plant", "deg", "Trunk",
              "How far the hips have opened toward home at foot plant (0 = side-on, 90 = square).",
              "pelvis_rotation_fp", needs_heading=True),
    MetricDef("torso_rotation_fp", "Torso rotation at foot plant", "deg", "Trunk",
              "How far the shoulders have opened toward home at foot plant (0 = side-on, 90 = square).",
              "torso_rotation_fp", needs_heading=True),
    MetricDef("hip_shoulder_sep_fp", "Hip-shoulder separation at foot plant", "deg", "Trunk",
              "Pelvis rotation minus torso rotation at foot plant.",
              "rotation_hip_shoulder_separation_fp"),
    MetricDef("max_hip_shoulder_sep", "Peak hip-shoulder separation", "deg", "Trunk",
              "Largest pelvis-minus-torso rotation from just before foot plant to release.",
              "max_rotation_hip_shoulder_separation"),
    MetricDef("trunk_flexion_fp", "Trunk forward tilt at foot plant", "deg", "Trunk",
              "Forward lean of the trunk at foot plant, in the trunk's own facing direction.",
              "torso_anterior_tilt_fp"),
    MetricDef("trunk_flexion_br", "Trunk forward tilt at release", "deg", "Trunk",
              "Forward lean of the trunk at ball release.", "torso_anterior_tilt_br"),
    MetricDef("trunk_lateral_br", "Trunk side tilt at release", "deg", "Trunk",
              "Lean of the trunk toward the glove side at release (+ = glove side).",
              "torso_lateral_tilt_br"),
    MetricDef("shoulder_abduction_fp", "Shoulder abduction at foot plant", "deg", "Arm",
              "Elevation of the throwing upper arm relative to the trunk at foot plant (90 = elbow at shoulder height).",
              "shoulder_abduction_fp"),
    MetricDef("shoulder_horiz_abduction_fp", "Shoulder horizontal abduction at foot plant", "deg", "Arm",
              "How far the throwing elbow is behind the shoulder line at foot plant.",
              "shoulder_horizontal_abduction_fp"),
    MetricDef("shoulder_er_fp", "Shoulder external rotation at foot plant", "deg", "Arm",
              "Forearm rotation at foot plant: 0 = forearm pointing forward, 90 = pointing up ('arm up on time').",
              "shoulder_external_rotation_fp"),
    MetricDef("elbow_flexion_fp", "Elbow flexion at foot plant", "deg", "Arm",
              "Throwing elbow bend at foot plant.", "elbow_flexion_fp"),
    MetricDef("max_shoulder_er", "Max shoulder external rotation (layback)", "deg", "Arm",
              "Peak forearm layback between foot plant and release.", "max_shoulder_external_rotation"),
    MetricDef("elbow_flexion_mer", "Elbow flexion at max layback", "deg", "Arm",
              "Throwing elbow bend at maximum external rotation.", "elbow_flexion_mer"),
    MetricDef("arm_slot", "Arm slot", "deg", "Arm",
              "Forearm angle at release seen from home plate: 0 = sidearm, 90 = straight over the top.",
              "arm_slot"),
    MetricDef("max_pelvis_rot_velo", "Peak pelvis rotation speed", "deg/s", "Sequencing",
              "Fastest hip rotation toward home.", "max_pelvis_rotational_velo"),
    MetricDef("max_torso_rot_velo", "Peak torso rotation speed", "deg/s", "Sequencing",
              "Fastest shoulder/trunk rotation toward home.", "max_torso_rotational_velo"),
    MetricDef("max_elbow_ext_velo", "Peak elbow extension speed", "deg/s", "Sequencing",
              "Fastest elbow straightening near release.", "max_elbow_extension_velo"),
    MetricDef("max_shoulder_ir_velo", "Peak shoulder internal rotation speed", "deg/s", "Sequencing",
              "Fastest forearm rotation forward after layback (very hard to see on video).",
              "max_shoulder_internal_rotational_velo"),
    MetricDef("pelvis_torso_timing", "Pelvis-to-torso peak timing", "ms", "Sequencing",
              "Time from peak pelvis rotation speed to peak torso rotation speed (+ = pelvis first).",
              "timing_peak_torso_to_peak_pelvis_rot_velo", 1000.0),
    MetricDef("fp_to_br_time", "Foot plant to release", "ms", "Sequencing",
              "Time from lead-foot plant to ball release."),
)
METRIC_KEYS = tuple(m.key for m in METRICS)
METRIC_BY_KEY = {m.key: m for m in METRICS}


# Throwing-arm metrics and the stretch of time the arm must really be tracked for them to be reported:
# (from event, seconds before it, to event, seconds after it).
ARM_WINDOWS = {
    "shoulder_abduction_fp": ("fp", 0.02, "fp", 0.02),
    "shoulder_horiz_abduction_fp": ("fp", 0.02, "fp", 0.02),
    "shoulder_er_fp": ("fp", 0.02, "fp", 0.02),
    "elbow_flexion_fp": ("fp", 0.02, "fp", 0.02),
    "max_shoulder_er": ("mer", 0.02, "mer", 0.02),
    "elbow_flexion_mer": ("mer", 0.02, "mer", 0.02),
    "arm_slot": ("br", 0.02, "br", 0.02),
    "max_elbow_ext_velo": ("mer", 0.03, "br", 0.04),
    "max_shoulder_ir_velo": ("mer", 0.01, "br", 0.06),
}


def arm_untracked(m: Motion, ev: Events, key: str) -> bool:
    """True when `key` depends on throwing-arm samples that were bridged (blurred / untracked)."""
    if key not in ARM_WINDOWS or m.arm_valid is None:
        return False
    e0, before, e1, after = ARM_WINDOWS[key]
    i0, i1 = getattr(ev, e0), getattr(ev, e1)
    sel = (m.t >= m.t[i0] - before) & (m.t <= m.t[i1] + after)
    return bool((~m.arm_valid[sel]).any())


def metric_definitions() -> list[dict]:
    return [asdict(m) for m in METRICS]


def _window(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.flatnonzero((t >= lo) & (t <= hi))


def _peak(x: np.ndarray, idx: np.ndarray) -> tuple[float, int | None]:
    if idx.size == 0 or not np.isfinite(x[idx]).any():
        return float("nan"), None
    i = int(idx[np.nanargmax(x[idx])])
    return float(x[i]), i


def normalise_er(er: np.ndarray, fp: int) -> np.ndarray:
    """Shift the unwrapped ER series by a multiple of 360 so its foot-contact value is in (-90, 270]."""
    v = er[fp]
    if not np.isfinite(v):
        return er
    return er + 360.0 * np.floor((270.0 - v) / 360.0)


def compute_metrics(m: Motion, kin: dict[str, np.ndarray], ev: Events,
                    height_per_leg_length: float = DEFAULT_HEIGHT_PER_LEG_LENGTH) -> dict[str, float | None]:
    t = m.t
    fp, mer, br = ev.fp, ev.mer, ev.br
    er = normalise_er(kin["shoulder_er"], fp)
    d = m.pos[fp, J["l_ankle"]] - m.pos[fp, J["r_ankle"]]

    out: dict[str, float] = {}
    if m.track_forward is not None and ev.pkh is not None:
        step = m.track["l_ankle"][fp] - m.track["r_ankle"][ev.pkh]
        out["stride_length"] = float((step @ m.track_forward) / m.track_scale / height_per_leg_length * 100.0)
    else:
        out["stride_length"] = float("nan")
    out["stride_angle"] = float(np.degrees(np.arctan2(-d[1], d[0])))
    out["lead_knee_flexion_fp"] = float(kin["lead_knee_flexion"][fp])
    out["lead_knee_flexion_br"] = float(kin["lead_knee_flexion"][br])
    out["lead_knee_extension"] = out["lead_knee_flexion_fp"] - out["lead_knee_flexion_br"]

    if "pelvis_fwd_velo" in kin:
        start = t[ev.pkh] if ev.pkh is not None else t[fp] - 0.7
        out["max_pelvis_fwd_velo"], _ = _peak(kin["pelvis_fwd_velo"], _window(t, start, t[fp] + 0.05))
    else:
        out["max_pelvis_fwd_velo"] = float("nan")

    out["pelvis_rotation_fp"] = float(kin["pelvis_rot"][fp])
    out["torso_rotation_fp"] = float(kin["torso_rot"][fp])
    out["hip_shoulder_sep_fp"] = float(kin["hip_shoulder_sep"][fp])
    out["max_hip_shoulder_sep"], _ = _peak(kin["hip_shoulder_sep"], _window(t, t[fp] - 0.1, t[br]))
    out["trunk_flexion_fp"] = float(kin["trunk_flexion"][fp])
    out["trunk_flexion_br"] = float(kin["trunk_flexion"][br])
    out["trunk_lateral_br"] = float(kin["trunk_lateral"][br])

    out["shoulder_abduction_fp"] = float(kin["shoulder_abduction"][fp])
    out["shoulder_horiz_abduction_fp"] = float(kin["shoulder_horiz_abduction"][fp])
    out["shoulder_er_fp"] = float(er[fp])
    out["elbow_flexion_fp"] = float(kin["elbow_flexion"][fp])
    out["max_shoulder_er"] = float(er[mer])
    out["elbow_flexion_mer"] = float(kin["elbow_flexion"][mer])
    out["arm_slot"] = float(kin["arm_slot"][br])

    rot_win = _window(t, t[fp] - 0.15, t[br] + 0.05)
    out["max_pelvis_rot_velo"], ip = _peak(kin["pelvis_rot_velo"], rot_win)
    out["max_torso_rot_velo"], it = _peak(kin["torso_rot_velo"], rot_win)
    out["max_elbow_ext_velo"], _ = _peak(kin["elbow_ext_velo"], _window(t, t[mer] - 0.03, t[br] + 0.04))
    ir = -np.gradient(er, t)
    out["max_shoulder_ir_velo"], _ = _peak(ir, _window(t, t[mer] - 0.01, t[br] + 0.06))
    out["pelvis_torso_timing"] = float((t[it] - t[ip]) * 1000.0) if ip is not None and it is not None else float("nan")
    out["fp_to_br_time"] = float((t[br] - t[fp]) * 1000.0)

    if not m.heading_known:
        for k in METRIC_KEYS:
            if METRIC_BY_KEY[k].needs_heading:
                out[k] = float("nan")
    for k in ARM_WINDOWS:
        if arm_untracked(m, ev, k):
            out[k] = float("nan")
    return {k: (None if not np.isfinite(out[k]) else round(out[k], 4)) for k in METRIC_KEYS}

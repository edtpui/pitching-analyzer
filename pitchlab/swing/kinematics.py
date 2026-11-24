"""Joint-angle and segment-rotation time series from the canonical 12-joint hitting skeleton.

Definitions are geometric (no inverse kinematics) so they can be computed identically from noisy video
keypoints and from the Driveline joint centres. Rotation follows the K-Vest / Driveline hitting
convention: pelvis and torso rotation 0 deg = square to the plate (chest facing the plate, the stance
orientation), + = turned open toward the pitcher, - = coiled toward the catcher. Hip-shoulder separation
is pelvis rotation minus torso rotation (+ = hips ahead of the shoulders). Knee/elbow flexion 0 = straight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signals import derivative, unwrap_deg
from .skeleton import J, leg_length

Z = np.array([0.0, 0.0, 1.0])

TRACK_KEYS = ("lead_ankle", "rear_ankle", "lead_knee", "mid_hip", "hands")


@dataclass
class Motion:
    """One swing in the canonical, right-hander-normalised lab frame (see skeleton.py).

    `pos` may be pelvis-centred per frame (MediaPipe world landmarks are), so it only supports
    within-frame geometry. Movement across frames (stride, foot plant, forward move) uses `track`:
    absolute positions in a fixed frame - the lab for Driveline, image pixels from a static camera for video."""

    t: np.ndarray                         # (T,) seconds, strictly increasing
    pos: np.ndarray                       # (T, 12, 3) joint centres
    track: dict[str, np.ndarray]          # TRACK_KEYS -> (T, D) absolute positions
    track_up: np.ndarray                  # (D,) unit "up" in track coordinates
    track_forward: np.ndarray | None      # (D,) unit vector toward the pitcher in track coordinates, None if unseen
    track_scale: float                    # leg length in track units
    heading_known: bool                   # +x of `pos` is the true direction to the pitcher
    fps: float
    hands_valid: np.ndarray | None = None  # (T,) both wrists/elbows really tracked (False = bridged)

    def __post_init__(self) -> None:
        if self.pos.ndim != 3 or self.pos.shape[1:] != (len(J), 3):
            raise ValueError(f"pos must be (T, {len(J)}, 3), got {self.pos.shape}")
        if len(self.t) != len(self.pos):
            raise ValueError("t and pos length differ")
        if len(self.t) > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError("t must be strictly increasing")
        missing = set(TRACK_KEYS) - set(self.track)
        if missing:
            raise ValueError(f"track is missing {sorted(missing)}")
        if not (np.isfinite(self.track_scale) and self.track_scale > 0):
            raise ValueError("track_scale must be a positive leg length")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = _dot(_unit(a), _unit(b))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _horizontal(v: np.ndarray) -> np.ndarray:
    out = np.array(v, dtype=float, copy=True)
    out[..., 2] = 0.0
    return out


def _heading(right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Anterior horizontal unit vector of a segment from its right-pointing (rear-side) axis, and its
    rotation: 0 = facing the plate (-y), 90 = facing the pitcher (+x). Unwrapped and shifted by whole
    turns so the clip median lies in (-135, 225]."""
    anterior = _unit(np.cross(Z, _horizontal(right)))
    rot = unwrap_deg(np.degrees(np.arctan2(anterior[:, 1], anterior[:, 0])) + 90.0)
    med = np.nanmedian(rot) if np.isfinite(rot).any() else 0.0
    return anterior, rot + 360.0 * np.floor((225.0 - med) / 360.0)


def _wrap180(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def joint(pos: np.ndarray, name: str) -> np.ndarray:
    return pos[:, J[name]]


def angular_speed(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Angular speed (deg/s) of the direction of a (T, 3) segment vector."""
    return np.degrees(np.linalg.norm(derivative(_unit(v), t), axis=-1))


def kinematics(m: Motion) -> dict[str, np.ndarray]:
    """All time series used by event detection and metrics. Rotational velocities in deg/s; linear
    speeds in leg lengths per second."""
    p = m.pos
    t = m.t
    rs, re, rw = joint(p, "rear_shoulder"), joint(p, "rear_elbow"), joint(p, "rear_wrist")
    ls, le, lw = joint(p, "lead_shoulder"), joint(p, "lead_elbow"), joint(p, "lead_wrist")
    rh, lh = joint(p, "rear_hip"), joint(p, "lead_hip")
    lk, la = joint(p, "lead_knee"), joint(p, "lead_ankle")
    mid_hip = (rh + lh) / 2.0
    mid_sh = (rs + ls) / 2.0
    trunk = mid_sh - mid_hip
    hands = (rw + lw) / 2.0
    L = leg_length(p)

    # Segment headings (the rear side is the hitter's right side after normalisation).
    pelvis_ant, pelvis_rot = _heading(rh - lh)
    torso_ant, torso_rot = _heading(rs - ls)
    torso_lead = np.cross(Z, torso_ant)

    # Trunk tilt resolved in the torso's own heading: forward bend, and side bend toward the rear leg.
    tz = trunk[:, 2]
    trunk_flexion = np.degrees(np.arctan2(_dot(trunk, torso_ant), tz))
    trunk_lateral = np.degrees(np.arctan2(-_dot(trunk, torso_lead), tz))      # + = toward the rear leg

    lead_knee_flexion = 180.0 - _angle_between(lh - lk, la - lk)
    rear_elbow_flexion = 180.0 - _angle_between(rs - re, rw - re)
    lead_elbow_flexion = 180.0 - _angle_between(ls - le, lw - le)
    # rear upper arm against the spine: 90 = rear elbow at shoulder height ("elbow up"), lower = tucked
    rear_shoulder_abduction = _angle_between(re - rs, -trunk)

    s = m.track_scale
    # Speeds as the camera sees them, relative to the pelvis (rotation invariant, no depth noise): event
    # detection. 3D speeds relative to the pelvis: metrics.
    hand_speed_img = np.linalg.norm(derivative(m.track["hands"] - m.track["mid_hip"], t), axis=-1) / s
    hand_speed = np.linalg.norm(derivative(hands - mid_hip, t), axis=-1) / L
    foot_pos = m.track["lead_ankle"] / s
    foot_speed = np.linalg.norm(derivative(foot_pos, t), axis=-1)
    foot_height = foot_pos @ m.track_up
    knee_height = (m.track["lead_knee"] @ m.track_up) / s

    out = {
        "pelvis_rot": pelvis_rot,
        "torso_rot": torso_rot,
        "hip_shoulder_sep": _wrap180(pelvis_rot - torso_rot),
        "pelvis_rot_velo": derivative(pelvis_rot, t),
        "torso_rot_velo": derivative(torso_rot, t),
        "trunk_flexion": trunk_flexion,
        "trunk_lateral": trunk_lateral,
        "lead_knee_flexion": lead_knee_flexion,
        "rear_elbow_flexion": rear_elbow_flexion,
        "lead_elbow_flexion": lead_elbow_flexion,
        "rear_shoulder_abduction": rear_shoulder_abduction,
        "lead_arm_speed": angular_speed(le - ls, t),
        "hand_speed": hand_speed,
        "hand_speed_img": hand_speed_img,
        "foot_speed": foot_speed,
        "foot_pos": foot_pos,
        "foot_height": foot_height,
        "knee_height": knee_height,
    }
    if m.track_forward is not None:
        out["pelvis_fwd_velo"] = derivative(m.track["mid_hip"] @ m.track_forward, t) / s
    out["hands_valid"] = m.hands_valid if m.hands_valid is not None else np.ones(len(t), dtype=bool)
    return out

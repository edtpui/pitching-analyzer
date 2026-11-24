"""Canonical 12-joint hitting skeleton shared by the video pipeline and the Driveline reference data.

All analysis runs in a right-handed-hitter-normalised laboratory frame (the Driveline hitting convention):
  +x  from home plate toward the pitcher
  +y  toward the right-handed batter's box (behind a right-handed hitter's back in his stance)
  +z  up
A right-handed hitter in his stance faces -y with his lead (left) side toward +x. Left-handed hitters are
mirrored (y -> -y) so every formula can assume a right-hander. Joint names use the rear (catcher-side)
and lead (pitcher-side) sides for both arms and legs: for a right-handed hitter rear = right, lead = left.
"""
from __future__ import annotations

import numpy as np

JOINTS: tuple[str, ...] = (
    "rear_shoulder", "rear_elbow", "rear_wrist",
    "lead_shoulder", "lead_elbow", "lead_wrist",
    "rear_hip", "rear_knee", "rear_ankle",
    "lead_hip", "lead_knee", "lead_ankle",
)
J = {name: i for i, name in enumerate(JOINTS)}
ARM_JOINTS = ("rear_elbow", "rear_wrist", "lead_elbow", "lead_wrist")

# MediaPipe pose landmark indices (anatomical side of the subject).
MP = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30, "l_foot": 31, "r_foot": 32,
}
# Left/right landmark pairs that MediaPipe can swap in profile views, grouped by limb.
MP_LEG_PAIRS = [(23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
MP_ARM_PAIRS = [(13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
MP_ARM_INDICES = list(range(13, 23))

# Bones drawn on the overlay (MediaPipe indices).
MP_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Driveline hitting landmark column prefixes, by anatomical side.
_DRIVELINE_SIDE = {
    "shoulder": ("rsjc", "lsjc"), "elbow": ("rejc", "lejc"), "wrist": ("rwjc", "lwjc"),
    "hip": ("right_hip", "left_hip"), "knee": ("rkjc", "lkjc"), "ankle": ("rajc", "lajc"),
}


def driveline_columns(bats: str) -> dict[str, str]:
    """Canonical joint -> Driveline landmark column prefix. The rear side is the right side for a
    right-handed hitter and the left side for a left-handed one."""
    if bats not in ("R", "L"):
        raise ValueError(f"bats must be 'R' or 'L', got {bats!r}")
    rear_i, lead_i = (0, 1) if bats == "R" else (1, 0)
    out = {}
    for joint in JOINTS:
        side, part = joint.split("_", 1)
        out[joint] = _DRIVELINE_SIDE[part][rear_i if side == "rear" else lead_i]
    return out


def mediapipe_indices(bats: str) -> list[int]:
    """MediaPipe landmark index for each canonical joint (same order as JOINTS)."""
    if bats not in ("R", "L"):
        raise ValueError(f"bats must be 'R' or 'L', got {bats!r}")
    r, l = ("r", "l") if bats == "R" else ("l", "r")
    return [
        MP[f"{r}_shoulder"], MP[f"{r}_elbow"], MP[f"{r}_wrist"],
        MP[f"{l}_shoulder"], MP[f"{l}_elbow"], MP[f"{l}_wrist"],
        MP[f"{r}_hip"], MP[f"{r}_knee"], MP[f"{r}_ankle"],
        MP[f"{l}_hip"], MP[f"{l}_knee"], MP[f"{l}_ankle"],
    ]


def mirror_y(pos: np.ndarray) -> np.ndarray:
    """Mirror a (..., 3) array across the x-z plane (left-handed -> right-handed equivalent)."""
    out = np.array(pos, dtype=float, copy=True)
    out[..., 1] *= -1.0
    return out


def leg_length_projected(hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l) -> float:
    """Leg length from 2D (projected) joint tracks: the 90th percentile over frames of thigh + shank,
    i.e. the least foreshortened views. Used for image tracks and their simulated counterparts."""
    def seg(a, b):
        return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)
    vals = np.concatenate([seg(hip_r, knee_r) + seg(knee_r, ankle_r), seg(hip_l, knee_l) + seg(knee_l, ankle_l)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no finite leg segments")
    return float(np.percentile(vals, 90))


def leg_length(pos: np.ndarray) -> float:
    """Median thigh + shank length over the clip, averaged over both legs (same units as `pos`)."""
    def seg(a: str, b: str) -> np.ndarray:
        return np.linalg.norm(pos[:, J[a]] - pos[:, J[b]], axis=-1)

    rear = seg("rear_hip", "rear_knee") + seg("rear_knee", "rear_ankle")
    lead = seg("lead_hip", "lead_knee") + seg("lead_knee", "lead_ankle")
    vals = np.concatenate([rear, lead])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no finite leg segments")
    return float(np.median(vals))

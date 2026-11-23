"""Canonical 12-joint pitching skeleton shared by the video pipeline and the Driveline reference data.

All analysis runs in a right-handed-pitcher-normalised laboratory frame (the Driveline convention):
  +x  from the mound toward home plate
  +y  toward first base (the pitcher's left as he faces home)
  +z  up
Left-handed pitchers are mirrored (y -> -y) so every formula can assume a right-hander. Joint names use
the throwing ("t_") / glove ("g_") side for arms and the rear ("r_") / lead ("l_") side for legs.
"""
from __future__ import annotations

import numpy as np

JOINTS: tuple[str, ...] = (
    "t_shoulder", "t_elbow", "t_wrist",
    "g_shoulder", "g_elbow", "g_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
)
J = {name: i for i, name in enumerate(JOINTS)}
ARM_JOINTS = ("t_elbow", "t_wrist", "g_elbow", "g_wrist")

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

# Bones drawn on the overlay (MediaPipe indices).
MP_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

DRIVELINE_COLUMNS = {
    "t_shoulder": "shoulder_jc", "t_elbow": "elbow_jc", "t_wrist": "wrist_jc",
    "g_shoulder": "glove_shoulder_jc", "g_elbow": "glove_elbow_jc", "g_wrist": "glove_wrist_jc",
    "r_hip": "rear_hip", "r_knee": "rear_knee_jc", "r_ankle": "rear_ankle_jc",
    "l_hip": "lead_hip", "l_knee": "lead_knee_jc", "l_ankle": "lead_ankle_jc",
}


def mediapipe_indices(throws: str) -> list[int]:
    """MediaPipe landmark index for each canonical joint. The rear leg is on the throwing side."""
    if throws not in ("R", "L"):
        raise ValueError(f"throws must be 'R' or 'L', got {throws!r}")
    t, g = ("r", "l") if throws == "R" else ("l", "r")
    return [
        MP[f"{t}_shoulder"], MP[f"{t}_elbow"], MP[f"{t}_wrist"],
        MP[f"{g}_shoulder"], MP[f"{g}_elbow"], MP[f"{g}_wrist"],
        MP[f"{t}_hip"], MP[f"{t}_knee"], MP[f"{t}_ankle"],
        MP[f"{g}_hip"], MP[f"{g}_knee"], MP[f"{g}_ankle"],
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

    rear = seg("r_hip", "r_knee") + seg("r_knee", "r_ankle")
    lead = seg("l_hip", "l_knee") + seg("l_knee", "l_ankle")
    vals = np.concatenate([rear, lead])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no finite leg segments")
    return float(np.median(vals))

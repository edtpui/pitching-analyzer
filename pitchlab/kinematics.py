"""Joint-angle and segment-rotation time series from the canonical 12-joint skeleton.

Definitions are geometric (no inverse kinematics) so they can be computed identically from noisy video
keypoints and from the Driveline joint centres. Angle conventions follow Driveline's where possible:
pelvis/torso rotation 0 deg = closed (side-on, facing third base for a right-hander), 90 deg = square to
home; hip-shoulder separation = pelvis rotation - torso rotation; knee/elbow flexion 0 deg = straight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signals import derivative, unwrap_deg
from .skeleton import J, leg_length

Z = np.array([0.0, 0.0, 1.0])


TRACK_KEYS = ("l_ankle", "r_ankle", "l_knee", "mid_hip", "t_wrist", "t_shoulder")


@dataclass
class Motion:
    """One pitch in the canonical, right-hander-normalised lab frame (see skeleton.py).

    `pos` may be pelvis-centred per frame (MediaPipe world landmarks are), so it only supports
    within-frame geometry. Movement across frames (stride, foot plant, momentum) uses `track`: absolute
    positions in a fixed frame - the lab for Driveline, image pixels from a static camera for video."""

    t: np.ndarray                         # (T,) seconds, strictly increasing
    pos: np.ndarray                       # (T, 12, 3) joint centres
    track: dict[str, np.ndarray]          # TRACK_KEYS -> (T, D) absolute positions
    track_up: np.ndarray                  # (D,) unit "up" in track coordinates
    track_forward: np.ndarray | None      # (D,) unit vector toward home in track coordinates, None if unseen
    track_scale: float                    # leg length in track units
    heading_known: bool                   # +x of `pos` is the true direction to home plate
    fps: float
    arm_valid: np.ndarray | None = None   # (T,) throwing elbow/wrist really tracked (False = bridged)

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
    """Anterior horizontal unit vector of a segment from its right-pointing axis, and its rotation in
    Driveline's convention (0 = facing third base / closed, 90 = facing home), unwrapped and shifted by
    whole turns so the clip median lies in (-90, 270]."""
    anterior = _unit(np.cross(Z, _horizontal(right)))
    rot = unwrap_deg(np.degrees(np.arctan2(anterior[:, 1], anterior[:, 0])) + 90.0)
    med = np.nanmedian(rot) if np.isfinite(rot).any() else 0.0
    return anterior, rot + 360.0 * np.floor((270.0 - med) / 360.0)


def _wrap180(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def joint(pos: np.ndarray, name: str) -> np.ndarray:
    return pos[:, J[name]]


def kinematics(m: Motion) -> dict[str, np.ndarray]:
    """All time series used by event detection and metrics. Rotational velocities in deg/s; linear
    speeds in leg lengths per second (wrist) or the foot track's own unit per second (foot)."""
    p = m.pos
    t = m.t
    ts, te, tw = joint(p, "t_shoulder"), joint(p, "t_elbow"), joint(p, "t_wrist")
    gs = joint(p, "g_shoulder")
    rh, rk, ra = joint(p, "r_hip"), joint(p, "r_knee"), joint(p, "r_ankle")
    lh, lk, la = joint(p, "l_hip"), joint(p, "l_knee"), joint(p, "l_ankle")
    mid_hip = (rh + lh) / 2.0
    mid_sh = (ts + gs) / 2.0
    trunk = mid_sh - mid_hip
    L = leg_length(p)

    # Segment headings (rear hip / throwing shoulder are the pitcher's right side after normalisation).
    pelvis_ant, pelvis_rot = _heading(rh - lh)
    torso_ant, torso_rot = _heading(ts - gs)
    torso_left = np.cross(Z, torso_ant)

    # Trunk tilt resolved in the torso's own heading (anatomical flexion / lateral flexion).
    tz = trunk[:, 2]
    trunk_flexion = np.degrees(np.arctan2(_dot(trunk, torso_ant), tz))
    trunk_lateral = np.degrees(np.arctan2(_dot(trunk, torso_left), tz))     # + toward glove side

    lead_knee_flexion = 180.0 - _angle_between(lh - lk, la - lk)
    elbow_flexion = 180.0 - _angle_between(ts - te, tw - te)

    # Thorax frame: z_T along the spine, y_T toward the glove shoulder, x_T anterior.
    z_t = _unit(trunk)
    y_raw = gs - ts
    y_t = _unit(y_raw - _dot(y_raw, z_t)[:, None] * z_t)
    x_t = np.cross(y_t, z_t)
    hum = te - ts
    fore = tw - te
    hum_u = _unit(hum)
    shoulder_abduction = _angle_between(hum, -z_t)
    shoulder_horiz_abduction = np.degrees(np.arctan2(-_dot(hum, x_t), -_dot(hum, y_t)))
    e1 = _unit(x_t - _dot(x_t, hum_u)[:, None] * hum_u)    # 0 deg ER: forearm points anteriorly
    e2 = np.cross(hum_u, e1)                                # 90 deg ER: forearm points along the spine
    shoulder_er = unwrap_deg(np.degrees(np.arctan2(_dot(fore, e2), _dot(fore, e1))))

    # Arm slot: shoulder-to-wrist vector projected on the lab frontal (y-z) plane; 0 = sidearm,
    # 90 = straight over the top. (The forearm alone matches Driveline equally well on lab data but
    # swings so fast at release that a 10 ms release-timing error ruins it on video.)
    arm = tw - ts
    arm_slot = np.degrees(np.arctan2(arm[:, 2], -arm[:, 1]))

    s = m.track_scale
    # Release is found from the wrist as the camera sees it (relative to the pelvis): MediaPipe's
    # per-frame depth estimate for a fast-moving wrist jumps by tens of centimetres, which made the
    # 3D speed peak during arm cocking instead of at release on real clips.
    wrist_speed = np.linalg.norm(derivative(m.track["t_wrist"] - m.track["mid_hip"], t), axis=-1) / s
    foot_pos = m.track["l_ankle"] / s
    foot_speed = np.linalg.norm(derivative(foot_pos, t), axis=-1)
    knee_height = (m.track["l_knee"] @ m.track_up) / s

    out = {
        "pelvis_rot": pelvis_rot,
        "torso_rot": torso_rot,
        "hip_shoulder_sep": _wrap180(pelvis_rot - torso_rot),
        "pelvis_rot_velo": derivative(pelvis_rot, t),
        "torso_rot_velo": derivative(torso_rot, t),
        "trunk_flexion": trunk_flexion,
        "trunk_lateral": trunk_lateral,
        "lead_knee_flexion": lead_knee_flexion,
        "lead_knee_ext_velo": -derivative(lead_knee_flexion, t),
        "elbow_flexion": elbow_flexion,
        "elbow_ext_velo": -derivative(elbow_flexion, t),
        "shoulder_abduction": shoulder_abduction,
        "shoulder_horiz_abduction": shoulder_horiz_abduction,
        "shoulder_er": shoulder_er,
        "shoulder_ir_velo": -derivative(shoulder_er, t),
        "arm_slot": arm_slot,
        "wrist_speed": wrist_speed,
        "foot_speed": foot_speed,
        "foot_pos": foot_pos,
        "knee_height": knee_height,
    }
    if m.track_forward is not None:
        out["pelvis_fwd_velo"] = derivative(m.track["mid_hip"] @ m.track_forward, t) / s
        # how far the throwing hand is in front of the throwing shoulder, toward home (leg lengths)
        out["hand_ahead"] = ((m.track["t_wrist"] - m.track["t_shoulder"]) @ m.track_forward) / s
    out["arm_valid"] = m.arm_valid if m.arm_valid is not None else np.ones(len(t), dtype=bool)
    return out

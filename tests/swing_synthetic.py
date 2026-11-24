"""Synthetic test inputs: canonical skeletons with known angles, and MediaPipe-like pose tracks rendered
from Driveline lab data through a virtual camera."""
from __future__ import annotations

import numpy as np

from pitchlab.pose import N_LANDMARKS, PoseTrack
from pitchlab.swing.signals import resample
from pitchlab.swing.skeleton import JOINTS, MP, J, mediapipe_indices, mirror_y


def skeleton(**joints) -> np.ndarray:
    """(1, 12, 3) skeleton from keyword positions. Unspecified joints give a right-handed hitter square to the
    plate (facing -y, lead side toward the pitcher at +x): rear arm out to the side with the forearm pointing at
    the plate, lead arm hanging straight down."""
    base = {
        "rear_shoulder": (-0.2, 0, 1.5), "lead_shoulder": (0.2, 0, 1.5),
        "rear_elbow": (-0.5, 0, 1.5), "rear_wrist": (-0.5, -0.28, 1.5),
        "lead_elbow": (0.2, 0, 1.2), "lead_wrist": (0.2, 0, 0.92),
        "rear_hip": (-0.15, 0, 1.0), "lead_hip": (0.15, 0, 1.0),
        "rear_knee": (-0.15, 0, 0.55), "lead_knee": (0.15, 0, 0.55),
        "rear_ankle": (-0.15, 0, 0.1), "lead_ankle": (0.15, 0, 0.1),
    }
    base.update(joints)
    return np.array([[base[j] for j in JOINTS]], dtype=float)


def camera_basis(view: str, bats: str, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (rows = camera x right, y down, z forward in lab coords) and camera position, matching
    pitchlab.swing.driveline.NOMINAL_CAMERAS. side = facing the hitter's chest from across the plate (a right-handed
    hitter faces -y, a left-hander +y in raw lab coordinates); behind = behind the catcher."""
    from pitchlab.swing.driveline import NOMINAL_CAMERAS
    cam = NOMINAL_CAMERAS[view]
    up = np.array([0.0, 0.0, 1.0])
    cx, cy = float(centre[0]), float(centre[1])
    if view == "side":
        if bats == "R":
            fwd, pos = np.array([0.0, 1.0, 0.0]), np.array([cx, cy - cam.distance, cam.height])
        else:
            fwd, pos = np.array([0.0, -1.0, 0.0]), np.array([cx, cy + cam.distance, cam.height])
    elif view == "behind":
        fwd, pos = np.array([1.0, 0.0, 0.0]), np.array([cx - cam.distance, cy, cam.height])
    else:
        raise ValueError(view)
    y_cam = -up
    z_cam = fwd
    x_cam = np.cross(y_cam, z_cam)
    return np.stack([x_cam, y_cam, z_cam]), pos


def driveline_track(trial, view: str = "side", fps: float = 240.0, width: int = 1280, height: int = 720,
                    focal: float = 1100.0) -> PoseTrack:
    """Render a Driveline swing as a MediaPipe-style PoseTrack seen by a static camera."""
    t = np.arange(trial.t[0], trial.t[-1], 1.0 / fps)
    canon = resample(trial.pos, trial.t, t)
    raw = mirror_y(canon) if trial.bats == "L" else canon          # back to true lab coordinates
    T = len(t)
    lm = np.zeros((T, N_LANDMARKS, 3))
    idx = mediapipe_indices(trial.bats)
    for j, mp_i in enumerate(idx):
        lm[:, mp_i] = raw[:, j]
    up = np.array([0.0, 0.0, 1.0])
    toward_plate = np.array([0.0, -1.0, 0.0]) if trial.bats == "R" else np.array([0.0, 1.0, 0.0])
    head = (lm[:, MP["l_shoulder"]] + lm[:, MP["r_shoulder"]]) / 2 + 0.25 * up
    for i in range(0, 11):
        lm[:, i] = head
    for side in ("l", "r"):
        wrist = lm[:, MP[f"{side}_wrist"]]
        for k in ((17, 19, 21) if side == "l" else (18, 20, 22)):
            lm[:, k] = wrist
        ankle = lm[:, MP[f"{side}_ankle"]]
        lm[:, MP[f"{side}_heel"]] = ankle - 0.05 * toward_plate - 0.04 * up
        lm[:, MP[f"{side}_foot"]] = ankle + 0.15 * toward_plate - 0.06 * up

    mid_hip = (lm[:, MP["l_hip"]] + lm[:, MP["r_hip"]]) / 2
    R, cam = camera_basis(view, trial.bats, np.mean(mid_hip, axis=0))
    world = (lm - mid_hip[:, None, :]) @ R.T
    pc = (lm - cam) @ R.T
    image = np.stack([focal * pc[..., 0] / pc[..., 2] + width / 2, focal * pc[..., 1] / pc[..., 2] + height / 2], axis=-1)
    return PoseTrack(t=t, frame_index=np.arange(T), image=image, visibility=np.full((T, N_LANDMARKS), 0.99),
                     world=world, detected=np.ones(T, dtype=bool), width=width, height=height)


def canonical_index(name: str) -> int:
    return J[name]

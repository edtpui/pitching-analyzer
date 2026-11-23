"""Synthetic test inputs: canonical skeletons with known angles, and MediaPipe-like pose tracks rendered
from Driveline lab data through a virtual camera."""
from __future__ import annotations

import numpy as np

from pitchlab.pose import N_LANDMARKS, PoseTrack
from pitchlab.signals import resample
from pitchlab.skeleton import JOINTS, MP, J, mediapipe_indices, mirror_y


def skeleton(**joints) -> np.ndarray:
    """(1, 12, 3) skeleton from keyword positions; unspecified joints get a neutral standing pose."""
    base = {
        "t_shoulder": (0, -0.2, 1.5), "g_shoulder": (0, 0.2, 1.5),
        "t_elbow": (0, -0.5, 1.5), "t_wrist": (0, -0.5, 1.78),
        "g_elbow": (0, 0.5, 1.5), "g_wrist": (0, 0.75, 1.5),
        "r_hip": (0, -0.15, 1.0), "l_hip": (0, 0.15, 1.0),
        "r_knee": (0, -0.15, 0.55), "l_knee": (0, 0.15, 0.55),
        "r_ankle": (0, -0.15, 0.1), "l_ankle": (0, 0.15, 0.1),
    }
    base.update(joints)
    return np.array([[base[j] for j in JOINTS]], dtype=float)


def camera_basis(view: str, throws: str, centre_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (rows = camera x right, y down, z forward in lab coords) and camera position, matching
    pitchlab.driveline.NOMINAL_CAMERAS. side = arm side of the pitcher (third-base side for a
    right-hander, first-base side for a left-hander); behind = from second base."""
    from pitchlab.driveline import NOMINAL_CAMERAS
    cam = NOMINAL_CAMERAS[view]
    up = np.array([0.0, 0.0, 1.0])
    if view == "side":
        fwd = np.array([0.0, 1.0, 0.0]) if throws == "R" else np.array([0.0, -1.0, 0.0])
        pos = np.array([centre_x, -cam.distance if throws == "R" else cam.distance, cam.height])
    elif view == "behind":
        fwd = np.array([1.0, 0.0, 0.0])
        pos = np.array([centre_x - cam.distance, 0.0, cam.height])
    else:
        raise ValueError(view)
    y_cam = -up
    z_cam = fwd
    x_cam = np.cross(y_cam, z_cam)
    return np.stack([x_cam, y_cam, z_cam]), pos


def driveline_track(trial, view: str = "side", fps: float = 240.0, width: int = 1280, height: int = 720,
                    focal: float = 1100.0) -> PoseTrack:
    """Render a Driveline trial as a MediaPipe-style PoseTrack seen by a static camera."""
    t = np.arange(trial.t[0], trial.t[-1], 1.0 / fps)
    canon = resample(trial.pos, trial.t, t)
    raw = mirror_y(canon) if trial.throws == "L" else canon          # back to true lab coordinates
    T = len(t)
    lm = np.zeros((T, N_LANDMARKS, 3))
    idx = mediapipe_indices(trial.throws)
    for j, mp_i in enumerate(idx):
        lm[:, mp_i] = raw[:, j]
    fwd = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    head = (lm[:, MP["l_shoulder"]] + lm[:, MP["r_shoulder"]]) / 2 + 0.25 * up
    for i in range(0, 11):
        lm[:, i] = head
    for side in ("l", "r"):
        wrist = lm[:, MP[f"{side}_wrist"]]
        for k in ((17, 19, 21) if side == "l" else (18, 20, 22)):
            lm[:, k] = wrist
        ankle = lm[:, MP[f"{side}_ankle"]]
        lm[:, MP[f"{side}_heel"]] = ankle - 0.05 * fwd - 0.04 * up
        lm[:, MP[f"{side}_foot"]] = ankle + 0.15 * fwd - 0.06 * up

    mid_hip = (lm[:, MP["l_hip"]] + lm[:, MP["r_hip"]]) / 2
    R, cam = camera_basis(view, trial.throws, float(np.mean(mid_hip[:, 0])))
    world = (lm - mid_hip[:, None, :]) @ R.T
    pc = (lm - cam) @ R.T
    image = np.stack([focal * pc[..., 0] / pc[..., 2] + width / 2, focal * pc[..., 1] / pc[..., 2] + height / 2], axis=-1)
    return PoseTrack(t=t, frame_index=np.arange(T), image=image, visibility=np.full((T, N_LANDMARKS), 0.99),
                     world=world, detected=np.ones(T, dtype=bool), width=width, height=height)


def canonical_index(name: str) -> int:
    return J[name]

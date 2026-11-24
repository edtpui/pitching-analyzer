"""Measuring the ball where the fitted trajectory says it is.

Every frame of the flight is revisited at the predicted position and the ball measured by its outer edge
(blob.measure_crop). Frames the detector missed (the ball in front of the pitcher, low contrast) are
recovered, and the track is extended backward toward the release until the ball is no longer where it
should be. Colour patches of the ball are kept for the spin estimate. Only small crops around the
predictions are held in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .blob import Crop, Measurement, crop_at, measure_crop
from .detect import frame_times
from .geometry import PLATE_FRONT_Y, Camera
from .trajectory import Track, Trajectory, model_diameter

SPIN_PATCH_MIN_D = 12.0      # px: smaller balls are not kept for the seam fit


@dataclass
class Measured:
    track: Track
    rows: list[Measurement]
    patches: dict[int, dict] = field(default_factory=dict)   # frame -> {"img", "bg", "center", "radius", "t", ...}


def _spin_patch(c: Crop, m: Measurement) -> dict:
    half = int(np.ceil(1.25 * m.d))
    M = np.array([[1.0, 0.0, half + 0.5 - (m.u - c.x0)], [0.0, 1.0, half + 0.5 - (m.v - c.y0)]])
    size = (2 * half + 1, 2 * half + 1)
    img = cv2.warpAffine(c.img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bg = cv2.warpAffine(c.bg, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return {"img": img, "bg": bg, "center": (half + 0.5, half + 0.5), "radius": m.d / 2, "t": m.t,
            "uv": (m.u, m.v), "elongation": m.elongation}


def remeasure(path, background: np.ndarray, traj: Trajectory, camera: Camera, track: Track,
              capture_fps: float | None, start_s: float | None, end_s: float | None,
              back_frames: int = 40) -> Measured:
    """Re-measure every frame from `back_frames` before the track to just past the plate crossing. The
    backward run stops at the second consecutive frame without the ball where it should be."""
    first, last = int(track.frame.min()), int(track.frame.max())
    t_plate = traj.time_at_y(PLATE_FRONT_Y)
    crops: dict[int, Crop] = {}
    for i, t, frame in frame_times(path, start_s, end_s, capture_fps):
        if i < first - back_frames:
            continue
        if i > last and (t_plate is None or t > t_plate + 0.03):
            break
        P = traj.pos(t)
        uv, z = camera.project(P)
        if z[0] <= 0.3:
            continue
        d_pred = max(float(model_diameter(P[None, :], camera)[0]), 2.0)
        c = crop_at(frame, background, uv[0], d_pred, t, camera=camera)
        if c is not None:
            crops[i] = c
    rows: dict[int, Measurement] = {}
    patches: dict[int, dict] = {}

    def take(i: int) -> bool:
        m = measure_crop(crops[i])
        if m is None:
            return False
        m.frame = i
        rows[i] = m
        if m.d >= SPIN_PATCH_MIN_D:
            patches[i] = _spin_patch(crops[i], m)
        return True

    for i in sorted(k for k in crops if k >= first):
        take(i)
    misses = 0
    for i in sorted((k for k in crops if k < first), reverse=True):
        if take(i):
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    ms = [rows[k] for k in sorted(rows)]
    tr = Track(frame=np.array([m.frame for m in ms], int), t=np.array([m.t for m in ms]),
               uv=np.array([[m.u, m.v] for m in ms]).reshape(-1, 2), d=np.array([m.d for m in ms]),
               sig_uv=np.array([m.sig_uv for m in ms]), d_kind=np.array([m.kind for m in ms], int))
    return Measured(track=tr, rows=ms, patches=patches)

"""MediaPipe pose landmarker over a video, tracking one subject."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")

from .config import MODELS_DIR  # noqa: E402

N_LANDMARKS = 33
MODEL_VARIANTS = ("lite", "full", "heavy")


@dataclass
class PoseTrack:
    t: np.ndarray            # (T,) seconds (container time)
    frame_index: np.ndarray  # (T,) index into the decoded video
    image: np.ndarray        # (T, 33, 2) pixel coordinates in the analysed frame size (NaN if undetected)
    visibility: np.ndarray   # (T, 33)
    world: np.ndarray        # (T, 33, 3) metres, camera axes (x right, y down, z away), hip-centred
    detected: np.ndarray     # (T,) bool
    width: int
    height: int

    def save(self, path: Path) -> None:
        np.savez_compressed(path, t=self.t, frame_index=self.frame_index, image=self.image,
                            visibility=self.visibility, world=self.world, detected=self.detected,
                            size=np.array([self.width, self.height]))

    @classmethod
    def load(cls, path: Path) -> "PoseTrack":
        d = np.load(path)
        w, h = (int(v) for v in d["size"])
        return cls(t=d["t"], frame_index=d["frame_index"], image=d["image"], visibility=d["visibility"],
                   world=d["world"], detected=d["detected"].astype(bool), width=w, height=h)


def model_path(variant: str) -> Path:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown pose model {variant!r}")
    p = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run scripts/fetch_assets.py")
    return p


def _bbox(lms: np.ndarray) -> tuple[np.ndarray, float]:
    lo, hi = lms.min(axis=0), lms.max(axis=0)
    return (lo + hi) / 2.0, float(np.prod(np.maximum(hi - lo, 0.0)))


def estimate(frames: Iterable[tuple[int, float, np.ndarray]], variant: str = "full",
             expected_frames: int | None = None,
             progress: Callable[[int], None] | None = None) -> PoseTrack:
    """Run the landmarker in VIDEO mode. When several people are detected, the subject is the largest
    person in the first detection and is followed by proximity afterwards."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path(variant))),
        running_mode=vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)

    ts, idxs, imgs, vis, world, det = [], [], [], [], [], []
    width = height = 0
    last_ms = -1
    prev_center: np.ndarray | None = None
    frames_since_seen = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for k, (i, t, frame) in enumerate(frames):
            height, width = frame.shape[:2]
            ms = max(int(round(t * 1000.0)), last_ms + 1)
            last_ms = ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
            img = np.full((N_LANDMARKS, 2), np.nan)
            v = np.zeros(N_LANDMARKS)
            wl = np.full((N_LANDMARKS, 3), np.nan)
            found = False
            choice = None
            if res.pose_landmarks:
                cands = []
                for pi, pl in enumerate(res.pose_landmarks):
                    xy = np.array([[p.x * width, p.y * height] for p in pl])
                    center, area = _bbox(xy)
                    cands.append((pi, xy, center, area))
                if prev_center is None:
                    choice = max(cands, key=lambda c: c[3])
                else:
                    near = min(cands, key=lambda c: float(np.linalg.norm(c[2] - prev_center)))
                    # reject a different person appearing elsewhere; allow more travel after dropouts
                    max_jump = float(np.hypot(width, height)) * (0.12 + 0.08 * frames_since_seen)
                    if float(np.linalg.norm(near[2] - prev_center)) <= max_jump:
                        choice = near
            if choice is not None:
                pi, xy, center, _ = choice
                prev_center = center
                frames_since_seen = 0
                img = xy
                v = np.array([float(p.visibility if p.visibility is not None else 0.0)
                              for p in res.pose_landmarks[pi]])
                if res.pose_world_landmarks and len(res.pose_world_landmarks) > pi:
                    wl = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[pi]])
                found = True
            else:
                frames_since_seen += 1
            ts.append(t)
            idxs.append(i)
            imgs.append(img)
            vis.append(v)
            world.append(wl)
            det.append(found)
            if progress is not None:
                progress(k + 1)
    if not ts:
        raise ValueError("the video contains no frames in the selected range")
    return PoseTrack(t=np.array(ts), frame_index=np.array(idxs), image=np.array(imgs), visibility=np.array(vis),
                     world=np.array(world), detected=np.array(det), width=width, height=height)

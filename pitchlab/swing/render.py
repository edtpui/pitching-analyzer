"""Skeleton overlay video and event key frames."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .skeleton import MP, MP_EDGES
from ..video_io import H264Writer, iter_frames, scaled_size

# BGR colours: rear (top-hand) arm stands out, lead arm second, lead leg green, the rest neutral.
REAR_ARM = (40, 170, 255)
LEAD_ARM = (230, 160, 60)
BODY = (235, 235, 235)
LEAD_LEG = (120, 220, 120)
EVENT_LABELS = {"lift": "Leg lift", "load": "Load (max hip coil)", "fp": "Foot plant", "contact": "Contact"}


def _edge_colour(a: int, b: int, bats: str) -> tuple[int, int, int]:
    rear, lead = ("r", "l") if bats == "R" else ("l", "r")
    rear_arm = {MP[f"{rear}_shoulder"], MP[f"{rear}_elbow"], MP[f"{rear}_wrist"]}
    lead_arm = {MP[f"{lead}_shoulder"], MP[f"{lead}_elbow"], MP[f"{lead}_wrist"]}
    lead_leg = {23, 25, 27, 29, 31} if bats == "R" else {24, 26, 28, 30, 32}
    if a in rear_arm and b in rear_arm:
        return REAR_ARM
    if a in lead_arm and b in lead_arm:
        return LEAD_ARM
    if a in lead_leg and b in lead_leg:
        return LEAD_LEG
    return BODY


def draw_skeleton(frame: np.ndarray, pts: np.ndarray, bats: str, scale: float) -> None:
    th = max(2, int(round(3 * scale)))
    for a, b in MP_EDGES:
        pa, pb = pts[a], pts[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(frame, tuple(int(v) for v in pa), tuple(int(v) for v in pb),
                     _edge_colour(a, b, bats), th, cv2.LINE_AA)
    for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        p = pts[i]
        if np.isfinite(p).all():
            cv2.circle(frame, (int(p[0]), int(p[1])), th + 1, (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(p[0]), int(p[1])), th, (255, 255, 255), -1, cv2.LINE_AA)


def _label(frame: np.ndarray, text: str, scale: float, y: int | None = None) -> None:
    fs = 0.6 * scale
    th = max(1, int(round(1.5 * scale)))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    y = y if y is not None else int(10 * scale) + h
    cv2.rectangle(frame, (int(8 * scale), y - h - int(6 * scale)), (int(8 * scale) + w + int(10 * scale), y + base + int(2 * scale)),
                  (20, 20, 20), -1)
    cv2.putText(frame, text, (int(13 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def render_overlay(video_path: Path, out_path: Path, frame_index: np.ndarray, image_pts: np.ndarray,
                   analysis_size: tuple[int, int], bats: str, fps: float, max_side: int,
                   start_s: float | None, end_s: float | None,
                   progress: Callable[[float], None] | None = None) -> tuple[int, int]:
    """Write the analysed frames with the skeleton drawn on them. Returns the output size."""
    lookup = {int(fi): k for k, fi in enumerate(frame_index)}
    out_size: tuple[int, int] | None = None
    writer: H264Writer | None = None
    total = len(frame_index)
    done = 0
    try:
        for i, _, frame in iter_frames(video_path, max_side=max_side, start_s=start_s, end_s=end_s):
            k = lookup.get(i)
            if k is None:
                continue
            if writer is None:
                out_size = scaled_size(frame.shape[1], frame.shape[0], max_side)
                writer = H264Writer(out_path, out_size, fps)
            sx = frame.shape[1] / analysis_size[0]
            sy = frame.shape[0] / analysis_size[1]
            pts = image_pts[k] * np.array([sx, sy])
            scale = frame.shape[0] / 540.0
            draw_skeleton(frame, pts, bats, scale)
            _label(frame, f"frame {k + 1}", scale)
            writer.write(frame)
            done += 1
            if progress and done % 10 == 0:
                progress(done / total)
    finally:
        if writer is not None:
            writer.close()
    if out_size is None:
        raise RuntimeError("no frames were rendered")
    return out_size


def render_keyframes(video_path: Path, out_dir: Path, frame_index: np.ndarray, image_pts: np.ndarray,
                     analysis_size: tuple[int, int], bats: str, events: dict[str, int | None],
                     max_side: int, start_s: float | None, end_s: float | None) -> dict[str, str]:
    """JPEG of each event frame with the skeleton. Returns event -> file name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted: dict[int, list[tuple[str, int]]] = {}
    for name, k in events.items():
        if k is not None:
            wanted.setdefault(int(frame_index[k]), []).append((name, k))
    if not wanted:
        return {}
    last = max(wanted)
    files: dict[str, str] = {}
    for i, _, frame in iter_frames(video_path, max_side=max_side, start_s=start_s, end_s=end_s):
        for name, k in wanted.get(i, []):
            img = frame.copy()
            sx = img.shape[1] / analysis_size[0]
            sy = img.shape[0] / analysis_size[1]
            scale = img.shape[0] / 540.0
            draw_skeleton(img, image_pts[k] * np.array([sx, sy]), bats, scale)
            _label(img, EVENT_LABELS.get(name, name), scale)
            fname = f"{name}.jpg"
            cv2.imwrite(str(out_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            files[name] = fname
        if i >= last:
            break
    return files

"""Pictures of a ball-flight analysis: the calibration overlay, the tracer video and the seam-fit strip."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..video_io import H264Writer, scaled_size
from .detect import frame_times
from .geometry import IN, Camera, plate_points, rubber_points, zone_corners
from .spin import N_C, R_C, _SEAM, seam_map
from .trajectory import Trajectory

ZONE = (255, 255, 255)
PLATE = (210, 210, 210)
TRAIL = (40, 170, 255)        # BGR orange, the throwing-arm colour of the mechanics overlay
BALL = (60, 220, 255)
MEASURED = (120, 220, 120)


def _poly(img, cam: Camera, pts3: np.ndarray, colour, th: int, closed: bool = True, scale: float = 1.0) -> None:
    uv, z = cam.project(pts3)
    if np.any(z <= 0.05) or not np.all(np.isfinite(uv)):
        return
    p = np.round((uv * scale - 0.5) * 16).astype(np.int32)
    cv2.polylines(img, [p], closed, colour, th, cv2.LINE_AA, shift=4)


def draw_field(img: np.ndarray, cam: Camera, distance_ft: float, mound_in: float, zone_bottom_in: float,
               zone_top_in: float, scale: float = 1.0) -> None:
    """Plate, rubber, strike zone and the line between them, drawn into img (scaled from camera pixels)."""
    th = max(1, int(round(img.shape[0] / 540)))
    _poly(img, cam, plate_points(), PLATE, th, scale=scale)
    rb = rubber_points(distance_ft, mound_in)
    _poly(img, cam, np.vstack([rb, rb[::-1] + [0.0, 6 * IN, 0.0]]), PLATE, th, scale=scale)
    zc = zone_corners(zone_bottom_in, zone_top_in)
    _poly(img, cam, zc, ZONE, th, scale=scale)
    for f in (1 / 3, 2 / 3):                                   # zone thirds
        a, b = zc[0] + f * (zc[3] - zc[0]), zc[1] + f * (zc[2] - zc[1])
        _poly(img, cam, np.vstack([a, b]), ZONE, 1, closed=False, scale=scale)
        a, b = zc[0] + f * (zc[1] - zc[0]), zc[3] + f * (zc[2] - zc[3])
        _poly(img, cam, np.vstack([a, b]), ZONE, 1, closed=False, scale=scale)
    line = np.array([[0.0, 17 * IN, 0.0], [0.0, distance_ft * 0.3048, mound_in * IN]])
    _poly(img, cam, line, (170, 170, 170), 1, closed=False, scale=scale)


def calibration_preview(image: np.ndarray, cam: Camera, distance_ft: float, mound_in: float, zone_bottom_in: float,
                        zone_top_in: float, marks: np.ndarray | None = None, max_side: int = 1600) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        out = cv2.resize(out, scaled_size(w, h, max_side), interpolation=cv2.INTER_AREA)
        s = out.shape[1] / w
    draw_field(out, cam, distance_ft, mound_in, zone_bottom_in, zone_top_in, scale=s)
    if marks is not None:
        for p in marks:
            cv2.circle(out, (int(p[0] * s), int(p[1] * s)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def _label(frame: np.ndarray, lines: list[str], scale: float) -> None:
    fs = 0.55 * scale
    th = max(1, int(round(1.4 * scale)))
    y = int(12 * scale)
    for text in lines:
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y += tht + int(6 * scale)
        cv2.rectangle(frame, (int(8 * scale), y - tht - int(5 * scale)), (int(18 * scale) + tw, y + base + int(2 * scale)),
                      (20, 20, 20), -1)
        cv2.putText(frame, text, (int(13 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def render_tracer(video_path: Path, out_path: Path, cam: Camera, traj: Trajectory, t_release: float, t_plate: float,
                  measured: dict[int, tuple[float, float]], setup, fps_out: float, max_side: int,
                  start_s: float | None, end_s: float | None, capture_fps: float | None, first_frame: int,
                  last_frame: int, summary: list[str],
                  progress: Callable[[float], None] | None = None) -> tuple[tuple[int, int], int, int]:
    """The analysed frames with the strike zone, the fitted flight (trail up to now) and the measured ball.
    Returns (size, first source frame, frame count)."""
    t_track = np.linspace(t_release, t_plate, 160)
    trail_uv, trail_z = cam.project(traj.pos(t_track))
    writer = None
    size = None
    n = 0
    total = max(1, last_frame - first_frame + 1)
    try:
        for i, t, frame in frame_times(video_path, start_s, end_s, capture_fps):
            if i < first_frame:
                continue
            if i > last_frame:
                break
            h, w = frame.shape[:2]
            if writer is None:
                size = scaled_size(w, h, max_side)
                writer = H264Writer(out_path, size, fps_out)
            s = size[0] / w
            img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if s != 1.0 else frame.copy()
            sc = size[1] / 540.0
            draw_field(img, cam, setup.distance_ft, setup.mound_in, setup.zone_bottom_in, setup.zone_top_in, scale=s)
            upto = (t_track <= t) & (trail_z > 0.05)
            if upto.sum() >= 2:
                p = np.round((trail_uv[upto] * s - 0.5) * 16).astype(np.int32)
                cv2.polylines(img, [p], False, TRAIL, max(2, int(round(2.5 * sc))), cv2.LINE_AA, shift=4)
            if t_release <= t <= t_plate:
                uv, z = cam.project(traj.pos(t))
                if z[0] > 0.05:
                    c = tuple(np.round(uv[0] * s - 0.5).astype(int))
                    cv2.circle(img, c, max(3, int(round(4 * sc))), BALL, max(1, int(round(1.5 * sc))), cv2.LINE_AA)
            if i in measured:
                u, v = measured[i]
                cv2.drawMarker(img, (int(u * s), int(v * s)), MEASURED, cv2.MARKER_CROSS, max(6, int(8 * sc)), 1,
                               cv2.LINE_AA)
            lines = [f"frame {i - first_frame + 1}"]
            if t > t_plate:
                lines += summary
            _label(img, lines, sc)
            writer.write(img)
            n += 1
            if progress and n % 10 == 0:
                progress(min(1.0, n / total))
    finally:
        if writer is not None:
            writer.close()
    if size is None:
        raise RuntimeError("no frames were rendered")
    return size, first_frame, n


def seam_strip(patches: dict[int, dict], orientations: dict[int, list[list[float]]], out_path: Path,
               per_row: int = 10, cell: int = 96) -> list[int]:
    """Ball crops used for the spin fit, each beside the fitted seam drawn on its seam map. Returns the
    frames shown, in order."""
    frames = sorted(int(k) for k in orientations)
    if not frames:
        return []
    rows = (len(frames) + per_row - 1) // per_row
    sheet = np.full((rows * cell * 2, per_row * cell, 3), 24, np.uint8)
    for j, k in enumerate(frames):
        p = patches[k]
        cx, cy = p["center"]
        s = R_C / p["radius"]
        M = np.array([[s, 0.0, N_C / 2 - s * (cx - 0.5) - 0.5], [0.0, s, N_C / 2 - s * (cy - 0.5) - 0.5]])
        can = cv2.warpAffine(p["img"], M, (N_C, N_C), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        smap = seam_map(p)
        sm = np.clip(128 + 40 * smap, 0, 255).astype(np.uint8)
        sm = cv2.cvtColor(sm, cv2.COLOR_GRAY2BGR)
        R = np.asarray(orientations[k])
        q = _SEAM @ R.T
        vis = q[:, 2] < -0.1
        overlay = cv2.resize(sm, (cell, cell), interpolation=cv2.INTER_CUBIC)
        pts = np.column_stack([N_C / 2 + R_C * q[:, 0], N_C / 2 + R_C * q[:, 1]]) * (cell / N_C)
        for a in range(len(q)):
            b = (a + 1) % len(q)
            if vis[a] and vis[b]:
                cv2.line(overlay, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)),
                         (60, 220, 255), 1, cv2.LINE_AA)
        r, c = divmod(j, per_row)
        sheet[r * 2 * cell:(r * 2 + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(can, (cell, cell),
                                                                                  interpolation=cv2.INTER_CUBIC)
        sheet[(r * 2 + 1) * cell:(r * 2 + 2) * cell, c * cell:(c + 1) * cell] = overlay
        cv2.putText(sheet, str(j + 1), (c * cell + 4, r * 2 * cell + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return frames

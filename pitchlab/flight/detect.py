"""Ball candidates in every frame.

The camera is on a tripod, so a per-pixel median over frames sampled across the clip is the static scene
(people who move are averaged out). In each frame, pixels that differ from it by more than the noise
level form blobs; those inside the image region a pitch can pass through (the corridor from the rubber to
the plate, projected through the calibrated camera) and of a size a baseball can have there are the
candidates. The tracker decides which of them is the ball.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from ..video_io import iter_frames
from .blob import crop_at, measure_crop
from .geometry import FT, IN, Camera

BALL_RADIUS = 1.45 * IN
CORRIDOR_HALF_WIDTH = 1.7        # m either side of the mound-plate line
CORRIDOR_TOP = 3.3               # m above the ground
CORRIDOR_BACK = 1.5              # m behind the back tip of the plate (catcher, backstop net)
MIN_THRESHOLD = 9.0              # grey levels
NOISE_K = 5.0                    # threshold in robust noise SDs
MAX_CANDIDATES = 40              # per frame, strongest first

# candidate columns
# U, V, D: the measured centre and diameter when REFINED (else the thresholded blob's; D then runs large),
# DKIND says which diameter D is (blob.KIND_MINOR / KIND_AREA)
U, V, D, MAJOR, MINOR, PEAK, AREA, SAT, REFINED, DKIND = range(10)
NCOL = 10


@dataclass
class FrameCandidates:
    index: int                   # frame index in the decoded file
    t: float                     # seconds (capture time)
    cands: np.ndarray            # (n, NCOL)
    gain: float = 1.0            # brightness correction applied (flicker / exposure)


def frame_times(path: Path, start_s: float | None, end_s: float | None,
                capture_fps: float | None, max_side: int | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Frames with capture-time stamps: the container's, or index / capture_fps for slow-motion exports
    that play back slowed down."""
    for i, t, frame in iter_frames(path, max_side=max_side, start_s=start_s, end_s=end_s):
        yield i, (i / capture_fps if capture_fps else t), frame


def compute_background(path: Path, frame_count: int, start_s: float | None = None, end_s: float | None = None,
                       n: int = 31, progress: Callable[[float], None] | None = None) -> tuple[np.ndarray, int]:
    """Per-pixel median (BGR uint8) of about `n` frames spread over the selected range, and the number of
    frames in the range."""
    stride = max(1, frame_count // n) if frame_count > 0 else 4
    keep: list[np.ndarray] = []
    count = 0
    for k, (_, _, frame) in enumerate(iter_frames(path, start_s=start_s, end_s=end_s)):
        if k % stride == 0:
            keep.append(frame)
        count += 1
        if progress and k % 20 == 0 and frame_count > 0:
            progress(min(1.0, k / frame_count))
    if not keep:
        raise ValueError("the video contains no frames in the selected range")
    stack = np.stack(keep)
    bg = np.empty(stack.shape[1:], np.uint8)
    for c in range(3):
        bg[..., c] = np.median(stack[..., c], axis=0).astype(np.uint8)
    return bg, count


class Corridor:
    """The 3D box a pitch travels through and its image footprint."""

    def __init__(self, camera: Camera, distance_ft: float):
        self.camera = camera
        y1 = distance_ft * FT + 0.6
        self.lo = np.array([-CORRIDOR_HALF_WIDTH, -CORRIDOR_BACK, 0.0])
        self.hi = np.array([CORRIDOR_HALF_WIDTH, y1, CORRIDOR_TOP])
        g = [np.linspace(self.lo[k], self.hi[k], n) for k, n in enumerate((9, 60, 9))]
        X, Y, Z = np.meshgrid(*g, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        uv, z = camera.project(pts)
        good = z > 0.4
        w, h = camera.width, camera.height
        self.mask = np.zeros((h, w), np.uint8)
        if good.sum() >= 3:
            hull = cv2.convexHull(np.clip(uv[good], -4 * w, 5 * w).astype(np.float32))
            cv2.fillConvexPoly(self.mask, np.round(hull).astype(np.int32), 255)
            dist = np.linalg.norm(pts[good] - camera.center, axis=1)
            margin = int(np.ceil(camera.f * BALL_RADIUS / max(dist.min(), 0.5) * 2 + 12))
            self.mask = cv2.dilate(self.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
            self.d_max = float(2 * camera.f * BALL_RADIUS / max(dist.min(), 0.5))
        else:
            self.d_max = 0.0
        ys, xs = np.nonzero(self.mask)
        self.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, 0, 0)

    def diameter_range(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) smallest and largest image diameter (px) a ball on each pixel's ray can have inside the box;
        NaN where the ray misses the box."""
        cam = self.camera
        rays = cam.rays(uv)
        C = cam.center
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (self.lo - C) / rays
            t2 = (self.hi - C) / rays
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        tmin = np.maximum(tmin, 0.4)
        out = np.full((len(uv), 2), np.nan)
        hit = tmax > tmin
        cos_t = (rays @ cam.R[2])[hit]
        out[hit, 0] = 2 * cam.f * BALL_RADIUS / (tmax[hit] * cos_t ** 1.5)
        out[hit, 1] = 2 * cam.f * BALL_RADIUS / (tmin[hit] * cos_t ** 1.5)
        return out


def _robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def detect(path: Path, corridor: Corridor, background: np.ndarray, start_s: float | None = None,
           end_s: float | None = None, capture_fps: float | None = None, expected_frames: int = 0,
           progress: Callable[[float], None] | None = None) -> list[FrameCandidates]:
    x0, y0, x1, y1 = corridor.bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("the pitching corridor is not in the picture; check the calibration")
    bg = np.ascontiguousarray(background[y0:y1, x0:x1])
    roi = np.ascontiguousarray(corridor.mask[y0:y1, x0:x1])
    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(roi)
    pick = rng.choice(len(ys), size=min(len(ys), 20000), replace=False)
    sample_idx = (ys[pick], xs[pick])
    bg_level = float(np.median(bg[sample_idx].astype(np.float32).mean(axis=1)))
    max_area = max(60.0, np.pi / 4 * corridor.d_max ** 2 * 3.0)
    out: list[FrameCandidates] = []
    for k, (i, t, frame) in enumerate(frame_times(path, start_s, end_s, capture_fps)):
        img = frame[y0:y1, x0:x1]
        level = float(np.median(img[sample_idx].astype(np.float32).mean(axis=1)))
        gain = bg_level / level if level > 1 else 1.0
        if abs(gain - 1.0) > 0.01:
            img = cv2.convertScaleAbs(img, alpha=gain)
        d3 = cv2.absdiff(img, bg)
        c0, c1, c2 = cv2.split(d3)
        diff = cv2.GaussianBlur(cv2.max(cv2.max(c0, c1), c2), (3, 3), 0.7)
        sigma = _robust_sigma(diff[sample_idx].astype(np.float32))
        thr = max(MIN_THRESHOLD, NOISE_K * sigma)
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, roi)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        area_all = stats[1:, cv2.CC_STAT_AREA]
        side = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        keep = np.flatnonzero((area_all >= 3) & (area_all <= max_area)
                              & (side <= 4.5 * np.maximum(3.0, 2 * np.sqrt(area_all / np.pi)))) + 1
        rows = []
        for c in keep:
            bx, by, bw, bh, area = stats[c]
            sl = (slice(by, by + bh), slice(bx, bx + bw))
            m = labels[sl] == c
            wgt = np.where(m, diff[sl], 0).astype(np.float32)
            mo = cv2.moments(wgt)
            if mo["m00"] <= 0:
                continue
            cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
            mxx, myy, mxy = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
            tr, det = mxx + myy, mxx * myy - mxy ** 2
            disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
            l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
            px = img[sl][m].astype(np.float32)
            sat = float(np.mean(px.max(axis=1) - px.min(axis=1)) / max(1.0, np.mean(px.max(axis=1))))
            u, v, d_thr = cx + 0.5 + bx + x0, cy + 0.5 + by + y0, 2 * np.sqrt(area / np.pi)
            row = [u, v, d_thr, 4 * np.sqrt(l1), 4 * np.sqrt(l2), float(wgt.max()), float(area), sat, 0.0, 0.0]
            crop = crop_at(frame, background, (u, v), d_thr, t, camera=corridor.camera)
            if crop is not None:
                if abs(gain - 1.0) > 0.01:
                    crop.img = cv2.convertScaleAbs(crop.img, alpha=gain)
                m = measure_crop(crop, gate=0.5 * d_thr + 2.0)
                if m is not None:
                    row[U], row[V], row[D], row[REFINED], row[DKIND] = m.u, m.v, m.d, 1.0, float(m.kind)
            rows.append(row)
        cands = np.array(rows, float).reshape(-1, NCOL)
        if len(cands):
            rng_d = corridor.diameter_range(cands[:, [U, V]])
            ok = np.isfinite(rng_d[:, 0]) & (cands[:, D] <= 1.8 * rng_d[:, 1] + 3.0) & (cands[:, D] >= 0.4 * rng_d[:, 0] - 1.5)
            cands = cands[ok]
            if len(cands) > MAX_CANDIDATES:
                cands = cands[np.argsort(-cands[:, PEAK] * np.sqrt(cands[:, AREA]))[:MAX_CANDIDATES]]
        out.append(FrameCandidates(index=i, t=t, cands=cands, gain=gain))
        if progress and k % 10 == 0 and expected_frames:
            progress(min(1.0, k / expected_frames))
    return out

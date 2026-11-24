"""Centre and size of one ball image.

Inside a ball the colour difference from the empty background swings widely (seams, shading, a darker
limb), so thresholds, a single ball colour or a steepest-gradient search all lock onto seams and shrink the
ball. Its outer boundary is a clean step from ball to background: on each of 48 radial lines the outermost
point still above the noise floor is found, the edge level just inside it is read, and the edge is placed
where the profile falls through half that level (a blurred step - optics, compression, and motion blur
along that line alike - passes half height exactly at the true edge). An ellipse through the edge points
gives the centre and the minor axis: an off-axis sphere images stretched radially, so its minor axis is
2 f r / Z exactly. Shared by detection (every candidate) and the per-frame re-measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PATCH_PAD = 6
GATE_MIN_PX = 3.0
MIN_PEAK = 12.0               # grey levels
N_RAYS = 48
RAY_STEP = 0.2                # px

KIND_AREA, KIND_MINOR = 0, 1


@dataclass
class Crop:
    t: float
    img: np.ndarray              # BGR uint8
    bg: np.ndarray
    x0: int
    y0: int
    uv_pred: np.ndarray
    d_pred: float
    radial: np.ndarray | None = None   # unit image direction away from the principal point
    stretch: float = 1.0               # 1 / cos(off-axis angle): the sphere's radial elongation


@dataclass
class Measurement:
    frame: int
    t: float
    u: float
    v: float
    d: float                     # minor-axis diameter (KIND_MINOR) or area-equivalent diameter (KIND_AREA)
    kind: int
    contrast: float              # median colour difference from the background over the ball, grey levels
    elongation: float            # major / minor axis
    sig_uv: float


def crop_at(frame: np.ndarray, background: np.ndarray, uv, d_pred: float, t: float = 0.0,
            camera=None) -> Crop | None:
    """Crop around uv. With the camera, the crop also carries the sphere's perspective stretch there."""
    h, w = frame.shape[:2]
    half = int(np.ceil(1.6 * d_pred + PATCH_PAD))
    x0, y0 = int(np.floor(uv[0])) - half, int(np.floor(uv[1])) - half
    x1, y1 = x0 + 2 * half + 1, y0 + 2 * half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    radial, stretch = None, 1.0
    if camera is not None:
        off = np.array([uv[0] - camera.cx, uv[1] - camera.cy])
        r = float(np.linalg.norm(off))
        if r > 1e-6:
            radial = off / r
            stretch = float(np.sqrt(1.0 + (r / camera.f) ** 2))      # 1 / cos(theta)
    return Crop(t=t, img=frame[y0:y1, x0:x1].copy(), bg=background[y0:y1, x0:x1].copy(), x0=x0, y0=y0,
                uv_pred=np.asarray(uv, float), d_pred=float(d_pred), radial=radial, stretch=stretch)


def _fit_circle(p: np.ndarray, w: np.ndarray) -> tuple[float, float, float] | None:
    """Weighted circle through points: algebraic (Kasa) start, then a few Gauss-Newton steps on the
    geometric distance with Huber weights."""
    if len(p) < 4:
        return None
    A = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    sw = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(A * sw[:, None], -(p ** 2).sum(axis=1) * sw, rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r2 = cx * cx + cy * cy - sol[2]
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    for _ in range(8):
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        dist = np.maximum(np.hypot(dx, dy), 1e-9)
        res = dist - r
        s = 1.4826 * np.median(np.abs(res)) + 0.05
        hub = np.where(np.abs(res) <= 1.5 * s, 1.0, 1.5 * s / np.abs(res))
        ww = w * hub
        J = np.column_stack([-dx / dist, -dy / dist, -np.ones(len(p))])
        step, *_ = np.linalg.lstsq(J * np.sqrt(ww)[:, None], -res * np.sqrt(ww), rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-3:
            break
    return (float(cx), float(cy), float(r)) if r > 0 else None


def _edge_points(D: np.ndarray, cx: float, cy: float, r_max: float, t_low: float) -> tuple[np.ndarray, np.ndarray]:
    """Half-level crossings of the outer edge on radial lines from (cx, cy), in OpenCV pixel-centre
    coordinates of the crop, and the edge level at each. D must already be zero outside the ball's
    neighbourhood."""
    ang = np.linspace(0.0, 2 * np.pi, N_RAYS, endpoint=False)
    rs = np.arange(0.0, r_max, RAY_STEP)
    mx = (cx + np.outer(np.cos(ang), rs)).astype(np.float32)
    my = (cy + np.outer(np.sin(ang), rs)).astype(np.float32)
    prof = cv2.remap(D, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    above = prof >= t_low
    inner = int(round(2.0 / RAY_STEP))
    pts, levels = [], []
    for k in range(N_RAYS):
        idx = np.flatnonzero(above[k])
        if len(idx) == 0:
            continue
        j = int(idx[-1])                                   # outermost sample above the noise floor
        if j >= len(rs) - 2:
            continue                                       # the edge is past r_max: not this ball's edge
        lo = max(0, j - inner)
        m = lo + int(np.argmax(prof[k, lo:j + 1]))         # edge level: the highest value just inside
        half = 0.5 * prof[k, m]
        seg = prof[k, m:]
        below = np.flatnonzero(seg < half)
        if len(below) == 0:
            continue
        q = m + int(below[0])
        a, b = prof[k, q - 1], prof[k, q]
        r = rs[q - 1] + (a - half) / max(a - b, 1e-6) * RAY_STEP
        pts.append((cx + r * np.cos(ang[k]), cy + r * np.sin(ang[k])))
        levels.append(2.0 * half)
    return np.array(pts, np.float32).reshape(-1, 2), np.array(levels, np.float32)


def measure_crop(c: Crop, gate: float | None = None) -> Measurement | None:
    I = c.img.astype(np.float32)
    B = c.bg.astype(np.float32)
    D = np.sqrt(((I - B) ** 2).sum(axis=2))
    hh, ww = D.shape
    uv, d_pred = c.uv_pred, c.d_pred
    pu, pv = uv[0] - c.x0 - 0.5, uv[1] - c.y0 - 0.5     # prediction in crop pixel-centre coordinates
    r_gate = gate if gate is not None else max(GATE_MIN_PX, 0.6 * d_pred)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    near = np.hypot(xx - pu, yy - pv) <= r_gate + 0.5 * d_pred
    if not near.any() or near.all():
        return None
    peak = float(D[near].max())
    if peak < MIN_PEAK:
        return None
    ring = D[~near]
    noise = float(1.4826 * np.median(np.abs(ring - np.median(ring))))
    t_low = max(4.0 * noise, 0.12 * peak, 8.0)
    mask = (D > t_low).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    dist = np.hypot(cents[1:, 0] - pu, cents[1:, 1] - pv)
    k = int(np.argmin(dist)) + 1
    if dist[k - 1] > r_gate:
        return None
    bx, by, bw, bh, area_px = stats[k]
    if bx == 0 or by == 0 or bx + bw >= ww or by + bh >= hh:
        return None                                  # part of something bigger than the ball
    comp = (labels == k).astype(np.uint8)
    near_comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    Dm = np.where(near_comp, D, 0.0).astype(np.float32)
    contrast = float(np.median(D[comp > 0]))
    cx0, cy0 = float(cents[k, 0]), float(cents[k, 1])
    r0 = float(np.sqrt(area_px / np.pi))
    pts, levels = _edge_points(Dm, cx0, cy0, 1.6 * r0 + 4.0, t_low)
    fit = None
    if len(pts) >= 0.5 * N_RAYS:
        # every edge point, weighted by the square root of its edge level: a shadowed side's edge is still
        # placed right by the half-level rule, only noisier; dropping it would leave a partial arc, on which
        # a circle's radius trades off against its centre
        strong = levels >= 0.2 * np.median(levels)
        q = pts[strong].astype(np.float64)
        ang = np.sort(np.arctan2(q[:, 1] - cy0, q[:, 0] - cx0))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi])) if len(ang) else np.array([2 * np.pi])
        if len(q) >= 8 and 2 * np.pi - gaps.max() >= 1.5 * np.pi:  # edges all round the rim
            # undo the sphere's radial perspective stretch about the crop centre, so the outline is a circle
            if c.radial is not None and c.stretch > 1.0001:
                rad = np.asarray(c.radial)
                rel = q - [cx0, cy0]
                along = rel @ rad
                q = q - np.outer(along * (1.0 - 1.0 / c.stretch), rad)
            circ = _fit_circle(q, np.sqrt(levels[strong].astype(np.float64)))
            if circ is not None:
                ccx, ccy, r = circ
                if c.radial is not None and c.stretch > 1.0001:       # back to image coordinates
                    rad = np.asarray(c.radial)
                    along = (np.array([ccx, ccy]) - [cx0, cy0]) @ rad
                    ccx, ccy = np.array([ccx, ccy]) + along * (c.stretch - 1.0) * rad
                fit = (float(ccx), float(ccy), 2.0 * r)
    if fit is not None:
        ex, ey, d = fit
        (_, _), (w, h), _ = cv2.fitEllipse(pts)
        elong = max(w, h) / max(min(w, h), 1e-6)
        if elong > 3.5:
            return None
        u, v, kind = ex + c.x0 + 0.5, ey + c.y0 + 0.5, KIND_MINOR
        sig = 0.3 + 0.02 * d + 0.1 * (elong - 1.0) * d
    else:
        u, v = cx0 + c.x0 + 0.5, cy0 + c.y0 + 0.5
        d, kind, elong = 2.0 * r0, KIND_AREA, 1.0
        sig = 0.6 + 0.06 * d
    if np.hypot(u - uv[0], v - uv[1]) > r_gate or not (0.4 * d_pred <= d <= 2.0 * d_pred + 1.5):
        return None
    return Measurement(frame=-1, t=c.t, u=u, v=v, d=float(d), kind=kind, contrast=contrast, elongation=float(elong),
                       sig_uv=float(sig))

"""Spin from the seams.

Every frame in which the ball is large enough is resampled to a canonical patch (ball radius 20 px) and
turned into a seam map: how much darker each pixel is than a smooth shading surface fitted to the ball,
plus how red it is (the stitching). A baseball's seam is a known closed curve on the sphere; for 6000
orientations spread evenly over all rotations, the seam's visible half is projected onto the patch and the
seam map averaged along it, and the best few orientations per frame are refined.

The ball spins at a constant rate about a fixed axis during the 0.1-0.2 s it is large in the picture.
Relative rotations between consecutive frames (with the seam's three 180-degree symmetries) vote for an
angular velocity; the winner is refined jointly with the starting orientation over all frames. The view
direction of every frame comes from the calibrated camera, so the ball's own motion across the picture
does not masquerade as spin.

Limits: a seam pattern repeats after half a turn about some axes, so spin rates are unambiguous only
below a quarter turn per frame - 3600 rpm at 240 fps, 1800 rpm at 120 fps (too low for most pitches).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .geometry import Camera
from .trajectory import Trajectory

SEAM_B = 0.30                 # seam curve shape (a + b = 1, z amplitude 2 sqrt(ab))
N_SEAM = 240
N_ROT = 6000
R_C = 20.0                    # canonical ball radius, px
N_C = 48                      # canonical patch size, px
MIN_RADIUS_PX = 7.0           # balls smaller than 14 px across do not show their seams
MAX_ELONGATION = 1.5          # stronger motion blur smears the seams
MIN_FRAMES = 6
MAX_FRAMES = 20
MAX_RPM = 3600.0              # physical ceiling for a pitched baseball (and the aliasing limit at 240 fps)
MIN_FPS = 200.0
D2 = np.array([np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])])


def seam_points(b: float = SEAM_B, n: int = N_SEAM) -> np.ndarray:
    a = 1.0 - b
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([a * np.cos(t) + b * np.cos(3 * t), a * np.sin(t) - b * np.sin(3 * t),
                         2 * np.sqrt(a * b) * np.sin(2 * t)])
    return p / np.linalg.norm(p, axis=1, keepdims=True)


_SEAM = seam_points()
_ROTS = Rotation.random(N_ROT, random_state=12345).as_matrix()


@dataclass
class SpinResult:
    ok: bool
    reason: str | None = None
    rpm: float | None = None
    axis_world: list[float] | None = None          # unit vector, right-hand rule
    efficiency: float | None = None                # transverse share of the spin, 0-1
    gyro_deg: float | None = None
    active_rpm: float | None = None
    tilt_deg: float | None = None                  # Magnus direction, clock-face angle from behind the pitcher
    frames: list[int] = field(default_factory=list)
    mean_score: float | None = None                # seam-map z-score along the fitted seam, averaged over frames
    margin: float | None = None                    # best minus best alternative hypothesis, per frame
    confidence: str | None = None                  # high | medium | low
    alternative_rpm: float | None = None
    orientations: dict[int, list[list[float]]] = field(default_factory=dict)   # frame -> patch-frame rotation


def seam_map(patch: dict) -> np.ndarray:
    """Canonical seam map (N_C x N_C): z-scored darkness against a fitted shading surface plus redness,
    zero outside the inner disc."""
    img = patch["img"]
    cx, cy = patch["center"]
    s = R_C / patch["radius"]
    M = np.array([[s, 0.0, N_C / 2 - s * (cx - 0.5) - 0.5], [0.0, s, N_C / 2 - s * (cy - 0.5) - 0.5]])
    can = cv2.warpAffine(img, M, (N_C, N_C), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    b, g, r = can[..., 0], can[..., 1], can[..., 2]
    Y = 0.114 * b + 0.587 * g + 0.299 * r
    yy, xx = np.mgrid[0:N_C, 0:N_C].astype(np.float32) + 0.5
    xn, yn = (xx - N_C / 2) / R_C, (yy - N_C / 2) / R_C
    disc = xn ** 2 + yn ** 2 < 0.85 ** 2
    A = np.column_stack([np.ones(disc.sum()), xn[disc], yn[disc], xn[disc] ** 2, yn[disc] ** 2, xn[disc] * yn[disc]])
    y = Y[disc]
    w = np.ones_like(y)
    coef = np.zeros(6)
    for _ in range(4):                                # robust: seams are dark outliers
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        res = y - A @ coef
        sd = 1.4826 * np.median(np.abs(res)) + 1.0
        w = np.where(res < -2.0 * sd, 0.2, 1.0)
    shade = A @ coef
    dark = np.clip((shade - y) / np.maximum(shade, 8.0), 0.0, None)
    red = (r[disc] - 0.5 * (g[disc] + b[disc])) / np.maximum(y, 8.0)
    red = np.clip(red - np.median(red), 0.0, None)
    smap = np.zeros((N_C, N_C), np.float32)
    smap[disc] = dark + 0.5 * red
    smap = cv2.GaussianBlur(smap, (0, 0), 1.0)
    vals = smap[disc]
    sd = vals.std()
    out = np.zeros_like(smap)
    if sd > 1e-6:
        out[disc] = (vals - vals.mean()) / sd
    return out


def _score(smap: np.ndarray, rots: np.ndarray) -> np.ndarray:
    """Mean seam-map value along the visible seam for each patch-frame rotation (n, 3, 3)."""
    q = np.einsum("nij,kj->nki", rots, _SEAM)                  # (n, K, 3)
    w = np.clip((-q[..., 2] - 0.15) / 0.35, 0.0, 1.0)            # facing the camera (patch z points away)
    x = N_C / 2 + R_C * q[..., 0] - 0.5
    y = N_C / 2 + R_C * q[..., 1] - 0.5
    vals = map_coordinates(smap, [y.ravel(), x.ravel()], order=1, mode="constant", cval=0.0).reshape(x.shape)
    return (w * vals).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-6)


def _score_frames(maps: np.ndarray, rots: np.ndarray) -> np.ndarray:
    """Like _score, but one rotation per map: maps (F, N_C, N_C), rots (F, 3, 3) -> (F,)."""
    q = np.einsum("fij,kj->fki", rots, _SEAM)                  # (F, K, 3)
    w = np.clip((-q[..., 2] - 0.15) / 0.35, 0.0, 1.0)
    x = N_C / 2 + R_C * q[..., 0] - 0.5
    y = N_C / 2 + R_C * q[..., 1] - 0.5
    f = np.broadcast_to(np.arange(len(maps))[:, None], x.shape).astype(float)
    vals = map_coordinates(maps, [f.ravel(), y.ravel(), x.ravel()], order=1, mode="constant", cval=0.0).reshape(x.shape)
    return (w * vals).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-6)


def _rot_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) between two orientations, modulo the seam's symmetries."""
    best = np.pi
    for Q in D2:
        c = (np.trace(a.T @ b @ Q) - 1) / 2
        best = min(best, float(np.arccos(np.clip(c, -1, 1))))
    return best


def _candidates(smap: np.ndarray, k: int = 8) -> list[tuple[np.ndarray, float]]:
    sc = _score(smap, _ROTS)
    order = np.argsort(-sc)
    picked: list[np.ndarray] = []
    for i in order[:300]:
        if all(_rot_dist(_ROTS[i], p) > np.radians(25) for p in picked):
            picked.append(_ROTS[i])
            if len(picked) >= k:
                break
    out = []
    for R0 in picked:
        f = lambda v: -float(_score(smap, (Rotation.from_rotvec(v).as_matrix() @ R0)[None])[0])
        res = minimize(f, np.zeros(3), method="Nelder-Mead", options={"xatol": 2e-3, "fatol": 1e-4, "maxiter": 80})
        out.append((Rotation.from_rotvec(res.x).as_matrix() @ R0, -float(res.fun)))
    return sorted(out, key=lambda c: -c[1])


def _view(camera: Camera, uv) -> np.ndarray:
    """Rotation from a patch frame (x right, y down, z along the line of sight to the ball) to world."""
    d = np.array([(uv[0] - camera.cx) / camera.f, (uv[1] - camera.cy) / camera.f, 1.0])
    d /= np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s = np.linalg.norm(axis)
    Q = np.eye(3) if s < 1e-9 else Rotation.from_rotvec(axis / s * np.arctan2(s, d @ z)).as_matrix()
    return camera.R.T @ Q


def estimate_spin(patches: dict[int, dict], camera: Camera, traj: Trajectory, fps: float) -> SpinResult:
    if fps < MIN_FPS:
        return SpinResult(False, f"seams need 240 fps slow motion (this clip is {fps:.0f} fps)")
    usable = {k: p for k, p in patches.items() if p["radius"] >= MIN_RADIUS_PX and p["elongation"] <= MAX_ELONGATION}
    # the largest images carry the seams best; 20 frames span well over a full turn at any pitch's spin
    frames = sorted(sorted(usable, key=lambda k: -usable[k]["radius"])[:MAX_FRAMES])
    if len(frames) < MIN_FRAMES:
        return SpinResult(False, f"the ball is large and sharp enough to show its seams in only {len(frames)} frames "
                                 f"(need {MIN_FRAMES}); film closer or with more light")
    maps = {k: seam_map(usable[k]) for k in frames}
    views = {k: _view(camera, usable[k]["uv"]) for k in frames}
    times = {k: usable[k]["t"] for k in frames}
    cands = {k: _candidates(maps[k]) for k in frames}
    dt_frame = float(np.median(np.diff([times[k] for k in frames])))
    w_max = MAX_RPM * 2 * np.pi / 60

    # vote: angular velocities implied by candidate pairs in neighbouring frames
    votes = []                          # (omega (3,), weight, pair index)
    for pi, (a, b) in enumerate(zip(frames[:-1], frames[1:])):
        dt = times[b] - times[a]
        if dt > 2.5 * dt_frame:
            continue
        top_a = max(c[1] for c in cands[a])
        top_b = max(c[1] for c in cands[b])
        for Ra, sa in cands[a]:
            Wa = views[a] @ Ra
            for Rb, sb in cands[b]:
                for Q in D2:
                    Wb = views[b] @ Rb @ Q
                    om = Rotation.from_matrix(Wb @ Wa.T).as_rotvec() / dt
                    if np.linalg.norm(om) <= w_max:
                        votes.append((om, max(sa, 0) / max(top_a, 1e-6) * max(sb, 0) / max(top_b, 1e-6), pi))
    if not votes:
        return SpinResult(False, "no consistent seam orientation from frame to frame")
    om_all = np.array([v[0] for v in votes])
    wt = np.array([v[1] for v in votes])
    pid = np.array([v[2] for v in votes])
    strong = np.argsort(-wt)[:400]

    def support(om: np.ndarray) -> float:
        h = max(12.0, 0.06 * np.linalg.norm(om))
        k = wt * np.exp(-np.sum((om_all - om) ** 2, axis=1) / (2 * h * h))
        return float(sum(k[pid == p].max() for p in np.unique(pid)))

    sup = np.array([support(om_all[i]) for i in strong])
    order = strong[np.argsort(-sup)]

    t0 = times[frames[0]]
    stack = np.stack([maps[k] for k in frames])
    vt = np.stack([views[k].T for k in frames])
    dts = np.array([times[k] - t0 for k in frames])

    def total(params: np.ndarray, W0: np.ndarray) -> float:
        om, dv = params[:3], params[3:]
        W0r = Rotation.from_rotvec(dv).as_matrix() @ W0
        Wk = Rotation.from_rotvec(np.outer(dts, om)).as_matrix() @ W0r
        return float(_score_frames(stack, vt @ Wk).mean())

    def refine(om0: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        # starting orientation: the first frame's candidate that best agrees with om0 over all frames
        best = (-np.inf, None, None)
        for Ra, _ in cands[frames[0]]:
            for Q in D2:
                W0 = views[frames[0]] @ Ra @ Q
                v = total(np.concatenate([om0, np.zeros(3)]), W0)
                if v > best[0]:
                    best = (v, W0, om0)
        _, W0, _ = best
        res = minimize(lambda p: -total(p, W0), np.concatenate([om0, np.zeros(3)]), method="Powell",
                       options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 3000})
        om = res.x[:3]
        W0r = Rotation.from_rotvec(res.x[3:]).as_matrix() @ W0
        return -float(res.fun), om, W0r

    # refine the strongest few distinct hypotheses
    hyps: list[tuple[float, np.ndarray, np.ndarray]] = []
    for i in order:
        om = om_all[i]
        if any(np.linalg.norm(om - h[1]) < max(25.0, 0.12 * np.linalg.norm(om)) for h in hyps):
            continue
        hyps.append(refine(om))
        if len(hyps) >= 3:
            break
    hyps.sort(key=lambda h: -h[0])
    best_score, om, W0 = hyps[0]
    if np.linalg.norm(om) > w_max:
        return SpinResult(False, "the seam fit points to a spin faster than a pitch can have")
    alt = next((h for h in hyps[1:] if np.linalg.norm(h[1] - om) > max(30.0, 0.15 * np.linalg.norm(om))), None)
    margin = best_score - alt[0] if alt else None

    rpm = float(np.linalg.norm(om) * 60 / (2 * np.pi))
    axis = om / max(np.linalg.norm(om), 1e-12)
    t_mid = 0.5 * (times[frames[0]] + times[frames[-1]])
    vh = traj.vel(t_mid)
    vh = vh / np.linalg.norm(vh)
    om_perp = om - np.dot(om, vh) * vh
    eff = float(np.linalg.norm(om_perp) / max(np.linalg.norm(om), 1e-12))
    gyro = float(np.degrees(np.arcsin(min(1.0, abs(np.dot(axis, vh))))))
    magnus = np.cross(om_perp, vh)
    tilt = float(np.degrees(np.arctan2(-magnus[0], magnus[2])) % 360.0)
    conf = "low"
    if best_score >= 0.9 and (margin is None or margin >= 0.25):
        conf = "high"
    elif best_score >= 0.6 and (margin is None or margin >= 0.1):
        conf = "medium"
    orient = {}
    for k in frames:
        Wk = Rotation.from_rotvec(om * (times[k] - t0)).as_matrix() @ W0
        orient[int(k)] = (views[k].T @ Wk).tolist()
    return SpinResult(True, None, rpm=rpm, axis_world=axis.tolist(), efficiency=eff, gyro_deg=gyro,
                      active_rpm=rpm * eff, tilt_deg=tilt, frames=[int(k) for k in frames], mean_score=best_score,
                      margin=margin, confidence=conf,
                      alternative_rpm=None if alt is None else float(np.linalg.norm(alt[1]) * 60 / (2 * np.pi)),
                      orientations=orient)

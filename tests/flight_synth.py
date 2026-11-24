"""Synthetic ball-flight clips with known ground truth.

A pitch is integrated with gravity, drag and Magnus lift (Nathan's lift model) from a given release; a
seamed, spinning ball is ray-traced into a perspective camera over a textured field (home plate, mound,
rubber), with a crude moving pitcher as a distractor, exposure motion blur, sensor noise and H.264 4:2:0
compression. The seam curve and stitch texture deliberately differ from the estimator's seam model, so the
spin tests do not grade the estimator on its own template.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pitchlab.flight.geometry import FT, IN, MPH, Camera, plate_points, rubber_points
from pitchlab.video_io import H264Writer

G = 9.80665
BALL_R = 1.45 * IN
BALL_M = 0.145
RHO = 1.204


def lift_coefficient(s: float) -> float:
    return 1.5 * s if s < 0.1 else 0.09 + 0.6 * s


@dataclass
class Pitch:
    t: np.ndarray            # (N,) s, 0 = release
    pos: np.ndarray          # (N, 3) m
    vel: np.ndarray          # (N, 3) m/s
    omega: np.ndarray        # (3,) rad/s, world frame (constant)
    r0: Rotation             # ball orientation at release (ball -> world)

    def at(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.atleast_1d(t)
        p = np.column_stack([np.interp(t, self.t, self.pos[:, k]) for k in range(3)])
        v = np.column_stack([np.interp(t, self.t, self.vel[:, k]) for k in range(3)])
        return p, v

    def orientation(self, t: float) -> Rotation:
        return Rotation.from_rotvec(self.omega * t) * self.r0

    def plate_time(self, y: float = 17 * IN) -> float:
        k = int(np.argmax(self.pos[:, 1] < y))
        a, b = self.pos[k - 1, 1], self.pos[k, 1]
        return float(self.t[k - 1] + (a - y) / (a - b) * (self.t[k] - self.t[k - 1]))


def simulate_pitch(release: np.ndarray, speed_mph: float, vert_deg: float, horiz_deg: float, spin_rpm: float,
                   spin_axis: np.ndarray, cd: float = 0.33, rho: float = RHO, dt: float = 2e-5,
                   seed: int = 0) -> Pitch:
    """Release at `release` (m) moving toward the plate at `speed_mph`, `vert_deg` above horizontal and
    `horiz_deg` toward first base. Spin `spin_rpm` about `spin_axis` (world, right-hand rule)."""
    v = speed_mph * MPH
    el, az = np.radians(vert_deg), np.radians(horiz_deg)
    v0 = v * np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    axis = np.asarray(spin_axis, float)
    omega = spin_rpm * 2 * np.pi / 60.0 * axis / np.linalg.norm(axis)
    k = rho * np.pi * BALL_R ** 2 / (2 * BALL_M)

    def acc(vel):
        s = np.linalg.norm(vel)
        vh = vel / s
        a = -k * cd * s * vel + np.array([0.0, 0.0, -G])
        w_perp = omega - np.dot(omega, vh) * vh
        wp = np.linalg.norm(w_perp)
        if wp > 1e-9:
            d = np.cross(w_perp, vh)
            a = a + k * lift_coefficient(BALL_R * wp / s) * s * s * d / np.linalg.norm(d)
        return a

    ts, ps, vs = [0.0], [np.asarray(release, float)], [v0]
    p, vel, t = ps[0].copy(), v0.copy(), 0.0
    while p[1] > -8.0 and t < 2.0:          # well past a camera behind the plate
        k1v = acc(vel); k1p = vel
        k2v = acc(vel + 0.5 * dt * k1v); k2p = vel + 0.5 * dt * k1v
        k3v = acc(vel + 0.5 * dt * k2v); k3p = vel + 0.5 * dt * k2v
        k4v = acc(vel + dt * k3v); k4p = vel + dt * k3v
        p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        vel = vel + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
        ts.append(t); ps.append(p.copy()); vs.append(vel.copy())
    r0 = Rotation.random(random_state=seed)
    return Pitch(t=np.array(ts), pos=np.array(ps), vel=np.array(vs), omega=omega, r0=r0)


def look_at(position, target, hfov_deg: float, size: tuple[int, int]) -> Camera:
    w, h = size
    f = w / 2.0 / np.tan(np.radians(hfov_deg) / 2)
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ np.asarray(position, float)
    return Camera(f=f, cx=w / 2.0, cy=h / 2.0, rvec=cv2.Rodrigues(R)[0].ravel(), tvec=t, width=w, height=h)


# ------------------------------------------------------------------------------------ seams (renderer's own)
def seam_curve(b: float = 0.27, n: int = 6000) -> np.ndarray:
    a = 1.0 - b
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([a * np.cos(t) + b * np.cos(3 * t), a * np.sin(t) - b * np.sin(3 * t),
                            2 * np.sqrt(a * b) * np.sin(2 * t)])


class BallTexture:
    """White leather, red double-row stitching (108 stitches) around a groove."""

    def __init__(self, b: float = 0.27, logo: bool = False):
        pts = seam_curve(b)
        self.tree = cKDTree(pts)
        self.n = len(pts)
        self.logo = logo

    def colour(self, local: np.ndarray) -> np.ndarray:
        """BGR albedo (0-1) at unit ball-local surface points (N, 3)."""
        d, idx = self.tree.query(local)
        phase = (idx / self.n * 108.0) % 1.0
        stitch = (d < 0.115) & (np.abs(phase - 0.5) < 0.30 + 0.9 * d)   # V-shaped stitches
        groove = d < 0.022
        col = np.tile(np.array([0.86, 0.90, 0.92]), (len(local), 1))
        col[stitch] = [0.16, 0.14, 0.72]
        col[groove] = [0.30, 0.32, 0.36]
        if self.logo:
            lg = np.linalg.norm(local - np.array([0.0, 0.0, 1.0]), axis=1) < 0.22
            col[lg & ~stitch] = [0.25, 0.25, 0.25]
        return col


# ------------------------------------------------------------------------------------ scene
def field_background(cam: Camera, distance_ft: float, mound_in: float, seed: int = 1) -> np.ndarray:
    w, h = cam.width, cam.height
    rng = np.random.default_rng(seed)
    res = 0.01                                   # m per texture pixel
    x0, x1, y0, y1 = -9.0, 9.0, -9.0, distance_ft * FT + 9.0
    tw, th = int((x1 - x0) / res), int((y1 - y0) / res)
    noise = cv2.resize(rng.normal(0, 1, (th // 40 + 2, tw // 40 + 2)).astype(np.float32), (tw, th))
    fine = rng.normal(0, 1, (th, tw)).astype(np.float32)
    tex = np.empty((th, tw, 3), np.float32)
    tex[:] = (52, 118, 62)
    tex += (8 * noise + 6 * fine)[..., None] * np.array([0.6, 1.0, 0.6], np.float32)

    def to_px(xy):
        xy = np.atleast_2d(xy)
        return np.column_stack([(xy[:, 0] - x0) / res, (y1 - xy[:, 1]) / res])

    dirt = (92, 128, 168)
    c_home = to_px([0.0, 0.0])[0]
    c_mound = to_px([0.0, distance_ft * FT - 0.46])[0]
    cv2.circle(tex, tuple(int(v) for v in c_home), int(3.96 / res), dirt, -1)
    cv2.circle(tex, tuple(int(v) for v in c_mound), int(2.74 / res), dirt, -1)
    tex += 5 * fine[..., None]
    tex = np.clip(tex, 0, 255).astype(np.uint8)

    R, t = cam.R, cam.tvec
    H = cam.K @ np.column_stack([R[:, 0], R[:, 1], t])
    T = np.array([[res, 0.0, x0], [0.0, -res, y1], [0.0, 0.0, 1.0]])
    half = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])   # OpenCV: pixel centres at integers
    ground = cv2.warpPerspective(tex, half @ H @ T, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(52, 118, 62))
    # sky where the pixel ray does not hit the ground in front of the camera
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = cam.rays(np.column_stack([uu.ravel(), vv.ravel()])).reshape(h, w, 3)
    sky = rays[..., 2] > -0.012
    grad = np.clip((rays[..., 2] + 0.05) * 2.5, 0, 1)[..., None]
    sky_col = (1 - grad) * np.array([225, 215, 200]) + grad * np.array([220, 170, 120])
    img = np.where(sky[..., None], sky_col, ground).astype(np.uint8)
    # chalk and the plate drawn in the image (exact coverage), not in the 1 cm texture
    for sx in (-1, 1):            # batter's boxes, 3 in lines
        xa, xb, ya, yb = sx * 0.31, sx * 1.53, -0.6, 1.23
        lw = 3 * IN
        for q in ([[xa, ya], [xb, ya], [xb, ya + lw], [xa, ya + lw]], [[xa, yb - lw], [xb, yb - lw], [xb, yb], [xa, yb]],
                  [[xa, ya], [xa + sx * lw, ya], [xa + sx * lw, yb], [xa, yb]], [[xb - sx * lw, ya], [xb, ya], [xb, yb], [xb - sx * lw, yb]]):
            _fill_exact(img, cam, np.column_stack([np.array(q), np.zeros(4)]), (222, 222, 222))
    _fill_exact(img, cam, plate_points(), (238, 238, 238))
    # rubber: front face and top face of a 24 x 6 in slab standing 1.5 in out of the mound
    rb = rubber_points(distance_ft, mound_in)
    top = np.vstack([rb, rb[::-1] + [0.0, 6 * IN, 0.0]])
    front = np.vstack([rb, rb[::-1] - [0.0, 0.0, 1.5 * IN]])
    for poly, col in ((front, (205, 205, 205)), (top, (242, 242, 242))):
        _fill_exact(img, cam, poly, col)
    return img


def _fill_exact(img: np.ndarray, cam: Camera, poly3d: np.ndarray, colour) -> None:
    """Blend a projected convex polygon into img with exact (8x supersampled) pixel coverage."""
    uv, z = cam.project(poly3d)
    if np.any(z <= 0):
        return
    s = 8
    lo = np.maximum(np.floor(uv.min(axis=0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(uv.max(axis=0)).astype(int) + 2, [img.shape[1], img.shape[0]])
    if np.any(hi <= lo):
        return
    m = np.zeros(((hi[1] - lo[1]) * s, (hi[0] - lo[0]) * s), np.uint8)
    pts = np.round(((uv - lo) * s - 0.5) * 16).astype(np.int32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 255, lineType=cv2.LINE_8, shift=4)
    cov = cv2.resize(m, (hi[0] - lo[0], hi[1] - lo[1]), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    roi = img[lo[1]:hi[1], lo[0]:hi[0]].astype(np.float32)
    img[lo[1]:hi[1], lo[0]:hi[0]] = np.clip(roi * (1 - cov) + np.array(colour, np.float32) * cov, 0, 255).astype(np.uint8)


def _draw_capsule(img, cam: Camera, a, b, radius_m, colour):
    uv, z = cam.project(np.vstack([a, b]))
    if np.any(z <= 0.2):
        return
    th = max(1, int(round(2 * cam.f * radius_m / z.mean())))
    cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), colour, th, cv2.LINE_AA)


@dataclass
class Scene:
    cam: Camera
    pitch: Pitch
    distance_ft: float = 60.5
    mound_in: float = 10.0
    fps: float = 240.0
    t_start: float = -0.15           # clip start, s relative to release
    t_end: float | None = None       # default: 0.08 s after the plate
    exposure_s: float = 1 / 2000
    noise: float = 2.0
    pitcher: bool = True
    texture: BallTexture = field(default_factory=BallTexture)
    throws: str = "R"
    light: np.ndarray | None = None  # world direction toward the sun; default high behind the camera
    lead_empty_s: float = 0.0        # empty field (no pitcher, no ball) before t_start, as before he steps on
    catch_y: float | None = None     # m: the ball disappears into the catcher's glove here (None: flies on)


@dataclass
class Truth:
    frame_times: np.ndarray          # s relative to release, per written frame
    release_frame: float             # fractional frame index of release
    ball_uv: np.ndarray              # (F, 2) projected centre at mid-exposure (NaN in the hand before release)
    ball_d: np.ndarray               # (F,) area-equivalent diameter, px
    plate_px: np.ndarray             # (5, 2) true image positions of the plate corners (PLATE_XY order)
    rubber_px: np.ndarray            # (2, 2)
    scene: Scene = None


def _hand(scene: Scene, t: float) -> np.ndarray:
    """Throwing hand: rides the ball's line into release, then follows through down and across."""
    p_rel, v_rel = scene.pitch.at(0.0)
    p_rel, v_rel = p_rel[0], v_rel[0]
    if t <= 0:
        return p_rel + v_rel * max(t, -0.045)
    side = -1.0 if scene.throws == "R" else 1.0
    s = min(t, 0.12)
    return p_rel + np.array([-side * 6.0 * s, -9.0 * s, -10.0 * s])


def render_clip(path: Path, scene: Scene, container_fps: float | None = None) -> Truth:
    cam, pitch = scene.cam, scene.pitch
    bg = field_background(cam, scene.distance_ft, scene.mound_in)
    t_plate = pitch.plate_time()
    t_end = scene.t_end if scene.t_end is not None else t_plate + 0.08
    t0 = scene.t_start - scene.lead_empty_s
    n = int(np.floor((t_end - t0) * scene.fps)) + 1
    times = t0 + np.arange(n) / scene.fps
    rng = np.random.default_rng(7)
    noise_bank = [rng.normal(0, scene.noise, (cam.height, cam.width, 3)).astype(np.float32) for _ in range(3)]
    if scene.light is None:                           # sun high behind the camera's left shoulder
        light = -cam.R[2] - 0.4 * cam.R[0] + np.array([0.0, 0.0, 1.6])
    else:
        light = np.asarray(scene.light, float)
    light = light / np.linalg.norm(light)
    side = -1.0 if scene.throws == "R" else 1.0
    uv_true = np.full((n, 2), np.nan)
    d_true = np.full(n, np.nan)
    with H264Writer(path, (cam.width, cam.height), container_fps or scene.fps) as wr:
        for k, tk in enumerate(times):
            img = bg.astype(np.float32)
            empty = tk < scene.t_start
            if scene.pitcher and not empty:
                frame = img.copy()
                prog = np.clip((tk + 0.15) / 0.3, 0, 1)
                y_hip = scene.distance_ft * FT - 0.35 - 1.1 * prog
                hip = np.array([side * 0.25 - side * 0.3 * prog, y_hip, 0.95 + scene.mound_in * IN * 0.5])
                shoulder_c = hip + np.array([0.0, -0.25 * prog, 0.52])
                sh = shoulder_c + np.array([side * 0.2, 0.0, 0.0])
                foot_l = np.array([-side * 0.1, scene.distance_ft * FT - 0.3 - 1.5 * prog, 0.08])
                foot_r = np.array([side * 0.3, scene.distance_ft * FT, 0.12])
                col = (110, 105, 100)
                _draw_capsule(frame, cam, hip, shoulder_c, 0.17, col)
                _draw_capsule(frame, cam, hip, foot_l, 0.08, (200, 200, 200))
                _draw_capsule(frame, cam, hip, foot_r, 0.08, (200, 200, 200))
                _draw_capsule(frame, cam, shoulder_c + [0, 0, 0.12], shoulder_c + [0, 0, 0.3], 0.11, (60, 60, 70))
                _draw_capsule(frame, cam, sh, _hand(scene, tk), 0.045, col)
                img = frame
            # ball: in the hand before release, in flight after
            sub = np.linspace(-0.5, 0.5, 5) * scene.exposure_s
            centres = []
            for ds in sub:
                ts = tk + ds
                if ts <= 0:
                    centres.append((_hand(scene, ts) + np.array([0.0, -0.02, 0.0]), pitch.orientation(0.0)))
                else:
                    pos = pitch.at(ts)[0][0]
                    if scene.catch_y is not None and pos[1] < scene.catch_y:
                        continue
                    centres.append((pos, pitch.orientation(ts)))
            uv_c, z_c = cam.project(np.array([c for c, _ in centres])) if centres else (None, None)
            if not empty and centres and np.all(z_c > 0.3):
                r_px = cam.f * BALL_R / z_c.min() * 1.4 + 2
                lo = np.floor(uv_c.min(axis=0) - r_px).astype(int)
                hi = np.ceil(uv_c.max(axis=0) + r_px).astype(int)
                lo = np.maximum(lo, 0)
                hi = np.minimum(hi, [cam.width - 1, cam.height - 1])
                if np.all(hi > lo):
                    ss = 3
                    gx = lo[0] + (np.arange((hi[0] - lo[0] + 1) * ss) + 0.5) / ss
                    gy = lo[1] + (np.arange((hi[1] - lo[1] + 1) * ss) + 0.5) / ss
                    U, V = np.meshgrid(gx, gy)
                    rays = cam.rays(np.column_stack([U.ravel(), V.ravel()]))
                    C = cam.center
                    acc_c = np.zeros((len(rays), 3))
                    acc_a = np.zeros(len(rays))
                    for P, rot in centres:
                        oc = C - P
                        bq = rays @ oc
                        disc = bq * bq - (oc @ oc - BALL_R ** 2)
                        hit = disc > 0
                        if not hit.any():
                            continue
                        s = -bq[hit] - np.sqrt(disc[hit])
                        X = C + s[:, None] * rays[hit]
                        nrm = (X - P) / BALL_R
                        local = rot.inv().apply(nrm)
                        alb = scene.texture.colour(local)
                        shade = 0.42 + 0.62 * np.clip(nrm @ light, 0, None)
                        acc_c[hit] += alb * shade[:, None] * 255.0
                        acc_a[hit] += 1.0
                    acc_c /= len(centres)
                    acc_a /= len(centres)
                    hh, ww = (hi[1] - lo[1] + 1), (hi[0] - lo[0] + 1)
                    col = acc_c.reshape(hh, ss, ww, ss, 3).mean(axis=(1, 3))
                    alpha = acc_a.reshape(hh, ss, ww, ss).mean(axis=(1, 3))
                    roi = img[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ball = np.where(alpha[..., None] > 0, col / np.maximum(alpha[..., None], 1e-6), 0)
                    roi[:] = roi * (1 - alpha[..., None]) + ball * alpha[..., None]
            if tk > 0:
                P = pitch.at(tk)[0]
                uv, z = cam.project(P)
                caught = scene.catch_y is not None and P[0][1] < scene.catch_y
                inside = (not caught and z[0] > 0.5 and -5 <= uv[0, 0] <= cam.width + 5 and -5 <= uv[0, 1] <= cam.height + 5)
                if inside:
                    uv_true[k] = uv[0]
                    rho = np.linalg.norm(P[0] - cam.center)
                    d_true[k] = 2 * cam.f * BALL_R * np.sqrt(rho) / z[0] ** 1.5
            shift = (int(rng.integers(cam.height)), int(rng.integers(cam.width)))
            img = img + np.roll(noise_bank[k % 3], shift, axis=(0, 1))
            wr.write(np.clip(img, 0, 255).astype(np.uint8))
    plate_uv, _ = cam.project(plate_points())
    rub_uv, _ = cam.project(rubber_points(scene.distance_ft, scene.mound_in))
    return Truth(frame_times=times, release_frame=float(-t0 * scene.fps), ball_uv=uv_true, ball_d=d_true,
                 plate_px=plate_uv, rubber_px=rub_uv, scene=scene)


# ------------------------------------------------------------------------------------ standard setups
def fastball(seed: int = 0, throws: str = "R") -> Pitch:
    side = -1.0 if throws == "R" else 1.0
    release = np.array([side * 0.55, (60.5 - 6.2) * FT, 1.80])
    # backspin with arm-side tilt (~1:00 from the pitcher's view) and some gyro
    axis = np.array([-1.0, side * 0.25, side * 0.45])
    return simulate_pitch(release, 90.0, -1.6, side * -2.6, 2300.0, axis, seed=seed)


def curveball(seed: int = 3, throws: str = "R") -> Pitch:
    side = -1.0 if throws == "R" else 1.0
    release = np.array([side * 0.55, (60.5 - 5.8) * FT, 1.85])
    axis = np.array([0.9, side * 0.3, side * -0.55])       # topspin with glove-side tilt
    return simulate_pitch(release, 76.0, 2.2, side * -0.6, 2650.0, axis, seed=seed)


def behind_plate_camera(size=(1920, 1080)) -> Camera:
    return look_at([0.25, -4.6, 1.35], [0.0, 8.0, 0.45], 64.0, size)


def behind_pitcher_camera(size=(1920, 1080)) -> Camera:
    return look_at([-1.0, 60.5 * FT + 5.0, 2.6], [0.0, 9.0, 0.0], 60.0, size)


def release_camera(size=(1920, 1080), throws: str = "R") -> Camera:
    """PitchLab's set-up: a phone a couple of metres from the release, on the arm side and slightly behind,
    aimed down the line so the ball is large as it leaves the hand and the plate is in the far picture."""
    side = -1.0 if throws == "R" else 1.0
    return look_at([side * 1.6, 60.5 * FT + 0.25, 1.7], [0.0, 8.0, 0.9], 65.0, size)


def pitchlab_camera(size=(1080, 1920), throws: str = "R", behind_ft: float = 4.0, side_ft: float = 1.5,
                    height_ft: float = 4.8, hfov_deg: float = 36.0) -> Camera:
    """PitchLab's own set-up, read off its app's screenshots: a phone upright (portrait) a few feet behind the
    rubber on the throwing-arm side, about chest high, aimed at the catcher. The ball is large as it leaves
    the hand; home plate is a thin sliver about 40 px wide far down the picture."""
    side = -1.0 if throws == "R" else 1.0
    return look_at([side * side_ft * FT, (60.5 + behind_ft) * FT, height_ft * FT], [0.0, 0.2, 0.55], hfov_deg, size)

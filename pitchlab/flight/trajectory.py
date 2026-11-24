"""The pitch trajectory and the numbers derived from it.

Model: constant acceleration in three dimensions (nine parameters, the PITCHf/x model, which follows a
full drag-and-lift trajectory to within about half an inch over a pitch). It is fitted to the ball's image
positions through the calibrated camera. A single camera cannot tell a pitch from a scaled copy of it
about the camera centre (every ray is the same), so the scale comes from:

* the ball's apparent size (a 2.9 in ball's image diameter fixes its distance), with a size factor shared
  by all frames (5% prior) so a systematic size error is not averaged away as if it were noise;
* drag: the deceleration along the flight is rho*Cd*A/(2m)*v^2, which a scaled copy violates (Cd 0.33,
  given +-18%);
* when the release is visible as a moving ball, the release distance from the rubber (extension), as a
  loose prior;
* filmed from behind the release (PitchLab's set-up), where the ball flies straight away from the lens and
  its far half is too small to size, the catch: the track ends where the ball vanishes into the glove, just
  behind home plate.

Movement is reported the TrackMan way: from release to the front of home plate, relative to a spinless
pitch (gravity and drag removed from the fitted acceleration). The spin axis implied by movement and an
active-spin estimate (Nathan's lift model) come from the same Magnus acceleration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .geometry import FT, G, IN, MPH, PLATE_FRONT_Y, PLATE_HALF_WIDTH, Camera, camera_view

BALL_RADIUS = 1.45 * IN
BALL_MASS = 0.145
BALL_AREA = np.pi * BALL_RADIUS ** 2
CD_NOMINAL = 0.33
CD_SIGMA_FRAC = 0.18
SIZE_SIGMA_FLOOR = 0.025         # relative, random from frame to frame
SIZE_SIGMA_PX = 0.35
SIZE_SCALE_SIGMA = 0.05          # systematic size error shared by all frames (lighting, edge judgement)
SIZE_MIN_PX = 9.0                # smaller images are not used for distance (a shadowed half vanishes)
EXTENSION_SIGMA_FT = 0.8
DEFAULT_EXTENSION_FT = 6.0       # at 60.5 ft; scaled with the pitching distance for youth fields
LIFT_MODEL = "Nathan 2008: C_L = 1.5 S (S < 0.1), 0.09 + 0.6 S"


def air_density(altitude_ft: float = 0.0) -> float:
    return 1.204 * float(np.exp(-altitude_ft * FT / 8400.0))


def drag_k(altitude_ft: float = 0.0) -> float:
    """Drag deceleration per (m/s)^2 at the nominal drag coefficient."""
    return air_density(altitude_ft) * CD_NOMINAL * BALL_AREA / (2 * BALL_MASS)


@dataclass
class Setup:
    distance_ft: float = 60.5
    mound_in: float = 10.0
    throws: str = "R"                       # R | L
    zone_bottom_in: float = 18.0
    zone_top_in: float = 42.0
    altitude_ft: float = 0.0
    pitcher_height_in: float | None = None

    @property
    def extension_prior_ft(self) -> float:
        if self.pitcher_height_in:
            return 0.95 * self.pitcher_height_in / 12.0
        return DEFAULT_EXTENSION_FT * self.distance_ft / 60.5


@dataclass
class Track:
    """Ball measurements: one row per frame in which the ball was found."""
    frame: np.ndarray
    t: np.ndarray
    uv: np.ndarray                 # (N, 2) px
    d: np.ndarray                  # (N,) image diameter, px (NaN = not measured)
    sig_uv: np.ndarray             # (N,) px
    d_kind: np.ndarray | None = None   # (N,) blob.KIND_MINOR (minor axis) or KIND_AREA (area-equivalent)

    def __post_init__(self) -> None:
        if self.d_kind is None:
            self.d_kind = np.zeros(len(self.t), int)

    def __len__(self) -> int:
        return len(self.t)

    def subset(self, keep: np.ndarray) -> "Track":
        return Track(self.frame[keep], self.t[keep], self.uv[keep], self.d[keep], self.sig_uv[keep], self.d_kind[keep])


@dataclass
class Trajectory:
    t_ref: float
    p: np.ndarray                  # position at t_ref, m
    v: np.ndarray                  # velocity at t_ref, m/s
    a: np.ndarray                  # constant acceleration, m/s^2

    @classmethod
    def from_theta(cls, theta: np.ndarray, t_ref: float) -> "Trajectory":
        return cls(t_ref, np.array(theta[0:3]), np.array(theta[3:6]), np.array(theta[6:9]))

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.p, self.v, self.a])

    def pos(self, t) -> np.ndarray:
        dt = np.asarray(t, float) - self.t_ref
        return self.p + np.multiply.outer(dt, self.v) + 0.5 * np.multiply.outer(dt * dt, self.a)

    def vel(self, t) -> np.ndarray:
        dt = np.asarray(t, float) - self.t_ref
        return self.v + np.multiply.outer(dt, self.a)

    def time_at_y(self, y: float) -> float | None:
        """First time the ball, travelling toward the plate, reaches field coordinate y."""
        a, b, c = 0.5 * self.a[1], self.v[1], self.p[1] - y
        if abs(a) < 1e-9:
            return None if b == 0 else self.t_ref - c / b
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        roots = [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]
        # the physical root: where the velocity still points toward the plate
        good = [r for r in roots if b + 2 * a * r < 0]
        return None if not good else self.t_ref + min(good, key=abs)

    def as_dict(self) -> dict:
        return {"t_ref": self.t_ref, "p": self.p.tolist(), "v": self.v.tolist(), "a": self.a.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(float(d["t_ref"]), np.array(d["p"]), np.array(d["v"]), np.array(d["a"]))


@dataclass
class Fit:
    traj: Trajectory
    rms_px: float                  # image residual RMS of the inliers
    inliers: np.ndarray            # (N,) bool
    size_ratio: float              # median measured / modelled diameter (scale check)
    cov: np.ndarray | None         # (9, 9) parameter covariance from measurement noise
    extension_prior_used: bool = False
    size_scale: float = 1.0        # fitted systematic size factor (measured / modelled)
    catch_prior_used: bool = False
    notes: list[str] = field(default_factory=list)


def model_diameter(P: np.ndarray, camera: Camera, kind: np.ndarray | int = 0) -> np.ndarray:
    """Image diameter (px) of the ball at world points P (N, 3). A sphere off the optical axis images as an
    ellipse stretched radially by 1/cos(theta): its minor (tangential) axis is 2 f r / Z exactly, its
    area-equivalent diameter 2 f r sqrt(rho) / Z^1.5 (Z depth along the axis, rho distance from the camera).
    kind 1 = minor axis, 0 = area-equivalent (blob.KIND_MINOR / KIND_AREA)."""
    pc = camera.to_cam(np.atleast_2d(P))
    z = np.maximum(pc[:, 2], 1e-3)
    rho = np.linalg.norm(pc, axis=1)
    return np.where(np.asarray(kind) == 1, 2 * camera.f * BALL_RADIUS / z,
                    2 * camera.f * BALL_RADIUS * np.sqrt(rho) / z ** 1.5)


SMALL_BALL_PX = 10.0             # ball images smaller than this are centred to about 1 px, not the
SMALL_BALL_SIG_PX = 0.8          # ~0.6 px their edge fit states (measured on synthetic clips): added in quadrature
HUBER_K = 2.5                    # image / size residuals beyond this many SDs are down-weighted


def _residuals(theta, t_ref, t, uv, sig_uv, d, sig_d, use_d, d_minor, cam_R, cam_t, cam_f, cam_c, cam_C, k_drag,
               t_mid, anchors, scale_sigma, w_img, w_size):
    """Measurement residuals carry Huber weights (outlier detections); the physics priors stay quadratic so
    a bad detection cannot buy an impossible deceleration. theta[9] is the log of a size scale shared by
    every frame: the systematic part of the size error (blur, compression, how the ball's edge is judged)
    must not be averaged away over a hundred frames as if it were noise."""
    p0, v0, a, log_k = theta[0:3], theta[3:6], theta[6:9], theta[9]
    dt = t - t_ref
    P = p0 + dt[:, None] * v0 + 0.5 * (dt * dt)[:, None] * a
    pc = P @ cam_R.T + cam_t
    z = pc[:, 2]
    zc = np.maximum(z, 0.05)
    uvm = cam_f * pc[:, :2] / zc[:, None] + cam_c
    r_img = ((uvm - uv) / sig_uv[:, None] * w_img[:, None]).ravel()
    behind = np.minimum(z - 0.05, 0.0) * 1e3
    rho = np.linalg.norm(P - cam_C, axis=1)
    dm = np.exp(log_k) * 2 * cam_f * BALL_RADIUS * np.where(d_minor, 1.0 / zc, np.sqrt(rho) / zc ** 1.5)
    r_size = np.where(use_d, (dm - np.nan_to_num(d)) / sig_d, 0.0) * w_size
    v_mid = v0 + (t_mid - t_ref) * a
    s = np.linalg.norm(v_mid)
    a_par = (a + np.array([0.0, 0.0, G])) @ v_mid / max(s, 1e-6)
    r_drag = (a_par + k_drag * s * s) / (CD_SIGMA_FRAC * k_drag * s * s + 1e-6)
    out = [r_img, r_size, behind, [r_drag, log_k / scale_sigma]]
    for t_e, y_mean, y_sig in anchors:
        y_e = p0[1] + (t_e - t_ref) * v0[1] + 0.5 * (t_e - t_ref) ** 2 * a[1]
        out.append([(y_e - y_mean) / y_sig])
    return np.concatenate(out)


def initial_guess(track: Track, camera: Camera, setup: Setup) -> Trajectory:
    """Distances from the apparent sizes give rough 3D points; a weighted least-squares quadratic through
    them seeds the fit. Falls back to a nominal 80 mph pitch from the release area to the plate."""
    t = track.t
    t_ref = float(np.median(t))
    ok = np.isfinite(track.d) & (track.d >= 3.0)
    traj = None
    if ok.sum() >= 4:
        rays = camera.rays(track.uv[ok])
        cos_t = rays @ camera.R[2]
        # distance from the camera: minor axis d = 2 f r / (rho cos), area-equivalent d = 2 f r / (rho cos^1.5)
        power = np.where(track.d_kind[ok] == 1, 1.0, 1.5)
        rho = 2 * camera.f * BALL_RADIUS / (track.d[ok] * cos_t ** power)
        X = camera.center + rho[:, None] * rays
        dt = t[ok] - t_ref
        A = np.column_stack([np.ones_like(dt), dt, 0.5 * dt * dt])
        w = np.sqrt(track.d[ok])
        coef = np.linalg.lstsq(A * w[:, None], X * w[:, None], rcond=None)[0]
        traj = Trajectory(t_ref, coef[0], coef[1], coef[2])
        speed = np.linalg.norm(traj.vel(t_ref)) / MPH
        if not (25.0 <= speed <= 120.0) or traj.v[1] >= 0:
            traj = None
    if traj is None:
        p_rel = np.array([0.0, (setup.distance_ft - setup.extension_prior_ft) * FT, 1.8])
        p_plate = np.array([0.0, PLATE_FRONT_Y, 0.8])
        dist = np.linalg.norm(p_plate - p_rel)
        v = (p_plate - p_rel) / dist * 80 * MPH
        # place the nominal pitch so its midpoint is seen at the median observation's ray at mid-distance
        mid_ray = camera.rays(track.uv[[len(track) // 2]])[0]
        mid = 0.5 * (p_rel + p_plate)
        depth = np.dot(mid - camera.center, mid_ray)
        p_ref = camera.center + depth * mid_ray
        traj = Trajectory(t_ref, p_ref, v, np.array([0.0, 0.0, -0.5 * G]))
    return traj


def fit_trajectory(track: Track, camera: Camera, setup: Setup, init: Trajectory | None = None,
                   ext: tuple[float, float, float] | None = None, robust: bool = True,
                   size_scale_sigma: float = SIZE_SCALE_SIGMA, catch: tuple[float, float, float] | None = None) -> Fit:
    """ext and catch: (t, y_mean, y_sigma) priors on where the ball is along the line at time t."""
    if len(track) < 5:
        raise ValueError("at least five ball positions are needed")
    t = track.t
    init = init or initial_guess(track, camera, setup)
    t_ref = init.t_ref
    d = track.d
    n = len(t)
    use_d = np.isfinite(d) & (d >= SIZE_MIN_PX)
    sig_d = SIZE_SIGMA_PX + SIZE_SIGMA_FLOOR * np.nan_to_num(d, nan=1.0)
    kind = track.d_kind
    w_img, w_size = np.ones(n), np.ones(n)
    base = (t_ref, t, track.uv, track.sig_uv, d, sig_d, use_d, kind == 1, camera.R, camera.tvec, camera.f,
            np.array([camera.cx, camera.cy]), camera.center, drag_k(setup.altitude_ft), float(0.5 * (t.min() + t.max())),
            tuple(a for a in (ext, catch) if a is not None), size_scale_sigma)
    theta = np.append(init.theta, 0.0)

    small = ~(np.nan_to_num(d, nan=0.0) >= SMALL_BALL_PX)
    sig_uv = np.where(small, np.hypot(track.sig_uv, SMALL_BALL_SIG_PX), track.sig_uv)
    for _ in range(4 if robust else 1):
        args = base[:3] + (sig_uv,) + base[4:] + (w_img, w_size)
        sol = least_squares(_residuals, theta, args=args, method="trf", x_scale="jac", max_nfev=2000)
        theta = sol.x
        tr = Trajectory.from_theta(theta, t_ref)
        P = tr.pos(t)
        uvm, _ = camera.project(P)
        e_img = np.linalg.norm(uvm - track.uv, axis=1) / sig_uv
        e_size = np.where(use_d, np.abs(np.exp(theta[9]) * model_diameter(P, camera, kind) - np.nan_to_num(d)) / sig_d,
                          0.0)
        w_img = np.sqrt(np.where(e_img <= HUBER_K, 1.0, HUBER_K / np.maximum(e_img, 1e-9)))
        w_size = np.sqrt(np.where(e_size <= HUBER_K, 1.0, HUBER_K / np.maximum(e_size, 1e-9)))
    traj = Trajectory.from_theta(theta, t_ref)
    P = traj.pos(t)
    uvm, _ = camera.project(P)
    err = np.linalg.norm(uvm - track.uv, axis=1)
    inl = err <= np.maximum(4.0 * np.median(err) + 1.0, 3.0 * sig_uv)
    dm = model_diameter(P, camera, kind)
    size_ratio = float(np.median(d[use_d] / dm[use_d])) if use_d.any() else float("nan")
    cov = None
    try:
        J = sol.jac
        dof = max(1, len(sol.fun) - len(theta))
        s2 = max(1.0, 2 * sol.cost / dof)
        cov = np.linalg.pinv(J.T @ J)[:9, :9] * s2
    except (np.linalg.LinAlgError, ValueError):
        cov = None
    return Fit(traj=traj, rms_px=float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("nan"),
               inliers=inl, size_ratio=size_ratio, cov=cov, extension_prior_used=ext is not None,
               size_scale=float(np.exp(theta[9])), catch_prior_used=catch is not None)


RELEASE_WINDOW_FT = 6.0          # first tracked point this close to the expected release = release seen...
RELEASE_MAX_SIZE_SHIFT = 0.10    # ...unless anchoring there needs the ball sizes off by more than this
RELEASE_LATE_FRAMES = 2.0        # the ball is first measured clear of the hand about two frames after release
RELEASE_LATE_SIGMA_FRAMES = 1.2


CATCH_Y_FT = -0.5                # where the glove takes the ball: just behind the back tip of the plate...
CATCH_SIGMA_FT = 1.0             # ...give or take a foot (catcher's set-up, reach)
CATCH_WINDOW_FT = 5.0            # last tracked point this close to the glove = the ball was followed into it
CATCH_EDGE_PX = 12.0             # a track ending this close to the picture's edge left the frame instead
CATCH_MIN_TAIL_FRAMES = 3        # ...and one ending this close to the end of the clip just ran out of video


@dataclass
class PitchFit:
    fit: Fit
    release_observed: bool
    t_release: float
    y_release_expected: float      # m
    release_source: str = "auto"   # auto (first tracked frame) | manual (set by the user) | estimated (not seen)
    caught: bool = False           # the catch anchored the far end of the flight

    def release_time(self, traj: Trajectory) -> float:
        if self.release_observed:
            return self.t_release
        t = traj.time_at_y(self.y_release_expected)
        return self.t_release if t is None else t


MANUAL_RELEASE_SIGMA_FRAMES = 0.7   # a release frame picked by eye


def catch_anchor(track: Track, camera: Camera, setup: Setup, traj: Trajectory, check: bool = True,
                 t_clip_end: float | None = None) -> tuple[float, float, float] | None:
    """(t, y, sigma) for the last tracked frame if the ball was followed into the catcher's glove. Only for a
    camera at the release looking down the line (the plate end of the flight is otherwise either sized well
    enough or hidden behind the catcher). The last sighting is up to one frame before the glove. With check,
    the track must end inside the picture, before the end of the clip (t_clip_end: the last frame searched;
    unknown = no catch), and within CATCH_WINDOW_FT of the glove by the unanchored fit, so a ball lost early,
    leaving the frame or still flying when the video stops is not pinned to the plate."""
    if camera_view(camera, setup.distance_ft) != "release" or len(track) < 5:
        return None
    t_last = float(track.t[-1])
    dt_frame = float(np.median(np.diff(track.t)))
    step = float(np.linalg.norm(traj.vel(t_last))) * dt_frame
    y_mean = CATCH_Y_FT * FT + 0.5 * step
    if check:
        if t_clip_end is None or t_clip_end - t_last < CATCH_MIN_TAIL_FRAMES * dt_frame:
            return None
        u, v = track.uv[-1]
        if not (CATCH_EDGE_PX <= u <= camera.width - CATCH_EDGE_PX and CATCH_EDGE_PX <= v <= camera.height - CATCH_EDGE_PX):
            return None
        if abs(traj.pos(t_last)[1] - y_mean) > CATCH_WINDOW_FT * FT:
            return None
    return t_last, y_mean, float(np.hypot(CATCH_SIGMA_FT * FT, 0.3 * step))


def fit_pitch(track: Track, camera: Camera, setup: Setup, init: Trajectory | None = None,
              observed: bool | None = None, t_release_manual: float | None = None,
              caught: bool | None = None, t_clip_end: float | None = None) -> PitchFit:
    """Fit, then decide whether the release was seen: if the first tracked point lies near where pitchers
    release (pitching distance minus extension), the ball was followed out of the hand and the release
    distance becomes a constraint on the scale - the time-of-flight-over-a-known-distance measurement, which
    does not depend on how precisely the camera's focal length was solved or on the ball's apparent size.
    The ball is measured clear of the hand about two frames after it leaves it, so the prior sits that far
    down the flight. If anchoring there would need the ball sizes off by more than 10%, the track most
    likely started late and the anchor is dropped. Filmed from behind the release, the end of the track is
    anchored the same way at the catch (catch_anchor). `observed` and `caught` force the decisions (used to
    keep them fixed across calibration samples; t_clip_end is the time of the last frame searched). `t_release_manual`, a release time the user picked,
    anchors the release there instead."""
    y_exp = (setup.distance_ft - setup.extension_prior_ft) * FT
    t_first = float(track.t[0])
    dt_frame = float(np.median(np.diff(track.t))) if len(track) > 1 else 1 / 240.0
    fit0 = fit_trajectory(track, camera, setup, init=init)
    catch = None
    if caught is None or caught:
        catch = catch_anchor(track, camera, setup, fit0.traj, check=caught is None, t_clip_end=t_clip_end)
        if catch is not None and caught is None:
            fit_c = fit_trajectory(track, camera, setup, init=fit0.traj, catch=catch)
            if abs(np.log(fit_c.size_scale)) > RELEASE_MAX_SIZE_SHIFT:
                catch = None
            else:
                fit0 = fit_c
        elif catch is not None:
            fit0 = fit_trajectory(track, camera, setup, init=fit0.traj, catch=catch)
    got = catch is not None
    speed = float(np.linalg.norm(fit0.traj.vel(t_first)))
    if t_release_manual is not None:
        sig = np.hypot(EXTENSION_SIGMA_FT * FT, MANUAL_RELEASE_SIGMA_FRAMES * dt_frame * speed)
        fit1 = fit_trajectory(track, camera, setup, init=fit0.traj, ext=(t_release_manual, y_exp, sig), catch=catch)
        return PitchFit(fit1, True, t_release_manual, y_exp, "manual", got)
    late_m = RELEASE_LATE_FRAMES * dt_frame * speed
    t_release = t_first - RELEASE_LATE_FRAMES * dt_frame
    decide = observed is None
    if decide:
        y_first = fit0.traj.pos(t_first)[1]
        observed = abs(y_first + late_m - y_exp) <= RELEASE_WINDOW_FT * FT
    if not observed:
        t_rel = fit0.traj.time_at_y(y_exp)
        return PitchFit(fit0, False, t_first if t_rel is None else t_rel, y_exp, "estimated", got)
    sig = np.hypot(EXTENSION_SIGMA_FT * FT, RELEASE_LATE_SIGMA_FRAMES * dt_frame * speed)
    fit1 = fit_trajectory(track, camera, setup, init=fit0.traj, ext=(t_first, y_exp - late_m, sig), catch=catch)
    if decide and abs(np.log(fit1.size_scale)) > RELEASE_MAX_SIZE_SHIFT:
        t_rel = fit0.traj.time_at_y(y_exp)
        return PitchFit(fit0, False, t_first if t_rel is None else t_rel, y_exp, "estimated", got)
    return PitchFit(fit1, True, t_release, y_exp, "auto", got)


# ------------------------------------------------------------------------------------------ metrics
def lift_coefficient(s: float) -> float:
    return 1.5 * s if s < 0.1 else 0.09 + 0.6 * s


def spin_factor_from_lift(cl: float) -> float:
    return cl / 1.5 if cl < 0.15 else (cl - 0.09) / 0.6


def clock_from_angle(angle_deg: float) -> str:
    """Tilt as a clock face seen from behind the pitcher: 0 deg = 12:00, clockwise."""
    minutes = int(round((angle_deg % 360.0) / 360.0 * 720.0)) % 720
    h, m = divmod(minutes, 60)
    return f"{12 if h == 0 else h}:{m:02d}"


def pitch_metrics(traj: Trajectory, setup: Setup, t_release: float) -> dict:
    """Velocity, release, plate location, movement and the movement-implied spin of a fitted pitch.
    Lengths in the units baseball uses (mph, ft, in)."""
    t_plate = traj.time_at_y(PLATE_FRONT_Y)
    out: dict = {"t_release": t_release, "t_plate": t_plate}
    p_rel, v_rel = traj.pos(t_release), traj.vel(t_release)
    out["release_speed_mph"] = float(np.linalg.norm(v_rel) / MPH)
    out["release_height_ft"] = float(p_rel[2] / FT)
    out["release_side_ft"] = float(p_rel[0] / FT)
    out["extension_ft"] = float(setup.distance_ft - p_rel[1] / FT)
    out["vert_release_angle_deg"] = float(np.degrees(np.arcsin(v_rel[2] / np.linalg.norm(v_rel))))
    out["horz_release_angle_deg"] = float(np.degrees(np.arctan2(v_rel[0], -v_rel[1])))
    if t_plate is None or t_plate <= t_release:
        return out
    p_pl, v_pl = traj.pos(t_plate), traj.vel(t_plate)
    T = t_plate - t_release
    out["flight_time_s"] = float(T)
    out["plate_speed_mph"] = float(np.linalg.norm(v_pl) / MPH)
    out["plate_x_in"] = float(p_pl[0] / IN)
    out["plate_z_in"] = float(p_pl[2] / IN)
    r = BALL_RADIUS
    out["strike"] = bool(abs(p_pl[0]) <= PLATE_HALF_WIDTH + r
                         and setup.zone_bottom_in * IN - r <= p_pl[2] <= setup.zone_top_in * IN + r)
    out["vert_approach_angle_deg"] = float(np.degrees(np.arctan2(v_pl[2], -v_pl[1])))
    out["horz_approach_angle_deg"] = float(np.degrees(np.arctan2(v_pl[0], -v_pl[1])))
    # Magnus acceleration: the fitted acceleration minus gravity, minus its component along the velocity
    v_mid = traj.vel(0.5 * (t_release + t_plate))
    vh = v_mid / np.linalg.norm(v_mid)
    a_ng = traj.a + np.array([0.0, 0.0, G])
    a_m = a_ng - np.dot(a_ng, vh) * vh
    hb = 0.5 * a_m[0] * T * T / IN                 # + toward first base (catcher's view)
    ivb = 0.5 * a_m[2] * T * T / IN
    arm = -1.0 if setup.throws == "R" else 1.0     # arm side: third base for a right-hander
    out["ivb_in"] = float(ivb)
    out["hb_in"] = float(hb)
    out["hb_arm_side_in"] = float(hb * arm)
    out["vb_in"] = float(0.5 * (a_m[2] - G) * T * T / IN)
    # spin axis implied by movement (clock face from behind the pitcher; arm-side run of a right-hander is
    # the pitcher's right, i.e. -x)
    tilt = float(np.degrees(np.arctan2(-a_m[0], a_m[2])))
    out["movement_tilt_deg"] = tilt % 360.0
    out["movement_tilt_clock"] = clock_from_angle(tilt)
    out["movement_axis_deg"] = float((tilt + 180.0) % 360.0)       # Statcast convention: 180 = pure backspin
    speed = float(np.linalg.norm(v_mid))
    am = float(np.linalg.norm(a_m))
    cl = 2 * BALL_MASS * am / (air_density(setup.altitude_ft) * BALL_AREA * speed * speed)
    out["lift_coefficient"] = cl
    out["active_spin_est_rpm"] = float(spin_factor_from_lift(cl) * speed / BALL_RADIUS * 60 / (2 * np.pi))
    out["magnus_accel_g"] = am / G
    return out


METRIC_SD_KEYS = ("release_speed_mph", "plate_speed_mph", "ivb_in", "hb_in", "plate_x_in", "plate_z_in",
                  "release_height_ft", "release_side_ft", "extension_ft", "movement_tilt_deg", "active_spin_est_rpm",
                  "flight_time_s")


def _circ_sd(deg: np.ndarray) -> float:
    a = np.radians(deg)
    R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    return float(np.degrees(np.sqrt(max(0.0, -2 * np.log(max(R, 1e-12))))))


def metric_uncertainty(pf: PitchFit, track: Track, cameras: list[Camera], setup: Setup, n_noise: int = 40,
                       seed: int = 0) -> dict[str, float]:
    """One-SD uncertainty of the reported metrics: calibration (refit with every click-noise camera
    sample, the release decision held fixed) and measurement noise (parameter covariance), combined in
    quadrature."""
    fit = pf.fit
    base = pitch_metrics(fit.traj, setup, pf.release_time(fit.traj))
    calib: dict[str, list[float]] = {k: [] for k in METRIC_SD_KEYS}
    sub = track.subset(fit.inliers)
    for cam in cameras:
        try:
            pf2 = fit_pitch(sub, cam, setup, init=fit.traj, observed=pf.release_observed,
                            t_release_manual=pf.t_release if pf.release_source == "manual" else None, caught=pf.caught)
        except (ValueError, np.linalg.LinAlgError):
            continue
        m = pitch_metrics(pf2.fit.traj, setup, pf2.release_time(pf2.fit.traj))
        for k in METRIC_SD_KEYS:
            if k in m and m[k] is not None:
                calib[k].append(m[k])
    noise: dict[str, list[float]] = {k: [] for k in METRIC_SD_KEYS}
    if fit.cov is not None and np.all(np.isfinite(fit.cov)):
        rng = np.random.default_rng(seed)
        try:
            draws = rng.multivariate_normal(fit.traj.theta, fit.cov, size=n_noise, method="eigh")
        except (np.linalg.LinAlgError, ValueError):
            draws = []
        for th in draws:
            tr = Trajectory.from_theta(th, fit.traj.t_ref)
            m = pitch_metrics(tr, setup, pf.release_time(tr))
            for k in METRIC_SD_KEYS:
                if k in m and m[k] is not None:
                    noise[k].append(m[k])
    out = {}
    for k in METRIC_SD_KEYS:
        if k not in base:
            continue
        sd = []
        for vals in (calib[k], noise[k]):
            if len(vals) >= 5:
                sd.append(_circ_sd(np.array(vals)) if k == "movement_tilt_deg" else float(np.std(vals)))
        out[k] = float(np.sqrt(np.sum(np.square(sd)))) if sd else None
    return out

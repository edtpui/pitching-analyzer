"""Field geometry and the pinhole camera solved from points marked on a still frame.

Field frame (metres, the Statcast convention): origin at the back tip of home plate on the ground, x toward
the first-base side (the catcher's right), y toward the pitcher, z up. The camera is a pinhole with square
pixels, the principal point at the image centre and no lens distortion (phone main cameras correct theirs);
its focal length and pose are solved from the five corners of home plate and the two front corners of the
pitching rubber, whose positions are known once the pitching distance and mound height are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

FT = 0.3048
IN = 0.0254
MPH = 0.44704
G = 9.80665

# Home plate, around the pentagon: back tip, first-base corner, front first-base corner, front third-base
# corner, third-base corner.
PLATE_XY = np.array([[0.0, 0.0], [8.5, 8.5], [8.5, 17.0], [-8.5, 17.0], [-8.5, 8.5]]) * IN
PLATE_FRONT_Y = 17.0 * IN
PLATE_HALF_WIDTH = 8.5 * IN
RUBBER_HALF_WIDTH = 12.0 * IN

# pitching distance (front of the rubber to the back tip of home plate, ft) -> usual mound height (in)
PITCHING_DISTANCES_FT = {60.5: 10.0, 54.0: 8.0, 50.0: 8.0, 46.0: 6.0}

TYPICAL_HFOV_DEG = 66.0       # phone main camera in slow motion; a weak prior, only decisive if the marks are not
FOCAL_PRIOR_SIGMA_LOG = 0.5   # ...e^0.5 = a factor 1.65 at one sigma
CLICK_SIGMA_PX = 1.0          # assumed accuracy of a zoomed-in click, for the calibration uncertainty


class CalibrationError(ValueError):
    """The marked points do not define a camera."""


def plate_points() -> np.ndarray:
    return np.column_stack([PLATE_XY, np.zeros(5)])


def rubber_points(distance_ft: float, mound_in: float) -> np.ndarray:
    """Front edge of the pitching rubber: third-base end, first-base end."""
    y, z = distance_ft * FT, mound_in * IN
    return np.array([[-RUBBER_HALF_WIDTH, y, z], [RUBBER_HALF_WIDTH, y, z]])


def zone_corners(bottom_in: float, top_in: float) -> np.ndarray:
    """Strike zone at the front of the plate (where Statcast reports plate location): 4 corners."""
    x, y = PLATE_HALF_WIDTH, PLATE_FRONT_Y
    b, t = bottom_in * IN, top_in * IN
    return np.array([[-x, y, b], [x, y, b], [x, y, t], [-x, y, t]])


@dataclass
class Camera:
    f: float                 # focal length, pixels
    cx: float
    cy: float
    rvec: np.ndarray         # world -> camera rotation (Rodrigues vector)
    tvec: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        self.rvec = np.asarray(self.rvec, float).reshape(3)
        self.tvec = np.asarray(self.tvec, float).reshape(3)
        self._R = cv2.Rodrigues(self.rvec.reshape(3, 1))[0]

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    @property
    def center(self) -> np.ndarray:
        return -self._R.T @ self.tvec

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / 2.0 / self.f)))

    def to_cam(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) @ self._R.T + self.tvec

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates and camera depth (along the optical axis) of world points (N, 3)."""
        pc = self.to_cam(np.atleast_2d(X))
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = self.f * pc[:, :2] / z[:, None] + np.array([self.cx, self.cy])
        return uv, z

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit world directions of the rays through pixels (N, 2)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        d = np.column_stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))])
        d = d @ self._R            # camera -> world (R^T applied to row vectors)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def as_dict(self) -> dict:
        return {"f": self.f, "cx": self.cx, "cy": self.cy, "rvec": self.rvec.tolist(), "tvec": self.tvec.tolist(),
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(f=float(d["f"]), cx=float(d["cx"]), cy=float(d["cy"]), rvec=np.array(d["rvec"]),
                   tvec=np.array(d["tvec"]), width=int(d["width"]), height=int(d["height"]))


@dataclass
class Calibration:
    camera: Camera
    rms_px: float
    residuals_px: np.ndarray      # (7,) reprojection error of each mark: 5 plate corners (PLATE_XY order), 2 rubber
    plate_px: np.ndarray          # (5, 2) the marks, reordered to PLATE_XY
    rubber_px: np.ndarray         # (2, 2) third-base end, first-base end
    distance_ft: float
    mound_in: float
    camera_height_ft: float | None = None
    samples: list[Camera] = field(default_factory=list)   # cameras re-solved with click noise

    @property
    def world_points(self) -> np.ndarray:
        return np.vstack([plate_points(), rubber_points(self.distance_ft, self.mound_in)])

    @property
    def image_points(self) -> np.ndarray:
        return np.vstack([self.plate_px, self.rubber_px])

    def view(self) -> str:
        return camera_view(self.camera, self.distance_ft)


def camera_view(camera: Camera, distance_ft: float) -> str:
    """behind_plate | behind_pitcher | side, from where the camera stands relative to the pitching line."""
    x, y, _ = camera.center
    d = distance_ft * FT
    if y < PLATE_FRONT_Y and abs(x) < 1.2 * (PLATE_FRONT_Y - y) + 1.0:
        return "behind_plate"
    if y > d - 1.0 and abs(x) < 1.2 * (y - d) + 2.0:
        return "behind_pitcher"
    return "side"


def _convex_order(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    return np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))


CAMERA_HEIGHT_SIGMA = 1.5 * IN    # a tape-measured lens height


def _residuals(params: np.ndarray, obj: np.ndarray, img: np.ndarray, cx: float, cy: float, logf0: float,
               sigma_px: float, height: float | None) -> np.ndarray:
    logf, rvec, tvec = params[0], params[1:4], params[4:7]
    f = np.exp(logf)
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    r = (proj.reshape(-1, 2) - img).ravel()
    extra = [sigma_px * (logf - logf0) / FOCAL_PRIOR_SIGMA_LOG]
    if height is not None:
        R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
        extra.append(sigma_px * ((-R.T @ tvec)[2] - height) / CAMERA_HEIGHT_SIGMA)
    return np.append(r, extra)


def _valid(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> bool:
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    depth = (obj @ R.T + np.asarray(tvec, float).reshape(3))[:, 2]
    center = -R.T @ np.asarray(tvec, float).reshape(3)
    return bool(np.all(depth > 0.3) and center[2] > 0.05)


def _solve(obj: np.ndarray, img: np.ndarray, size: tuple[int, int],
           init: tuple[float, np.ndarray, np.ndarray] | None = None,
           height: float | None = None) -> tuple[Camera, float] | None:
    """Focal length and pose for fixed correspondences. Without `init`, a focal-length sweep with SQPnP
    seeds a joint Levenberg-Marquardt refinement."""
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    logf0 = float(np.log(w / 2.0 / np.tan(np.radians(TYPICAL_HFOV_DEG / 2))))
    seeds = []
    if init is not None:
        seeds.append(np.concatenate([[np.log(init[0])], init[1], init[2]]))
    else:
        best = []
        for f in np.geomspace(0.25, 8.0, 56) * max(w, h):
            K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            try:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                continue
            if not ok or not _valid(obj, rvec, tvec):
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
            best.append((err, f, rvec.ravel(), tvec.ravel()))
        best.sort(key=lambda b: b[0])
        seeds = [np.concatenate([[np.log(f)], r, t]) for _, f, r, t in best[:3]]
    out = None
    for x0 in seeds:
        try:
            sol = least_squares(_residuals, x0, args=(obj, img, cx, cy, logf0, CLICK_SIGMA_PX, height), method="lm",
                                x_scale="jac", max_nfev=400)
        except ValueError:
            continue
        if not _valid(obj, sol.x[1:4], sol.x[4:7]):
            continue
        cost = float(np.sum(sol.fun[: 2 * len(obj)] ** 2))
        if out is None or cost < out[1]:
            out = (Camera(f=float(np.exp(sol.x[0])), cx=cx, cy=cy, rvec=sol.x[1:4], tvec=sol.x[4:7],
                          width=w, height=h), cost)
    return out


def calibrate(plate_px, rubber_px, size: tuple[int, int], distance_ft: float = 60.5,
              mound_in: float | None = None, camera_height_ft: float | None = None, n_samples: int = 32,
              seed: int = 0) -> Calibration:
    """Solve the camera from the five home-plate corners and the two front corners of the rubber, clicked in
    any order. Every assignment of the clicks to the plate's corners (5 rotations x 2 directions) and the
    rubber's ends is tried; the best one with the camera above the ground and in front of every mark wins.
    A measured lens height, if given, is a further constraint (it pins the camera distance, and with it the
    focal length, independently of how precisely the small, far rubber was marked)."""
    plate_px = np.asarray(plate_px, float).reshape(-1, 2)
    rubber_px = np.asarray(rubber_px, float).reshape(-1, 2)
    if plate_px.shape != (5, 2) or rubber_px.shape != (2, 2):
        raise CalibrationError("mark all five corners of home plate and both front corners of the rubber")
    if not (np.all(np.isfinite(plate_px)) and np.all(np.isfinite(rubber_px))):
        raise CalibrationError("a marked point is not a number")
    if not (30.0 <= distance_ft <= 70.0):
        raise CalibrationError("pitching distance must be between 30 and 70 ft")
    if mound_in is None:
        mound_in = PITCHING_DISTANCES_FT.get(distance_ft, 10.0)
    if not (0.0 <= mound_in <= 18.0):
        raise CalibrationError("mound height must be between 0 and 18 in")
    if camera_height_ft is not None and not (0.5 <= camera_height_ft <= 30.0):
        raise CalibrationError("camera height must be between 0.5 and 30 ft")
    height = None if camera_height_ft is None else camera_height_ft * FT
    w, h = int(size[0]), int(size[1])
    span = np.ptp(plate_px, axis=0)
    if max(span) < 8.0:
        raise CalibrationError("the plate corners are too close together; zoom in and mark them again")
    if np.linalg.norm(rubber_px[0] - rubber_px[1]) < 3.0:
        raise CalibrationError("the two rubber corners are on top of each other")

    obj = np.vstack([plate_points(), rubber_points(distance_ft, mound_in)])
    order = _convex_order(plate_px)
    best = None
    for shift in range(5):
        for step in (1, -1):
            idx = [int(order[(shift + step * k) % 5]) for k in range(5)]
            for rub in (rubber_px, rubber_px[::-1]):
                img = np.vstack([plate_px[idx], rub])
                sol = _solve(obj, img, (w, h), height=height)
                if sol is not None and (best is None or sol[1] < best[1]):
                    best = (sol[0], sol[1], img)
    if best is None:
        raise CalibrationError("no camera position fits these marks; check that each point is on the right corner")
    cam, _, img = best
    proj, _ = cam.project(obj)
    res = np.linalg.norm(proj - img, axis=1)
    calib = Calibration(camera=cam, rms_px=float(np.sqrt(np.mean(res ** 2))), residuals_px=res,
                        plate_px=img[:5], rubber_px=img[5:], distance_ft=float(distance_ft), mound_in=float(mound_in),
                        camera_height_ft=camera_height_ft)
    rng = np.random.default_rng(seed)
    init = (cam.f, cam.rvec, cam.tvec)
    for _ in range(n_samples):
        noisy = img + rng.normal(0.0, CLICK_SIGMA_PX, img.shape)
        hs = None if height is None else height + rng.normal(0.0, CAMERA_HEIGHT_SIGMA)
        s = _solve(obj, noisy, (w, h), init=init, height=hs)
        if s is not None:
            calib.samples.append(s[0])
    return calib


def calibration_from_dict(d: dict, n_samples: int = 32) -> Calibration:
    """Re-solve a stored calibration (marks + setup) so the camera and its samples are reproducible."""
    return calibrate(d["plate_px"], d["rubber_px"], tuple(d["image_size"]), d["distance_ft"], d["mound_in"],
                     camera_height_ft=d.get("camera_height_ft"), n_samples=n_samples)


# ------------------------------------------------------------------------------------------ image refinement
# Clicks are good to about a pixel, and the focal length - and with it every speed - inherits the error of
# the small, far marks (a 1 px error on a 40 px rubber is 2.5%). Home plate and the rubber are white on dirt,
# so the camera is refined until their rendered outlines line up with the picture: hundreds of edge pixels
# instead of seven clicks. The clicks stay in the fit as a loose prior.
RUBBER_THICKNESS = 1.0 * IN        # how far the slab stands out of the mound (front face seen from the plate)
REFINE_BAND_PX = 5
REFINE_PHOTO_SIGMA = 6.0           # grey levels of texture and noise along the outlines
REFINE_CLICK_SIGMA = 2.0           # px: the refinement may move a corner this far from its click at one SD
REFINE_MAX_SHIFT_PX = 4.0          # larger moves mean the outlines locked onto something else: keep the clicks
REFINE_SS = 4                      # supersampling of the rendered coverage


def _slab_corners(distance_ft: float, mound_in: float, thickness: float = RUBBER_THICKNESS) -> np.ndarray:
    rb = rubber_points(distance_ft, mound_in)
    back = rb + np.array([0.0, 6.0 * IN, 0.0])
    top = np.vstack([rb, back[::-1]])
    return np.vstack([top, top - np.array([0.0, 0.0, thickness])])


def _coverage(poly_uv: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Anti-aliased coverage (0-1) of a convex image polygon over the pixel block [x0, x0+w) x [y0, y0+h).
    Image coordinates put pixel centres at i + 0.5; OpenCV's drawing puts them at integers, hence the -0.5."""
    s = REFINE_SS
    pts = np.round(((poly_uv - [x0, y0]) * s - 0.5) * 16).astype(np.int32)
    m = np.zeros((h * s, w * s), np.uint8)
    cv2.fillConvexPoly(m, pts, 255, lineType=cv2.LINE_8, shift=4)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


REFINE_EDGE_SIGMA = 0.7            # px: optics + compression blur of a real edge
# passes of (edge sigma px, band half-width px): coarse-to-fine first, sharp-only as the fallback
REFINE_SCHEDULES = (((2.0, 8), (1.2, 6), (0.7, 5), (0.7, 5)), ((0.7, 5), (0.7, 5)))


def _soft_coverage(poly_uv: np.ndarray, xy: np.ndarray, sigma: float = REFINE_EDGE_SIGMA) -> np.ndarray:
    """Smooth coverage of a convex polygon at pixel centres xy (N, 2): signed distance to the polygon
    (largest distance past any edge line), through an error-function edge of width REFINE_EDGE_SIGMA.
    Continuous in the vertices, so the camera can be optimised with ordinary derivatives."""
    from scipy.special import erfc
    p = poly_uv
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area2 < 0:
        p = p[::-1]
    e = np.roll(p, -1, axis=0) - p
    n = np.column_stack([e[:, 1], -e[:, 0]])                 # outward normals for counter-clockwise order (y down)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    sd = ((xy[:, None, :] - p[None, :, :]) * n[None, :, :]).sum(axis=2).max(axis=1)
    return 0.5 * erfc(sd / (np.sqrt(2.0) * sigma))


def _refine_stage(cam0: Camera, img: np.ndarray, calib: Calibration, sigma: float = REFINE_EDGE_SIGMA,
                  band_px: int = REFINE_BAND_PX):
    """One photometric pass: edge bands from `cam0`, Levenberg-Marquardt on the camera. Returns
    (camera, least-squares result, camera_of, per-object contrast) or a reason string."""
    h_img, w_img = img.shape[:2]
    plate3d = plate_points()
    slab3d = _slab_corners(calib.distance_ft, calib.mound_in)
    obj = calib.world_points
    clicks = calib.image_points

    def polys(cam: Camera) -> list[np.ndarray] | None:
        out = []
        for pts3 in (plate3d, slab3d):
            uv, z = cam.project(pts3)
            if np.any(z <= 0.2):
                return None
            out.append(cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(np.float64))
        return out

    p0 = polys(cam0)
    if p0 is None:
        return "marks behind the camera"
    blocks = []
    k_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1,) * 2)
    for j, poly in enumerate(p0):
        lo = np.maximum(np.floor(poly.min(axis=0)).astype(int) - band_px - 3, 0)
        hi = np.minimum(np.ceil(poly.max(axis=0)).astype(int) + band_px + 3, [w_img, h_img])
        if np.any(hi - lo < 4):
            return "plate or rubber outside the picture"
        bw, bh = hi[0] - lo[0], hi[1] - lo[1]
        inside = (_coverage(poly, lo[0], lo[1], bw, bh) > 0.5).astype(np.uint8)
        band = (cv2.dilate(inside, k_band) > 0) & ~(cv2.erode(inside, k_band) > 0)
        if j == 1:
            # how far the rubber stands out of the mound is unknown: drop pixels whose coverage depends on it
            thin, thick = (cv2.convexHull(cam0.project(_slab_corners(calib.distance_ft, calib.mound_in, t))[0]
                                          .astype(np.float32)).reshape(-1, 2).astype(np.float64)
                           for t in (0.25 * IN, 2.0 * IN))
            band &= np.abs(_coverage(thin, lo[0], lo[1], bw, bh) - _coverage(thick, lo[0], lo[1], bw, bh)) < 0.05
        if band.sum() < 12:
            return "plate or rubber too small to refine"
        ys, xs = np.nonzero(band)
        blocks.append((np.column_stack([xs + lo[0] + 0.5, ys + lo[1] + 0.5]), img[lo[1]:hi[1], lo[0]:hi[0]][band]))

    logf0 = float(np.log(cam0.width / 2.0 / np.tan(np.radians(TYPICAL_HFOV_DEG / 2))))
    height = None if calib.camera_height_ft is None else calib.camera_height_ft * FT
    x_base = np.concatenate([[np.log(cam0.f)], cam0.rvec, cam0.tvec])
    # scale each parameter so one unit moves the outline vertices by at most one pixel
    verts = np.vstack([plate3d, slab3d])
    uv0 = cam0.project(verts)[0]
    sc = np.ones(7)
    for k in range(7):
        p = x_base.copy()
        p[k] += 1e-6
        moved = Camera(f=float(np.exp(p[0])), cx=cam0.cx, cy=cam0.cy, rvec=p[1:4], tvec=p[4:7], width=cam0.width,
                       height=cam0.height).project(verts)[0]
        sc[k] = 1e-6 / max(np.abs(moved - uv0).max(), 1e-12)

    def camera_of(x: np.ndarray) -> Camera:
        p = x_base + x * sc
        return Camera(f=float(np.exp(p[0])), cx=cam0.cx, cy=cam0.cy, rvec=p[1:4], tvec=p[4:7], width=cam0.width,
                      height=cam0.height)

    def photo_residuals(x: np.ndarray) -> np.ndarray:
        cam = camera_of(x)
        ps = polys(cam)
        out = []
        for (xy, vals), poly in zip(blocks, ps if ps is not None else [None, None]):
            if poly is None:
                out.append(np.full(len(vals), 50.0))
                continue
            cov = _soft_coverage(poly, xy, sigma)
            A = np.column_stack([np.ones_like(cov), cov])
            coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
            out.append((vals - A @ coef) / REFINE_PHOTO_SIGMA)
        uv, _ = cam.project(obj)
        out.append(((uv - clicks) / REFINE_CLICK_SIGMA).ravel())
        out.append([CLICK_SIGMA_PX * (np.log(cam.f) - logf0) / FOCAL_PRIOR_SIGMA_LOG])
        if height is not None:
            out.append([CLICK_SIGMA_PX * (cam.center[2] - height) / CAMERA_HEIGHT_SIGMA])
        return np.concatenate(out)

    def jac(x: np.ndarray) -> np.ndarray:
        cols = []
        for k in range(7):
            e = np.zeros(7)
            e[k] = 0.02
            cols.append((photo_residuals(x + e) - photo_residuals(x - e)) / 0.04)
        return np.column_stack(cols)

    sol = least_squares(photo_residuals, np.zeros(7), jac=jac, method="lm", x_scale="jac", max_nfev=100)
    cam = camera_of(sol.x)
    contrast = []
    for (xy, vals), poly in zip(blocks, polys(cam)):
        cov = _soft_coverage(poly, xy, sigma)
        A = np.column_stack([np.ones_like(cov), cov])
        contrast.append(float(np.linalg.lstsq(A, vals, rcond=None)[0][1]))
    return cam, sol, camera_of, contrast


def refine_with_image(calib: Calibration, gray: np.ndarray, n_samples: int = 32, seed: int = 0) -> tuple[Calibration, dict]:
    """Photometric refinement of a click calibration against a greyscale picture of the empty field,
    coarse to fine: blurred edges in wide bands first (so a 2-px click error on a 4-px-tall rubber still
    overlaps it), then sharp edges in narrow bands, each pass re-deriving its bands from the last. Returns
    the refined calibration, or the original with the reason if the refinement is not trusted."""
    cam0 = calib.camera
    if (gray.shape[1], gray.shape[0]) != (cam0.width, cam0.height):
        return calib, {"used": False, "reason": "image size differs from the calibration"}
    img = gray.astype(np.float32)
    obj, clicks = calib.world_points, calib.image_points
    info: dict = {"used": False, "reason": "the outlines did not lock on clearly; the marks are used as clicked"}
    result = None
    # wide bands can catch nearby chalk when the plate is tiny (filmed from behind the pitcher): if the
    # coarse-to-fine schedule is not trusted, try sharp edges only, starting from the clicks
    for schedule in REFINE_SCHEDULES:
        cam = cam0
        stage = None
        for sigma, band in schedule:
            extra = np.sqrt(max(sigma ** 2 - REFINE_EDGE_SIGMA ** 2, 0.0))
            im = cv2.GaussianBlur(img, (0, 0), extra) if extra > 0.05 else img
            stage = _refine_stage(cam, im, calib, sigma=sigma, band_px=band)
            if isinstance(stage, str):
                return calib, {"used": False, "reason": stage}
            cam = stage[0]
        cam, sol, camera_of, contrast = stage
        shift = np.linalg.norm(cam.project(obj)[0] - cam0.project(obj)[0], axis=1)
        info = {"max_shift_px": float(shift.max()), "contrast": contrast, "focal_change_pct": 100 * (cam.f / cam0.f - 1),
                "schedule": "coarse-to-fine" if len(schedule) > 2 else "fine"}
        if shift.max() <= REFINE_MAX_SHIFT_PX and min(contrast) >= 15.0:
            result = (cam, sol, camera_of)
            break
    if result is None:
        info.update(used=False, reason="the outlines did not lock on clearly; the marks are used as clicked")
        return calib, info
    cam, sol, camera_of = result
    proj = cam.project(obj)[0]
    res = np.linalg.norm(proj - clicks, axis=1)
    out = Calibration(camera=cam, rms_px=float(np.sqrt(np.mean(res ** 2))), residuals_px=res, plate_px=calib.plate_px,
                      rubber_px=calib.rubber_px, distance_ft=calib.distance_ft, mound_in=calib.mound_in,
                      camera_height_ft=calib.camera_height_ft)
    # uncertainty: parameter covariance of the photometric fit
    try:
        J = sol.jac
        dof = max(1, len(sol.fun) - 7)
        s2 = max(1.0, 2 * sol.cost / dof)
        cov = np.linalg.pinv(J.T @ J) * s2
        rng = np.random.default_rng(seed)
        for x in rng.multivariate_normal(sol.x, cov, size=n_samples, method="eigh"):
            out.samples.append(camera_of(x))
    except (np.linalg.LinAlgError, ValueError):
        pass
    info.update(used=True, reason=None, refined_points_px=proj.tolist())
    return out, info

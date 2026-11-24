"""From per-frame candidates to the ball's track.

1. Tracklets: chains of candidates in consecutive frames (gaps of up to two frames) that move smoothly,
   grown from every nearby pair with a constant-acceleration image prediction.
2. Each long-enough tracklet is fitted as a pitch through the calibrated camera; it is kept only if the
   result is a pitch (30-110 mph toward the plate, released above the mound, crossing near the plate).
3. The best one absorbs the other tracklets its trajectory passes through (the ball is often lost for a few
   frames in front of the pitcher's body).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import D, DKIND, REFINED, U, V, FrameCandidates
from .geometry import FT, MPH, PLATE_FRONT_Y, Camera
from .trajectory import Fit, Setup, Track, fit_trajectory, model_diameter

MIN_TRACKLET = 4
MIN_EXTENT_PX = 8.0           # a tracklet must cross at least this much of the picture
MAX_GAP = 2
MAX_SEED_STEP = 0.09          # fraction of the image diagonal a ball can move between frames


@dataclass
class Tracklet:
    nodes: list[tuple[int, int]]           # (frame position, candidate index)

    def frames(self) -> list[int]:
        return [f for f, _ in self.nodes]


def _grow(frames: list[FrameCandidates], nodes: list[tuple[int, int]], used: set, direction: int) -> list[tuple[int, int]]:
    nodes = list(nodes)
    n = len(frames)
    while True:
        f_last, c_last = nodes[-1]
        p = np.array([frames[f_last].cands[c_last, U], frames[f_last].cands[c_last, V]])
        d_last = frames[f_last].cands[c_last, D]
        f_prev, c_prev = nodes[-2]
        q = np.array([frames[f_prev].cands[c_prev, U], frames[f_prev].cands[c_prev, V]])
        vel = (p - q) / abs(f_last - f_prev)
        acc = np.zeros(2)
        if len(nodes) >= 3:
            f3, c3 = nodes[-3]
            r = np.array([frames[f3].cands[c3, U], frames[f3].cands[c3, V]])
            v_old = (q - r) / abs(f_prev - f3)
            acc = (vel - v_old) / max(1.0, 0.5 * abs(f_last - f3))
        best = None
        for g in range(1, MAX_GAP + 2):
            j = f_last + direction * g
            if not (0 <= j < n):
                break
            c = frames[j].cands
            if not len(c):
                continue
            pred = p + vel * g + 0.5 * acc * g * g
            gate = 2.5 + 0.5 * d_last + 0.3 * np.linalg.norm(vel) * g
            dist = np.hypot(c[:, U] - pred[0], c[:, V] - pred[1])
            ratio = c[:, D] / max(d_last, 1.0)
            ok = (dist <= gate) & (ratio >= 0.6 ** g) & (ratio <= 1.7 ** g)
            ok &= np.array([(j, k) not in used for k in range(len(c))])
            if ok.any():
                k = int(np.argmin(np.where(ok, dist / gate, np.inf)))
                best = (j, k)
                break
        if best is None:
            break
        nodes.append(best)
    return nodes


def build_tracklets(frames: list[FrameCandidates], width: int, height: int) -> list[Tracklet]:
    diag = float(np.hypot(width, height))
    max_step = MAX_SEED_STEP * diag
    out: list[Tracklet] = []
    claimed: set[tuple[int, int]] = set()
    n = len(frames)
    for i in range(n):
        ci = frames[i].cands
        for a in range(len(ci)):
            if (i, a) in claimed:
                continue
            for g in (1, 2):
                j = i + g
                if j >= n or not len(frames[j].cands):
                    continue
                cj = frames[j].cands
                dist = np.hypot(cj[:, U] - ci[a, U], cj[:, V] - ci[a, V])
                ratio = cj[:, D] / max(ci[a, D], 1.0)
                for b in np.flatnonzero((dist <= max_step * g) & (ratio > 0.5) & (ratio < 2.0)):
                    if (j, int(b)) in claimed:
                        continue
                    nodes = _grow(frames, [(i, a), (j, int(b))], claimed, +1)
                    if len(nodes) >= MIN_TRACKLET:
                        out.append(Tracklet(nodes))
                        claimed.update(nodes)
                        break
                if (i, a) in claimed:
                    break
    return out


def _track_of(frames: list[FrameCandidates], nodes: list[tuple[int, int]]) -> Track:
    """Detections as a Track. Only unmixed (REFINED) diameters are used for distance: a thresholded blob's
    size carries a blur halo that no single scale factor can absorb."""
    nodes = sorted(nodes)
    c = np.array([frames[f].cands[k] for f, k in nodes])
    sig = 0.6 + 0.05 * c[:, D]
    d = np.where(c[:, REFINED] > 0, c[:, D], np.nan)
    return Track(frame=np.array([frames[f].index for f, _ in nodes]), t=np.array([frames[f].t for f, _ in nodes]),
                 uv=c[:, [U, V]], d=d, sig_uv=sig, d_kind=c[:, DKIND].astype(int))


SELECT_SIZE_SCALE_SIGMA = 0.10


def plausible(fit: Fit, track: Track, setup: Setup) -> tuple[bool, str]:
    """Is this fitted trajectory a pitch?"""
    tr = fit.traj
    t0, t1 = track.t[0], track.t[-1]
    v0 = tr.vel(t0)
    speed = np.linalg.norm(v0) / MPH
    if not (30.0 <= speed <= 115.0):
        return False, f"speed {speed:.0f} mph"
    if v0[1] > -10.0:
        return False, "not moving toward the plate"
    t_pl = tr.time_at_y(PLATE_FRONT_Y)
    if t_pl is None or t_pl < t0:
        return False, "never reaches the plate"
    p_pl = tr.pos(t_pl)
    if abs(p_pl[0]) > 1.8 or not (-0.5 <= p_pl[2] <= 3.0):
        return False, "crosses far from the plate"
    t_rel = tr.time_at_y((setup.distance_ft - setup.extension_prior_ft) * FT)
    if t_rel is not None:
        p_rel = tr.pos(t_rel)
        if abs(p_rel[0]) > 2.5 or not (0.2 <= p_rel[2] <= 3.2):
            return False, "does not come from the mound"
    if t_pl - t1 > 0.35:
        return False, "track ends far from the plate"
    d_med = float(np.nanmedian(track.d)) if np.isfinite(track.d).any() else 5.0
    if fit.rms_px > max(3.0, 0.35 * d_med):
        return False, f"rough track ({fit.rms_px:.1f} px)"
    return True, ""


def select_track(frames: list[FrameCandidates], tracklets: list[Tracklet], camera: Camera,
                 setup: Setup) -> tuple[Track, Fit, list[str]] | None:
    """The pitch: the most supported tracklet that fits as one, plus the tracklets on its trajectory."""
    notes: list[str] = []
    ranked = sorted(tracklets, key=lambda tl: -len(tl.nodes))[:8]
    valid: list[tuple[int, Tracklet, Track, Fit]] = []
    for tl in ranked:
        if len(tl.nodes) < 5:
            continue
        tr = _track_of(frames, tl.nodes)
        # something that stays on one pixel fits "a pitch straight down a camera ray" perfectly (a planted foot,
        # a fielder standing still); a ball in flight always crosses the picture
        extent = float(np.linalg.norm(np.ptp(tr.uv, axis=0)))
        d_med = float(np.nanmedian(tr.d)) if np.isfinite(tr.d).any() else 0.0
        if extent < max(MIN_EXTENT_PX, d_med):
            continue
        try:
            fit = fit_trajectory(tr, camera, setup, size_scale_sigma=SELECT_SIZE_SCALE_SIGMA, robust=False)
        except (ValueError, np.linalg.LinAlgError):
            continue
        ok, why = plausible(fit, tr, setup)
        if ok:
            valid.append((int(fit.inliers.sum()), tl, tr, fit))
    if not valid:
        return None
    valid.sort(key=lambda v: -v[0])
    _, core, track, fit = valid[0]
    nodes = set(core.nodes)
    # absorb other tracklets lying on this trajectory
    for tl in sorted(tracklets, key=lambda t: -len(t.nodes)):
        if tl is core or set(tl.nodes) & nodes:
            continue
        frames_taken = {f for f, _ in nodes}
        if any(f in frames_taken for f, _ in tl.nodes):
            continue
        tr2 = _track_of(frames, tl.nodes)
        P = fit.traj.pos(tr2.t)
        uvm, z = camera.project(P)
        err = np.linalg.norm(uvm - tr2.uv, axis=1)
        d_pred = model_diameter(P, camera)
        if np.all(z > 0) and np.all(err <= np.maximum(4.0, 0.5 * d_pred)):
            nodes |= set(tl.nodes)
            merged = _track_of(frames, list(nodes))
            try:
                fit = fit_trajectory(merged, camera, setup, init=fit.traj, size_scale_sigma=SELECT_SIZE_SCALE_SIGMA,
                                     robust=False)
            except (ValueError, np.linalg.LinAlgError):
                break
    track = _track_of(frames, list(nodes))
    t_main = fit.traj.time_at_y(PLATE_FRONT_Y)
    for _, _, _, other in valid[1:]:
        t_other = other.traj.time_at_y(PLATE_FRONT_Y)
        if t_main is not None and t_other is not None and abs(t_other - t_main) > 0.3:
            notes.append("More than one ball flight was found; the best-supported one is analysed.")
            break
    return track, fit, notes

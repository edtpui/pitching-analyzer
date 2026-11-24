"""Turn raw MediaPipe output into a `Motion` in the right-hander-normalised lab frame.

Steps: low-visibility masking -> left/right swap repair -> one-frame spike removal -> batting-side
identification -> masking of blurred hand frames -> gap filling -> smoothing -> provisional contact /
foot-plant detection (side-on signals first, 3D signals if the stride shows a camera behind or in front)
-> lab frame from the stride at foot plant (camera-square or stride-aligned heading) -> mirroring of
left-handed hitters.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .events import DEFAULT_CALIBRATION, calib_for_view, detect_contact, detect_fp
from .kinematics import Motion, kinematics
from ..pose import PoseTrack
from .signals import CUTOFF_ARM_HZ, CUTOFF_BODY_HZ, fill_gaps, lowpass, max_gap_s, smooth_skeleton, unwrap_deg
from .skeleton import (J, MP, MP_ARM_INDICES, MP_ARM_PAIRS, MP_LEG_PAIRS, leg_length, leg_length_projected,
                       mediapipe_indices, mirror_y)

VISIBILITY_MIN = 0.3
HANDS_TRUST_VISIBILITY = 0.6  # wrists/elbows below this are treated as missing (motion blur, bat occlusion)
SWAP_RATIO = 0.5              # swap when the crossed assignment is this much closer to the previous frame
SWAP_MAX_GAP_FRAMES = 3       # never judge a swap against a reference frame older than this
MIN_FRAMES = 30
ARM_DECISION_PAIRS = [(13, 14), (15, 16)]   # elbows, wrists
HAND_MP = [13, 14, 15, 16]                  # both elbows and wrists
SIDE_ROTATION_MIN_DEG = 60.0  # net hip + shoulder turn through the swing that makes the batting side certain
SWING_WINDOW_S = (0.30, 0.06)  # around peak hand speed: from the load to just after contact

# camera coordinates (x right, y down, z away) -> provisional z-up frame (x right, y away, z up)
CAM_TO_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


class PreprocessError(ValueError):
    """The clip cannot be analysed (no person, too short, swing not visible...)."""


@dataclass
class Prepared:
    motion: Motion
    bats: str                       # batting side used for the analysis
    bats_detected: str              # what the body rotation in the video shows
    camera_view: str                # side | behind | front
    heading_known: bool
    image_smooth: np.ndarray        # (T, 33, 2) repaired + smoothed pixel landmarks for drawing
    frame_index: np.ndarray         # (T,) decoded-video frame index of each sample
    quality: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _swap_pairs(arr: np.ndarray, f: int, pairs: list[tuple[int, int]]) -> None:
    for a, b in pairs:
        arr[f, [a, b]] = arr[f, [b, a]]


def _cost(image: np.ndarray, f: int, g: int, pairs) -> tuple[float, float]:
    """(keep, cross) assignment cost of frame f against reference frame g for left/right pairs."""
    left = image[f, [p[0] for p in pairs]]
    right = image[f, [p[1] for p in pairs]]
    pl, pr = image[g, [p[0] for p in pairs]], image[g, [p[1] for p in pairs]]
    keep = np.linalg.norm(left - pl, axis=1).sum() + np.linalg.norm(right - pr, axis=1).sum()
    cross = np.linalg.norm(left - pr, axis=1).sum() + np.linalg.norm(right - pl, axis=1).sum()
    return float(keep), float(cross)


def repair_swaps(image: np.ndarray, world: np.ndarray, vis: np.ndarray) -> dict[str, int]:
    """In-place repair of frames where MediaPipe labelled the left limb as the right one and vice versa.

    Each frame is judged only against the previous judged frame, and only if that frame is at most
    SWAP_MAX_GAP_FRAMES old. Arms are judged on elbows and wrists, only when all four are confidently
    tracked, and a swap also needs the elbows to agree: both hands sit together on the bat, so the wrists
    alone cannot tell the arms apart, but a real MediaPipe swap moves the whole arm."""
    counts = {}
    groups = (
        ("legs", MP_LEG_PAIRS, MP_LEG_PAIRS, 0.0),
        ("arms", MP_ARM_PAIRS, ARM_DECISION_PAIRS, HANDS_TRUST_VISIBILITY),
    )
    for name, pairs, decide, min_vis in groups:
        idx = [i for p in decide for i in p]
        n_swaps = 0
        prev = None
        for f in range(len(image)):
            if not np.isfinite(image[f, idx]).all() or (min_vis and vis[f, idx].min() < min_vis):
                continue
            if prev is not None and f - prev <= SWAP_MAX_GAP_FRAMES:
                keep, cross = _cost(image, f, prev, decide)
                swap = cross < SWAP_RATIO * keep
                if swap and name == "arms":
                    ke, ce = _cost(image, f, prev, decide[:1])
                    swap = ce < ke
                if swap:
                    for arr in (image, world, vis):
                        _swap_pairs(arr, f, pairs)
                    n_swaps += 1
            prev = f
        counts[name] = n_swaps
    return counts


def median3(x: np.ndarray) -> np.ndarray:
    """Running 3-sample median along time (NaN-aware): removes one-frame outliers, keeps edges."""
    out = np.array(x, dtype=float, copy=True)
    if len(x) >= 3:
        stack = np.stack([x[:-2], x[1:-1], x[2:]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
        out[1:-1] = np.where(np.isfinite(x[1:-1]), med, x[1:-1])
    return out


def _leg_length_px(img_c: dict[str, np.ndarray]) -> float:
    try:
        return leg_length_projected(img_c["rear_hip"], img_c["rear_knee"], img_c["rear_ankle"],
                                    img_c["lead_hip"], img_c["lead_knee"], img_c["lead_ankle"])
    except ValueError as exc:
        raise PreprocessError("the legs were never visible") from exc


def _net_yaw(world_up: np.ndarray, a: int, b: int, t: np.ndarray, fps: float, sel: np.ndarray) -> float:
    """Net change (deg) of the horizontal direction of landmark b - landmark a over the samples `sel`.
    Counter-clockwise seen from above is positive. The direction of turn does not depend on which end of
    the line is which, so it survives left/right label swaps."""
    v = world_up[:, b] - world_up[:, a]
    yaw = unwrap_deg(np.degrees(np.arctan2(v[:, 1], v[:, 0])))
    yaw = lowpass(yaw, fps, 6.0)
    ok = sel & np.isfinite(yaw)
    if ok.sum() < 3:
        return 0.0
    idx = np.flatnonzero(ok)
    return float(yaw[idx[-1]] - yaw[idx[0]])


def detect_batting_side(world: np.ndarray, image: np.ndarray, t: np.ndarray, fps: float,
                        peak_t: float) -> tuple[str, bool, float]:
    """(side, confident, net turn in degrees). A right-handed hitter turns counter-clockwise seen from above
    (from facing the plate toward the pitcher), a left-hander clockwise; the hips and shoulders turn ~100 deg
    each through the swing. If the turn is too small to trust, the striding foot decides (it is the lead foot:
    the left one for a right-handed hitter)."""
    world_up = world @ CAM_TO_UP.T
    sel = (t >= peak_t - SWING_WINDOW_S[0]) & (t <= peak_t + SWING_WINDOW_S[1])
    turn = (_net_yaw(world_up, MP["l_hip"], MP["r_hip"], t, fps, sel)
            + _net_yaw(world_up, MP["l_shoulder"], MP["r_shoulder"], t, fps, sel))
    if abs(turn) >= SIDE_ROTATION_MIN_DEG:
        return ("R" if turn > 0 else "L"), True, turn
    stride = (t >= peak_t - 0.8) & (t <= peak_t - 0.05)

    def travel(i: int) -> float:
        d = np.linalg.norm(np.diff(image[stride, i], axis=0), axis=1)
        return float(np.nansum(d))

    left, right = travel(MP["l_ankle"]), travel(MP["r_ankle"])
    if max(left, right) == 0.0:
        return ("R" if turn >= 0 else "L"), False, turn
    return ("R" if left >= right else "L"), min(left, right) / max(left, right) < 0.6, turn


def prepare(track: PoseTrack, bats: str = "auto", alignment: str = "square",
            capture_fps: float | None = None, calib: dict | None = None) -> Prepared:
    if bats not in ("auto", "R", "L"):
        raise ValueError("bats must be auto, R or L")
    if alignment not in ("square", "angled"):
        raise ValueError("alignment must be square or angled")
    calib = {**DEFAULT_CALIBRATION, **(calib or {})}
    notes: list[str] = []
    n = len(track.t)
    if n < MIN_FRAMES:
        raise PreprocessError(f"only {n} frames - the clip is too short")
    if track.detected.mean() < 0.2:
        raise PreprocessError("no person was detected in most of the clip")

    if capture_fps:
        t = np.arange(n) / float(capture_fps)
    else:
        t = track.t - track.t[0]
    fps = 1.0 / float(np.median(np.diff(t)))

    image = track.image.copy()
    world = track.world.copy()
    vis = track.visibility.copy()
    low = vis < VISIBILITY_MIN
    image[low] = np.nan
    world[low] = np.nan
    swaps = repair_swaps(image, world, vis)
    world = median3(world)          # single-frame jumps from the pose model (depth especially)
    image = median3(image)

    # the swing = peak speed of the hands relative to the pelvis, in the image
    img_hip = (image[:, MP["l_hip"]] + image[:, MP["r_hip"]]) / 2.0
    img_hands = (image[:, MP["l_wrist"]] + image[:, MP["r_wrist"]]) / 2.0
    h_speed = np.linalg.norm(np.gradient(img_hands - img_hip, t, axis=0), axis=-1)
    if not np.isfinite(h_speed).any():
        raise PreprocessError("the hands were never tracked")
    peak_t = float(t[int(np.nanargmax(h_speed))])

    detected, confident, turn = detect_batting_side(world, image, t, fps, peak_t)
    used = detected
    if bats != "auto" and bats != detected:
        if not confident:
            used = bats      # weak evidence: trust the user
        else:
            notes.append(f"You selected {'right' if bats == 'R' else 'left'}-handed, but the body turns like a "
                         f"{'right' if detected == 'R' else 'left'}-handed hitter in the reconstructed pose (the pose "
                         "model may have mirrored the body). The swing was analysed as the video shows it.")
    elif bats == "auto" and not confident:
        notes.append("The batting side could not be determined reliably from the video; select it explicitly "
                     "if the analysis looks mirrored.")

    # Blurred hands (low landmark confidence around contact, or the bat hiding them): their positions are
    # guesses, so treat them as missing and bridge them instead.
    hands_valid = (track.visibility[:, HAND_MP].min(axis=1) >= HANDS_TRUST_VISIBILITY) & \
        np.isfinite(image[:, HAND_MP]).all(axis=(1, 2))
    swing = np.abs(t - peak_t) <= 0.4
    if hands_valid[swing].mean() < 0.5:
        hands_valid = np.isfinite(image[:, HAND_MP]).all(axis=(1, 2))
        notes.append("The hands were tracked with low confidence throughout the swing; hand and arm metrics "
                     "may be unreliable.")
    image[np.ix_(~hands_valid, MP_ARM_INDICES)] = np.nan
    world[np.ix_(~hands_valid, MP_ARM_INDICES)] = np.nan

    max_gap = max_gap_s(fps)
    image = fill_gaps(image, t, max_gap)
    world = fill_gaps(world, t, max_gap)

    idx = mediapipe_indices(used)
    names = ("rear_shoulder", "rear_elbow", "rear_wrist", "lead_shoulder", "lead_elbow", "lead_wrist",
             "rear_hip", "rear_knee", "rear_ankle", "lead_hip", "lead_knee", "lead_ankle")
    pos_cam = world[:, idx]
    if not np.isfinite(pos_cam).any(axis=(1, 2)).any():
        raise PreprocessError("no usable 3D landmarks")
    pos_up = smooth_skeleton(pos_cam @ CAM_TO_UP.T, fps)

    img_c = {nm: image[:, i] for nm, i in zip(names, idx)}
    img_track = {
        "lead_ankle": img_c["lead_ankle"], "rear_ankle": img_c["rear_ankle"], "lead_knee": img_c["lead_knee"],
        "mid_hip": (img_c["rear_hip"] + img_c["lead_hip"]) / 2.0,
        "hands": (img_c["rear_wrist"] + img_c["lead_wrist"]) / 2.0,
    }
    img_track = {k: lowpass(v, fps, CUTOFF_ARM_HZ if k == "hands" else CUTOFF_BODY_HZ) for k, v in img_track.items()}
    scale_px = _leg_length_px(img_c)
    img_up = np.array([0.0, -1.0])

    # provisional events with rotation-invariant signals; the view decides which contact signal is right, and the
    # view comes from the stride at foot plant, so assume side-on first and redo the events if it is not
    prov = Motion(t=t, pos=pos_up, track=img_track, track_up=img_up, track_forward=None,
                  track_scale=scale_px, heading_known=False, fps=fps, hands_valid=hands_valid)
    kin0 = kinematics(prov)

    def stride_at_plant(view_guess: str) -> tuple[np.ndarray, int, int]:
        c = calib_for_view(calib, view_guess)
        contact, _ = detect_contact(t, kin0, c)
        # the height test needs a side-on camera; until the view is confirmed use the planted radius
        c_fp = c if view_guess != "side" else {**c, "fp_method": "radius"}
        fp = detect_fp(t, kin0["foot_speed"], kin0["foot_pos"], kin0["foot_height"], contact, c_fp)
        if fp is None:
            fp = int(np.argmin(np.abs(t - (t[contact] - calib["fp_to_contact_median_s"]))))
        return pos_up[fp, J["lead_ankle"]] - pos_up[fp, J["rear_ankle"]], fp, contact

    def view_of(stride: np.ndarray) -> str:
        if abs(stride[0]) >= abs(stride[1]):
            return "side"
        return "behind" if stride[1] > 0 else "front"

    stride, fp, contact = stride_at_plant("side")
    if not np.isfinite(stride).all():
        raise PreprocessError("the feet are not visible at foot plant")
    view = view_of(stride)
    if view != "side":
        stride2, fp2, contact2 = stride_at_plant(view)
        if np.isfinite(stride2).all() and view_of(stride2) == view:
            stride, fp, contact = stride2, fp2, contact2
    sx, sy = stride[0], stride[1]
    if alignment == "square":
        heading = np.array([np.sign(sx) or 1.0, 0.0, 0.0]) if view == "side" else \
            np.array([0.0, 1.0 if view == "behind" else -1.0, 0.0])
    else:
        h = np.array([sx, sy, 0.0])
        heading = h / np.linalg.norm(h)
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, heading)
    R = np.stack([heading, y, z])
    pos_lab = pos_up @ R.T
    if used == "L":
        pos_lab = mirror_y(pos_lab)

    forward_img = None
    if view == "side":
        dx = img_track["lead_ankle"][fp, 0] - img_track["rear_ankle"][fp, 0]
        if np.isfinite(dx) and dx != 0:
            forward_img = np.array([np.sign(dx), 0.0])
    if view != "side":
        notes.append("Camera is behind or in front of the hitter: stride length and the forward move need a "
                     "side-on view and are not reported.")
    if alignment == "angled":
        notes.append("Camera marked as angled: stride direction and pelvis/shoulder rotation angles are not "
                     "reported; the line to the pitcher is taken from the stride.")

    motion = Motion(t=t, pos=pos_lab, track=img_track, track_up=img_up, track_forward=forward_img,
                    track_scale=scale_px, heading_known=alignment == "square", fps=fps, hands_valid=hands_valid)

    image_smooth = lowpass(image, fps, CUTOFF_BODY_HZ)
    image_smooth[:, MP_ARM_INDICES] = lowpass(image[:, MP_ARM_INDICES], fps, CUTOFF_ARM_HZ)

    win = (t >= t[fp] - 0.4) & (t <= t[contact] + 0.1)
    lead_ankle_mp = MP[f"{'l' if used == 'R' else 'r'}_ankle"]
    key_vis = track.visibility[:, [MP["l_wrist"], MP["r_wrist"], lead_ankle_mp]]
    quality = {
        "frames": int(n),
        "fps_effective": fps,
        "pose_detected_fraction": float(track.detected.mean()),
        "pose_detected_fraction_window": float(track.detected[win].mean()) if win.any() else 0.0,
        "key_joint_visibility_window": float(np.nanmean(key_vis[win])) if win.any() else 0.0,
        "swaps_repaired": swaps,
        "hands_untracked_frames_window": int((~hands_valid[win]).sum()),
        "camera_view": view,
        "leg_length_px": scale_px,
        "leg_length_world_m": leg_length(pos_lab),
        "batting_side_turn_deg": turn,
    }
    for k in ("lead_ankle", "rear_ankle", "lead_knee", "mid_hip"):
        if not np.isfinite(img_track[k][win]).all():
            notes.append("Some lower-body landmarks are missing around foot plant; foot-plant timing and the "
                         "stride may be unreliable.")
            break
    return Prepared(motion=motion, bats=used, bats_detected=detected, camera_view=view,
                    heading_known=alignment == "square", image_smooth=image_smooth,
                    frame_index=track.frame_index, quality=quality, notes=notes)

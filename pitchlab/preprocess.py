"""Turn raw MediaPipe output into a `Motion` in the right-hander-normalised lab frame.

Steps: low-visibility masking -> left/right swap repair -> one-frame spike removal -> throwing-arm
identification -> masking of untrustworthy throwing-arm frames (blurred, mirrored body, lone samples)
-> gap filling -> smoothing -> provisional release / foot-plant detection -> lab frame from the stride
at foot plant (camera-square or stride-aligned heading) -> mirroring of left-handers.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .events import DEFAULT_CALIBRATION, detect_br, detect_fp, detect_pkh
from .kinematics import Motion, kinematics
from .pose import PoseTrack
from .signals import CUTOFF_ARM_HZ, CUTOFF_BODY_HZ, fill_gaps, lowpass, max_gap_s, smooth_skeleton
from .skeleton import (J, MP, MP_ARM_PAIRS, MP_LEG_PAIRS, leg_length, leg_length_projected,
                       mediapipe_indices, mirror_y)

VISIBILITY_MIN = 0.3
ARM_TRUST_VISIBILITY = 0.7  # throwing elbow/wrist below this are treated as missing (motion blur)
ARM_FALLBACK_VISIBILITY = 0.5   # ...lowered to this when the arm rarely reaches 0.7 during the delivery
ARM_TRUSTED_FRACTION_MIN = 0.4  # "rarely": in fewer than this share of the delivery's detected frames
SWAP_RATIO = 0.5            # swap when the crossed assignment is this much closer to the previous frame
SWAP_MAX_GAP_FRAMES = 3     # never judge a swap against a reference frame older than this
MIN_FRAMES = 30
ARM_MP_INDICES = list(range(13, 23))
ARM_DECISION_PAIRS = [(13, 14), (15, 16)]   # elbows, wrists

# camera coordinates (x right, y down, z away) -> provisional z-up frame (x right, y away, z up)
CAM_TO_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


class PreprocessError(ValueError):
    """The clip cannot be analysed (no person, too short, pitch not visible...)."""


@dataclass
class Prepared:
    motion: Motion
    throws: str                     # handedness used for the analysis
    throws_detected: str            # faster wrist (what the video shows)
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


def repair_swaps(image: np.ndarray, world: np.ndarray, vis: np.ndarray) -> dict[str, np.ndarray]:
    """In-place repair of frames where MediaPipe labelled the left limb as the right one and vice versa.
    Returns, for "legs" and "arms", a per-frame mask of the frames that were swapped.

    Each frame is judged only against the previous judged frame, and only if that frame is at most
    SWAP_MAX_GAP_FRAMES old - comparing with a stale frame let one bad decision flip the arms for the
    rest of a real clip. Arms are judged on elbows and wrists, only when all four are confidently
    tracked (a blurred arm at release gives random positions), and a swap also needs the elbows to
    agree: the throwing hand legitimately sweeps past the glove hand in the follow-through, but MediaPipe
    swaps move the whole arm."""
    swapped = {}
    groups = (
        ("legs", MP_LEG_PAIRS, MP_LEG_PAIRS, 0.0),
        ("arms", MP_ARM_PAIRS, ARM_DECISION_PAIRS, ARM_TRUST_VISIBILITY),
    )
    for name, pairs, decide, min_vis in groups:
        idx = [i for p in decide for i in p]
        mask = np.zeros(len(image), dtype=bool)
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
                    mask[f] = True
            prev = f
        swapped[name] = mask
    return swapped


def trusted_arm_frames(conf: np.ndarray, finite: np.ndarray, legs_swapped: np.ndarray,
                       detected: np.ndarray, delivery: np.ndarray) -> tuple[np.ndarray, float]:
    """Frames whose throwing elbow and wrist positions can be trusted, and the confidence threshold used.

    - Confidence (the lower of elbow and wrist) must reach ARM_TRUST_VISIBILITY: a blurred arm gets
      low confidence and random positions. If the arm reaches it in fewer than ARM_TRUSTED_FRACTION_MIN
      of the delivery's pose-detected frames (dim light, dark clothes), ARM_FALLBACK_VISIBILITY is used
      instead - the worst frames, the blurred ones, are still dropped. Frames where the whole pose was
      lost say nothing about the arm and are left out of that count.
    - Frames on which the legs had to be swapped back are dropped: MediaPipe then mirrored the whole
      body (it confuses the pitcher's front and back as the trunk turns), and the arm labels cannot be
      repaired the same way - the glove arm is hidden, and the throwing-wrist label sits on the glove
      hand with high confidence.
    - A lone confident frame between low-confidence ones is dropped: in a blurred stretch MediaPipe now
      and then places the wrist on the glove or the torso with high confidence for a single frame. (Only
      blur makes a frame lone: a good frame between a mirrored stretch and the release blur is kept.)"""
    judged = delivery & detected
    threshold = ARM_TRUST_VISIBILITY
    if judged.any() and np.mean(conf[judged] >= ARM_TRUST_VISIBILITY) < ARM_TRUSTED_FRACTION_MIN:
        threshold = ARM_FALLBACK_VISIBILITY
    ok = (conf >= threshold) & finite
    lone = ok[1:-1] & ~ok[:-2] & ~ok[2:]
    ok[1:-1][lone] = False
    return ok & ~legs_swapped, threshold


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


def _peak_speed(x: np.ndarray, t: np.ndarray) -> float:
    v = np.linalg.norm(np.gradient(x, t, axis=0), axis=-1)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 98)) if v.size else 0.0


def _leg_length_px(img_c: dict[str, np.ndarray]) -> float:
    try:
        return leg_length_projected(img_c["r_hip"], img_c["r_knee"], img_c["r_ankle"],
                                    img_c["l_hip"], img_c["l_knee"], img_c["l_ankle"])
    except ValueError as exc:
        raise PreprocessError("the legs were never visible") from exc


def prepare(track: PoseTrack, throws: str = "auto", alignment: str = "square",
            capture_fps: float | None = None, calib: dict | None = None) -> Prepared:
    if throws not in ("auto", "R", "L"):
        raise ValueError("throws must be auto, R or L")
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

    # throwing arm = the faster wrist relative to the pelvis, in the image (MediaPipe's depth is too
    # jumpy for the arm at release to compare speeds in 3D)
    img_hip = (image[:, MP["l_hip"]] + image[:, MP["r_hip"]]) / 2.0
    speed_l = _peak_speed(image[:, MP["l_wrist"]] - img_hip, t)
    speed_r = _peak_speed(image[:, MP["r_wrist"]] - img_hip, t)
    if max(speed_l, speed_r) == 0.0:
        raise PreprocessError("the wrists were never tracked")
    detected = "R" if speed_r >= speed_l else "L"
    used = detected
    if throws != "auto" and throws != detected:
        ratio = min(speed_l, speed_r) / max(speed_l, speed_r)
        if ratio > 0.8:
            used = throws     # arms similar: trust the user
        else:
            notes.append(f"You selected {'right' if throws == 'R' else 'left'}-handed, but the "
                         f"{'right' if detected == 'R' else 'left'} arm moves much faster in the video "
                         "(the pose model may have mirrored the body). The faster arm was analysed as the "
                         "throwing arm.")

    # Throwing-arm frames whose positions are guesses (blur around release at 60 fps, a mirrored body)
    # are treated as missing and bridged instead. "Low confidence throughout" is judged over the
    # delivery only (0.6 s either side of the arm's peak speed): during the windup the arm is often
    # hidden behind the body, which is expected.
    side = "r" if used == "R" else "l"
    arm_idx = [MP[f"{side}_elbow"], MP[f"{side}_wrist"]]
    w_speed = np.linalg.norm(np.gradient(image[:, MP[f"{side}_wrist"]] - img_hip, t, axis=0), axis=-1)
    peak_t = t[int(np.nanargmax(w_speed))] if np.isfinite(w_speed).any() else t[len(t) // 2]
    arm_valid, arm_threshold = trusted_arm_frames(
        track.visibility[:, arm_idx].min(axis=1), np.isfinite(image[:, arm_idx]).all(axis=(1, 2)),
        swaps["legs"], track.detected, np.abs(t - peak_t) <= 0.6)
    if arm_threshold < ARM_TRUST_VISIBILITY:
        notes.append("The throwing arm was tracked with low confidence through most of the delivery (dim "
                     "light or dark clothing?); arm metrics may be unreliable.")
    hand = [i for i in range(13, 23) if (i % 2 == 1) == (side == "l")] + arm_idx
    image[np.ix_(~arm_valid, hand)] = np.nan
    world[np.ix_(~arm_valid, hand)] = np.nan

    max_gap = max_gap_s(fps)
    image = fill_gaps(image, t, max_gap)
    world = fill_gaps(world, t, max_gap)

    idx = mediapipe_indices(used)
    names = ("t_shoulder", "t_elbow", "t_wrist", "g_shoulder", "g_elbow", "g_wrist",
             "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle")
    pos_cam = world[:, idx]
    if not np.isfinite(pos_cam).any(axis=(1, 2)).any():
        raise PreprocessError("no usable 3D landmarks")
    pos_up = smooth_skeleton(pos_cam @ CAM_TO_UP.T, fps)

    img_c = {nm: image[:, i] for nm, i in zip(names, idx)}
    lead = "l" if used == "R" else "r"          # the glove-side leg strides
    img_track = {
        "l_ankle": img_c["l_ankle"], "r_ankle": img_c["r_ankle"], "l_knee": img_c["l_knee"],
        "mid_hip": (img_c["r_hip"] + img_c["l_hip"]) / 2.0,
        "t_wrist": img_c["t_wrist"], "t_shoulder": img_c["t_shoulder"],
        # lead heel and toe: points on the foot itself, for the moment it touches down
        "l_heel": image[:, MP[f"{lead}_heel"]], "l_toe": image[:, MP[f"{lead}_foot"]],
    }
    img_track = {k: lowpass(v, fps, CUTOFF_ARM_HZ if k == "t_wrist" else CUTOFF_BODY_HZ) for k, v in img_track.items()}
    scale_px = _leg_length_px(img_c)
    img_up = np.array([0.0, -1.0])

    # provisional events with rotation-invariant signals
    prov = Motion(t=t, pos=pos_up, track=img_track, track_up=img_up, track_forward=None,
                  track_scale=scale_px, heading_known=False, fps=fps, arm_valid=arm_valid)
    kin0 = kinematics(prov)
    br, _ = detect_br(t, kin0["wrist_speed_bridged"], calib)
    fp = detect_fp(t, kin0["foot_speed"], kin0["foot_pos"], br, detect_pkh(t, kin0["knee_height"], br), calib)
    if fp is None:
        fp = int(np.argmin(np.abs(t - (t[br] - calib["fp_to_br_median_s"]))))

    stride = pos_up[fp, J["l_ankle"]] - pos_up[fp, J["r_ankle"]]
    if not np.isfinite(stride).all():
        raise PreprocessError("the feet are not visible at foot plant")
    sx, sy = stride[0], stride[1]
    if abs(sx) >= abs(sy):
        view = "side"
    else:
        view = "behind" if sy > 0 else "front"
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
        dx = img_track["l_ankle"][fp, 0] - img_track["r_ankle"][fp, 0]
        if np.isfinite(dx) and dx != 0:
            forward_img = np.array([np.sign(dx), 0.0])
    if view != "side":
        notes.append("Camera is behind or in front of the pitcher: stride length and pelvis momentum "
                     "need a side-on view and are not reported.")
    if alignment == "angled":
        notes.append("Camera marked as angled: stride direction and pelvis/torso rotation at foot plant "
                     "are not reported; the line to home is taken from the stride.")

    motion = Motion(t=t, pos=pos_lab, track=img_track, track_up=img_up, track_forward=forward_img,
                    track_scale=scale_px, heading_known=alignment == "square", fps=fps, arm_valid=arm_valid)

    image_smooth = lowpass(image, fps, CUTOFF_BODY_HZ)
    image_smooth[:, ARM_MP_INDICES] = lowpass(image[:, ARM_MP_INDICES], fps, CUTOFF_ARM_HZ)

    win = (t >= t[fp] - 0.3) & (t <= t[br] + 0.1)
    key_vis = track.visibility[:, [MP[f"{side}_wrist"], MP[f"{side}_elbow"],
                                   MP[f"{'l' if used == 'R' else 'r'}_ankle"]]]
    quality = {
        "frames": int(n),
        "fps_effective": fps,
        "pose_detected_fraction": float(track.detected.mean()),
        "pose_detected_fraction_window": float(track.detected[win].mean()) if win.any() else 0.0,
        "key_joint_visibility_window": float(np.nanmean(key_vis[win])) if win.any() else 0.0,
        "swaps_repaired": {k: int(v.sum()) for k, v in swaps.items()},
        "arm_trust_visibility": arm_threshold,
        "arm_untracked_frames_window": int((~arm_valid[win]).sum()),
        "camera_view": view,
        "leg_length_px": scale_px,
        "leg_length_world_m": leg_length(pos_lab),
        "throwing_wrist_speed_ratio": float(min(speed_l, speed_r) / max(speed_l, speed_r)),
    }
    for k in ("l_ankle", "r_ankle", "l_knee", "mid_hip"):
        if not np.isfinite(img_track[k][win]).all():
            notes.append("Some lower-body landmarks are missing around foot plant; foot-plant timing and "
                         "stride may be unreliable.")
            break
    return Prepared(motion=motion, throws=used, throws_detected=detected, camera_view=view,
                    heading_known=alignment == "square", image_smooth=image_smooth,
                    frame_index=track.frame_index, quality=quality, notes=notes)

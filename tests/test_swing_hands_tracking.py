"""Hand and arm handling: left/right swap repair with both hands on the bat, batting-side detection from the
direction the body turns, and hand metrics withheld where the hands were bridged."""
import numpy as np
import pytest

from pitchlab.swing.events import Events
from pitchlab.swing.kinematics import TRACK_KEYS, Motion, kinematics
from pitchlab.swing.metrics import compute_metrics
from pitchlab.swing.preprocess import detect_batting_side, repair_swaps
from pitchlab.swing.skeleton import MP
from swing_synthetic import skeleton

LE, RE, LW, RW = 13, 14, 15, 16


def arms_track(n=20):
    """A right-handed hitter's arms seen side-on: both hands together on the bat moving right, the lead (left)
    elbow ahead, the rear (right) elbow behind: (image, world, vis)."""
    image = np.zeros((n, 33, 2))
    for f in range(n):
        image[f, LW] = (600 + 10 * f, 300)
        image[f, RW] = (595 + 10 * f, 305)          # the hands sit a few pixels apart on the handle
        image[f, LE] = (650 + 6 * f, 380)
        image[f, RE] = (520 + 6 * f, 360)
    world = np.zeros((n, 33, 3))
    vis = np.ones((n, 33))
    return image, world, vis


def test_whole_arm_swap_for_one_frame_is_repaired():
    image, world, vis = arms_track()
    image[10, [LE, RE]] = image[10, [RE, LE]]
    image[10, [LW, RW]] = image[10, [RW, LW]]
    counts = repair_swaps(image, world, vis)
    assert counts["arms"] == 1
    assert image[10, LE, 0] == pytest.approx(650 + 60)       # lead elbow back on the lead side


def test_hands_together_on_the_bat_are_not_a_swap():
    # the wrists trade places by a few pixels (both on the handle) but the elbows stay put: no swap
    image, world, vis = arms_track()
    image[8:12, [LW, RW]] = image[8:12, [RW, LW]]
    counts = repair_swaps(image, world, vis)
    assert counts["arms"] == 0


def body_turning(direction: float, n=240, fps=240.0):
    """World landmarks (camera axes: x right, y down, z away) of hips and shoulders turning 100 deg about the
    vertical between t = 0.76 and 1.0 s, inside the 0.7-1.06 s window the detector measures for a hand-speed
    peak at 1.0 s; direction +1 turns counter-clockwise seen from above (right-handed hitter)."""
    t = np.arange(n) / fps
    s = np.clip((t - 0.76) / 0.24, 0, 1)
    a = direction * np.radians(100) * (3 * s ** 2 - 2 * s ** 3)
    world = np.zeros((n, 33, 3))
    for (l, r), half in (((MP["l_hip"], MP["r_hip"]), 0.15), ((MP["l_shoulder"], MP["r_shoulder"]), 0.2)):
        v = np.stack([np.cos(a), np.zeros(n), np.sin(a)], axis=1) * half      # z away = "y" in the z-up frame
        world[:, r] = v
        world[:, l] = -v
    image = np.zeros((n, 33, 2))
    return t, world, image, fps


@pytest.mark.parametrize("direction,expected", [(1.0, "R"), (-1.0, "L")])
def test_batting_side_from_the_direction_of_turn(direction, expected):
    t, world, image, fps = body_turning(direction)
    side, confident, turn = detect_batting_side(world, image, t, fps, peak_t=1.0)
    assert (side, confident) == (expected, True)
    assert turn == pytest.approx(direction * 200, abs=15)
    swapped = world.copy()                                   # MediaPipe swapping every left/right label
    for a, b in ((MP["l_hip"], MP["r_hip"]), (MP["l_shoulder"], MP["r_shoulder"])):
        swapped[:, [a, b]] = swapped[:, [b, a]]
    assert detect_batting_side(swapped, image, t, fps, peak_t=1.0)[0] == expected


def test_striding_foot_decides_when_the_turn_is_unclear():
    t, world, image, fps = body_turning(0.0)
    image[:, MP["l_ankle"], 0] = np.interp(t, [0.3, 0.8], [400, 600])      # the left foot strides
    image[:, MP["r_ankle"], 0] = 300
    side, confident, turn = detect_batting_side(world, image, t, fps, peak_t=1.0)
    assert (side, confident) == ("R", True) and abs(turn) < 1
    image[:, MP["l_ankle"], 0] = 400                                          # neither foot moves: no confidence
    assert detect_batting_side(world, image, t, fps, peak_t=1.0)[1] is False


def test_bridged_hands_hide_hand_metrics():
    n, fps = 120, 240.0
    pos = np.repeat(skeleton(), n, axis=0)
    t = np.arange(n) / fps
    for name_i in (2, 5):                                    # wrists move so speeds are defined
        pos[:, name_i, 0] += 1.5 * t
    valid = np.ones(n, dtype=bool)
    valid[95:99] = False                                     # hands lost just before contact (sample 100)
    m = Motion(t=t, pos=pos, track={k: np.zeros((n, 3)) for k in TRACK_KEYS}, track_up=np.array([0.0, 0.0, 1.0]),
               track_forward=np.array([1.0, 0.0, 0.0]), track_scale=0.9, heading_known=True, fps=fps, hands_valid=valid)
    ev = Events(fp=70, contact=100, load=40)
    out = compute_metrics(m, kinematics(m), ev, height_m=1.8)
    assert out["max_hand_speed"] is None and out["max_lead_arm_speed"] is None
    assert out["lead_elbow_flexion_contact"] is None
    assert out["rear_elbow_flexion_fp"] is not None          # the hands were tracked at foot plant
    m.hands_valid = np.ones(n, dtype=bool)
    assert compute_metrics(m, kinematics(m), ev, height_m=1.8)["max_hand_speed"] is not None

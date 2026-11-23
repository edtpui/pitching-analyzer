"""Throwing-arm handling learned from a real 60 fps clip: MediaPipe swapped the arms for one frame at
release and lost the blurred arm for the next three."""
import numpy as np
import pytest

from pitchlab.events import detect_br_hand, detect_events
from pitchlab.preprocess import repair_swaps
from test_events import CAL, synthetic_signals

LE, RE, LW, RW = 13, 14, 15, 16


def arms_track(n=20):
    """Right (throwing) arm moving steadily to the right, left arm still: (image, world, vis)."""
    image = np.zeros((n, 33, 2))
    for f in range(n):
        image[f, RE] = (500 + 10 * f, 350)
        image[f, RW] = (480 + 12 * f, 280)
        image[f, LE] = (750, 400)
        image[f, LW] = (820, 450)
    world = np.zeros((n, 33, 3))
    vis = np.ones((n, 33))
    return image, world, vis


def test_whole_arm_swap_for_one_frame_is_repaired():
    image, world, vis = arms_track()
    image[10, [LE, RE]] = image[10, [RE, LE]]
    image[10, [LW, RW]] = image[10, [RW, LW]]
    counts = repair_swaps(image, world, vis)
    assert counts["arms"] == 1
    assert image[10, RW, 0] == pytest.approx(480 + 120)       # throwing wrist back on the throwing side


def test_follow_through_crossing_is_not_a_swap():
    # the throwing hand sweeps past the glove hand, but the elbows stay on their own sides
    image, world, vis = arms_track()
    for f in range(8, 14):
        image[f, RW] = (820 - 2 * (f - 8), 450 + 3 * (f - 8))   # throwing wrist arrives at the glove hand
    counts = repair_swaps(image, world, vis)
    assert counts["arms"] == 0


def test_low_confidence_frames_are_not_judged_and_stale_reference_is_dropped():
    image, world, vis = arms_track(30)
    vis[6:16, [RE, RW]] = 0.4                                  # blurred arm for 10 frames
    image[16:, [LE, RE]] = image[16:, [RE, LE]]                # labels differ after the gap...
    image[16:, [LW, RW]] = image[16:, [RW, LW]]
    counts = repair_swaps(image, world, vis)
    # ...but the last confident frame is 10 frames back, too old to judge against: no swaps
    assert counts["arms"] == 0


def test_release_from_hand_crossing_across_a_bridged_gap():
    t = np.arange(0, 1.0, 1 / 60)
    fp = 30                                           # foot plant at 0.5 s
    hand = np.full(len(t), -0.4)
    i0, i1 = 34, 39                                   # last tracked "behind" sample, first tracked "ahead"
    hand[i1:] = 0.6
    hand[i0:i1 + 1] = np.linspace(-0.4, 0.6, i1 - i0 + 1)      # what gap bridging produces
    br = detect_br_hand(t, hand, fp, {"br_hand_ahead": 0.41, "br_hand_offset_s": 0.0})
    # crossing of 0.41 between the bracketing samples, ~81% of the way from i0 to i1
    assert br == round(i0 + (0.41 + 0.4) / 1.0 * (i1 - i0))


def test_layback_is_estimated_when_the_arm_was_not_tracked():
    t, kin = synthetic_signals()
    br_auto = detect_events(t, kin, CAL).br
    kin["arm_valid"] = np.ones(len(t), dtype=bool)
    kin["arm_valid"][br_auto - 5:br_auto] = False             # arm lost in the 20 ms before release
    ev = detect_events(t, kin, CAL)
    assert ev.source["mer"] == "estimated"
    assert t[ev.br] - t[ev.mer] == pytest.approx(CAL.get("mer_to_br_median_s", 0.031), abs=1 / 240)
    assert any("blurred" in n for n in ev.notes)

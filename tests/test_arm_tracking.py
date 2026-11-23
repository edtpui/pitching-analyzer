"""Throwing-arm handling learned from real 60 fps clips. First clip: MediaPipe swapped the arms for one
frame at release and lost the blurred arm for the next three. Second clip (night, dark clothes): it
mirrored the whole body for several frames after foot plant, lost the arm from layback until well after
release, and then put the throwing wrist on the glove hand and the torso with high confidence."""
import numpy as np
import pytest

from pitchlab.events import detect_br_hand, detect_events, release_from_last_tracked
from pitchlab.preprocess import ARM_FALLBACK_VISIBILITY, ARM_TRUST_VISIBILITY, repair_swaps, trusted_arm_frames
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
    swapped = repair_swaps(image, world, vis)
    assert np.flatnonzero(swapped["arms"]).tolist() == [10] and not swapped["legs"].any()
    assert image[10, RW, 0] == pytest.approx(480 + 120)       # throwing wrist back on the throwing side


def test_follow_through_crossing_is_not_a_swap():
    # the throwing hand sweeps past the glove hand, but the elbows stay on their own sides
    image, world, vis = arms_track()
    for f in range(8, 14):
        image[f, RW] = (820 - 2 * (f - 8), 450 + 3 * (f - 8))   # throwing wrist arrives at the glove hand
    assert not repair_swaps(image, world, vis)["arms"].any()


def test_low_confidence_frames_are_not_judged_and_stale_reference_is_dropped():
    image, world, vis = arms_track(30)
    vis[6:16, [RE, RW]] = 0.4                                  # blurred arm for 10 frames
    image[16:, [LE, RE]] = image[16:, [RE, LE]]                # labels differ after the gap...
    image[16:, [LW, RW]] = image[16:, [RW, LW]]
    # ...but the last confident frame is 10 frames back, too old to judge against: no swaps
    assert not repair_swaps(image, world, vis)["arms"].any()


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


def _trust(conf, legs_swapped=None, detected=None):
    n = len(conf)
    return trusted_arm_frames(np.asarray(conf, dtype=float), np.ones(n, dtype=bool),
                              np.zeros(n, dtype=bool) if legs_swapped is None else legs_swapped,
                              np.ones(n, dtype=bool) if detected is None else detected, np.ones(n, dtype=bool))


def test_mirrored_body_and_lone_confident_frames_are_not_trusted():
    conf = np.full(20, 0.95)
    conf[[8, 10]] = 0.2                       # frame 9: one confident frame inside a blurred stretch
    legs = np.zeros(20, dtype=bool)
    legs[14:17] = True                        # the legs had to be swapped back: the whole body was mirrored
    ok, threshold = _trust(conf, legs)
    assert threshold == ARM_TRUST_VISIBILITY
    assert not ok[[8, 9, 10, 14, 15, 16]].any()
    assert ok[[7, 11, 13, 17]].all()


def test_dim_light_lowers_the_threshold_but_blurred_frames_stay_out():
    # The arm reaches 0.7 in only 25% of the delivery. This used to switch the check off entirely, so
    # the blurred frames around release were trusted and release landed on a glitch.
    conf = np.tile([0.9, 0.6, 0.6, 0.1], 10)
    ok, threshold = _trust(conf)
    assert threshold == ARM_FALLBACK_VISIBILITY
    assert ok[conf >= 0.5].all() and not ok[conf < 0.5].any()


def test_frames_without_any_pose_do_not_count_as_low_arm_confidence():
    conf = np.tile([0.9, 0.9, 0.0, 0.0], 10)                   # the pitcher was lost on half the frames
    _, threshold = _trust(conf, detected=conf > 0)
    assert threshold == ARM_TRUST_VISIBILITY


def blurred_release(lost_before=2, lost_after=9, fps=60.0, t_br=1.5):
    """Side-on signals at 60 fps in which the throwing arm is lost from `lost_before` frames before release
    to `lost_after` frames after it, then the wrist label sits on the torso (hand_ahead -0.14), as in the
    second real clip. The hand reaches its deepest layback (-0.45) 50 ms before release."""
    t, kin = synthetic_signals(fps=fps, t_br=t_br)
    ha = np.full(len(t), -0.1)
    lay = (t >= t_br - 0.15) & (t < t_br - 0.05)
    ha[lay] = np.linspace(-0.2, -0.45, lay.sum())                  # cocking into layback
    rise = t >= t_br - 0.05
    ha[rise] = np.minimum(-0.45 + (t[rise] - (t_br - 0.05)) * 17.2, 0.8)   # 0.41 at release
    ibr = int(np.argmin(np.abs(t - t_br)))
    lost = slice(ibr - lost_before, ibr + lost_after + 1)
    ha[lost] = np.nan
    ha[ibr + lost_after + 1:ibr + lost_after + 7] = -0.14          # wrist label on the torso
    ha[ibr + lost_after + 7:] = 0.2                                # follow-through, hand in front
    kin["hand_ahead"] = ha
    kin["arm_valid"] = np.isfinite(ha)
    kin["wrist_speed_bridged"] = kin["wrist_speed"].copy()        # the line across the gap is fast
    kin["wrist_speed"][lost] = np.nan
    return t, kin, ibr


def test_release_is_extrapolated_when_the_arm_is_lost_through_release():
    t, kin, ibr = blurred_release()
    ev = detect_events(t, kin, CAL)
    assert ev.br == ibr                       # hand last seen 50 ms before release, still laid back
    assert ev.source["br"] == "estimated" and ev.source["mer"] == "estimated"
    assert any("last tracked position" in n for n in ev.notes)
    assert ev.source["fp"] == "auto" and ev.fp < ev.mer < ev.br


def test_release_is_found_when_the_rough_release_lands_on_a_glitch():
    t, kin, ibr = blurred_release()
    kin["wrist_speed_bridged"] = kin["wrist_speed_bridged"] + 60 * np.exp(-((t - 1.26) / 0.02) ** 2)
    ev = detect_events(t, kin, CAL)           # rough release during the stride: no foot plant from it
    assert ev.br == ibr and ev.source["fp"] == "auto"


def test_mirrored_body_gap_while_the_arm_swings_back_is_not_release():
    t, kin, ibr = blurred_release()
    ha = kin["hand_ahead"]
    ha[82] = -0.6                             # arm swung far back, then the body was mirrored for 2 frames
    ha[83:85] = np.nan                        # ...and the hand came back tracked, still laid back
    fp = int(np.argmin(np.abs(t - 1.36)))
    assert release_from_last_tracked(t, ha, fp, {"br_hand_ahead": 0.41}) == ibr


def test_layback_is_estimated_when_the_arm_was_not_tracked():
    t, kin = synthetic_signals()
    br_auto = detect_events(t, kin, CAL).br
    kin["arm_valid"] = np.ones(len(t), dtype=bool)
    kin["arm_valid"][br_auto - 5:br_auto] = False             # arm lost in the 20 ms before release
    ev = detect_events(t, kin, CAL)
    assert ev.source["mer"] == "estimated"
    assert t[ev.br] - t[ev.mer] == pytest.approx(CAL.get("mer_to_br_median_s", 0.031), abs=1 / 240)
    assert any("blurred" in n for n in ev.notes)

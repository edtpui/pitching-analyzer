import numpy as np
import pytest

from pitchlab.swing.events import Events, calib_for_view, detect_contact, detect_events, validate_order

CAL = {"contact_method": "image", "contact_offset_s": 0.062, "contact_offset_hand3d_s": 0.060,
       "contact_offset_torso_s": 0.040, "fp_method": "height", "fp_speed_fraction": 0.3, "fp_height": 0.015,
       "fp_plant_radius": 0.03, "fp_to_contact_median_s": 0.117, "min_leg_lift": 0.06}
CAL_RADIUS = {**CAL, "fp_method": "radius"}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 3 * x ** 2 - 2 * x ** 3


def synthetic_signals(fps=240.0, t_contact=1.5, t_fp=1.37, t_load=1.23, t_lift=0.9, hover=0.0, lift_height=0.2):
    """Image hand speed peaks 62 ms before contact (plus a smaller follow-through bump), 3D hand speed 60 ms and
    shoulder rotation speed 40 ms before it; the lead foot travels 0.8 leg lengths toward the pitcher from the leg
    lift to foot plant (optionally hovering `hover` s above its landing spot first), then stays put; the pelvis
    coils to -25 deg at the load and turns open to ~95 deg; the knee rises `lift_height` leg lengths at the
    leg lift."""
    t = np.arange(0, 2.0, 1 / fps)
    hand = 10 * np.exp(-((t - (t_contact - 0.062)) / 0.03) ** 2) + 4 * np.exp(-((t - (t_contact + 0.2)) / 0.04) ** 2) + 0.5
    hand3d = 9 * np.exp(-((t - (t_contact - 0.060)) / 0.03) ** 2) + 0.5
    torso = 900 * np.exp(-((t - (t_contact - 0.040)) / 0.04) ** 2)
    t_arrive = t_fp - hover
    ph = smoothstep((t - t_lift) / (t_arrive - t_lift))
    x = 0.8 * ph
    y = 0.1 * np.sin(np.pi * ph)
    if hover:
        drop = np.clip((t - t_arrive) / hover, 0, 1)
        y = np.where(t >= t_arrive, 0.06 * (1 - drop), y + 0.06 * (t >= t_lift) * ph)
    foot_pos = np.stack([x, y], axis=1)                 # (toward the pitcher, up) in leg lengths
    foot_speed = np.linalg.norm(np.gradient(foot_pos, t, axis=0), axis=1)
    coil = -10 - 15 * smoothstep((t - (t_load - 0.4)) / 0.4)
    turn = -25 + 120 * smoothstep((t - t_load) / (t_contact + 0.05 - t_load))
    pelvis = np.where(t <= t_load, coil, turn)
    knee = lift_height * np.exp(-((t - t_lift) / 0.12) ** 2)
    return t, {"hand_speed_img": hand, "hand_speed": hand3d, "torso_rot_velo": torso, "foot_speed": foot_speed,
               "foot_pos": foot_pos, "foot_height": foot_pos[:, 1], "knee_height": knee, "pelvis_rot": pelvis}


def test_detects_all_events_in_order():
    t, kin = synthetic_signals()
    ev = detect_events(t, kin, CAL)
    assert t[ev.contact] == pytest.approx(1.5, abs=1 / 240)
    assert 1.31 < t[ev.fp] <= 1.38
    assert t[ev.load] == pytest.approx(1.23, abs=2 / 240)
    assert t[ev.lift] == pytest.approx(0.9, abs=2 / 240)
    assert ev.source == {"lift": "auto", "load": "auto", "fp": "auto", "contact": "auto"}
    validate_order(ev, len(t))


def test_contact_follows_the_calibrated_offset():
    t, kin = synthetic_signals()
    ev = detect_events(t, kin, {**CAL, "contact_offset_s": 0.04})
    assert t[ev.contact] == pytest.approx(1.5 - 0.022, abs=1 / 240)


def test_contact_from_behind_uses_3d_hands_and_shoulders():
    t, kin = synthetic_signals()
    kin["hand_speed_img"] = np.exp(-((t - 1.62) / 0.03) ** 2)       # from behind the image peak is meaningless
    kin["torso_rot_velo"] = -kin["torso_rot_velo"]                   # before mirroring a left-hander turns clockwise
    c = calib_for_view(CAL, "behind")
    assert c["contact_method"] == "3d" and c["fp_method"] == "radius"
    contact, _ = detect_contact(t, kin, c)
    assert t[contact] == pytest.approx(1.5, abs=1 / 240)
    assert calib_for_view(c, "side")["contact_method"] == "image"


@pytest.mark.parametrize("cal", [CAL, CAL_RADIUS], ids=["height", "radius"])
def test_hovering_foot_is_not_planted_until_it_lands(cal):
    # the foot stops travelling forward 80 ms early and slowly drops 0.06 leg lengths; speed alone would call
    # foot plant then, the planted height / radius test waits until it is (nearly) down
    t, kin = synthetic_signals(hover=0.08)
    ev = detect_events(t, kin, cal)
    assert t[ev.fp] >= 1.325


def test_toe_tap_swing_has_no_leg_lift():
    t, kin = synthetic_signals(lift_height=0.02)
    ev = detect_events(t, kin, CAL)
    assert ev.lift is None
    assert ev.source["lift"] == "missing"
    assert any("leg lift" in n for n in ev.notes)
    assert ev.source["fp"] == "auto" and ev.source["contact"] == "auto"


def test_clip_starting_after_the_load():
    t, kin = synthetic_signals()
    kin["pelvis_rot"] = np.linspace(0, 100, len(t))          # hips only ever turn open: the load is before the clip
    ev = detect_events(t, kin, CAL)
    assert ev.load is None
    assert ev.source["load"] == "missing"
    assert any("load" in n for n in ev.notes)


def test_load_is_not_searched_without_the_lab_frame():
    t, kin = synthetic_signals()
    del kin["pelvis_rot"]                                    # provisional, rotation-free kinematics
    ev = detect_events(t, kin, CAL)
    assert ev.load is None and ev.source["load"] == "missing"


@pytest.mark.parametrize("cal", [CAL, CAL_RADIUS], ids=["height", "radius"])
def test_foot_plant_fallback_when_foot_never_stops(cal):
    t, kin = synthetic_signals()
    kin["foot_pos"] = np.stack([3.0 * t, 0.5 + 0.2 * t], axis=1)
    kin["foot_height"] = kin["foot_pos"][:, 1]
    kin["foot_speed"] = np.full(len(t), 3.0)
    ev = detect_events(t, kin, cal)
    assert ev.source["fp"] == "estimated"
    assert t[ev.contact] - t[ev.fp] == pytest.approx(0.117, abs=1 / 240)


def test_untracked_hands_raise():
    t, kin = synthetic_signals()
    kin["hand_speed_img"] = np.full(len(t), np.nan)
    with pytest.raises(ValueError):
        detect_events(t, kin)


def test_blurred_hands_at_the_peak_are_noted():
    t, kin = synthetic_signals()
    valid = np.ones(len(t), dtype=bool)
    i = int(np.argmin(np.abs(t - (1.5 - 0.062))))
    valid[i - 2:i + 2] = False
    kin["hands_valid"] = valid
    ev = detect_events(t, kin, CAL)
    assert any("blurred" in n for n in ev.notes)


def test_reference_foot_plant_is_the_front_foot_down_flat():
    """On lab data, foot plant = first sample after the lead ankle's peak speed within 5 mm of its planted height."""
    from pitchlab.swing.driveline import kinematic_foot_plant
    from swing_synthetic import canonical_index, skeleton

    fps = 360.0
    t = np.arange(0, 1.6, 1 / fps)
    pos = np.repeat(skeleton(), len(t), axis=0)
    u = smoothstep((t - 0.8) / 0.4)                       # stride: 0.8 -> 1.2 s, 0.6 m forward
    lead = canonical_index("lead_ankle")
    pos[:, lead, 0] += 0.6 * u
    heel = np.where(t <= 1.2, u, np.clip((1.3 - t) / 0.1, 0, 1))      # lands on the toes 3 cm up, heel down by 1.3 s
    pos[:, lead, 2] += 0.12 * np.sin(np.pi * u) + 0.03 * heel
    contact = int(np.argmin(np.abs(t - 1.45)))
    i = kinematic_foot_plant(t, pos, contact)
    assert i is not None
    assert pos[i, lead, 2] - pos[contact, lead, 2] < 0.005 <= pos[i - 1, lead, 2] - pos[contact, lead, 2]
    assert 1.25 < t[i] < 1.3


@pytest.mark.parametrize("ev", [Events(fp=60, contact=50), Events(fp=50, contact=60, load=55),
                                Events(fp=50, contact=60, lift=52), Events(fp=50, contact=600)])
def test_validate_order_rejects_bad_events(ev):
    with pytest.raises(ValueError):
        validate_order(ev, 100)

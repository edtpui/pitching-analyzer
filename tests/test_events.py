import numpy as np
import pytest

from pitchlab.events import DEFAULT_CALIBRATION, Events, detect_contact, detect_events, validate_order

CAL = {"br_offset_s": 0.0, "fp_speed_fraction": 0.2, "fp_plant_radius": 0.03, "fp_to_br_median_s": 0.134}


def synthetic_signals(fps=240.0, t_br=1.5, t_fp=1.36, t_pkh=0.8, hover=0.0):
    """Wrist peaks at release; the lead foot travels 1.6 leg lengths toward home between leg lift and
    foot plant (optionally hovering `hover` s above its landing spot first), then stays put."""
    t = np.arange(0, 2.0, 1 / fps)
    wrist = np.exp(-((t - t_br) / 0.03) ** 2) * 30 + 0.5
    t_arrive = t_fp - hover
    ph = np.clip((t - t_pkh) / (t_arrive - t_pkh), 0, 1)
    x = 1.6 * (3 * ph ** 2 - 2 * ph ** 3)
    y = 0.25 * np.sin(np.pi * ph)
    if hover:
        drop = np.clip((t - t_arrive) / hover, 0, 1)
        y = np.where(t >= t_arrive, 0.06 * (1 - drop), y + 0.06 * (t >= t_pkh) * ph)
    foot_pos = np.stack([x, y], axis=1)
    foot_speed = np.linalg.norm(np.gradient(foot_pos, t, axis=0), axis=1)
    knee = np.exp(-((t - t_pkh) / 0.15) ** 2)
    er = 100 + 80 * np.exp(-((t - (t_br - 0.03)) / 0.02) ** 2)
    return t, {"wrist_speed": wrist, "foot_speed": foot_speed, "foot_pos": foot_pos, "knee_height": knee, "shoulder_er": er}


def test_detects_all_events_in_order():
    t, kin = synthetic_signals()
    ev = detect_events(t, kin, CAL)
    assert t[ev.br] == pytest.approx(1.5, abs=1 / 240)
    assert t[ev.mer] == pytest.approx(1.47, abs=2 / 240)
    assert t[ev.pkh] == pytest.approx(0.8, abs=2 / 240)
    assert 1.31 < t[ev.fp] <= 1.37
    assert ev.source == {"pkh": "auto", "fp": "auto", "mer": "auto", "br": "auto"}
    validate_order(ev, len(t))


def _with_foot_points(kin, t_heel=None, t_toe=None):
    """Add heel / toe tracks that settle at the given times (a copy of the foot path arriving then)."""
    for name, t_arrive in (("heel", t_heel), ("toe", t_toe)):
        if t_arrive is not None:
            _, point = synthetic_signals(t_fp=t_arrive)
            kin[f"{name}_pos"], kin[f"{name}_speed"] = point["foot_pos"], point["foot_speed"]
    return kin


def test_foot_plant_follows_the_heel_touchdown_when_the_ankle_point_keeps_creeping():
    # Real night clip: the heel landed 7 frames before MediaPipe's ankle point (on the lower shin) stopped
    # creeping forward, so the ankle rule put foot plant 4 frames after the foot was flat.
    t, kin = synthetic_signals()                                # ankle point settles at 1.36 s
    ankle_only = detect_events(t, dict(kin), CAL).fp
    kin = _with_foot_points(kin, t_heel=1.28, t_toe=1.31)
    contact = detect_contact(t, kin, int(np.argmin(np.abs(t - 1.5))), None, {**DEFAULT_CALIBRATION, **CAL})
    ev = detect_events(t, kin, CAL)
    assert 1.24 < t[contact] <= 1.28 + 1 / 240                 # touch-down from the heel
    assert t[ev.fp] == pytest.approx(t[contact] + DEFAULT_CALIBRATION["fc_to_fp_median_s"], abs=1 / 240)
    assert ev.fp < ankle_only and ev.source["fp"] == "auto"


def test_toe_first_landing_counts_as_touch_down():
    t, kin = synthetic_signals()
    kin = _with_foot_points(kin, t_heel=1.34, t_toe=1.29)
    contact = detect_contact(t, kin, int(np.argmin(np.abs(t - 1.5))), None, {**DEFAULT_CALIBRATION, **CAL})
    assert 1.25 < t[contact] <= 1.29 + 1 / 240


def test_ankle_rule_stands_when_the_foot_lands_with_it():
    # heel and toe settle with the ankle: touch-down + 19 ms comes after the ankle rule, which stays
    t, kin = synthetic_signals()
    ankle_only = detect_events(t, dict(kin), CAL).fp
    assert detect_events(t, _with_foot_points(kin, t_heel=1.36, t_toe=1.36), CAL).fp == ankle_only


def test_layback_ignores_the_undefined_rotation_of_a_straight_elbow():
    # At release the elbow is nearly straight and the rotation angle (taken from the forearm's direction)
    # swings by ~170 degrees on noise; on a simulated pitch that swing became a 354-degree "max layback".
    t, kin = synthetic_signals()
    br = int(np.argmin(np.abs(t - 1.5)))
    kin["elbow_flexion"] = np.where(t < 1.49, 90.0, 5.0)        # elbow straightens over the last 10 ms
    kin["shoulder_er"][br - 1:br + 1] = 354.0
    ev = detect_events(t, kin, CAL)
    assert t[ev.mer] == pytest.approx(1.47, abs=2 / 240) and ev.source["mer"] == "auto"


def test_hovering_foot_is_not_planted_until_it_lands():
    # the foot stops travelling forward 80 ms early (t = 1.28 s), slowly drops 0.06 leg lengths and lands at
    # 1.36 s; speed alone would call foot plant at 1.28 s, the planted-radius test waits until it is within
    # 0.03 leg lengths of the ground (1.32 s)
    t, kin = synthetic_signals(hover=0.08)
    ev = detect_events(t, kin, CAL)
    assert t[ev.fp] >= 1.315


def test_missing_leg_lift_is_reported():
    t, kin = synthetic_signals()
    kin["knee_height"] = np.linspace(1, 0, len(t))          # clip starts after the leg lift
    ev = detect_events(t, kin, CAL)
    assert ev.pkh is None
    assert ev.source["pkh"] == "missing"
    assert any("leg lift" in n for n in ev.notes)


def test_foot_plant_fallback_when_foot_never_stops():
    t, kin = synthetic_signals()
    kin["foot_pos"] = np.stack([3.0 * t, np.zeros_like(t)], axis=1)
    kin["foot_speed"] = np.full(len(t), 3.0)
    ev = detect_events(t, kin, CAL)
    assert ev.source["fp"] == "estimated"
    assert t[ev.br] - t[ev.fp] == pytest.approx(0.134, abs=1 / 240)


def test_untracked_wrist_raises():
    t, kin = synthetic_signals()
    kin["wrist_speed"] = np.full(len(t), np.nan)
    with pytest.raises(ValueError):
        detect_events(t, kin)


@pytest.mark.parametrize("ev", [Events(fp=50, mer=40, br=60), Events(fp=70, mer=75, br=60),
                                Events(fp=50, mer=55, br=60, pkh=55), Events(fp=50, mer=55, br=600)])
def test_validate_order_rejects_bad_events(ev):
    with pytest.raises(ValueError):
        validate_order(ev, 100)

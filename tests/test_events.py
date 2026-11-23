import numpy as np
import pytest

from pitchlab.events import Events, detect_events, validate_order

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


def test_hovering_foot_is_not_planted_until_it_lands():
    # the foot stops travelling forward 80 ms early (t = 1.28 s), slowly drops 0.06 leg lengths and lands at\n    # 1.36 s; speed alone would call foot plant at 1.28 s, the planted-radius test waits until it is within\n    # 0.03 leg lengths of the ground (1.32 s)
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

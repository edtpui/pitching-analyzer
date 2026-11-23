"""Geometric definitions on hand-built skeletons with known answers."""
import numpy as np
import pytest

from pitchlab.kinematics import TRACK_KEYS, Motion, kinematics
from pitchlab.metrics import normalise_er
from pitchlab.signals import fill_gaps, lowpass, resample, unwrap_deg
from synthetic import skeleton


def motion_of(pos: np.ndarray, frames: int = 40) -> Motion:
    pos = np.repeat(pos, frames, axis=0)
    t = np.arange(frames) / 240.0
    track = {k: np.zeros((frames, 3)) for k in TRACK_KEYS}
    return Motion(t=t, pos=pos, track=track, track_up=np.array([0.0, 0.0, 1.0]),
                  track_forward=np.array([1.0, 0.0, 0.0]), track_scale=0.9, heading_known=True, fps=240.0)


def k0(pos: np.ndarray) -> dict:
    return {k: v[0] for k, v in kinematics(motion_of(pos)).items()}


def test_square_stance_faces_home():
    k = k0(skeleton())
    assert k["pelvis_rot"] == pytest.approx(90.0)
    assert k["torso_rot"] == pytest.approx(90.0)
    assert k["hip_shoulder_sep"] == pytest.approx(0.0)


def test_closed_stance_is_zero_rotation():
    # right-hander side-on, glove side toward home: rear (right) hip toward second base
    k = k0(skeleton(r_hip=(-0.15, 0, 1.0), l_hip=(0.15, 0, 1.0), t_shoulder=(-0.2, 0, 1.5), g_shoulder=(0.2, 0, 1.5)))
    assert k["pelvis_rot"] == pytest.approx(0.0, abs=1e-9)
    assert k["torso_rot"] == pytest.approx(0.0, abs=1e-9)


def test_separation_sign_pelvis_ahead_of_torso():
    a = np.radians(45.0)   # pelvis rotated 45 deg toward home from closed, torso still closed
    rh = (-0.15 * np.cos(a), -0.15 * np.sin(a), 1.0)
    lh = (0.15 * np.cos(a), 0.15 * np.sin(a), 1.0)
    k = k0(skeleton(r_hip=rh, l_hip=lh, t_shoulder=(-0.2, 0, 1.5), g_shoulder=(0.2, 0, 1.5)))
    assert k["pelvis_rot"] == pytest.approx(45.0)
    assert k["hip_shoulder_sep"] == pytest.approx(45.0)


def test_elbow_and_shoulder_angles():
    # arm abducted 90 deg to the side, forearm straight up: elbow 90, abduction 90, ER 90, horiz ab 0
    k = k0(skeleton())
    assert k["elbow_flexion"] == pytest.approx(90.0)
    assert k["shoulder_abduction"] == pytest.approx(90.0)
    assert k["shoulder_er"] == pytest.approx(90.0)
    assert k["shoulder_horiz_abduction"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("wrist,expected", [((0.28, -0.5, 1.5), 0.0), ((-0.28, -0.5, 1.5), 180.0)])
def test_external_rotation_forward_and_layback(wrist, expected):
    k = k0(skeleton(t_wrist=wrist))
    assert abs(((k["shoulder_er"] - expected) + 180) % 360 - 180) < 1e-6


def test_horizontal_abduction_positive_when_elbow_behind():
    k = k0(skeleton(t_elbow=(-0.1, -0.5, 1.5), t_wrist=(-0.1, -0.5, 1.78)))
    assert k["shoulder_horiz_abduction"] == pytest.approx(np.degrees(np.arctan2(0.1, 0.3)))


def test_knee_flexion_and_trunk_tilt():
    tilt = np.radians(30)
    mid_hip = np.array([0, 0, 1.0])
    sh = mid_hip + 0.5 * np.array([np.sin(tilt), 0, np.cos(tilt)])
    k = k0(skeleton(l_knee=(0.45, 0.15, 1.0), l_ankle=(0.45, 0.15, 0.55),
                    t_shoulder=tuple(sh + [0, -0.2, 0]), g_shoulder=tuple(sh + [0, 0.2, 0])))
    assert k["lead_knee_flexion"] == pytest.approx(90.0)
    assert k["trunk_flexion"] == pytest.approx(30.0)
    assert k["trunk_lateral"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("wrist,slot", [((0, -0.2, 2.1), 90.0), ((0, -0.8, 1.5), 0.0)])
def test_arm_slot(wrist, slot):
    assert k0(skeleton(t_wrist=wrist))["arm_slot"] == pytest.approx(slot)


def test_normalise_er_shifts_whole_turns():
    er = np.array([400.0, 450.0, 530.0])
    out = normalise_er(er, 0)
    assert out[0] == pytest.approx(40.0)
    assert np.allclose(np.diff(out), np.diff(er))


def test_fill_gaps_respects_max_gap():
    t = np.arange(10) / 10.0
    x = np.arange(10, dtype=float)
    x[2] = np.nan            # short gap -> filled
    x[5:9] = np.nan          # long gap -> kept
    out = fill_gaps(x, t, max_gap_s=0.25)
    assert out[2] == pytest.approx(2.0)
    assert np.isnan(out[5:9]).all()


def test_lowpass_keeps_slow_signal_and_removes_noise():
    fs = 240.0
    t = np.arange(0, 2, 1 / fs)
    slow = np.sin(2 * np.pi * 2 * t)
    noisy = slow + 0.3 * np.sin(2 * np.pi * 60 * t)
    out = lowpass(noisy, fs, 12.0)
    assert np.max(np.abs(out[50:-50] - slow[50:-50])) < 0.02


def test_resample_and_unwrap():
    t = np.linspace(0, 1, 11)
    x = t * 10
    assert resample(x, t, np.array([0.05, 0.55])) == pytest.approx([0.5, 5.5])
    assert unwrap_deg(np.array([170.0, -170.0, -150.0])) == pytest.approx([170.0, 190.0, 210.0])

"""Geometric definitions on hand-built skeletons with known answers."""
import numpy as np
import pytest

from pitchlab.swing.kinematics import TRACK_KEYS, Motion, kinematics
from pitchlab.swing.signals import fill_gaps, lowpass, resample, unwrap_deg
from pitchlab.swing.skeleton import J
from swing_synthetic import skeleton


def motion_of(pos: np.ndarray, frames: int = 40, fps: float = 240.0) -> Motion:
    if len(pos) == 1:
        pos = np.repeat(pos, frames, axis=0)
    t = np.arange(len(pos)) / fps
    track = {k: np.zeros((len(pos), 3)) for k in TRACK_KEYS}
    return Motion(t=t, pos=pos, track=track, track_up=np.array([0.0, 0.0, 1.0]),
                  track_forward=np.array([1.0, 0.0, 0.0]), track_scale=0.9, heading_known=True, fps=fps)


def k0(pos: np.ndarray) -> dict:
    return {k: v[0] for k, v in kinematics(motion_of(pos)).items()}


def rotated(deg: float, half_width: float, z: float) -> tuple[tuple, tuple]:
    """(rear, lead) points of a segment turned `deg` toward the pitcher from square to the plate."""
    a = np.radians(deg)
    rear = (-half_width * np.cos(a), -half_width * np.sin(a), z)
    lead = (half_width * np.cos(a), half_width * np.sin(a), z)
    return rear, lead


def test_square_to_the_plate_is_zero_rotation():
    k = k0(skeleton())
    assert k["pelvis_rot"] == pytest.approx(0.0, abs=1e-9)
    assert k["torso_rot"] == pytest.approx(0.0, abs=1e-9)
    assert k["hip_shoulder_sep"] == pytest.approx(0.0, abs=1e-9)


def test_facing_the_pitcher_is_90():
    rh, lh = rotated(90, 0.15, 1.0)
    rs, ls = rotated(90, 0.2, 1.5)
    k = k0(skeleton(rear_hip=rh, lead_hip=lh, rear_shoulder=rs, lead_shoulder=ls))
    assert k["pelvis_rot"] == pytest.approx(90.0)
    assert k["torso_rot"] == pytest.approx(90.0)


def test_coil_toward_the_catcher_is_negative():
    rh, lh = rotated(-20, 0.15, 1.0)
    k = k0(skeleton(rear_hip=rh, lead_hip=lh))
    assert k["pelvis_rot"] == pytest.approx(-20.0)
    assert k["hip_shoulder_sep"] == pytest.approx(-20.0)      # hips behind the (square) shoulders


def test_separation_positive_when_hips_lead():
    rh, lh = rotated(45, 0.15, 1.0)
    k = k0(skeleton(rear_hip=rh, lead_hip=lh))
    assert k["pelvis_rot"] == pytest.approx(45.0)
    assert k["hip_shoulder_sep"] == pytest.approx(45.0)


def test_elbows_and_rear_elbow_height():
    # rear upper arm straight out sideways at shoulder height, forearm pointing at the plate; lead arm hanging
    k = k0(skeleton())
    assert k["rear_elbow_flexion"] == pytest.approx(90.0)
    assert k["rear_shoulder_abduction"] == pytest.approx(90.0)
    assert k["lead_elbow_flexion"] == pytest.approx(0.0, abs=1e-6)
    tucked = k0(skeleton(rear_elbow=(-0.2, 0, 1.2), rear_wrist=(-0.2, -0.28, 1.2)))
    assert tucked["rear_shoulder_abduction"] == pytest.approx(0.0, abs=1e-6)


def test_knee_flexion_and_forward_tilt_toward_the_plate():
    tilt = np.radians(30)
    mid_hip = np.array([0, 0, 1.0])
    sh = mid_hip + 0.5 * np.array([0, -np.sin(tilt), np.cos(tilt)])      # the plate is at -y
    k = k0(skeleton(lead_knee=(0.15, -0.45, 1.0), lead_ankle=(0.15, -0.45, 0.55),
                    rear_shoulder=tuple(sh + [-0.2, 0, 0]), lead_shoulder=tuple(sh + [0.2, 0, 0])))
    assert k["lead_knee_flexion"] == pytest.approx(90.0)
    assert k["trunk_flexion"] == pytest.approx(30.0)
    assert k["trunk_lateral"] == pytest.approx(0.0, abs=1e-9)


def test_side_bend_toward_the_rear_leg_is_positive():
    bend = np.radians(25)
    mid_hip = np.array([0, 0, 1.0])
    sh = mid_hip + 0.5 * np.array([-np.sin(bend), 0, np.cos(bend)])       # rear side is -x
    k = k0(skeleton(rear_shoulder=tuple(sh + [-0.2, 0, 0]), lead_shoulder=tuple(sh + [0.2, 0, 0])))
    assert k["trunk_lateral"] == pytest.approx(25.0)
    assert k["trunk_flexion"] == pytest.approx(0.0, abs=1e-9)


def test_lead_arm_angular_speed():
    fps, omega = 240.0, 600.0
    t = np.arange(60) / fps
    pos = np.repeat(skeleton(), len(t), axis=0)
    a = np.radians(omega) * t
    pos[:, J["lead_elbow"]] = pos[:, J["lead_shoulder"]] + 0.3 * np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
    k = kinematics(motion_of(pos, fps=fps))
    assert k["lead_arm_speed"][5:-5] == pytest.approx(omega, rel=1e-3)


def test_hand_speed_is_relative_to_the_pelvis():
    fps = 240.0
    t = np.arange(60) / fps
    pos = np.repeat(skeleton(), len(t), axis=0)
    for name in ("rear_wrist", "lead_wrist"):
        pos[:, J[name], 0] += 2.0 * t                       # hands move 2 m/s toward the pitcher
    k = kinematics(motion_of(pos, fps=fps))
    assert k["hand_speed"][5:-5] == pytest.approx(2.0 / 0.9)      # leg length = 0.45 + 0.45 m
    whole = np.repeat(skeleton(), len(t), axis=0)
    whole[:, :, 0] += 2.0 * t[:, None]                      # the whole body drifts: no relative hand speed
    assert np.allclose(kinematics(motion_of(whole, fps=fps))["hand_speed"], 0.0, atol=1e-9)


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

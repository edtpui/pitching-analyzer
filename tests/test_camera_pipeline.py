"""Driveline pitches rendered through a virtual camera must come out of the video preprocessing with the
same metrics as the lab data: this checks camera->lab rotation, handedness detection and mirroring,
view detection, image-space foot plant and the stride / momentum tracks."""
import numpy as np
import pytest

from conftest import needs_driveline, needs_reference
from pitchlab.driveline import simulate_video
from pitchlab.events import calib_for_view, detect_events
from pitchlab.kinematics import kinematics
from pitchlab.metrics import compute_metrics
from pitchlab.preprocess import prepare
from synthetic import driveline_track

pytestmark = [needs_driveline, needs_reference]

HEADING_FREE = ["lead_knee_flexion_fp", "lead_knee_flexion_br", "hip_shoulder_sep_fp", "max_hip_shoulder_sep",
                "trunk_flexion_fp", "trunk_flexion_br", "trunk_lateral_br", "shoulder_abduction_fp",
                "shoulder_horiz_abduction_fp", "elbow_flexion_fp", "max_shoulder_er", "arm_slot", "max_pelvis_rot_velo"]
TOL_DEG = 4.0


def pick(trials, throws, n=2):
    good = [t for t in trials if t.throws == throws and np.isfinite(t.events_s["fp"]) and 0.1 < t.events_s["br"] - t.events_s["fp"] < 0.2]
    return good[:n]


def lab_metrics(tr, reference, fps=240.0, view="side"):
    m = simulate_video(tr, fps, 0.0, np.random.default_rng(0), view=view, camera="nominal")
    kin = kinematics(m)
    ev = detect_events(m.t, kin, calib_for_view(reference.calibration, view))
    return compute_metrics(m, kin, ev, reference.height_per_leg_length), ev, m


@pytest.mark.parametrize("throws", ["R", "L"])
def test_side_view_matches_lab(trials, reference, throws):
    for tr in pick(trials, throws):
        lab, lab_ev, lab_m = lab_metrics(tr, reference)
        prep = prepare(driveline_track(tr, "side"), "auto", "square", None, reference.calibration)
        assert prep.throws == throws
        assert prep.camera_view == "side"
        kin = kinematics(prep.motion)
        ev = detect_events(prep.motion.t, kin, calib_for_view(reference.calibration, prep.camera_view))
        vid = compute_metrics(prep.motion, kin, ev, reference.height_per_leg_length)
        assert abs(prep.motion.t[ev.br] - lab_m.t[lab_ev.br]) <= 2.5 / 240
        assert abs(prep.motion.t[ev.fp] - lab_m.t[lab_ev.fp]) <= 0.02
        for k in HEADING_FREE + ["pelvis_rotation_fp", "torso_rotation_fp"]:
            tol = 25.0 if k == "max_pelvis_rot_velo" else TOL_DEG
            assert vid[k] == pytest.approx(lab[k], abs=tol), (tr.session_pitch, k, vid[k], lab[k])
        assert vid["stride_length"] == pytest.approx(lab["stride_length"], abs=5.0)
        assert vid["max_pelvis_fwd_velo"] is not None


@pytest.mark.parametrize("throws", ["R", "L"])
def test_behind_view_drops_translation_metrics(trials, reference, throws):
    tr = pick(trials, throws, 1)[0]
    lab, _, _ = lab_metrics(tr, reference, view="behind")
    prep = prepare(driveline_track(tr, "behind"), "auto", "square", None, reference.calibration)
    assert prep.camera_view == "behind"
    assert prep.motion.track_forward is None
    kin = kinematics(prep.motion)
    ev = detect_events(prep.motion.t, kin, calib_for_view(reference.calibration, prep.camera_view))
    vid = compute_metrics(prep.motion, kin, ev, reference.height_per_leg_length)
    assert vid["stride_length"] is None and vid["max_pelvis_fwd_velo"] is None
    for k in ["hip_shoulder_sep_fp", "trunk_flexion_br", "elbow_flexion_fp", "max_shoulder_er"]:
        assert vid[k] == pytest.approx(lab[k], abs=TOL_DEG + 2.0), (k, vid[k], lab[k])


def test_angled_camera_hides_heading_metrics(trials, reference):
    tr = pick(trials, "R", 1)[0]
    prep = prepare(driveline_track(tr, "side"), "auto", "angled", None, reference.calibration)
    kin = kinematics(prep.motion)
    ev = detect_events(prep.motion.t, kin, calib_for_view(reference.calibration, prep.camera_view))
    vid = compute_metrics(prep.motion, kin, ev, reference.height_per_leg_length)
    assert vid["stride_angle"] is None and vid["pelvis_rotation_fp"] is None and vid["torso_rotation_fp"] is None
    assert vid["hip_shoulder_sep_fp"] is not None


def test_wrong_hand_setting_is_overridden_by_arm_speed(trials, reference):
    tr = pick(trials, "R", 1)[0]
    prep = prepare(driveline_track(tr, "side"), "L", "square", None, reference.calibration)
    assert prep.throws == "R"
    assert any("faster arm" in n for n in prep.notes)


def test_capture_fps_override_rescales_time(trials, reference):
    tr = pick(trials, "R", 1)[0]
    track = driveline_track(tr, "side", fps=240.0)
    track.t = np.arange(len(track.t)) / 30.0          # slow-motion export: 240 fps footage played at 30 fps
    slow = prepare(track, "auto", "square", None, reference.calibration)
    fixed = prepare(track, "auto", "square", 240.0, reference.calibration)
    assert slow.quality["fps_effective"] == pytest.approx(30.0)
    assert fixed.quality["fps_effective"] == pytest.approx(240.0)


def test_depth_jumps_on_the_wrist_do_not_move_release(trials, reference):
    """Real clips showed MediaPipe's wrist depth jumping ~0.5 m between frames during arm cocking,
    which used to put release ~250 ms early. Release comes from the image track now."""
    tr = pick(trials, "R", 1)[0]
    clean = prepare(driveline_track(tr, "side"), "auto", "square", None, reference.calibration)
    kin = kinematics(clean.motion)
    br_clean = detect_events(clean.motion.t, kin, calib_for_view(reference.calibration, "side")).br
    track = driveline_track(tr, "side")
    rng = np.random.default_rng(1)
    for k in range(br_clean - 60, br_clean - 10, 4):           # spikes during cocking, alternate signs
        track.world[k, 16, 2] += 0.5 * (1 if rng.random() > 0.5 else -1)
    noisy = prepare(track, "auto", "square", None, reference.calibration)
    kin2 = kinematics(noisy.motion)
    br_noisy = detect_events(noisy.motion.t, kin2, calib_for_view(reference.calibration, "side")).br
    assert abs(br_noisy - br_clean) <= 1

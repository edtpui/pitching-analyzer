"""Driveline swings rendered through a virtual camera must come out of the video preprocessing with the same
metrics as the lab data: this checks camera->lab rotation, batting-side detection and mirroring, view
detection, image-space foot plant and contact, and the stride / forward-move tracks."""
import numpy as np
import pytest

from conftest import needs_swing_driveline as needs_driveline, needs_swing_reference as needs_reference
from pitchlab.swing.driveline import simulate_video
from pitchlab.swing.events import calib_for_view, detect_events
from pitchlab.swing.kinematics import kinematics
from pitchlab.swing.metrics import compute_metrics
from pitchlab.swing.preprocess import prepare
from swing_synthetic import driveline_track


# the hitting data and reference under the pitching tests' fixture names
@pytest.fixture(scope="session")
def trials(swing_trials):
    return swing_trials


@pytest.fixture(scope="session")
def reference(swing_reference):
    return swing_reference

pytestmark = [needs_driveline, needs_reference]

HEADING_FREE = ["max_lead_knee_flexion", "lead_knee_flexion_fp", "lead_knee_flexion_contact", "hip_shoulder_sep_fp",
                "max_hip_shoulder_sep", "trunk_flexion_load", "max_trunk_lateral", "rear_elbow_flexion_fp",
                "rear_shoulder_abduction_fp", "lead_elbow_flexion_contact"]
HEADING = ["stride_angle", "pelvis_rotation_load", "torso_rotation_load", "pelvis_rotation_fp", "torso_rotation_fp",
           "pelvis_rotation_contact"]
TOL_DEG = 4.0


def pick(trials, bats, n=2):
    good = [t for t in trials if t.bats == bats and np.isfinite(t.events_s["fp"])
            and 0.08 < t.events_s["contact"] - t.events_s["fp"] < 0.16 and t.events_s["fp"] - t.t[0] > 0.8]
    return good[:n]


def metrics_of(m, reference, view, height_m):
    kin = kinematics(m)
    ev = detect_events(m.t, kin, calib_for_view(reference.calibration, view))
    met = compute_metrics(m, kin, ev, reference.height_per_leg_length, height_m, reference.height_per_leg_length_3d)
    return met, ev


def lab_metrics(tr, reference, fps=240.0, view="side"):
    m = simulate_video(tr, fps, 0.0, np.random.default_rng(0), view=view, camera="nominal")
    met, ev = metrics_of(m, reference, view, tr.height_m)
    return met, ev, m


@pytest.mark.parametrize("bats", ["R", "L"])
def test_side_view_matches_lab(trials, reference, bats):
    for tr in pick(trials, bats):
        lab, lab_ev, lab_m = lab_metrics(tr, reference)
        prep = prepare(driveline_track(tr, "side"), "auto", "square", None, reference.calibration)
        assert prep.bats == bats
        assert prep.camera_view == "side"
        vid, ev = metrics_of(prep.motion, reference, prep.camera_view, tr.height_m)
        assert abs(prep.motion.t[ev.contact] - lab_m.t[lab_ev.contact]) <= 2.5 / 240
        assert abs(prep.motion.t[ev.fp] - lab_m.t[lab_ev.fp]) <= 0.02
        assert ev.load is not None and abs(prep.motion.t[ev.load] - lab_m.t[lab_ev.load]) <= 0.03
        for k in HEADING_FREE + HEADING:
            assert vid[k] == pytest.approx(lab[k], abs=TOL_DEG), (tr.session_swing, k, vid[k], lab[k])
        assert vid["max_pelvis_rot_velo"] == pytest.approx(lab["max_pelvis_rot_velo"], abs=30)
        assert vid["max_lead_arm_speed"] == pytest.approx(lab["max_lead_arm_speed"], abs=60)
        assert vid["max_hand_speed"] == pytest.approx(lab["max_hand_speed"], abs=1.0)
        assert vid["stride_length"] == pytest.approx(lab["stride_length"], abs=5.0)
        assert vid["max_pelvis_fwd_velo"] is not None


@pytest.mark.parametrize("bats", ["R", "L"])
def test_behind_view_drops_translation_metrics(trials, reference, bats):
    tr = pick(trials, bats, 1)[0]
    lab, _, _ = lab_metrics(tr, reference, view="behind")
    prep = prepare(driveline_track(tr, "behind"), "auto", "square", None, reference.calibration)
    assert prep.camera_view == "behind"
    assert prep.bats == bats
    assert prep.motion.track_forward is None
    vid, _ = metrics_of(prep.motion, reference, prep.camera_view, tr.height_m)
    assert vid["stride_length"] is None and vid["max_pelvis_fwd_velo"] is None
    for k in ["hip_shoulder_sep_fp", "max_trunk_lateral", "rear_elbow_flexion_fp", "lead_knee_flexion_contact"]:
        assert vid[k] == pytest.approx(lab[k], abs=TOL_DEG + 2.0), (k, vid[k], lab[k])


def test_angled_camera_hides_heading_metrics(trials, reference):
    tr = pick(trials, "R", 1)[0]
    prep = prepare(driveline_track(tr, "side"), "auto", "angled", None, reference.calibration)
    vid, _ = metrics_of(prep.motion, reference, prep.camera_view, tr.height_m)
    assert all(vid[k] is None for k in HEADING)
    assert vid["hip_shoulder_sep_fp"] is not None and vid["max_hip_shoulder_sep"] is not None


def test_wrong_batting_side_is_overridden_by_body_rotation(trials, reference):
    tr = pick(trials, "R", 1)[0]
    prep = prepare(driveline_track(tr, "side"), "L", "square", None, reference.calibration)
    assert prep.bats == "R"
    assert any("turns like" in n for n in prep.notes)
    assert prep.quality["batting_side_turn_deg"] > 60


def test_capture_fps_override_rescales_time(trials, reference):
    tr = pick(trials, "R", 1)[0]
    track = driveline_track(tr, "side", fps=240.0)
    track.t = np.arange(len(track.t)) / 30.0          # slow-motion export: 240 fps footage played at 30 fps
    slow = prepare(track, "auto", "square", None, reference.calibration)
    fixed = prepare(track, "auto", "square", 240.0, reference.calibration)
    assert slow.quality["fps_effective"] == pytest.approx(30.0)
    assert fixed.quality["fps_effective"] == pytest.approx(240.0)


def test_depth_jumps_on_the_hands_do_not_move_contact(trials, reference):
    """MediaPipe's depth for fast-moving hands jumps between frames; contact is timed from the image."""
    tr = pick(trials, "R", 1)[0]
    clean = prepare(driveline_track(tr, "side"), "auto", "square", None, reference.calibration)
    ct_clean = detect_events(clean.motion.t, kinematics(clean.motion), calib_for_view(reference.calibration, "side")).contact
    track = driveline_track(tr, "side")
    rng = np.random.default_rng(1)
    for k in range(ct_clean - 60, ct_clean - 5, 4):           # spikes through the swing, alternate signs
        track.world[k, [15, 16], 2] += 0.4 * (1 if rng.random() > 0.5 else -1)
    noisy = prepare(track, "auto", "square", None, reference.calibration)
    ct_noisy = detect_events(noisy.motion.t, kinematics(noisy.motion), calib_for_view(reference.calibration, "side")).contact
    assert abs(ct_noisy - ct_clean) <= 1

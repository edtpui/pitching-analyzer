"""Regression tests from the end-to-end debugging pass (real clip variants run through the container)."""
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import needs_driveline, needs_reference
from pitchlab import jobs as jobs_mod
from pitchlab.pipeline import RELEASE_SPEED_FAIL, Options, assemble, options_dict, release_check, write_json
from pitchlab.video_io import H264Writer, VideoError, open_readable


def test_unreadable_file_message_is_clean(tmp_path):
    bad = tmp_path / "clip.mov"
    bad.write_bytes(b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 4000)      # truncated container, no moov atom
    with pytest.raises(VideoError) as exc:
        open_readable(bad, tmp_path)
    msg = str(exc.value)
    assert "[in#" not in msg and "/" not in msg and "moov" not in msg
    assert "incomplete" in msg


def _check(peak_speed, fp_source="auto", view="side", fps=60.0):
    t = np.arange(0, 2, 1 / fps)
    wrist = np.where(np.abs(t - 1.5) < 0.05, peak_speed, 1.0)
    prep = SimpleNamespace(motion=SimpleNamespace(t=t), camera_view=view, quality={"fps_effective": fps})
    ref = SimpleNamespace(calibration={"release_hand_speed": {"side": {"60": {"median": 19.0}}, "behind": {"60": {"median": 13.0}}}})
    ev = SimpleNamespace(br=int(np.argmin(np.abs(t - 1.5))), source={"fp": fp_source})
    return release_check(prep, {"wrist_speed": wrist}, ev, ref)


def test_release_check_passes_a_real_pitch_and_fails_slow_motion_or_no_release():
    assert not _check(18.3)["fail"] and not _check(18.3)["warn"]          # the real 60 fps clip
    slow = _check(2.9)                                                     # the same clip as a 4x slow-motion export
    assert slow["fail"] and slow["ratio"] < RELEASE_SPEED_FAIL
    assert _check(5.3, fp_source="estimated", view="behind")["fail"]       # clip that ends before release
    # same clip in the container: 7.5 leg lengths/s, mistaken for a front view because foot plant failed
    assert _check(7.5, fp_source="estimated", view="behind")["fail"]
    assert not _check(13.0, fp_source="estimated", view="behind")["fail"]  # a real pitch filmed from behind
    borderline = _check(8.5)                                               # slower than anyone in the reference
    assert borderline["warn"] and not borderline["fail"]
    assert _check(8.5, fp_source="estimated")["fail"]                      # ...and the foot plant was not found either


def test_release_check_does_not_judge_an_extrapolated_release():
    # the arm was blurred out of tracking through release: the straight line across the gap is slow, but
    # blur means a fast arm (a slow-motion clip is sharp)
    chk = _check(2.9, fp_source="auto") | {}
    assert chk["fail"]
    t = np.arange(0, 2, 1 / 60)
    prep = SimpleNamespace(motion=SimpleNamespace(t=t), camera_view="side", quality={"fps_effective": 60.0})
    ref = SimpleNamespace(calibration={"release_hand_speed": {"side": {"60": {"median": 19.0}}}})
    ev = SimpleNamespace(br=90, source={"fp": "auto", "br": "estimated"})
    chk = release_check(prep, {"wrist_speed": np.full(len(t), 2.9)}, ev, ref)
    assert not chk["fail"] and not chk["warn"] and chk["skipped"]


def test_release_check_is_skipped_without_reference_values():
    t = np.arange(0, 2, 1 / 60)
    prep = SimpleNamespace(motion=SimpleNamespace(t=t), camera_view="side", quality={"fps_effective": 60.0})
    chk = release_check(prep, {"wrist_speed": np.ones(len(t))}, SimpleNamespace(br=90, source={}), SimpleNamespace(calibration={}))
    assert chk["ratio"] is None and not chk["fail"] and not chk["warn"]


def _job(tmp_path, monkeypatch, behaviour):
    q = jobs_mod.Jobs(tmp_path / "analyses")
    aid, d = q.create("clip.mp4", ".mp4", Options().validate())
    (d / "input.mp4").write_bytes(b"x")
    monkeypatch.setattr(jobs_mod, "run", behaviour)
    return q, aid


def test_restart_mid_analysis_reports_interruption_not_a_crash(tmp_path, monkeypatch):
    def mediapipe_refuses(*a, **k):
        raise RuntimeError("cannot schedule new futures after shutdown")
    q, aid = _job(tmp_path, monkeypatch, mediapipe_refuses)
    q._stopping.set()                       # what server shutdown does before the interpreter tears down
    q._run(aid)
    m = q.meta(aid)
    assert m["status"] == "failed" and m["error"] == jobs_mod.INTERRUPTED


def test_progress_after_shutdown_stops_the_job(tmp_path, monkeypatch):
    def run_with_progress(d, opts, path, progress):
        progress(0.5, "Estimating pose")
        raise AssertionError("should have been interrupted")
    q, aid = _job(tmp_path, monkeypatch, run_with_progress)
    q._stopping.set()
    q._run(aid)
    assert q.meta(aid)["error"] == jobs_mod.INTERRUPTED
    q.start()                               # a new server start accepts work again
    assert not q._stopping.is_set()


def test_real_crash_is_still_reported_as_internal_error(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise KeyError("missing")
    q, aid = _job(tmp_path, monkeypatch, boom)
    q._run(aid)
    assert q.meta(aid)["error"].startswith("Internal error (KeyError")


@needs_driveline
@needs_reference
def test_night_clip_with_mirrored_body_and_arm_lost_through_release(trials, reference):
    """A Driveline pitch through the video pipeline at 60 fps with the second real clip's failures: the
    whole body mirrored for 4 frames after foot plant, the throwing arm blurred (confidence 0.05) from 2
    frames before release to 9 after, the pitcher lost for 3 of those frames, then the throwing-wrist label
    on the torso with high confidence. Before the fix release landed 16 frames early, on the mirror glitch."""
    from pitchlab.events import calib_for_view, detect_events
    from pitchlab.kinematics import kinematics
    from pitchlab.preprocess import prepare
    from pitchlab.skeleton import MP, MP_ARM_PAIRS, MP_LEG_PAIRS
    from synthetic import driveline_track

    tr = next(t for t in trials if t.throws == "R" and all(np.isfinite(t.events_s[k]) for k in ("fp", "mer", "br"))
              and 0.1 < t.events_s["br"] - t.events_s["fp"] < 0.2)
    cal = calib_for_view(reference.calibration, "side")

    def events(track):
        prep = prepare(track, "R", "square", None, reference.calibration)
        kin = kinematics(prep.motion)
        ev = detect_events(prep.motion.t, kin, cal)
        return prep, kin, ev

    _, _, clean = events(driveline_track(tr, "side", fps=60.0))
    track = driveline_track(tr, "side", fps=60.0)
    ibr = int(np.argmin(np.abs(track.t - tr.events_s["br"])))
    for f in range(clean.fp + 1, clean.fp + 5):                       # mirrored body
        for a, b in MP_LEG_PAIRS + MP_ARM_PAIRS:
            for arr in (track.image, track.world, track.visibility):
                arr[f, [a, b]] = arr[f, [b, a]]
    track.visibility[ibr - 2:ibr + 10, [MP["r_elbow"], MP["r_wrist"], 18, 20, 22]] = 0.05   # blurred arm
    lost = slice(ibr + 1, ibr + 4)                                     # pitcher lost
    track.image[lost] = np.nan
    track.world[lost] = np.nan
    track.visibility[lost] = 0.0
    track.detected[lost] = False
    after = slice(ibr + 10, ibr + 16)                                  # wrist label on the torso
    track.image[after, MP["r_wrist"]] = (track.image[after, MP["r_hip"]] + track.image[after, MP["r_shoulder"]]) / 2
    track.world[after, MP["r_wrist"]] = (track.world[after, MP["r_hip"]] + track.world[after, MP["r_shoulder"]]) / 2

    prep, kin, ev = events(track)
    assert ev.source["br"] == "estimated" and abs(ev.br - clean.br) <= 2, (ev.br, clean.br)
    assert ev.source["fp"] == "auto" and abs(ev.fp - clean.fp) <= 1, (ev.fp, clean.fp)
    assert ev.source["mer"] == "estimated" and ev.fp < ev.mer < ev.br
    assert prep.quality["arm_trust_visibility"] == 0.7
    assert not kin["arm_valid"][clean.fp + 1:clean.fp + 5].any()
    assert not release_check(prep, kin, ev, reference)["fail"]


@pytest.fixture()
def done_analysis(trials, reference):
    """A finished analysis built from a Driveline pitch through a virtual camera (no MediaPipe)."""
    from pitchlab import server
    from pitchlab.events import calib_for_view, detect_events
    from pitchlab.kinematics import kinematics
    from pitchlab.preprocess import prepare
    from pitchlab.render import draw_skeleton
    from synthetic import driveline_track

    tr = next(t for t in trials if t.throws == "R" and np.isfinite(t.events_s["fp"]))
    track = driveline_track(tr, "side", fps=240.0)
    aid, d = server.jobs.create("synthetic.mp4", ".mp4", Options().validate())
    with H264Writer(d / "input.mp4", (640, 360), 240.0) as w:
        for k in range(len(track.t)):
            f = np.full((360, 640, 3), 70, np.uint8)
            draw_skeleton(f, track.image[k] * 0.5, "R", 0.7)
            w.write(f)
    track.save(d / "pose.npz")
    opts = Options().validate()
    prep = prepare(track, opts.throws, opts.alignment, None, reference.calibration)
    ev = detect_events(prep.motion.t, kinematics(prep.motion), calib_for_view(reference.calibration, prep.camera_view))
    video = {"width": 640, "height": 360, "fps": 240.0, "frame_count": len(track.t), "duration_s": 1.0, "codec": "avc1",
             "decoded_path": "input.mp4", "analysis_size": [1280, 720]}
    result = assemble(prep, ev, reference, opts, video)
    write_json(d / "result.json", result)
    meta = server.jobs.meta(aid) | {"status": "done", "options": options_dict(opts),
                                    "finished_utc": datetime.now(timezone.utc).isoformat(), "summary": {"warnings": 99}}
    write_json(d / "meta.json", meta)
    yield aid, result
    server.jobs.delete(aid)


@needs_driveline
@needs_reference
def test_event_edit_refreshes_list_summary_and_timeline(done_analysis):
    from pitchlab.server import app
    aid, result = done_analysis
    assert len(result["timeline_ms"]) == result["video"]["samples"]
    assert result["timeline_ms"][result["events"]["br"]["sample"]] == 0.0
    assert np.all(np.diff(result["timeline_ms"]) > 0)
    with TestClient(app) as c:
        fp = result["events"]["fp"]["sample"]
        r = c.post(f"/api/analyses/{aid}/events", json={"fp": fp + 2})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["result"]["events"]["fp"]["source"] == "manual"
        assert body["meta"]["summary"]["warnings"] == len(body["result"]["quality"]["warnings"])   # was 99
        listed = next(m for m in c.get("/api/analyses").json() if m["id"] == aid)
        assert listed["summary"]["warnings"] != 99
        kf = c.get(f"/api/analyses/{aid}/keyframes/fp.jpg")
        assert kf.status_code == 200
        img = cv2.imdecode(np.frombuffer(kf.content, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[1] == 640

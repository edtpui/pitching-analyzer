"""Ball-flight analysis: calibration, photometric refinement, trajectory fit and metrics, ball measurement,
seam spin, and the full upload -> mark -> analyze -> release-override flow on a synthetic clip.

The synthetic clips (tests/flight_synth.py) integrate a pitch with drag and Magnus lift and ray-trace a
seamed ball whose seam curve and stitching differ from the estimator's model."""
import time

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flight_synth import (BallTexture, Scene, behind_pitcher_camera, behind_plate_camera, curveball,
                          field_background, fastball, look_at, render_clip)
from pitchlab.flight.blob import KIND_MINOR, crop_at, measure_crop
from pitchlab.flight.detect import AREA, D, DKIND, NCOL, PEAK, REFINED, U, V, FrameCandidates
from pitchlab.flight.geometry import (FT, IN, MPH, CalibrationError, Camera, calibrate, camera_view, plate_points,
                                      refine_with_image, rubber_points)
from pitchlab.flight.spin import estimate_spin
from pitchlab.flight.track import build_tracklets, plausible, select_track
from pitchlab.flight.trajectory import (Setup, Track, Trajectory, clock_from_angle, fit_pitch, fit_trajectory,
                                        pitch_metrics)


def marks(cam: Camera, distance_ft=60.5, mound_in=10.0):
    return cam.project(plate_points())[0], cam.project(rubber_points(distance_ft, mound_in))[0]


def true_metrics(pitch, setup=Setup()):
    tp = pitch.plate_time()
    ts = np.linspace(0, tp, 200)
    P, _ = pitch.at(ts)
    A = np.column_stack([np.ones_like(ts), ts, 0.5 * ts * ts])
    coef = np.linalg.lstsq(A, P, rcond=None)[0]
    return pitch_metrics(Trajectory(0.0, coef[0], coef[1], coef[2]), setup, 0.0)


# ------------------------------------------------------------------------------------------ geometry
@pytest.mark.parametrize("make_cam,view", [(behind_plate_camera, "behind_plate"), (behind_pitcher_camera, "behind_pitcher")])
def test_calibration_recovers_the_camera_from_marks_in_any_order(make_cam, view):
    cam = make_cam()
    plate, rubber = marks(cam)
    order = [3, 0, 4, 2, 1]
    c = calibrate(plate[order], rubber[::-1], (cam.width, cam.height), 60.5, 10.0, n_samples=4)
    assert c.camera.f == pytest.approx(cam.f, rel=0.002)
    assert np.allclose(c.camera.center, cam.center, atol=0.01)
    assert c.rms_px < 0.05
    assert c.view() == view == camera_view(cam, 60.5)
    assert len(c.samples) == 4
    # the marks come back reordered to the plate's corner order: back tip first
    assert np.allclose(c.plate_px[0], plate[0], atol=1e-6)


def test_calibration_rejects_bad_marks():
    cam = behind_plate_camera()
    plate, rubber = marks(cam)
    size = (cam.width, cam.height)
    with pytest.raises(CalibrationError):
        calibrate(plate[:4], rubber, size)
    with pytest.raises(CalibrationError):
        calibrate(plate, rubber[:1], size)
    with pytest.raises(CalibrationError):
        calibrate(np.full((5, 2), 500.0), rubber, size)                 # all on one spot
    bad = plate.copy()
    bad[2] = np.nan
    with pytest.raises(CalibrationError):
        calibrate(bad, rubber, size)
    with pytest.raises(CalibrationError):
        calibrate(plate, rubber, size, distance_ft=90.0)


def test_photometric_refinement_sharpens_a_rough_calibration():
    """Clicks off by 1.5 px: the outline match brings the focal length (and every speed with it) close."""
    cam = behind_plate_camera()
    gray = cv2.cvtColor(field_background(cam, 60.5, 10.0), cv2.COLOR_BGR2GRAY)
    plate, rubber = marks(cam)
    errs_click, errs_ref = [], []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        c = calibrate(plate + rng.normal(0, 1.5, (5, 2)), rubber + rng.normal(0, 1.5, (2, 2)), (cam.width, cam.height),
                      60.5, 10.0, n_samples=0)
        r, info = refine_with_image(c, gray, n_samples=4)
        assert info["used"], info
        errs_click.append(abs(c.camera.f / cam.f - 1))
        errs_ref.append(abs(r.camera.f / cam.f - 1))
        world = np.vstack([plate_points(), rubber_points(60.5, 10.0)])
        assert np.abs(r.camera.project(world)[0] - np.vstack([plate, rubber])).max() < 1.2
    assert max(errs_ref) < 0.015
    assert np.mean(errs_ref) < np.mean(errs_click)


def test_refinement_declines_when_the_picture_has_no_field():
    cam = behind_plate_camera()
    plate, rubber = marks(cam)
    c = calibrate(plate, rubber, (cam.width, cam.height), 60.5, 10.0, n_samples=0)
    flat = np.full((cam.height, cam.width), 90, np.uint8)
    r, info = refine_with_image(c, flat)
    assert not info["used"] and info["reason"]
    assert r is c


# ------------------------------------------------------------------------------------------ trajectory
def _track_from_truth(pitch, cam, fps=240.0, noise=0.6, size_bias=1.0, start=0.004, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(start, pitch.plate_time(), 1 / fps)
    P, _ = pitch.at(t)
    uv, z = cam.project(P)
    d = 2 * cam.f * 1.45 * IN / z
    keep = (uv[:, 0] > 0) & (uv[:, 0] < cam.width) & (uv[:, 1] > 0) & (uv[:, 1] < cam.height)
    t, uv, d = t[keep], uv[keep], d[keep]
    return Track(frame=np.arange(len(t)), t=t, uv=uv + rng.normal(0, noise, uv.shape),
                 d=d * size_bias * (1 + rng.normal(0, 0.02, len(d))), sig_uv=np.full(len(t), 0.7),
                 d_kind=np.full(len(t), KIND_MINOR))


@pytest.mark.parametrize("pitch_fn", [fastball, curveball])
def test_fit_recovers_velocity_and_movement(pitch_fn):
    pitch, cam = pitch_fn(), behind_plate_camera()
    track = _track_from_truth(pitch, cam)
    pf = fit_pitch(track, cam, Setup())
    m = pitch_metrics(pf.fit.traj, Setup(), pf.release_time(pf.fit.traj))
    mt = true_metrics(pitch)
    assert pf.release_observed
    assert m["release_speed_mph"] == pytest.approx(mt["release_speed_mph"], abs=1.5)
    assert m["plate_speed_mph"] == pytest.approx(mt["plate_speed_mph"], abs=1.5)
    assert m["ivb_in"] == pytest.approx(mt["ivb_in"], abs=1.5)
    assert m["hb_in"] == pytest.approx(mt["hb_in"], abs=1.5)
    assert m["plate_x_in"] == pytest.approx(mt["plate_x_in"], abs=1.0)
    assert m["plate_z_in"] == pytest.approx(mt["plate_z_in"], abs=1.5)
    assert abs((m["movement_tilt_deg"] - mt["movement_tilt_deg"] + 180) % 360 - 180) < 8


def test_release_anchor_survives_a_size_bias():
    """Ball sizes 6% off: anchored at the seen release, the speed barely moves."""
    pitch, cam = fastball(), behind_plate_camera()
    v_true = np.linalg.norm(pitch.vel[0]) / MPH
    for bias in (0.94, 1.06):
        pf = fit_pitch(_track_from_truth(pitch, cam, size_bias=bias), cam, Setup())
        m = pitch_metrics(pf.fit.traj, Setup(), pf.release_time(pf.fit.traj))
        assert pf.release_observed
        assert m["release_speed_mph"] == pytest.approx(v_true, abs=2.0)


def test_manual_release_anchors_a_late_track():
    pitch, cam = fastball(), behind_plate_camera()
    track = _track_from_truth(pitch, cam, start=0.06)        # first 14 frames of the flight missing
    pf = fit_pitch(track, cam, Setup(), t_release_manual=0.0)
    assert pf.release_source == "manual" and pf.release_observed
    m = pitch_metrics(pf.fit.traj, Setup(), pf.release_time(pf.fit.traj))
    assert m["release_speed_mph"] == pytest.approx(np.linalg.norm(pitch.vel[0]) / MPH, abs=2.0)


def test_a_motionless_blob_is_never_taken_for_the_ball():
    """Something sitting on one pixel (a planted foot) fits 'a pitch straight down a camera ray'; the
    tracker must not pick it."""
    cam = behind_plate_camera()
    rows = []
    for k in range(60):
        c = np.zeros((1, NCOL))
        c[0, [U, V, D, PEAK, AREA, REFINED, DKIND]] = [955.0, 520.0, 6.0, 40.0, 28.0, 1.0, KIND_MINOR]
        rows.append(FrameCandidates(index=k, t=k / 240, cands=c))
    tracklets = build_tracklets(rows, cam.width, cam.height)
    assert tracklets and len(tracklets[0].nodes) == 60
    assert select_track(rows, tracklets, cam, Setup()) is None


def test_plausibility_rejects_a_ball_moving_away_from_the_plate():
    pitch, cam = fastball(), behind_plate_camera()
    track = _track_from_truth(pitch, cam)
    fit = fit_trajectory(track, cam, Setup())
    assert plausible(fit, track, Setup())[0]
    rev = Track(frame=track.frame, t=track.t, uv=track.uv[::-1].copy(), d=track.d[::-1].copy(), sig_uv=track.sig_uv,
                d_kind=track.d_kind)
    assert not plausible(fit_trajectory(rev, cam, Setup()), rev, Setup())[0]


def test_metric_conventions():
    assert clock_from_angle(0) == "12:00"
    assert clock_from_angle(45) == "1:30"
    assert clock_from_angle(-30) == "11:00"
    assert clock_from_angle(180) == "6:00"
    # a pitch that runs toward third base (-x) is arm-side for a right-hander, glove-side for a left-hander
    tr = Trajectory(0.0, np.array([0.0, 16.5, 1.8]), np.array([0.0, -40.0, -1.0]), np.array([-3.0, 7.0, -5.0]))
    r = pitch_metrics(tr, Setup(throws="R"), 0.0)
    l = pitch_metrics(tr, Setup(throws="L"), 0.0)
    assert r["hb_in"] < 0 and r["hb_arm_side_in"] > 0 and l["hb_arm_side_in"] < 0
    assert 0 < r["movement_tilt_deg"] < 90                 # up and to the pitcher's right: 12-3 o'clock
    t_plate = tr.time_at_y(17 * IN)
    assert tr.pos(t_plate)[1] == pytest.approx(17 * IN, abs=1e-6)


# ------------------------------------------------------------------------------------------ ball measurement
def _disc_frame(d_px, centre, shade_bottom=0.45, blur=0.8, size=(160, 120)):
    w, h = size
    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = (60, 120, 70)
    s = 8
    yy, xx = np.mgrid[0:h * s, 0:w * s].astype(np.float32)
    xx, yy = (xx + 0.5) / s, (yy + 0.5) / s
    r = np.hypot(xx - centre[0], yy - centre[1])
    inside = r <= d_px / 2
    lit = np.clip(1.0 - (yy - (centre[1] - d_px / 2)) / d_px * (1 - shade_bottom), shade_bottom, 1.0)
    img = np.empty((h * s, w * s, 3), np.float32)
    img[:] = (60, 120, 70)
    img[inside] = (np.array([225, 230, 235], np.float32)[None, :] * lit[inside, None])
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), blur)
    return np.clip(img, 0, 255).astype(np.uint8), bg


@pytest.mark.parametrize("d_px", [8.0, 14.0, 24.0])
def test_ball_edge_measurement(d_px):
    centre = (80.3, 60.7)
    frame, bg = _disc_frame(d_px, centre)
    m = measure_crop(crop_at(frame, bg, (centre[0] + 1.0, centre[1] - 0.8), d_px))
    assert m is not None and m.kind == KIND_MINOR
    assert m.d == pytest.approx(d_px, rel=0.08)
    assert np.hypot(m.u - centre[0], m.v - centre[1]) < 0.5


def test_ball_merged_with_a_larger_object_is_not_measured():
    frame, bg = _disc_frame(12.0, (80.0, 60.0))
    cv2.rectangle(frame, (60, 40), (150, 110), (230, 230, 230), -1)   # a jersey behind the ball
    assert measure_crop(crop_at(frame, bg, (80.0, 60.0), 12.0)) is None


# ------------------------------------------------------------------------------------------ spin
def _spin_patches(omega, n=20, radius=18.0, fps=240.0, seed=5):
    """Ball patches rendered orthographically (stitched texture, Lambert shading) for a camera looking
    straight at the ball, spinning at `omega` (world, rad/s)."""
    cam = look_at([0.0, -4.0, 1.0], [0.0, 10.0, 1.0], 60.0, (1280, 720))
    tex = BallTexture(b=0.27)
    R0 = Rotation.random(random_state=seed)
    half = int(np.ceil(1.25 * radius))
    size = 2 * half + 1
    V = cam.R.T                                    # patch frame -> world (ball on the optical axis)
    light = np.array([-0.3, -1.0, 1.2])
    light /= np.linalg.norm(light)
    patches = {}
    ss = 3
    for k in range(n):
        t = k / fps
        Rwb = Rotation.from_rotvec(np.asarray(omega) * t) * R0
        g = (np.arange(size * ss) + 0.5) / ss
        X, Y = np.meshgrid(g, g)
        x, y = (X - (half + 0.5)) / radius, (Y - (half + 0.5)) / radius
        r2 = x * x + y * y
        inside = r2 < 1.0
        n_p = np.stack([x, y, -np.sqrt(np.clip(1 - r2, 0, 1))], axis=-1)[inside]
        n_w = n_p @ V.T
        col = tex.colour(Rwb.inv().apply(n_w)) * (0.45 + 0.6 * np.clip(n_w @ light, 0, None))[:, None] * 255
        img = np.zeros((size * ss, size * ss, 3), np.float32)
        img[:] = (60, 120, 70)
        img[inside] = col
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = np.clip(img + np.random.default_rng(k).normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        patches[k] = {"img": img, "bg": np.full_like(img, (60, 120, 70)), "center": (half + 0.5, half + 0.5),
                      "radius": radius, "t": t, "uv": (cam.cx, cam.cy), "elongation": 1.0}
    traj = Trajectory(0.0, np.array([0.0, 0.0, 1.0]), np.array([0.0, -40.0, 0.0]), np.zeros(3))
    return patches, cam, traj


@pytest.mark.parametrize("rpm,axis", [(2300, [-1.0, -0.2, -0.4]), (2700, [1.0, 0.3, -0.5]), (1500, [-0.6, 0.0, 0.8])])
def test_spin_from_seams(rpm, axis):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    omega = rpm * 2 * np.pi / 60 * axis
    patches, cam, traj = _spin_patches(omega)
    sr = estimate_spin(patches, cam, traj, 240.0)
    assert sr.ok, sr.reason
    assert sr.rpm == pytest.approx(rpm, rel=0.03)
    assert np.degrees(np.arccos(np.clip(np.dot(sr.axis_world, axis), -1, 1))) < 10
    assert sr.confidence in ("high", "medium")


def test_spin_needs_slow_motion_and_a_large_ball():
    patches, cam, traj = _spin_patches(2300 * 2 * np.pi / 60 * np.array([-1.0, 0.0, 0.0]), n=8)
    assert not estimate_spin(patches, cam, traj, 120.0).ok
    small = {k: dict(p, radius=5.0) for k, p in patches.items()}
    assert not estimate_spin(small, cam, traj, 240.0).ok


# ------------------------------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from pitchlab.server import app
    with TestClient(app) as c:
        yield c


def test_flight_api_validation(client, tmp_path):
    assert client.get("/api/flights").status_code == 200
    assert client.get("/api/flights/0123456789ab").status_code == 404
    assert client.get("/api/flights/..%2F..%2Fetc").status_code == 404
    assert client.post("/api/flights", files={"file": ("notes.txt", b"x", "text/plain")}).status_code == 415
    assert client.post("/api/flights", files={"file": ("a.mp4", b"", "video/mp4")}).status_code == 422
    assert client.post("/api/flights", files={"file": ("a.mp4", b"x", "video/mp4")}, data={"throws": "X"}).status_code == 422
    assert client.post("/api/flights", files={"file": ("a.mp4", b"x", "video/mp4")}, data={"distance_ft": "abc"}).status_code == 422
    assert client.post("/api/flights", files={"file": ("a.mp4", b"x", "video/mp4")}, data={"zone_bottom_in": "40", "zone_top_in": "42"}).status_code == 422


def _wait(client, fid, statuses, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = client.get(f"/api/flights/{fid}").json()["meta"]
        if m["status"] in statuses:
            return m
        time.sleep(0.5)
    raise AssertionError(f"flight {fid} did not reach {statuses}")


def test_flight_end_to_end(client, tmp_path):
    """Upload a synthetic behind-the-plate fastball, mark the field on an empty frame, analyze, override
    the release, and read the results back."""
    cam = behind_plate_camera((1280, 720))
    clip = tmp_path / "pitch.mp4"
    truth = render_clip(clip, Scene(cam=cam, pitch=fastball(), lead_empty_s=0.25))
    r = client.post("/api/flights", files={"file": ("pitch.mp4", clip.read_bytes(), "video/mp4")},
                    data={"throws": "R", "label": "synthetic"})
    assert r.status_code == 202
    fid = r.json()["id"]
    assert client.post(f"/api/flights/{fid}/analyze").status_code in (409,)      # not prepared / not marked yet
    meta = _wait(client, fid, ("calibrate", "failed"))
    assert meta["status"] == "calibrate", meta.get("error")
    assert client.get(f"/api/flights/{fid}/background.jpg").status_code == 200
    assert client.get(f"/api/flights/{fid}/frames/0.jpg").status_code == 200
    assert client.post(f"/api/flights/{fid}/analyze").status_code == 409          # no calibration yet
    assert client.post(f"/api/flights/{fid}/calibration", json={"plate_px": [[1, 2]], "image": 0}).status_code == 422
    rng = np.random.default_rng(1)
    plate, rubber = truth.plate_px + rng.normal(0, 0.8, (5, 2)), truth.rubber_px + rng.normal(0, 0.8, (2, 2))
    r = client.post(f"/api/flights/{fid}/calibration", json={"plate_px": plate.tolist(), "rubber_px": rubber.tolist(), "image": 0})
    assert r.status_code == 200, r.text
    cal = r.json()
    assert cal["view"] == "behind_plate" and cal["overlay"]["zone"] is not None
    assert np.allclose(np.array(cal["camera_position_ft"]) * FT, cam.center, atol=0.25)
    assert client.post(f"/api/flights/{fid}/analyze").status_code == 202
    meta = _wait(client, fid, ("done", "failed"))
    assert meta["status"] == "done", meta.get("error")
    res = client.get(f"/api/flights/{fid}").json()["result"]
    p = res["pitch"]
    v_true = np.linalg.norm(truth.scene.pitch.vel[0]) / MPH
    assert p["release_speed_mph"] == pytest.approx(v_true, abs=6.0)
    assert res["uncertainty"]["release_speed_mph"] > 0
    assert p["strike"] is True
    assert res["spin"]["ok"] and res["spin"]["rpm"] == pytest.approx(2300, rel=0.05)
    assert client.get(f"/api/flights/{fid}/tracer.mp4").status_code == 200
    assert client.get(f"/api/flights/{fid}/seams.jpg").status_code == 200
    # the release frame picked by hand anchors the speed
    r = client.post(f"/api/flights/{fid}/release", json={"frame": int(round(truth.release_frame))})
    assert r.status_code == 200, r.text
    out = r.json()["result"]
    assert out["release"]["source"] == "manual"
    assert out["pitch"]["release_speed_mph"] == pytest.approx(v_true, abs=2.5)
    assert out["spin"]["rpm"] == res["spin"]["rpm"]
    summary = r.json()["meta"]["summary"]
    assert summary["spin_rpm"] and summary["release_seen"]
    assert client.post(f"/api/flights/{fid}/release", json={"frame": 10_000}).status_code == 422
    # copying the marks onto another clip reuses the camera
    assert client.post(f"/api/flights/{fid}/calibration", json={"copy_from": "0123456789ab"}).status_code == 404
    assert client.delete(f"/api/flights/{fid}").status_code == 204
    assert client.get(f"/api/flights/{fid}").status_code == 404

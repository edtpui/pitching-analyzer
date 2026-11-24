"""Swing HTTP API (/api/swings): validation, upload -> background job -> failure reporting, listing, deletion."""
import time

import pytest
from fastapi.testclient import TestClient

from conftest import needs_pose_model, needs_swing_reference as needs_reference
from test_render_video import make_clip

pytestmark = [needs_reference, needs_pose_model]


@pytest.fixture(scope="module")
def client():
    from pitchlab.server import app
    with TestClient(app) as c:
        yield c


def wait_terminal(client, aid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = client.get(f"/api/swings/{aid}").json()["meta"]
        if m["status"] in ("done", "failed"):
            return m
        time.sleep(0.5)
    raise AssertionError("analysis did not finish")


def test_health_and_reference(client):
    h = client.get("/api/health").json()
    assert h["swing_reference"]["ok"] is True
    assert "full" in h["pose_models"]
    ref = client.get("/api/swings/reference").json()
    assert ref["source"]["n_swings"] == 677
    assert {m["key"] for m in ref["metrics"]} >= {"hip_shoulder_sep_fp", "max_pelvis_rot_velo", "max_hand_speed"}
    assert set(ref["event_error"]["240"]) == {"lift", "load", "fp", "contact"}


def test_rejects_wrong_type_and_bad_options(client, tmp_path):
    r = client.post("/api/swings", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    clip = make_clip(tmp_path / "c.mp4")
    r = client.post("/api/swings", files={"file": ("c.mp4", clip.read_bytes(), "video/mp4")}, data={"bats": "X"})
    assert r.status_code == 422
    r = client.post("/api/swings", files={"file": ("c.mp4", clip.read_bytes(), "video/mp4")}, data={"height_cm": "abc"})
    assert r.status_code == 422
    r = client.post("/api/swings", files={"file": ("c.mp4", clip.read_bytes(), "video/mp4")}, data={"bat_speed_mph": "250"})
    assert r.status_code == 422
    r = client.post("/api/swings", files={"file": ("e.mp4", b"", "video/mp4")})
    assert r.status_code == 422


def test_clip_without_a_person_fails_cleanly(client, tmp_path):
    clip = make_clip(tmp_path / "c.mp4", n=60)
    r = client.post("/api/swings", files={"file": ("nobody.mp4", clip.read_bytes(), "video/mp4")},
                    data={"label": "no person", "model": "lite"})
    assert r.status_code == 202
    aid = r.json()["id"]
    meta = wait_terminal(client, aid)
    assert meta["status"] == "failed"
    assert "no person" in meta["error"].lower()
    assert any(m["id"] == aid for m in client.get("/api/swings").json())
    assert client.post(f"/api/swings/{aid}/events", json={"fp": 3}).status_code == 409
    assert client.post(f"/api/swings/{aid}/report").status_code == 409
    assert client.get(f"/api/swings/{aid}/keyframes/evil.jpg").status_code == 404
    assert client.delete(f"/api/swings/{aid}").status_code == 204
    assert client.get(f"/api/swings/{aid}").status_code == 404


def test_unknown_and_malformed_ids(client):
    assert client.get("/api/swings/0123456789ab").status_code == 404
    assert client.get("/api/swings/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/swings/not-an-id/overlay.mp4").status_code == 404


def test_index_and_static(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/swing.js").status_code == 200

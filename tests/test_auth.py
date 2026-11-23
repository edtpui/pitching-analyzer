"""Password gate used by the public container."""
import base64

import pytest
from fastapi.testclient import TestClient

from conftest import needs_reference

pytestmark = [needs_reference]


def basic(password: str, user: str = "coach") -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


@pytest.fixture()
def client(monkeypatch):
    from pitchlab import server
    monkeypatch.setattr(server, "PASSWORD", "s3cret-pass")
    with TestClient(server.app) as c:
        yield c


def test_everything_but_health_needs_the_password(client):
    assert client.get("/api/health").status_code == 200
    for path in ("/", "/static/app.js", "/api/analyses", "/api/reference"):
        r = client.get(path)
        assert r.status_code == 401, path
        assert r.headers["www-authenticate"].startswith("Basic")
    assert client.post("/api/analyses", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code == 401


def test_wrong_and_malformed_credentials_are_rejected(client):
    assert client.get("/api/analyses", headers=basic("wrong")).status_code == 401
    assert client.get("/api/analyses", headers={"Authorization": "Basic !!!notbase64"}).status_code == 401
    assert client.get("/api/analyses", headers={"Authorization": "Bearer s3cret-pass"}).status_code == 401


def test_right_password_any_user_name(client):
    assert client.get("/api/analyses", headers=basic("s3cret-pass")).status_code == 200
    assert client.get("/", headers=basic("s3cret-pass", user="")).status_code == 200

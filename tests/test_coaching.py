"""Result assembly, rule findings and the Claude report call (against a fake client - no API spend)."""
import json
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest

from conftest import needs_driveline, needs_reference
from pitchlab import coaching
from pitchlab.coaching import LLMUnavailable, claude_report, findings, rules_report
from pitchlab.events import calib_for_view, detect_events
from pitchlab.kinematics import kinematics
from pitchlab.pipeline import Options, _finite, assemble
from pitchlab.preprocess import prepare
from synthetic import driveline_track

pytestmark = [needs_driveline, needs_reference]


@pytest.fixture(scope="module")
def result(trials, reference):
    tr = next(t for t in trials if t.throws == "R" and np.isfinite(t.events_s["fp"]))
    prep = prepare(driveline_track(tr, "side"), "auto", "square", None, reference.calibration)
    kin = kinematics(prep.motion)
    ev = detect_events(prep.motion.t, kin, calib_for_view(reference.calibration, prep.camera_view))
    opts = Options(height_cm=tr.height_m * 100, weight_kg=tr.mass_kg).validate()
    video = {"width": 1280, "height": 720, "fps": 240.0, "frame_count": len(prep.motion.t), "duration_s": 1.0, "codec": "avc1"}
    return assemble(prep, ev, reference, opts, video), tr


def test_assemble_shapes(result):
    r, tr = result
    keys = {m["key"] for m in r["metrics"]}
    assert "hip_shoulder_sep_fp" in keys and len(r["metrics"]) == 26
    measured = [m for m in r["metrics"] if m["value"] is not None]
    assert len(measured) >= 24
    for m in measured:
        assert 0 <= m["percentile"] <= 100
        assert m["rating"] in ("high", "medium", "low")
    assert r["events"]["br"]["ms_from_release"] == 0
    assert -200 < r["events"]["fp"]["ms_from_release"] < -80
    assert len(r["series"]["t_ms"]) == len(r["series"]["pelvis_rot_velo"]) <= 600
    json.dumps(_finite(r), allow_nan=False)   # what write_json stores must be strict JSON


def test_predictions_use_body_variant(result):
    r, tr = result
    velo = next(p for p in r["predictions"] if p["target"] == "pitch_speed_mph")
    assert velo["variant"] == "kinematics+body"
    assert velo["reportable"] is True
    assert abs(velo["value"] - tr.pitch_speed_mph) < 12      # clean-ish input; loose sanity bound
    assert velo["what_if"] and all(isinstance(w["delta"], float) for w in velo["what_if"])


def test_findings_only_from_trusted_metrics(result):
    r, _ = result
    by_key = {m["key"]: m for m in r["metrics"]}
    for f in findings(r["metrics"]):
        assert by_key[f["metric"]]["rating"] != "low"
        assert f["kind"] in ("priority", "strength", "note")
    text = rules_report(r)
    assert text.startswith("## Summary") and "## Priorities" in text


def test_rules_fire_on_extreme_percentiles():
    metrics = [{"key": "hip_shoulder_sep_fp", "value": 10.0, "unit": "deg", "decimals": 0, "percentile": 5.0,
                "rating": "high", "velocity_correlation": 0.26},
               {"key": "max_hip_shoulder_sep", "value": 12.0, "unit": "deg", "decimals": 0, "percentile": 4.0,
                "rating": "high", "velocity_correlation": 0.28},
               {"key": "trunk_flexion_br", "value": 50.0, "unit": "deg", "decimals": 0, "percentile": 95.0,
                "rating": "high", "velocity_correlation": 0.21},
               {"key": "max_shoulder_ir_velo", "value": 9000.0, "unit": "deg/s", "decimals": 0, "percentile": 1.0,
                "rating": "low", "velocity_correlation": 0.1}]
    fs = findings(metrics)
    titles = [f["title"] for f in fs]
    assert "Limited hip-shoulder separation at foot plant" in titles
    assert "Limited peak hip-shoulder separation" not in titles          # suppressed duplicate
    assert "Good forward trunk tilt at release" in titles
    assert all(f["metric"] != "max_shoulder_ir_velo" for f in fs)        # low confidence never fires


class FakeClient:
    def __init__(self, response=None, exc=None, api_key="sk-ant-test"):
        self.api_key = api_key
        self.calls = []
        self._response, self._exc = response, exc
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self._exc:
            raise self._exc
        return self._response


def fake_response(text="## Summary\nGood.", stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)], model="claude-opus-5-5",
                           usage=SimpleNamespace(input_tokens=1200, output_tokens=400), _request_id="req_test")


def test_claude_report_request_shape(monkeypatch, result):
    r, _ = result
    fake = FakeClient(fake_response())
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake)
    out = claude_report(r)
    assert out["text"].startswith("## Summary") and out["source"] == "claude"
    kw = fake.calls[0]
    assert kw["model"] == coaching.ANTHROPIC_MODEL
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]
    assert kw["output_config"] == {"effort": "medium"}
    payload = json.loads(kw["messages"][0]["content"])
    assert {"metrics", "automatic_findings", "model_predictions", "data_quality_warnings"} <= set(payload)
    assert all(m["rating"] in ("high", "medium", "low") for m in payload["metrics"])


def test_claude_refusal_and_missing_credentials(monkeypatch, result):
    r, _ = result
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: FakeClient(fake_response(stop="refusal")))
    with pytest.raises(LLMUnavailable, match="declined"):
        claude_report(r)

    def no_key(**kw):
        raise anthropic.AnthropicError("no credentials")
    monkeypatch.setattr(anthropic, "Anthropic", no_key)
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        claude_report(r)

    unconfigured = FakeClient(fake_response(), api_key=None)          # SDK builds a client with no credentials
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: unconfigured)
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        claude_report(r)
    assert unconfigured.calls == []                                     # no request was attempted

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: FakeClient(exc=anthropic.APIConnectionError(request=SimpleNamespace())))
    with pytest.raises(LLMUnavailable, match="reach"):
        claude_report(r)

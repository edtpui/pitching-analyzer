"""Result assembly, rule findings and the Claude report call (against a fake client - no API spend)."""
import json
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest

from conftest import needs_swing_driveline as needs_driveline, needs_swing_reference as needs_reference
from pitchlab.swing import coaching
from pitchlab.swing.coaching import LLMUnavailable, claude_report, findings, rules_report
from pitchlab.swing.events import calib_for_view, detect_events
from pitchlab.swing.kinematics import kinematics
from pitchlab.swing.pipeline import Options, _finite, assemble
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
VIDEO = {"width": 1280, "height": 720, "fps": 240.0, "frame_count": 400, "duration_s": 1.7, "codec": "avc1"}


def _swing(trials):
    return next(t for t in trials if t.bats == "R" and np.isfinite(t.events_s["contact"])
                and t.events_s["contact"] - t.t[0] > 1.0 and np.isfinite(t.bat_speed_mph))


@pytest.fixture(scope="module")
def prepared(trials, reference):
    tr = _swing(trials)
    prep = prepare(driveline_track(tr, "side"), "auto", "square", None, reference.calibration)
    kin = kinematics(prep.motion)
    ev = detect_events(prep.motion.t, kin, calib_for_view(reference.calibration, prep.camera_view))
    return prep, ev, tr


@pytest.fixture(scope="module")
def result(prepared, reference):
    prep, ev, tr = prepared
    opts = Options(height_cm=tr.height_m * 100, weight_kg=tr.mass_kg).validate()
    return assemble(prep, ev, reference, opts, VIDEO), tr


def test_assemble_shapes(result):
    r, tr = result
    keys = {m["key"] for m in r["metrics"]}
    assert "hip_shoulder_sep_fp" in keys and len(r["metrics"]) == 25
    measured = [m for m in r["metrics"] if m["value"] is not None]
    assert len(measured) >= 23
    for m in measured:
        assert 0 <= m["percentile"] <= 100
        assert m["rating"] in ("high", "medium", "low")
    assert r["events"]["contact"]["ms_from_contact"] == 0
    assert -250 < r["events"]["fp"]["ms_from_contact"] < -30
    assert r["events"]["load"]["ms_from_contact"] < r["events"]["fp"]["ms_from_contact"]
    assert r["athlete"]["bats"] == "R"
    assert r["reference_group"]["n_swings"] == 677
    assert len(r["series"]["t_ms"]) == len(r["series"]["pelvis_rot_velo"]) == len(r["series"]["hand_speed_mph"]) <= 600
    json.dumps(_finite(r), allow_nan=False)   # what write_json stores must be strict JSON


def test_hand_speed_needs_the_height(prepared, reference):
    prep, ev, _ = prepared
    r = assemble(prep, ev, reference, Options().validate(), VIDEO)
    hand = next(m for m in r["metrics"] if m["key"] == "max_hand_speed")
    assert hand["value"] is None and "height" in hand["unavailable"]
    assert "hand_speed_mph" not in r["series"]
    assert all(p["variant"] == "kinematics" for p in r["predictions"])


def test_predictions_use_body_variant(result):
    r, tr = result
    bat = next(p for p in r["predictions"] if p["target"] == "bat_speed_mph")
    assert bat["variant"] == "kinematics+body"
    assert abs(bat["value"] - tr.bat_speed_mph) < 12      # clean-ish input; loose sanity bound
    assert bat["what_if"] and all(isinstance(w["delta"], float) for w in bat["what_if"])
    assert {p["target"] for p in r["predictions"]} == {"bat_speed_mph", "exit_velo_mph"}


def test_findings_only_from_trusted_metrics(result):
    r, _ = result
    by_key = {m["key"]: m for m in r["metrics"]}
    for f in findings(r["metrics"]):
        assert by_key[f["metric"]]["rating"] != "low"
        assert f["kind"] in ("priority", "strength", "note")
    text = rules_report(r)
    assert text.startswith("## Summary") and "## Priorities" in text


def test_rules_fire_on_extreme_percentiles():
    metrics = [{"key": "hip_shoulder_sep_fp", "value": 5.0, "unit": "deg", "decimals": 0, "percentile": 5.0,
                "rating": "medium", "bat_speed_correlation": 0.22},
               {"key": "max_hip_shoulder_sep", "value": 20.0, "unit": "deg", "decimals": 0, "percentile": 4.0,
                "rating": "high", "bat_speed_correlation": 0.25},
               {"key": "max_trunk_lateral", "value": 40.0, "unit": "deg", "decimals": 0, "percentile": 95.0,
                "rating": "high", "bat_speed_correlation": 0.3},
               {"key": "max_torso_rot_velo", "value": 1500.0, "unit": "deg/s", "decimals": 0, "percentile": 99.0,
                "rating": "low", "bat_speed_correlation": 0.1},
               {"key": "pelvis_torso_timing", "value": -40.0, "unit": "ms", "decimals": 0, "percentile": 1.0,
                "rating": "low", "bat_speed_correlation": None}]
    fs = findings(metrics)
    titles = [f["title"] for f in fs]
    assert "Hips and shoulders turning together" in titles
    assert "Limited peak hip-shoulder separation" not in titles          # suppressed duplicate
    assert "Good side bend" in titles
    assert all(f["metric"] not in ("max_torso_rot_velo", "pelvis_torso_timing") for f in fs)   # low confidence never fires
    sep = next(f for f in fs if f["metric"] == "hip_shoulder_sep_fp")
    assert "5 deg" in sep["detail"] and "5th percentile" in sep["detail"] and "r = +0.22 with bat speed" in sep["detail"]


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


def fake_response(text="## Summary\nGood.", stop="end_turn", model="claude-opus-5-5", stop_details=None):
    return SimpleNamespace(stop_reason=stop, stop_details=stop_details,
                           content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           model=model, usage=SimpleNamespace(input_tokens=1200, output_tokens=400), _request_id="req_test")


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
    assert kw["max_tokens"] >= 16000                 # adaptive thinking counts against the cap
    assert "thinking" not in kw and "temperature" not in kw
    payload = json.loads(kw["messages"][0]["content"])
    assert {"metrics", "automatic_findings", "model_estimates", "data_quality_warnings", "reference_group"} <= set(payload)
    assert all(m["rating"] in ("high", "medium", "low") for m in payload["metrics"])


def test_fallback_model_is_reported(monkeypatch, result):
    r, _ = result
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: FakeClient(fake_response(model="claude-opus-4-8")))
    assert claude_report(r)["model"] == "claude-opus-4-8"


def test_claude_refusal_and_missing_credentials(monkeypatch, result):
    r, _ = result
    refusal = fake_response(stop="refusal", stop_details=SimpleNamespace(category="cyber", explanation=""))
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: FakeClient(refusal))
    with pytest.raises(LLMUnavailable, match=r"declined.*cyber"):
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

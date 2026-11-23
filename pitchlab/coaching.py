"""Coaching findings: deterministic rules over the percentiles, plus an optional Claude-written report.

The rules only fire on metrics whose confidence rating is not "low", and every statement quotes the
measured value and its percentile so nothing is asserted without a number behind it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import ANTHROPIC_MODEL

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    metric: str
    direction: str          # "low" | "high"
    threshold: float        # percentile
    kind: str               # priority | strength | note
    title: str
    why: str
    cue: str = ""
    requires_not: tuple[str, ...] = ()   # skip if one of these findings already fired


RULES: tuple[Rule, ...] = (
    Rule("hip_shoulder_sep_fp", "low", 25, "priority", "Limited hip-shoulder separation at foot plant",
         "The hips and shoulders are only {value} apart when the front foot lands. Separation at landing "
         "lets the trunk stretch and then rotate hard; in the Driveline sample, pitchers with more "
         "separation threw harder{vcorr}.",
         "Let the hips start opening while the chest stays closed through landing. Drills: hip-lead "
         "walk-ins, step-behind throws holding a closed chest until the front foot is down."),
    Rule("max_hip_shoulder_sep", "low", 20, "priority", "Limited peak hip-shoulder separation",
         "Peak separation between pelvis and shoulders is {value}.",
         "Same theme as above: hips first, shoulders late. Med-ball rotational throws from a stride position.",
         requires_not=("hip_shoulder_sep_fp",)),
    Rule("torso_rotation_fp", "high", 80, "priority", "Shoulders opening early",
         "The shoulders have already turned {value} toward home at foot plant, more than most of the "
         "reference group. Opening early leaks the rotation the hips created and can make the arm drag.",
         "Keep the glove-side shoulder pointed at the target until the front foot is down. Drill: rocker "
         "throws with a closed front side."),
    Rule("lead_knee_extension", "low", 25, "priority", "Lead leg giving way after foot plant",
         "Between foot plant and release the lead knee changes by {value} (positive = straightening). A "
         "firm lead leg turns forward momentum into rotation; a softening knee absorbs it.",
         "Land on a slightly bent knee and push 'up and back' into the ground through release. Drills: "
         "lead-leg block throws, step-and-brace med-ball throws."),
    Rule("stride_length", "low", 20, "priority", "Short stride",
         "Stride length is {value} of estimated height.",
         "Drive further down the mound from the back hip rather than reaching with the front leg. Drill: "
         "stride to a marker set a few inches past your current landing spot."),
    Rule("max_pelvis_fwd_velo", "low", 25, "priority", "Limited momentum toward home",
         "Peak pelvis speed toward home is {value}.",
         "Lead with the front hip earlier and 'fall' toward the target instead of sitting on the back leg. "
         "Drill: hip-lead drop-step throws."),
    Rule("trunk_flexion_br", "low", 25, "priority", "Upright trunk at release",
         "The trunk is tilted forward {value} at release. Harder throwers in the Driveline sample tended to "
         "finish with more forward tilt{vcorr}.",
         "Finish over a firm front leg with the chest heading toward the glove-side thigh. Drill: towel drill "
         "or finish-and-hold throws."),
    Rule("shoulder_er_fp", "low", 15, "priority", "Arm late at foot plant",
         "At foot plant the forearm is only {value} up from pointing forward. When the arm is not up as "
         "the foot lands it has to catch up, which is commonly linked to extra arm stress.",
         "Start the arm path a beat earlier or shorten the arm action so the ball is up at landing. Drill: "
         "'ball up at landing' checkpoint throws, filmed side-on."),
    Rule("shoulder_abduction_fp", "low", 10, "priority", "Throwing elbow below the shoulder line at foot plant",
         "Shoulder abduction at foot plant is {value} (about 90 deg puts the elbow at shoulder height).",
         "Get the elbow up toward shoulder height as the front foot lands; check arm path with side-on video."),
    Rule("shoulder_horiz_abduction_fp", "high", 90, "note", "Elbow well behind the body at foot plant",
         "The throwing elbow is {value} behind the shoulder line at foot plant. A very deep 'scap load' is "
         "one common source of arm drag.",
         "Keep the elbow roughly in line with the shoulders at landing."),
    Rule("max_pelvis_rot_velo", "low", 25, "priority", "Slow hip rotation",
         "Peak pelvis rotation speed is {value}.",
         "Rotational power work: med-ball shot-put and scoop throws, hip mobility and lead-hip strength."),
    Rule("trunk_lateral_br", "high", 90, "note", "Large glove-side trunk tilt at release",
         "The trunk leans {value} toward the glove side at release. Tilt raises the arm slot; very large "
         "tilt is often paired with higher elbow loads in research.",
         ""),
    Rule("max_shoulder_er", "low", 15, "note", "Limited layback",
         "Max external rotation (layback) is {value}. This is often a mobility limit rather than a timing issue.",
         "Have shoulder external rotation and thoracic extension range checked by a qualified professional."),
    Rule("stride_angle", "high", 90, "note", "Closed (cross-body) stride",
         "The stride lands {value} closed of the line to home.", "Check landing direction with a line on the mound."),
    Rule("stride_angle", "low", 10, "note", "Open stride",
         "The stride lands {value} open of the line to home.", "Check landing direction with a line on the mound."),
    # strengths
    Rule("hip_shoulder_sep_fp", "high", 75, "strength", "Good hip-shoulder separation at foot plant",
         "Separation at foot plant is {value}."),
    Rule("lead_knee_extension", "high", 75, "strength", "Firm lead-leg block", "The lead knee straightens by {value} into release."),
    Rule("stride_length", "high", 80, "strength", "Long stride", "Stride length is {value} of estimated height."),
    Rule("trunk_flexion_br", "high", 75, "strength", "Good forward trunk tilt at release", "Forward tilt at release is {value}."),
    Rule("max_pelvis_rot_velo", "high", 75, "strength", "Fast hip rotation", "Peak pelvis rotation speed is {value}."),
    Rule("max_hip_shoulder_sep", "high", 75, "strength", "Large peak separation", "Peak separation is {value}."),
    Rule("max_pelvis_fwd_velo", "high", 75, "strength", "Strong momentum toward home", "Peak pelvis speed toward home is {value}."),
)

MAX_PRIORITIES = 4
MAX_STRENGTHS = 3


def ordinal(p: float) -> str:
    n = int(round(p))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fmt(entry: dict) -> str:
    v = entry["value"]
    unit = entry["unit"]
    d = entry.get("decimals", 0)
    if unit == "deg":
        return f"{v:.{d}f} deg"
    if unit == "% height":
        return f"{v:.{d}f}%"
    return f"{v:.{d}f} {unit}"


def findings(metrics: list[dict]) -> list[dict]:
    by_key = {m["key"]: m for m in metrics}
    fired: dict[str, dict] = {}
    out: list[dict] = []
    for rule in RULES:
        m = by_key.get(rule.metric)
        if not m or m["value"] is None or m["percentile"] is None or m["rating"] == "low":
            continue
        if any(r in fired for r in rule.requires_not):
            continue
        p = m["percentile"]
        if (rule.direction == "low" and p >= rule.threshold) or (rule.direction == "high" and p <= rule.threshold):
            continue
        vc = m.get("velocity_correlation")
        vcorr = f" (r = {vc:+.2f} with pitch speed)" if vc is not None and abs(vc) >= 0.15 else ""
        f = {
            "kind": rule.kind, "metric": rule.metric, "title": rule.title,
            "detail": rule.why.format(value=_fmt(m), vcorr=vcorr) + f" That is the {ordinal(p)} percentile of the reference group.",
            "cue": rule.cue, "percentile": p, "rating": m["rating"],
            "severity": abs(p - 50.0) * (1.0 if m["rating"] == "high" else 0.8),
        }
        if rule.kind != "strength":
            fired[rule.metric] = f
        out.append(f)
    pri = sorted([f for f in out if f["kind"] == "priority"], key=lambda f: -f["severity"])[:MAX_PRIORITIES]
    strengths = sorted([f for f in out if f["kind"] == "strength"], key=lambda f: -f["severity"])[:MAX_STRENGTHS]
    notes = [f for f in out if f["kind"] == "note"]
    return pri + strengths + notes


# ---------------------------------------------------------------------------------------------- Claude
SYSTEM_PROMPT = """You are an experienced pitching coach who also understands sports biomechanics. You
write a short coaching report for one pitch that was filmed on a phone and analysed automatically.

The user message is JSON produced by the analysis software. It contains measured metrics with the
pitcher's percentile against a reference group of 100 pitchers from the Driveline OpenBiomechanics
dataset (mostly college pitchers), a confidence rating per metric (high / medium / low), automatically
generated findings, model predictions, and data-quality warnings.

Rules:
- Use only the numbers in the JSON. Never invent measurements, percentiles, or velocities.
- Ignore metrics rated "low" except to say they could not be measured reliably.
- If data-quality warnings exist, say plainly how they limit the conclusions.
- No medical or injury diagnosis. You may say a pattern is commonly associated with arm stress and
  suggest seeing a qualified professional.
- Model predictions are estimates with the stated error; present them as such and skip any marked
  not reportable.
- Be specific and practical: for each priority give what was seen (with the number), why it matters,
  one coaching cue, and one or two drills.

Format (Markdown, about 350-550 words):
## Summary
## What's working
## Priorities
(numbered, at most three)
## How much to trust this
"""


def build_llm_payload(result: dict) -> dict:
    metrics = [
        {k: m[k] for k in ("label", "unit", "value", "percentile", "rating", "group")}
        | {"typical_range_p25_p75": [m["typical"]["25"], m["typical"]["75"]] if m.get("typical") else None}
        for m in result["metrics"] if m["value"] is not None
    ]
    preds = [
        {k: p[k] for k in ("label", "unit", "value", "cv_mae", "cv_r2", "reportable")}
        | {"largest_model_drivers": [
            {"metric": w["label"], "change_if_at_reference_median": round(w["delta"], 2)} for w in p["what_if"][:4]]}
        for p in result.get("predictions", [])
    ]
    return {
        "pitcher": result.get("athlete", {}),
        "video": {"fps": result["video"]["fps_effective"], "camera_view": result["quality"]["camera_view"]},
        "events_ms_before_release": result.get("event_summary", {}),
        "metrics": metrics,
        "automatic_findings": [{k: f[k] for k in ("kind", "title", "detail")} for f in result.get("findings", [])],
        "model_predictions": preds,
        "data_quality_warnings": result["quality"].get("warnings", []),
    }


def rules_report(result: dict) -> str:
    """Plain Markdown report built from the rule findings (used when Claude is not configured)."""
    fs = result.get("findings", [])
    pri = [f for f in fs if f["kind"] == "priority"]
    strengths = [f for f in fs if f["kind"] == "strength"]
    notes = [f for f in fs if f["kind"] == "note"]
    lines = ["## Summary"]
    if pri:
        count = "One area stands" if len(pri) == 1 else f"{len(pri)} areas stand"
        lines.append(f"{count} out against the Driveline reference group; the biggest is "
                     f"**{pri[0]['title'].lower()}**.")
    else:
        lines.append("No reliably measured metric falls in the bottom of the reference distribution.")
    lines.append("")
    lines.append("## What's working")
    lines += [f"- **{f['title']}.** {f['detail']}" for f in strengths] or ["- Nothing stands out above the 75th percentile."]
    lines.append("")
    lines.append("## Priorities")
    for i, f in enumerate(pri, 1):
        lines.append(f"{i}. **{f['title']}.** {f['detail']}" + (f" *Cue:* {f['cue']}" if f["cue"] else ""))
    if not pri:
        lines.append("- None flagged.")
    if notes:
        lines.append("")
        lines.append("## Notes")
        lines += [f"- **{f['title']}.** {f['detail']}" + (f" {f['cue']}" if f["cue"] else "") for f in notes]
    warnings = result["quality"].get("warnings", [])
    lines.append("")
    lines.append("## How much to trust this")
    lines.append("Metrics come from a single-camera pose model; ratings in the table show how well each one "
                 "survives video noise in simulation. Real footage is usually noisier than the simulation.")
    lines += [f"- {w}" for w in warnings]
    return "\n".join(lines)


class LLMUnavailable(RuntimeError):
    pass


def has_credentials(client) -> bool:
    """The SDK constructs a client without credentials and only fails at request time, so look at what it
    resolved: an API key, an auth token, or a credential provider (e.g. an `ant auth login` profile)."""
    return bool(getattr(client, "api_key", None) or getattr(client, "auth_token", None)
                or getattr(client, "credentials", None))


def claude_report(result: dict, model: str = ANTHROPIC_MODEL) -> dict:
    """Ask Claude for the report. Raises LLMUnavailable with a user-facing reason on failure."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency is in pyproject
        raise LLMUnavailable("the anthropic package is not installed") from exc
    try:
        client = anthropic.Anthropic(max_retries=2, timeout=120.0)
    except anthropic.AnthropicError as exc:
        raise LLMUnavailable("no Anthropic credentials found - set ANTHROPIC_API_KEY in .env") from exc
    if not has_credentials(client):
        raise LLMUnavailable("no Anthropic credentials found - set ANTHROPIC_API_KEY in .env")

    payload = json.dumps(build_llm_payload(result), sort_keys=True)
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable("the Anthropic API key was rejected") from exc
    except anthropic.PermissionDeniedError as exc:
        raise LLMUnavailable(f"the API key cannot use {model}") from exc
    except anthropic.NotFoundError as exc:
        raise LLMUnavailable(f"model {model} not found") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable("rate limited by the Anthropic API - try again in a minute") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable("could not reach the Anthropic API") from exc
    except anthropic.AnthropicError as exc:
        raise LLMUnavailable(f"Anthropic client error: {exc}") from exc

    if response.stop_reason == "refusal":
        raise LLMUnavailable("the model declined to write this report")
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise LLMUnavailable("the model returned no text")
    log.info("claude report: request %s, %s in / %s out tokens", response._request_id,
             response.usage.input_tokens, response.usage.output_tokens)
    return {
        "source": "claude", "model": response.model, "text": text,
        "truncated": response.stop_reason == "max_tokens",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }

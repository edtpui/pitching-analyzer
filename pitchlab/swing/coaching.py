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
    Rule("hip_shoulder_sep_fp", "low", 25, "priority", "Hips and shoulders turning together",
         "At foot plant the hips are only {value} ahead of the shoulders. Hips that start before the shoulders "
         "stretch the trunk and let it rotate harder; in the Driveline sample, hitters with more separation "
         "here swung faster{vcorr}.",
         "Let the front hip start to open as the front foot lands while the chest stays closed. Drills: "
         "step-and-hold tee swings (land, feel the hips start, then swing), med-ball hip-lead scoop throws."),
    Rule("max_hip_shoulder_sep", "low", 20, "priority", "Limited peak hip-shoulder separation",
         "The largest gap between hip and shoulder rotation through the swing is {value}{vcorr}.",
         "Same theme: hips first, shoulders late. Rotational med-ball throws from a stride position, "
         "separation tee work with a pause at foot plant.",
         requires_not=("hip_shoulder_sep_fp",)),
    Rule("torso_rotation_fp", "high", 80, "priority", "Shoulders opening early",
         "The shoulders have already turned {value} toward the pitcher at foot plant, more than most of the "
         "reference group. Opening early leaks the rotation the hips create and pulls the barrel off the "
         "ball's path, especially on outer-half pitches.",
         "Keep the front shoulder closed until the front foot is down. Drills: opposite-field tee work, "
         "front-toss to the outer half, 'show the back pocket' feel in the stride."),
    Rule("lead_knee_extension", "low", 25, "priority", "Front leg not bracing",
         "Between foot plant and contact the front knee changes by {value} (positive = straightening). A "
         "front leg that firms up turns the forward move into rotation; a knee that keeps bending absorbs "
         "it{vcorr}.",
         "Land on a flexed front knee, then push it straight into the ground as the hips turn. Drills: "
         "front-leg 'post up' swings, step-and-brace med-ball throws."),
    Rule("lead_knee_flexion_contact", "high", 85, "note", "Soft front knee at contact",
         "The front knee is still bent {value} at contact.",
         "Check that the front side firms up through contact.", requires_not=("lead_knee_extension",)),
    Rule("max_pelvis_fwd_velo", "low", 25, "priority", "Limited forward move",
         "The pelvis peaks at {value} toward the pitcher during the stride. A controlled move forward "
         "puts momentum into the swing that the front leg then turns into rotation{vcorr}.",
         "Let the back hip push the body forward into the stride instead of stepping out with the front "
         "foot only. Drills: walk-through or step-back swings, hip-lead stride holds."),
    Rule("max_pelvis_rot_velo", "low", 25, "priority", "Slow hip rotation",
         "Peak pelvis rotation speed is {value}{vcorr}.",
         "Rotational power and hip mobility work: med-ball scoop and shot-put throws, hip internal-rotation "
         "mobility, front-hip strength."),
    Rule("max_trunk_lateral", "low", 20, "priority", "Little side bend",
         "The trunk bends only {value} toward the back leg through the swing. Side bend (back shoulder below "
         "the front shoulder) is how hitters get the barrel onto the pitch's plane, most of all on low "
         "pitches{vcorr}.",
         "Let the back shoulder work down and under as you turn, keeping the spine angle. Drills: low-tee "
         "swings, high-tee to low-tee alternation, 'tilt and turn' dry swings."),
    Rule("stride_length", "low", 10, "note", "Short stride",
         "Stride length at foot plant is {value} of estimated height.",
         "Not a fault by itself; check that the short stride still carries the body forward (forward-move speed)."),
    Rule("stride_length", "high", 90, "note", "Long stride",
         "Stride length at foot plant is {value} of estimated height. Very long strides can lower the eyes and "
         "make it harder to rotate over a firm front leg.",
         "Check head height from stance to contact on the overlay video."),
    Rule("stride_angle", "high", 90, "note", "Closed stride",
         "The stride lands {value} toward the plate from the line to the pitcher.",
         "A closed landing can block the hips; check it against a line drawn toward the pitcher."),
    Rule("stride_angle", "low", 10, "note", "Open stride",
         "The stride lands {value} away from the plate ('in the bucket').",
         "An open landing can pull the front side out early; check it against a line drawn toward the pitcher."),
    Rule("torso_rotation_load", "low", 10, "note", "Large shoulder coil in the load",
         "At maximum hip coil the shoulders are turned {value} toward the catcher, more than almost everyone in "
         "the reference group.",
         "A deep coil is fine if the barrel still arrives on time; if it does not, quiet the shoulder turn-in."),
    Rule("pelvis_rotation_load", "high", 90, "note", "Little hip coil in the load",
         "At maximum coil the pelvis is turned only {value} (0 = square to the plate).",
         "Some coil of the front hip in the stride lets the hips build speed over a longer path."),
    Rule("trunk_flexion_load", "low", 10, "note", "Upright posture in the load",
         "The trunk is bent forward only {value} toward the plate at maximum coil.",
         "A little more hinge at the hips gives the side bend room to work."),
    Rule("max_hand_speed", "low", 15, "note", "Low hand speed",
         "Peak hand speed is {value}.",
         "Hand speed follows from the body: work on the priorities above before chasing it directly."),
    # strengths
    Rule("hip_shoulder_sep_fp", "high", 75, "strength", "Good hip-shoulder separation at foot plant",
         "Separation at foot plant is {value}."),
    Rule("max_hip_shoulder_sep", "high", 75, "strength", "Large peak separation", "Peak separation is {value}."),
    Rule("max_pelvis_rot_velo", "high", 75, "strength", "Fast hip rotation", "Peak pelvis rotation speed is {value}."),
    Rule("lead_knee_extension", "high", 75, "strength", "Firm front-leg brace",
         "The front knee straightens by {value} from foot plant to contact."),
    Rule("max_pelvis_fwd_velo", "high", 75, "strength", "Strong forward move",
         "The pelvis reaches {value} toward the pitcher."),
    Rule("max_trunk_lateral", "high", 75, "strength", "Good side bend", "Peak side bend is {value}."),
    Rule("max_hand_speed", "high", 75, "strength", "Fast hands", "Peak hand speed is {value}."),
    Rule("max_lead_arm_speed", "high", 75, "strength", "Fast lead arm", "Peak lead upper-arm speed is {value}."),
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
        bc = m.get("bat_speed_correlation")
        vcorr = f" (r = {bc:+.2f} with bat speed)" if bc is not None and abs(bc) >= 0.15 else ""
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
SYSTEM_PROMPT = """You are an experienced baseball hitting coach who also understands sports biomechanics.
You write a short coaching report for one swing that was filmed on a phone and analysed automatically.

The user message is JSON produced by the analysis software. It contains measured metrics with the
hitter's percentile against a reference group from the Driveline OpenBiomechanics hitting dataset (mostly
college hitters hitting machine-thrown pitches), a confidence rating per metric (high / medium / low),
automatically generated findings, model estimates, and data-quality warnings.

Rules:
- Use only the numbers in the JSON. Never invent measurements, percentiles, bat speeds or exit velocities.
- Ignore metrics rated "low" except to say they could not be measured reliably.
- If data-quality warnings exist, say plainly how they limit the conclusions.
- No medical or injury diagnosis.
- Model estimates are estimates with the stated error; present them as such and skip any marked not
  reportable. A single swing also depends on the pitch it met, which the analysis cannot see.
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
        "hitter": result.get("athlete", {}),
        "reference_group": result.get("reference_group", {}),
        "video": {"fps": result["video"]["fps_effective"], "camera_view": result["quality"]["camera_view"]},
        "events_ms_before_contact": result.get("event_summary", {}),
        "metrics": metrics,
        "automatic_findings": [{k: f[k] for k in ("kind", "title", "detail")} for f in result.get("findings", [])],
        "model_estimates": preds,
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
                 "survives video noise in simulation. Real footage is usually noisier than the simulation, and "
                 "one swing also depends on the pitch it met.")
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
    """Ask Claude for the report. Raises LLMUnavailable with a user-facing reason on failure.

    Thinking is adaptive (always on for claude-opus-5-5) and counts against max_tokens, so the cap leaves
    room for it; server-side refusal fallback is enabled, so `response.model` names whichever model wrote it."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dependency is in pyproject
        raise LLMUnavailable("the anthropic package is not installed") from exc
    try:
        client = anthropic.Anthropic(max_retries=2, timeout=180.0)
    except anthropic.AnthropicError as exc:
        raise LLMUnavailable("no Anthropic credentials found - set ANTHROPIC_API_KEY in .env") from exc
    if not has_credentials(client):
        raise LLMUnavailable("no Anthropic credentials found - set ANTHROPIC_API_KEY in .env")

    payload = json.dumps(build_llm_payload(result), sort_keys=True)
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
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
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise LLMUnavailable("the model declined to write this report" + (f" ({category})" if category else ""))
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise LLMUnavailable("the model returned no text")
    log.info("claude report: request %s, model %s, %s in / %s out tokens", getattr(response, "_request_id", None),
             response.model, response.usage.input_tokens, response.usage.output_tokens)
    return {
        "source": "claude", "model": response.model, "text": text,
        "truncated": response.stop_reason == "max_tokens",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }

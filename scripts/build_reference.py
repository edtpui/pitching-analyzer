"""Build the reference bundle from the Driveline OpenBiomechanics pitching data.

Steps (all reproducible from a fixed seed):
  1. Definition validity - this project's geometric metrics computed from Driveline's joint centres at
     Driveline's own events, correlated with Driveline's published point-of-interest (POI) values. Metrics
     with r >= 0.7 get a linear "Driveline scale" mapping (mean and SD matched) for display.
  2. Event-detector calibration on simulated 240 fps video (resampling + joint noise + the app's
     smoothing): ball-release offset and the foot-plant speed fraction, against Driveline's events.
  3. Event-detection error and metric reliability (simulated-video metric at detected events vs the
     clean metric at true events) at 240, 120 and 60 fps.
  4. Reference distributions (simulated 240 fps, detected events) overall, by velocity band and level.
  5. Velocity / elbow-torque models trained on simulated features, athlete-grouped CV.

Outputs: assets/reference/reference.json, models.joblib, reference_metrics.csv
Usage:   .venv\\Scripts\\python.exe scripts\\build_reference.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitchlab import __version__  # noqa: E402
from pitchlab.config import REFERENCE_DIR  # noqa: E402
from pitchlab.driveline import (Trial, arm_dropout_p, clean_motion, load_trials, side_leg_length_m,  # noqa: E402
                                simulate_video, true_event_indices)
from pitchlab.driveline import ARM_DROPOUT_P  # noqa: E402
from pitchlab.events import DEFAULT_CALIBRATION, Events, calib_for_view, detect_br, detect_events, detect_fp, detect_pkh  # noqa: E402
from pitchlab.kinematics import kinematics  # noqa: E402
from pitchlab.metrics import METRIC_KEYS, METRICS, compute_metrics, metric_definitions  # noqa: E402
from pitchlab.model import BODY_FEATURES, FEATURES, TARGETS, train  # noqa: E402

SEED = 20261120
FPS_BUCKETS = (240, 120, 60)
NOMINAL_NOISE = 1.0
TRAIN_CONFIGS = [(240, 0.5), (240, 1.0), (240, 1.5), (120, 1.0), (120, 1.5), (60, 1.0)]
MASK_HEADING_P = 0.25
BEHIND_VIEW_P = 0.35          # share of training clips simulated from behind (no stride / momentum)
VELO_BANDS = [(0, 80), (80, 85), (85, 90), (90, 200)]
LEVELS = ("high_school", "college", "independent", "milb")
MAPPING_MIN_R = 0.7
PCTS = (5, 10, 25, 50, 75, 90, 95)


def valid_truth(tr: Trial) -> bool:
    e = tr.events_s
    if not all(np.isfinite(e[k]) for k in ("fp", "mer", "br")):
        return False
    return 0.06 < e["br"] - e["fp"] < 0.30 and e["fp"] < e["mer"] < e["br"]


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def metrics_frame(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(METRIC_KEYS)).astype(float)


def detected_metrics(tr: Trial, m, calib: dict, ratio: float) -> tuple[dict, Events, dict]:
    kin = kinematics(m)
    ev = detect_events(m.t, kin, calib)
    return compute_metrics(m, kin, ev, ratio), ev, kin


def main() -> int:
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    trials = load_trials()
    print(f"loaded {len(trials)} Driveline pitches from {len({t.user for t in trials})} athletes")
    truth_ok = {t.session_pitch for t in trials if valid_truth(t)}
    print(f"{len(truth_ok)} pitches have usable force-plate/ball events")

    # height per image-plane leg length, exactly as the video pipeline measures it (side-on)
    ratio = float(np.median([t.height_m / side_leg_length_m(t.pos) for t in trials]))

    # 1. definition validity at true events, clean data
    clean_true: dict[str, dict] = {}
    for tr in trials:
        if tr.session_pitch not in truth_ok:
            continue
        m = clean_motion(tr)
        kin = kinematics(m)
        ix = true_event_indices(tr, m.t)
        ev = Events(fp=ix["fp"], mer=ix["mer"], br=ix["br"], pkh=ix["pkh"])
        clean_true[tr.session_pitch] = compute_metrics(m, kin, ev, ratio)
    clean_df = metrics_frame(clean_true)
    poi = pd.DataFrame({t.session_pitch: t.poi for t in trials}).T
    validity: dict[str, dict] = {}
    mapping: dict[str, dict] = {}
    for md in METRICS:
        if not md.driveline:
            continue
        a = clean_df[md.key].to_numpy()
        b = poi.loc[clean_df.index, md.driveline].astype(float).to_numpy() * md.driveline_scale
        r = pearson(a, b)
        ok = np.isfinite(a) & np.isfinite(b)
        validity[md.key] = {"driveline_column": md.driveline, "r": r, "n": int(ok.sum()),
                            "mean_this": float(np.mean(a[ok])), "mean_driveline": float(np.mean(b[ok])),
                            "sd_this": float(np.std(a[ok])), "sd_driveline": float(np.std(b[ok]))}
        if r is not None and r >= MAPPING_MIN_R:
            scale = float(np.std(b[ok]) / np.std(a[ok]))
            mapping[md.key] = {"scale": scale, "offset": float(np.mean(b[ok]) - scale * np.mean(a[ok]))}
    print("validity r:", {k: round(v["r"], 2) for k, v in validity.items() if v["r"] is not None})

    # 2. calibrate the detectors on simulated 240 fps video (no arm blur: calibration needs clean timing)
    by_sp = {t.session_pitch: t for t in trials}
    clean240 = {sp: simulate_video(by_sp[sp], 240, NOMINAL_NOISE, rng) for sp in truth_ok}
    kin_clean = {sp: kinematics(m) for sp, m in clean240.items()}
    behind240 = {sp: simulate_video(by_sp[sp], 240, NOMINAL_NOISE, rng, view="behind") for sp in truth_ok}
    kin_behind = {sp: kinematics(m) for sp, m in behind240.items()}
    true_ix = lambda sp, m, k: int(np.argmin(np.abs(m.t - by_sp[sp].events_s[k])))

    # release, side-on: how far the hand is in front of the shoulder at Driveline's release
    br_hand_ahead = float(np.median([kin_clean[sp]["hand_ahead"][true_ix(sp, m, "br")] for sp, m in clean240.items()]))
    # release, other views: peak wrist speed + offset
    offsets = []
    for sp, m in behind240.items():
        _, peak = detect_br(m.t, kin_behind[sp]["wrist_speed"], {"br_offset_s": 0.0})
        offsets.append(by_sp[sp].events_s["br"] - m.t[peak])
    br_offset = float(np.median(offsets))

    def best_fp_params(sims: dict, kins: dict, br_of) -> tuple[float, float]:
        """Speed fraction and planted-position radius that best reproduce Driveline's foot plant,
        given the release index the pipeline would use for that view (`br_of(sp, m, kin)`)."""
        brs = {sp: br_of(sp, m, kins[sp]) for sp, m in sims.items()}
        scores = []
        for frac in np.round(np.arange(0.1, 0.401, 0.05), 3):
            for radius in (0.02, 0.03, 0.04, 0.05):
                errs = []
                for sp, m in sims.items():
                    kin, br = kins[sp], brs[sp]
                    pkh = detect_pkh(m.t, kin["knee_height"], br)
                    fp = detect_fp(m.t, kin["foot_speed"], kin["foot_pos"], br, pkh,
                                   {"fp_speed_fraction": float(frac), "fp_plant_radius": radius})
                    errs.append(np.nan if fp is None else m.t[fp] - by_sp[sp].events_s["fp"])
                errs = np.array(errs)
                # bias + typical error + tail; failed detections count against the setting
                score = abs(np.nanmedian(errs)) + np.nanmedian(np.abs(errs)) + 0.25 * np.nanpercentile(np.abs(errs), 90)
                scores.append((score, int(np.isnan(errs).sum()), float(frac), radius))
        best = min(scores)
        return best[2], best[3]

    # side-on, the final foot plant is found against the (accurate) hand-crossing release
    fp_frac, fp_radius = best_fp_params(clean240, kin_clean, lambda sp, m, kin: true_ix(sp, m, "br"))
    fp_frac_behind, fp_radius_behind = best_fp_params(
        behind240, kin_behind, lambda sp, m, kin: detect_br(m.t, kin["wrist_speed"], {"br_offset_s": br_offset})[0])

    # residual offset of the hand-crossing release with the full detector chain
    base = {**DEFAULT_CALIBRATION, "br_offset_s": br_offset, "fp_speed_fraction": fp_frac, "fp_plant_radius": fp_radius,
            "br_hand_ahead": br_hand_ahead, "br_hand_offset_s": 0.0}
    resid = [by_sp[sp].events_s["br"] - m.t[detect_events(m.t, kin_clean[sp], base).br] for sp, m in clean240.items()]
    br_hand_offset = float(np.median(resid))
    sims240 = {tr.session_pitch: simulate_video(tr, 240, NOMINAL_NOISE, rng, dropout_p=arm_dropout_p(240)) for tr in trials}
    true_fp_br = [by_sp[s].events_s["br"] - by_sp[s].events_s["fp"] for s in truth_ok]
    true_mer_br = [by_sp[s].events_s["br"] - by_sp[s].events_s["mer"] for s in truth_ok]
    calib = {
        **DEFAULT_CALIBRATION,
        "br_offset_s": br_offset,
        "br_hand_ahead": br_hand_ahead,
        "br_hand_offset_s": br_hand_offset,
        "fp_speed_fraction": fp_frac,
        "fp_plant_radius": fp_radius,
        "fp_speed_fraction_behind": fp_frac_behind,
        "fp_plant_radius_behind": fp_radius_behind,
        "fp_to_br_median_s": float(np.median(true_fp_br)),
        "fp_to_br_p05_s": float(np.percentile(true_fp_br, 5)),
        "fp_to_br_p95_s": float(np.percentile(true_fp_br, 95)),
        "mer_to_br_median_s": float(np.median(true_mer_br)),
        "height_per_leg_length": ratio,
    }
    print("calibration:", {k: round(v, 4) for k, v in calib.items()})

    # 3. event errors + reliability per frame rate
    event_error: dict[str, dict] = {}
    reliability: dict[str, dict] = {str(f): {} for f in FPS_BUCKETS}
    detected_by_fps: dict[int, pd.DataFrame] = {}
    for fps in FPS_BUCKETS:
        rows, errs = {}, {k: [] for k in ("pkh", "fp", "mer", "br")}
        for tr in trials:
            m = sims240[tr.session_pitch] if fps == 240 else simulate_video(tr, fps, NOMINAL_NOISE, rng, dropout_p=arm_dropout_p(fps))
            met, ev, _ = detected_metrics(tr, m, calib, ratio)
            rows[tr.session_pitch] = met
            if tr.session_pitch in truth_ok:
                for k in errs:
                    v = getattr(ev, k)
                    true_t = tr.events_s[k]
                    errs[k].append(np.nan if v is None or not np.isfinite(true_t) else m.t[v] - true_t)
        df = metrics_frame(rows)
        detected_by_fps[fps] = df
        event_error[str(fps)] = {
            k: {"median_ms": float(np.nanmedian(v) * 1000), "mae_ms": float(np.nanmedian(np.abs(v)) * 1000),
                "p90_abs_ms": float(np.nanpercentile(np.abs(v), 90) * 1000), "n": int(np.isfinite(v).sum())}
            for k, v in ((k, np.array(v, dtype=float)) for k, v in errs.items())}
        common = clean_df.index
        for key in METRIC_KEYS:
            a = df.loc[common, key].to_numpy()
            b = clean_df.loc[common, key].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            reliability[str(fps)][key] = {
                "r": pearson(a, b),
                "mae": float(np.median(np.abs(a[ok] - b[ok]))) if ok.sum() else None,
                "bias": float(np.median(a[ok] - b[ok])) if ok.sum() else None,
            }
        print(f"{fps} fps events:", {k: round(v["mae_ms"], 1) for k, v in event_error[str(fps)].items()})

    def rating(key: str, fps: int) -> str:
        rr = reliability[str(fps)][key]["r"]
        rv = validity.get(key, {}).get("r")
        if rr is None:
            return "low"
        if rr >= 0.8 and (rv is None or rv >= 0.8):
            return "high"
        if rr >= 0.6 and (rv is None or rv >= 0.6):
            return "medium"
        return "low"

    ratings = {str(f): {k: rating(k, f) for k in METRIC_KEYS} for f in FPS_BUCKETS}

    # 4. reference distributions (simulated 240 fps, detected events)
    ref = detected_by_fps[240]
    info = pd.DataFrame({t.session_pitch: {"velo": t.pitch_speed_mph, "level": t.playing_level,
                                           "user": t.user, "height_m": t.height_m, "mass_kg": t.mass_kg}
                         for t in trials}).T
    distributions: dict[str, dict] = {}
    for key in METRIC_KEYS:
        v = ref[key].dropna()
        if v.empty:
            continue
        d = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
             "percentiles": {str(p): float(np.percentile(v, p)) for p in PCTS},
             "values": [round(float(x), 3) for x in np.sort(v.to_numpy())],
             "by_velocity": {}, "by_level": {}}
        for lo, hi in VELO_BANDS:
            sel = v[(info.loc[v.index, "velo"].astype(float) >= lo) & (info.loc[v.index, "velo"].astype(float) < hi)]
            if sel.size >= 10:
                d["by_velocity"][f"{lo}-{hi}"] = {"n": int(sel.size),
                                                  **{str(p): float(np.percentile(sel, p)) for p in (25, 50, 75)}}
        for lev in LEVELS:
            sel = v[info.loc[v.index, "level"] == lev]
            if sel.size >= 10:
                d["by_level"][lev] = {"n": int(sel.size), **{str(p): float(np.percentile(sel, p)) for p in (25, 50, 75)}}
        distributions[key] = d
    # per-velocity-band correlation of each metric with pitch speed (what separates harder throwers)
    velo_corr = {k: pearson(ref[k].to_numpy(), info.loc[ref.index, "velo"].astype(float).to_numpy())
                 for k in METRIC_KEYS}

    # 5. models on simulated features
    train_rows = []
    for tr in trials:
        tgt = {"pitch_speed_mph": tr.pitch_speed_mph, "elbow_varus_moment": tr.poi.get("elbow_varus_moment", np.nan)}
        for fps, noise in TRAIN_CONFIGS:
            realistic = fps == 240 and noise == NOMINAL_NOISE
            heading = realistic or rng.random() > MASK_HEADING_P
            view = "side" if realistic or rng.random() > BEHIND_VIEW_P else "behind"
            if realistic:
                m = sims240[tr.session_pitch]
            else:
                m = simulate_video(tr, fps, noise, rng, heading_known=heading, view=view, dropout_p=arm_dropout_p(fps))
            row_calib = calib_for_view(calib, view)
            met, _, _ = detected_metrics(tr, m, row_calib, ratio)
            train_rows.append({**{k: (np.nan if met[k] is None else met[k]) for k in FEATURES},
                               "height_m": tr.height_m, "mass_kg": tr.mass_kg, **tgt, "user": tr.user,
                               "session_pitch": tr.session_pitch, "fps": fps, "noise": noise, "view": view,
                               "realistic": realistic})
    train_df = pd.DataFrame(train_rows)
    models, model_summary = train(train_df, train_df["realistic"].to_numpy())
    for k, v in model_summary.items():
        print(f"model {k}: {v['algorithm']} MAE {v['mae']:.2f} R2 {v['r2']:.2f} "
              f"(mean baseline MAE {v['baseline_mean_mae']:.2f}, body-only {v['body_only_mae']:.2f})")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": __version__,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "source": {
            "name": "Driveline OpenBiomechanics Project - baseball pitching",
            "url": "https://github.com/drivelineresearch/openbiomechanics",
            "license": "CC BY-NC-SA 4.0 (non-commercial; additional exclusions - https://openbiomechanics.org/#terms)",
            "n_pitches": len(trials), "n_athletes": len({t.user for t in trials}),
            "n_with_valid_events": len(truth_ok),
            "levels": info["level"].value_counts().to_dict(),
            "velocity_mph": {"mean": float(info["velo"].astype(float).mean()),
                             "min": float(info["velo"].astype(float).min()),
                             "max": float(info["velo"].astype(float).max())},
        },
        "simulation": {"arm_dropout_p": {str(k): v for k, v in ARM_DROPOUT_P.items()}, "fps": list(FPS_BUCKETS), "nominal_noise_scale": NOMINAL_NOISE, "reference_view": "side", "behind_view_share": BEHIND_VIEW_P,
                       "train_configs": TRAIN_CONFIGS},
        "calibration": calib,
        "metrics": metric_definitions(),
        "validity": validity,
        "mapping": mapping,
        "reliability": reliability,
        "ratings": ratings,
        "event_error": event_error,
        "distributions": distributions,
        "velocity_correlation": velo_corr,
        "models": model_summary,
        "model_features": FEATURES, "body_features": BODY_FEATURES, "targets": TARGETS,
    }
    (REFERENCE_DIR / "reference.json").write_text(
        json.dumps(_finite(bundle), indent=1, allow_nan=False, default=_json_default), encoding="utf-8")
    joblib.dump(models, REFERENCE_DIR / "models.joblib")
    out = ref.join(info)
    out.index.name = "session_pitch"
    out.to_csv(REFERENCE_DIR / "reference_metrics.csv", float_format="%.4f")
    print(f"wrote {REFERENCE_DIR} in {time.time() - t_start:.0f} s")
    return 0


def _finite(o):
    """Recursively replace NaN/inf floats with None so the bundle is strict JSON."""
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    raise SystemExit(main())

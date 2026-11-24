"""Build the reference bundle from the Driveline OpenBiomechanics hitting data.

Steps (all reproducible from a fixed seed):
  1. Definition validity - this project's geometric metrics computed from Driveline's joint centres at the
     true events (Driveline's contact; foot plant, load and leg lift found with the app's definitions on the
     clean lab data), correlated with Driveline's published point-of-interest (POI) values where Driveline
     measures the same thing at the same instant. Values are always displayed as measured (no rescaling onto
     Driveline's scale), so the metrics table, the time-series charts and the coaching text agree.
  2. Event-detector calibration on simulated 240 fps video (resampling + joint noise + the app's smoothing +
     a perspective camera): contact offsets (side-on: after peak image hand speed; behind: after the 3D
     hand-speed and shoulder-rotation peaks) and the foot-plant thresholds (side-on: speed fraction and
     planted height; behind: speed fraction and planted radius).
  3. Event-detection error and metric reliability (simulated-video metric at detected events vs the clean
     metric at true events) at 240, 120 and 60 fps.
  4. Reference distributions (simulated 240 fps side view, detected events) overall, by bat-speed band and
     by playing level.
  5. Bat speed / exit velocity models trained on simulated features, athlete-grouped CV.

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

from pitchlab.swing import __version__  # noqa: E402
from pitchlab.swing.config import REFERENCE_DIR  # noqa: E402
from pitchlab.swing.driveline import (FOOT_FLAT_HEIGHT_M, HANDS_DROPOUT_P, Trial, clean_motion,  # noqa: E402
                                hands_dropout_p, kinematic_foot_plant, load_trials, side_leg_length_m,
                                simulate_video)
from pitchlab.swing.events import (DEFAULT_CALIBRATION, TORSO_SEARCH_S, Events, calib_for_view, detect_contact,  # noqa: E402
                             detect_events, detect_fp, detect_lift, detect_load)
from pitchlab.swing.kinematics import kinematics  # noqa: E402
from pitchlab.swing.metrics import METRIC_KEYS, METRICS, compute_metrics, metric_definitions  # noqa: E402
from pitchlab.swing.model import BODY_FEATURES, FEATURES, TARGETS, train  # noqa: E402
from pitchlab.swing.skeleton import leg_length  # noqa: E402

SEED = 20261122
FPS_BUCKETS = (240, 120, 60)
NOMINAL_NOISE = 1.0
TRAIN_CONFIGS = [(240, 0.5), (240, 1.0), (240, 1.5), (120, 1.0), (120, 1.5), (60, 1.0)]
MASK_HEADING_P = 0.25
BEHIND_VIEW_P = 0.35          # share of training clips simulated from behind (no stride / forward move)
BAT_SPEED_BANDS = [(0, 62), (62, 66), (66, 70), (70, 200)]
LEVELS = ("high_school", "college", "independent", "milb")
PCTS = (5, 10, 25, 50, 75, 90, 95)
EVENT_KEYS = ("lift", "load", "fp", "contact")


def true_plant_and_contact(tr: Trial) -> tuple[int, int] | None:
    """(foot-plant index, contact index) on the lab data, or None when the swing has no usable events:
    Driveline's contact, and foot plant by the app's foot-flat definition."""
    ct = tr.events_s["contact"]
    if not (np.isfinite(ct) and tr.t[0] + 0.5 < ct < tr.t[-1] - 0.05):
        return None
    c = int(np.argmin(np.abs(tr.t - ct)))
    fp = kinematic_foot_plant(tr.t, tr.pos, c)
    if fp is None or not (tr.t[fp] > tr.t[0] + 0.4 and 0.03 < tr.t[c] - tr.t[fp] < 0.3):
        return None
    return fp, c


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def metrics_frame(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(METRIC_KEYS)).astype(float)


def main() -> int:
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    trials = load_trials()
    by_sp = {t.session_swing: t for t in trials}
    print(f"loaded {len(trials)} Driveline swings from {len({t.user for t in trials})} hitters")
    truth_idx = {t.session_swing: e for t in trials if (e := true_plant_and_contact(t)) is not None}
    truth_ok = list(truth_idx)
    print(f"{len(truth_ok)} swings have usable foot-plant / contact events")

    # height per leg length, exactly as the video pipeline measures it: side-on image (stride) and 3D (hand speed)
    ratio = float(np.median([t.height_m / side_leg_length_m(t.pos) for t in trials]))
    ratio3d = float(np.median([t.height_m / leg_length(t.pos) for t in trials]))

    def scale_kw(tr: Trial) -> dict:
        return {"height_per_leg_length": ratio, "height_m": tr.height_m, "height_per_leg_length_3d": ratio3d}

    # 1. definition validity at true events, clean data (load and leg lift are kinematic definitions, found here)
    clean_true: dict[str, dict] = {}
    clean_events: dict[str, Events] = {}
    for sp in truth_ok:
        tr = by_sp[sp]
        m = clean_motion(tr)
        kin = kinematics(m)
        fp, ct = truth_idx[sp]
        load = detect_load(m.t, kin["pelvis_rot"], ct, fp)
        lift = detect_lift(m.t, kin["knee_height"], fp, DEFAULT_CALIBRATION["min_leg_lift"])
        ev = Events(fp=fp, contact=ct, load=load, lift=lift)
        clean_events[sp] = ev
        clean_true[sp] = compute_metrics(m, kin, ev, **scale_kw(tr))
    clean_df = metrics_frame(clean_true)
    poi = pd.DataFrame({t.session_swing: t.poi for t in trials}).T
    validity: dict[str, dict] = {}
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
    print("validity r:", {k: round(v["r"], 2) for k, v in validity.items() if v["r"] is not None})

    # 2. calibrate the detectors on simulated 240 fps video (no hand blur: calibration needs clean timing)
    side240 = {sp: simulate_video(by_sp[sp], 240, NOMINAL_NOISE, rng) for sp in truth_ok}
    kin_side = {sp: kinematics(m) for sp, m in side240.items()}
    behind240 = {sp: simulate_video(by_sp[sp], 240, NOMINAL_NOISE, rng, view="behind") for sp in truth_ok}
    kin_behind = {sp: kinematics(m) for sp, m in behind240.items()}

    true_t = {sp: (by_sp[sp].t[fp], by_sp[sp].t[ct]) for sp, (fp, ct) in truth_idx.items()}   # (foot plant, contact)

    def timing_diag(errs: np.ndarray) -> dict:
        errs = errs[np.isfinite(errs)]
        err = np.abs(errs)
        return {"median_ms": float(np.median(errs) * 1000), "median_abs_ms": float(np.median(err) * 1000),
                "p90_abs_ms": float(np.percentile(err, 90) * 1000), "share_over_50ms": float(np.mean(err > 0.05)),
                "n": int(errs.size)}

    # contact, side-on: peak image hand speed + offset
    offs = np.array([true_t[sp][1] - m.t[detect_contact(m.t, kin_side[sp], {"contact_method": "image", "contact_offset_s": 0.0})[1]]
                     for sp, m in side240.items()])
    contact_off = float(np.median(offs))
    contact_diag = timing_diag(offs - contact_off)
    # contact, behind: 3D hand-speed peak and shoulder-rotation peak, each + its own offset, averaged
    off_h, off_t = [], []
    for sp, m in behind240.items():
        kin = kin_behind[sp]
        peak = int(np.nanargmax(kin["hand_speed"]))
        near = np.flatnonzero((m.t >= m.t[peak] - TORSO_SEARCH_S) & (m.t <= m.t[peak] + TORSO_SEARCH_S))
        off_h.append(true_t[sp][1] - m.t[peak])
        off_t.append(true_t[sp][1] - m.t[near[np.nanargmax(np.abs(kin["torso_rot_velo"][near]))]])
    contact_3d = {"contact_method": "3d", "contact_offset_hand3d_s": float(np.median(off_h)),
                  "contact_offset_torso_s": float(np.median(off_t))}
    contact_diag_behind = timing_diag(np.array([m.t[detect_contact(m.t, kin_behind[sp], contact_3d)[0]] - true_t[sp][1]
                                                for sp, m in behind240.items()]))
    print(f"contact side offset {contact_off * 1000:.1f} ms {contact_diag}; behind {contact_3d} {contact_diag_behind}")

    def best_fp_params(sims: dict, kins: dict, contact_calib: dict, method: str, sizes: tuple) -> tuple[float, float]:
        """Speed fraction and planted height / radius that best reproduce the foot plant, given the contact index
        the pipeline would use for that view."""
        cts = {sp: detect_contact(m.t, kins[sp], contact_calib)[0] for sp, m in sims.items()}
        key = "fp_height" if method == "height" else "fp_plant_radius"
        scores = []
        for frac in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
            for size in sizes:
                c = {"fp_method": method, "fp_speed_fraction": frac, key: size}
                errs = []
                for sp, m in sims.items():
                    k = kins[sp]
                    fp = detect_fp(m.t, k["foot_speed"], k["foot_pos"], k["foot_height"], cts[sp], c)
                    errs.append(np.nan if fp is None else m.t[fp] - true_t[sp][0])
                errs = np.array(errs)
                # bias + typical error + tail; failed detections count against the setting
                score = abs(np.nanmedian(errs)) + np.nanmedian(np.abs(errs)) + 0.25 * np.nanpercentile(np.abs(errs), 90)
                scores.append((score + 0.01 * int(np.isnan(errs).sum()), frac, size))
        best = min(scores)
        return best[1], best[2]

    fp_frac, fp_height = best_fp_params(side240, kin_side, {"contact_method": "image", "contact_offset_s": contact_off},
                                        "height", (0.005, 0.01, 0.015, 0.02, 0.025, 0.03))
    fp_frac_behind, fp_radius_behind = best_fp_params(behind240, kin_behind, contact_3d, "radius",
                                                      (0.02, 0.03, 0.04, 0.05, 0.06))
    true_fp_ct = np.array([c - f for f, c in true_t.values()])
    load_ct = np.array([by_sp[s].t[ev.contact] - by_sp[s].t[ev.load] for s, ev in clean_events.items() if ev.load is not None])
    calib = {
        **DEFAULT_CALIBRATION,
        "contact_offset_s": contact_off,
        **{k: v for k, v in contact_3d.items() if k != "contact_method"},
        "fp_speed_fraction": fp_frac,
        "fp_height": fp_height,
        "fp_speed_fraction_behind": fp_frac_behind,
        "fp_plant_radius_behind": fp_radius_behind,
        "fp_to_contact_median_s": float(np.median(true_fp_ct)),
        "fp_to_contact_p05_s": float(np.percentile(true_fp_ct, 5)),
        "fp_to_contact_p95_s": float(np.percentile(true_fp_ct, 95)),
        "load_to_contact_median_s": float(np.median(load_ct)),
        "height_per_leg_length": ratio,
        "height_per_leg_length_3d": ratio3d,
    }
    print("calibration:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in calib.items()})
    # how the foot-flat foot plant sits relative to Driveline's force-plate events (information only)
    vs_force = {k: timing_diag(np.array([true_t[sp][0] - by_sp[sp].events_s[k] for sp in truth_ok]))
                for k in ("fc", "fp")}
    print("foot-flat plant minus force-plate events:", vs_force)

    # 3. event errors + reliability per frame rate
    def detected(tr: Trial, m, c: dict) -> tuple[dict, Events]:
        kin = kinematics(m)
        ev = detect_events(m.t, kin, c)
        return compute_metrics(m, kin, ev, **scale_kw(tr)), ev

    event_error: dict[str, dict] = {}
    reliability: dict[str, dict] = {str(f): {} for f in FPS_BUCKETS}
    detected_by_fps: dict[int, pd.DataFrame] = {}
    sims240: dict[str, object] = {}
    for fps in FPS_BUCKETS:
        rows, errs = {}, {k: [] for k in EVENT_KEYS}
        for tr in trials:
            m = simulate_video(tr, fps, NOMINAL_NOISE, rng, dropout_p=hands_dropout_p(fps))
            if fps == 240:
                sims240[tr.session_swing] = m
            met, ev = detected(tr, m, calib)
            rows[tr.session_swing] = met
            if tr.session_swing in clean_events:
                truth = clean_events[tr.session_swing]
                ref_t = by_sp[tr.session_swing].t
                for k in EVENT_KEYS:
                    v, tv = getattr(ev, k), getattr(truth, k)
                    errs[k].append(np.nan if v is None or tv is None else m.t[v] - ref_t[tv])
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
        print(f"{fps} fps reliability r:", {k: (None if v["r"] is None else round(v["r"], 2)) for k, v in reliability[str(fps)].items()})

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
    info = pd.DataFrame({t.session_swing: {"bat_speed": t.bat_speed_mph, "exit_velo": t.exit_velo_mph,
                                           "level": t.playing_level, "user": t.user, "bats": t.bats,
                                           "height_m": t.height_m, "mass_kg": t.mass_kg}
                         for t in trials}).T
    bat_speed = info["bat_speed"].astype(float)
    distributions: dict[str, dict] = {}
    for key in METRIC_KEYS:
        v = ref[key].dropna()
        if v.empty:
            continue
        d = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
             "percentiles": {str(p): float(np.percentile(v, p)) for p in PCTS},
             "values": [round(float(x), 3) for x in np.sort(v.to_numpy())],
             "by_bat_speed": {}, "by_level": {}}
        for lo, hi in BAT_SPEED_BANDS:
            sel = v[(bat_speed.loc[v.index] >= lo) & (bat_speed.loc[v.index] < hi)]
            if sel.size >= 10:
                d["by_bat_speed"][f"{lo}-{hi}"] = {"n": int(sel.size),
                                                   **{str(p): float(np.percentile(sel, p)) for p in (25, 50, 75)}}
        for lev in LEVELS:
            sel = v[info.loc[v.index, "level"] == lev]
            if sel.size >= 10:
                d["by_level"][lev] = {"n": int(sel.size), **{str(p): float(np.percentile(sel, p)) for p in (25, 50, 75)}}
        distributions[key] = d
    # correlation of each metric with bat speed across the reference (what separates faster swings)
    bat_corr = {k: pearson(ref[k].to_numpy(), bat_speed.loc[ref.index].to_numpy()) for k in METRIC_KEYS}

    # 5. models on simulated features
    train_rows = []
    for tr in trials:
        tgt = {"bat_speed_mph": tr.bat_speed_mph, "exit_velo_mph": tr.exit_velo_mph}
        for fps, noise in TRAIN_CONFIGS:
            realistic = fps == 240 and noise == NOMINAL_NOISE
            heading = realistic or rng.random() > MASK_HEADING_P
            view = "side" if realistic or rng.random() > BEHIND_VIEW_P else "behind"
            if realistic:
                m = sims240[tr.session_swing]
            else:
                m = simulate_video(tr, fps, noise, rng, heading_known=heading, view=view, dropout_p=hands_dropout_p(fps))
            met, _ = detected(tr, m, calib_for_view(calib, view))
            train_rows.append({**{k: (np.nan if met[k] is None else met[k]) for k in FEATURES},
                               "height_m": tr.height_m, "mass_kg": tr.mass_kg, **tgt, "user": tr.user,
                               "session_swing": tr.session_swing, "fps": fps, "noise": noise, "view": view,
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
            "name": "Driveline OpenBiomechanics Project - baseball hitting",
            "url": "https://github.com/drivelineresearch/openbiomechanics",
            "license": "CC BY-NC-SA 4.0 (non-commercial; additional exclusions - https://openbiomechanics.org/#terms)",
            "n_swings": len(trials), "n_athletes": len({t.user for t in trials}),
            "n_with_valid_events": len(truth_ok),
            "bats": info["bats"].value_counts().to_dict(),
            "levels": info["level"].value_counts().to_dict(),
            "bat_speed_mph": {"mean": float(bat_speed.mean()), "min": float(bat_speed.min()), "max": float(bat_speed.max())},
            "exit_velo_mph": {"mean": float(info["exit_velo"].astype(float).mean()),
                              "min": float(info["exit_velo"].astype(float).min()),
                              "max": float(info["exit_velo"].astype(float).max())},
            "pitch": "about 65 mph from about 40 ft",
        },
        "simulation": {"hands_dropout_p": {str(k): v for k, v in HANDS_DROPOUT_P.items()}, "fps": list(FPS_BUCKETS),
                       "nominal_noise_scale": NOMINAL_NOISE, "reference_view": "side", "behind_view_share": BEHIND_VIEW_P,
                       "train_configs": TRAIN_CONFIGS},
        "calibration": calib,
        "calibration_diagnostics": {"contact_side": contact_diag, "contact_behind": contact_diag_behind,
                                    "foot_flat_height_m": FOOT_FLAT_HEIGHT_M,
                                    "foot_plant_minus_force_plate": {"fc10": vs_force["fc"], "fp100": vs_force["fp"]}},
        "metrics": metric_definitions(),
        "validity": validity,
        "reliability": reliability,
        "ratings": ratings,
        "event_error": event_error,
        "distributions": distributions,
        "bat_speed_correlation": bat_corr,
        "models": model_summary,
        "model_features": FEATURES, "body_features": BODY_FEATURES, "targets": TARGETS,
    }
    (REFERENCE_DIR / "reference.json").write_text(
        json.dumps(_finite(bundle), indent=1, allow_nan=False, default=_json_default), encoding="utf-8")
    joblib.dump(models, REFERENCE_DIR / "models.joblib")
    out = ref.join(info)
    out.index.name = "session_swing"
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

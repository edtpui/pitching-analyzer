"""Predictive models trained on the Driveline data: bat speed and exit velocity from video-measurable
mechanics (optionally plus height and body mass).

Models are trained on simulated-video features (noise + frame-rate + smoothing applied to the lab data),
evaluated with athlete-grouped cross-validation, and the better of a gradient-boosted tree model and a
ridge regression is kept per target. Explanations are one-at-a-time counterfactuals: "if this metric were
at the reference median, the prediction would change by X". They describe the model, not causation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import METRIC_BY_KEY, METRIC_KEYS

FEATURES = [k for k in METRIC_KEYS if k != "lead_knee_extension"]     # extension = fp - contact flexion
# without the hitter's height the metrics that need it are never available, so that variant leaves them out
KIN_FEATURES = [k for k in FEATURES if not METRIC_BY_KEY[k].needs_height]
BODY_FEATURES = ["height_m", "mass_kg"]
TARGETS = {
    "bat_speed_mph": {"label": "Bat speed", "unit": "mph", "decimals": 1},
    "exit_velo_mph": {"label": "Exit velocity", "unit": "mph", "decimals": 1},
}
VARIANTS = ("kinematics", "kinematics+body")
MIN_R2_TO_REPORT = 0.25     # below this the prediction is not shown as a number


def candidate_models() -> dict[str, object]:
    return {
        "gradient_boosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
            min_samples_leaf=20, random_state=0),
        "ridge": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                               RidgeCV(alphas=np.logspace(-2, 3, 30))),
    }


def features_for(variant: str) -> list[str]:
    return KIN_FEATURES if variant == "kinematics" else FEATURES + BODY_FEATURES


def group_cv(model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions with folds grouped by athlete (no athlete in both train and test)."""
    from sklearn.base import clone

    pred = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[train], y[train])
        pred[test] = m.predict(X.iloc[test])
    return pred


@dataclass
class TrainedTarget:
    target: str
    variant: str
    algorithm: str
    features: list[str]
    model: object
    cv: dict
    reference_median: dict[str, float]


def train(df: pd.DataFrame, realistic_mask: np.ndarray) -> tuple[dict[str, TrainedTarget], dict]:
    """Fit every target x variant. `df` holds FEATURES, BODY_FEATURES, targets and `user`.
    `realistic_mask` marks rows simulated at 240 fps / nominal noise from the side; CV scores are reported
    on those rows (the conditions the app recommends)."""
    groups = df["user"].to_numpy()
    out: dict[str, TrainedTarget] = {}
    summary: dict = {}
    for target in TARGETS:
        y = df[target].to_numpy(dtype=float)
        ok = np.isfinite(y)
        base_pred = np.full(ok.sum(), np.nan)
        # mean-only baseline, athlete-grouped
        for tr, te in GroupKFold(n_splits=5).split(df[ok], y[ok], groups[ok]):
            base_pred[te] = y[ok][tr].mean()
        mask_real = realistic_mask[ok]
        baseline_mae = float(mean_absolute_error(y[ok][mask_real], base_pred[mask_real]))
        body_only = group_cv(candidate_models()["ridge"], df.loc[ok, BODY_FEATURES], y[ok], groups[ok])
        body_mae = float(mean_absolute_error(y[ok][mask_real], body_only[mask_real]))
        for variant in VARIANTS:
            feats = features_for(variant)
            X = df.loc[ok, feats]
            scores = {}
            for name, model in candidate_models().items():
                p = group_cv(model, X, y[ok], groups[ok])
                scores[name] = {
                    "mae": float(mean_absolute_error(y[ok][mask_real], p[mask_real])),
                    "r2": float(r2_score(y[ok][mask_real], p[mask_real])),
                    "mae_all_conditions": float(mean_absolute_error(y[ok], p)),
                }
            best = min(scores, key=lambda k: scores[k]["mae"])
            model = candidate_models()[best]
            model.fit(X, y[ok])
            cv = {"algorithm": best, "candidates": scores, "mae": scores[best]["mae"], "r2": scores[best]["r2"],
                  "baseline_mean_mae": baseline_mae, "body_only_mae": body_mae,
                  "n_rows": int(ok.sum()), "n_athletes": int(len(np.unique(groups[ok]))),
                  "reportable": scores[best]["r2"] >= MIN_R2_TO_REPORT}
            key = f"{target}|{variant}"
            out[key] = TrainedTarget(target, variant, best, feats, model, cv,
                                     {f: float(np.nanmedian(df.loc[ok, f])) for f in feats})
            summary[key] = cv
    return out, summary


def predict(models: dict[str, TrainedTarget], metrics: dict[str, float | None],
            body: dict[str, float | None] | None) -> list[dict]:
    """Predictions for every target, choosing the body-aware variant when height and mass are known."""
    has_body = bool(body) and all(body.get(k) is not None for k in BODY_FEATURES)
    variant = "kinematics+body" if has_body else "kinematics"
    results = []
    for target, info in TARGETS.items():
        tm = models.get(f"{target}|{variant}")
        if tm is None:
            continue
        row: dict[str, float | None] = {f: metrics.get(f) for f in FEATURES}
        if has_body:
            row.update({k: float(body[k]) for k in BODY_FEATURES})
        X = pd.DataFrame([{f: (np.nan if row.get(f) is None else row[f]) for f in tm.features}])
        if tm.algorithm == "ridge" and X.isna().all(axis=None):
            continue
        value = float(tm.model.predict(X)[0])
        what_if = []
        for f in tm.features:
            if f in BODY_FEATURES or pd.isna(X.at[0, f]):
                continue
            X2 = X.copy()
            X2.at[0, f] = tm.reference_median[f]
            delta = float(tm.model.predict(X2)[0]) - value
            if abs(delta) > 1e-9:
                md = METRIC_BY_KEY[f]
                what_if.append({"metric": f, "label": md.label, "current": float(X.at[0, f]),
                                "reference_median": tm.reference_median[f], "delta": delta})
        what_if.sort(key=lambda d: -abs(d["delta"]))
        results.append({
            "target": target, **info, "variant": variant, "algorithm": tm.algorithm,
            "value": value, "cv_mae": tm.cv["mae"], "cv_r2": tm.cv["r2"],
            "baseline_mae": tm.cv["baseline_mean_mae"], "body_only_mae": tm.cv["body_only_mae"],
            "reportable": tm.cv["reportable"],
            "missing_features": [f for f in tm.features if pd.isna(X.at[0, f])],
            "what_if": what_if[:8],
        })
    return results

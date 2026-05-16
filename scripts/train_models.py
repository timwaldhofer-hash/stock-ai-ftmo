#!/usr/bin/env python3
"""
Etappe 5 — AI Model Training

Trains two independent gradient-boosted classifiers on
data/datasets/train_samples.parquet:

  Main AI        full feature set (signal quality + setup geometry + microstructure)
                 threshold: ai.main_ai_threshold_normal  (default 0.70)

  Independent AI microstructure-only feature subset
                 (volume, relative volume, price/candle structure, VWAP distance,
                  RSI, momentum, ATR, S/R distance, time-of-day)
                 threshold: ai.independent_ai_threshold_normal  (default 0.60)

Split strategy: chronological 70 / 10 / 20  (fit / calibration / test).
The calibration set is used with Platt scaling so probability outputs are
well-calibrated before applying the decision thresholds.

Outputs
-------
models/main_ai.pkl                  calibrated main AI model (joblib)
models/main_ai_meta.json            feature list + threshold
models/independent_ai.pkl           calibrated independent AI model (joblib)
models/independent_ai_meta.json     feature list + threshold
data/model_reports/main_ai_report.json
data/model_reports/independent_ai_report.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.features import (
    INDEPENDENT_AI_FEATURES,
    MAIN_AI_FEATURES,
    add_relative_volume,
    engineer_features,
)

_DATASET_PATH = ROOT / "data" / "datasets" / "train_samples.parquet"
_MODELS_DIR = ROOT / "models"
_REPORTS_DIR = ROOT / "data" / "model_reports"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("train_models")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def time_split(
    df: pd.DataFrame,
    fit_ratio: float = 0.70,
    cal_ratio: float = 0.10,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Chronological boolean masks (fit, calibration, test).
    No shuffling — later timestamps are always in test.
    """
    n = len(df)
    fit_end = int(n * fit_ratio)
    cal_end = int(n * (fit_ratio + cal_ratio))

    fit_mask = pd.Series(False, index=df.index)
    cal_mask = pd.Series(False, index=df.index)
    test_mask = pd.Series(False, index=df.index)

    fit_mask.iloc[:fit_end] = True
    cal_mask.iloc[fit_end:cal_end] = True
    test_mask.iloc[cal_end:] = True

    return fit_mask, cal_mask, test_mask


def _build_base_pipeline(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    seed: int = 42,
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=seed,
        )),
    ])


def train_model(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> CalibratedClassifierCV:
    """
    Fit a GradientBoosting pipeline on X_fit, then apply Platt scaling on the
    held-out calibration set to avoid any temporal leakage from CV folds.
    Uses FrozenEstimator (sklearn>=1.6) when available; falls back to cv='prefit'.
    """
    pipeline = _build_base_pipeline(n_estimators, learning_rate, max_depth)
    pipeline.fit(X_fit, y_fit)

    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    else:
        calibrated = CalibratedClassifierCV(pipeline, cv="prefit", method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def _extract_feature_importances(
    model: CalibratedClassifierCV,
    feature_names: list,
) -> dict:
    try:
        inner = model.calibrated_classifiers_[0].estimator
        # Unwrap FrozenEstimator if present (sklearn>=1.6 calibration path)
        base_pipeline = getattr(inner, "estimator", inner)
        clf = base_pipeline.named_steps["clf"]
        importances = clf.feature_importances_
        ranked = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(ranked)
    except Exception:
        return {}


def evaluate_model(
    model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    feature_names: list,
    model_name: str,
    n_fit: int,
    n_cal: int,
    split_dates: dict,
) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # Calibration curve (reliability diagram data)
    try:
        n_bins = min(10, max(5, int(len(y_test) / 20)))
        frac_pos, mean_pred = calibration_curve(
            y_test, proba, n_bins=n_bins, strategy="quantile"
        )
        cal_data = {
            "fraction_of_positives": frac_pos.tolist(),
            "mean_predicted_value": mean_pred.tolist(),
        }
    except Exception:
        cal_data = {"fraction_of_positives": [], "mean_predicted_value": []}

    # ROC-AUC (needs at least two classes)
    try:
        roc_auc: Optional[float] = float(roc_auc_score(y_test, proba))
    except ValueError:
        roc_auc = None

    return {
        "model": model_name,
        "threshold": threshold,
        "split_dates": split_dates,
        "sample_counts": {
            "n_fit": n_fit,
            "n_cal": n_cal,
            "n_test": int(len(y_test)),
        },
        "class_balance": {
            "positive_rate_test": round(float(y_test.mean()), 4),
            "predicted_positive_rate": round(float(y_pred.mean()), 4),
        },
        "metrics_at_threshold": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        },
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "brier_score": round(float(brier_score_loss(y_test, proba)), 4),
        "calibration": cal_data,
        "feature_importances": _extract_feature_importances(model, feature_names),
    }


def log_report_summary(log: logging.Logger, report: dict, threshold: float) -> None:
    m = report["metrics_at_threshold"]
    roc = report["roc_auc"]
    log.info(
        f"  ROC-AUC: {roc}  |  "
        f"Precision@{threshold}: {m['precision']}  |  "
        f"Recall: {m['recall']}  |  "
        f"F1: {m['f1']}  |  "
        f"Brier: {report['brier_score']}"
    )
    imps = report.get("feature_importances", {})
    if imps:
        top5 = list(imps.items())[:5]
        top5_str = ", ".join(f"{k}={v:.4f}" for k, v in top5)
        log.info(f"  Top-5 features: {top5_str}")


def save_model_and_meta(
    model: CalibratedClassifierCV,
    feature_names: list,
    threshold: float,
    model_path: Path,
    meta_path: Path,
) -> None:
    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(
            {"features": feature_names, "threshold_normal": threshold},
            f,
            indent=2,
        )


def save_report(report: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = setup_logging()

    ai_cfg = cfg.get("ai", {})
    main_threshold = float(ai_cfg.get("main_ai_threshold_normal", 0.70))
    indep_threshold = float(ai_cfg.get("independent_ai_threshold_normal", 0.60))

    if not _DATASET_PATH.exists():
        log.error(f"Dataset not found: {_DATASET_PATH}")
        log.error("Run  python scripts/build_training_dataset.py  first.")
        sys.exit(1)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & sort ────────────────────────────────────────────────────────
    log.info(f"Loading dataset: {_DATASET_PATH}")
    df = load_dataset(_DATASET_PATH)
    log.info(
        f"Loaded {len(df)} samples | "
        f"label=1 rate: {df['label'].mean():.3f} | "
        f"symbols: {df['symbol'].nunique()} | "
        f"date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    )

    # ── Time-based split (masks applied before feature engineering) ────────
    fit_mask, cal_mask, test_mask = time_split(df, fit_ratio=0.70, cal_ratio=0.10)

    def _ts_range(mask: pd.Series) -> str:
        sub = df.loc[mask, "timestamp"]
        return f"{sub.min().date()} to {sub.max().date()}"

    split_dates = {
        "fit": _ts_range(fit_mask),
        "calibration": _ts_range(cal_mask),
        "test": _ts_range(test_mask),
    }
    log.info(
        f"Split -> fit: {fit_mask.sum()} ({split_dates['fit']}) | "
        f"cal: {cal_mask.sum()} ({split_dates['calibration']}) | "
        f"test: {test_mask.sum()} ({split_dates['test']})"
    )

    # ── Feature engineering (fitted stats use only the fit set) ───────────
    df = engineer_features(df)
    df = add_relative_volume(df, fit_mask)

    y = df["label"].astype(int)

    # ══════════════════════════════════════════════════════════════════════
    # Main AI Model
    # ══════════════════════════════════════════════════════════════════════
    log.info("-" * 60)
    log.info("Training Main AI (full feature set, threshold=%.2f) ...", main_threshold)

    feat_main = [f for f in MAIN_AI_FEATURES if f in df.columns]
    missing_main = set(MAIN_AI_FEATURES) - set(feat_main)
    if missing_main:
        log.warning("Main AI - missing features skipped: %s", sorted(missing_main))

    X_main = df[feat_main]

    main_model = train_model(
        X_main[fit_mask], y[fit_mask],
        X_main[cal_mask], y[cal_mask],
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
    )

    main_report = evaluate_model(
        main_model,
        X_main[test_mask], y[test_mask],
        threshold=main_threshold,
        feature_names=feat_main,
        model_name="main_ai",
        n_fit=int(fit_mask.sum()),
        n_cal=int(cal_mask.sum()),
        split_dates=split_dates,
    )
    log_report_summary(log, main_report, main_threshold)

    save_model_and_meta(
        main_model, feat_main, main_threshold,
        _MODELS_DIR / "main_ai.pkl",
        _MODELS_DIR / "main_ai_meta.json",
    )
    save_report(main_report, _REPORTS_DIR / "main_ai_report.json")
    log.info("Saved -> models/main_ai.pkl  |  data/model_reports/main_ai_report.json")

    # ══════════════════════════════════════════════════════════════════════
    # Independent AI Model
    # ══════════════════════════════════════════════════════════════════════
    log.info("-" * 60)
    log.info(
        "Training Independent AI (microstructure subset, threshold=%.2f) ...",
        indep_threshold,
    )

    feat_indep = [f for f in INDEPENDENT_AI_FEATURES if f in df.columns]
    missing_indep = set(INDEPENDENT_AI_FEATURES) - set(feat_indep)
    if missing_indep:
        log.warning("Independent AI - missing features skipped: %s", sorted(missing_indep))

    X_indep = df[feat_indep]

    indep_model = train_model(
        X_indep[fit_mask], y[fit_mask],
        X_indep[cal_mask], y[cal_mask],
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
    )

    indep_report = evaluate_model(
        indep_model,
        X_indep[test_mask], y[test_mask],
        threshold=indep_threshold,
        feature_names=feat_indep,
        model_name="independent_ai",
        n_fit=int(fit_mask.sum()),
        n_cal=int(cal_mask.sum()),
        split_dates=split_dates,
    )
    log_report_summary(log, indep_report, indep_threshold)

    save_model_and_meta(
        indep_model, feat_indep, indep_threshold,
        _MODELS_DIR / "independent_ai.pkl",
        _MODELS_DIR / "independent_ai_meta.json",
    )
    save_report(indep_report, _REPORTS_DIR / "independent_ai_report.json")
    log.info(
        "Saved -> models/independent_ai.pkl  |  "
        "data/model_reports/independent_ai_report.json"
    )

    log.info("-" * 60)
    log.info("Training complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Walk-forward backtest — Etappe 7.

Monthly rolling 6-month-train / 1-month-test windows over the full
backtest period (settings.yaml: backtest.start_date → backtest.end_date).

For each test month M:
  train window : [M − 6 months, M − 1 day]
  test  window : [1st of M, last day of M]

Per-month outputs under  data/walk_forward/YYYY-MM/
  models/   main_ai.pkl, main_ai_meta.json,
            independent_ai.pkl, independent_ai_meta.json
  reports/  main_ai_report.json, independent_ai_report.json,
            backtest_summary.json
  trades/   trades_taken_ai.parquet, trades_rejected_ai.parquet,
            blocked_by_main_ai.parquet, blocked_by_independent_ai.parquet,
            trades_taken_rule.parquet, trades_rejected_rule.parquet

Aggregate outputs:
  data/walk_forward/aggregate_report.json
  data/walk_forward/monthly_performance.parquet

Usage:
    python scripts/run_walk_forward.py
    python scripts/run_walk_forward.py --symbols AAPL MSFT
    python scripts/run_walk_forward.py --start 2023-05 --end 2024-10
    python scripts/run_walk_forward.py --force
"""

import argparse
import dataclasses
import json
import logging
import math
import sys
from calendar import monthrange
from datetime import date, timedelta, time
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

from src.backtester.costs import (
    calc_fees,
    calc_slippage_cost,
    entry_price_after_slippage,
    exit_price_after_slippage,
)
from src.backtester.exits import check_exits
from src.backtester.risk import (
    calc_shares,
    check_daily_loss_ok,
    check_min_position,
    check_stop_distance,
)
from src.backtester.setup_detection import detect_setup
from src.backtester.simulator import simulate_symbol
from src.backtester.trade import RejectedSignal, SetupSignal, Trade
from src.models.features import (
    INDEPENDENT_AI_FEATURES,
    MAIN_AI_FEATURES,
    add_relative_volume,
    engineer_features,
)

TZ = "America/New_York"
_NO_NEW_TRADES_AT = time(15, 30)
_EOD_EXIT = time(15, 50)
_WF_DIR = ROOT / "data" / "walk_forward"
_TRAIN_WINDOW_MONTHS = 6
_MIN_TRAIN_SAMPLES = 50
_MIN_CAL_SAMPLES = 10


# ── Config & logging ───────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("walk_forward")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


# ── Date window helpers ────────────────────────────────────────────────────────

def _add_months(d: date, months: int) -> date:
    """Add or subtract calendar months from a date, clamping to month-end."""
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    last_day = monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _generate_test_months(
    start_ym: str, end_ym: str
) -> list[tuple[date, date, date, date]]:
    """
    Return list of (test_start, test_end, train_start, train_end) tuples,
    one per calendar month from start_ym to end_ym inclusive.
    """
    sy, sm = int(start_ym[:4]), int(start_ym[5:7])
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    cur = date(sy, sm, 1)
    end_date = date(ey, em, 1)
    months = []
    while cur <= end_date:
        _, last_day = monthrange(cur.year, cur.month)
        test_start = cur
        test_end = date(cur.year, cur.month, last_day)
        train_end = test_start - timedelta(days=1)
        train_start = _add_months(test_start, -_TRAIN_WINDOW_MONTHS)
        months.append((test_start, test_end, train_start, train_end))
        cur = _add_months(cur, 1)
    return months


# ── Feature loading ────────────────────────────────────────────────────────────

def _load_features(path: Path, start: date, end: date) -> pd.DataFrame:
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    else:
        df["timestamp"] = ts.dt.tz_convert(TZ)
    df = df[
        (df["timestamp"].dt.date >= start)
        & (df["timestamp"].dt.date <= end)
    ]
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Training dataset builder (mirrors build_training_dataset.py) ───────────────

def _is_earnings_bar(bar: pd.Series) -> bool:
    return bool(bar.get("is_earnings", False))


def _simulate_label(
    bars: pd.DataFrame,
    entry_idx: int,
    direction: int,
    entry_price: float,
    stop_price: float,
    profit_threshold_pct: float,
) -> tuple[int, str, int]:
    if direction == 1:
        profit_target = entry_price * (1.0 + profit_threshold_pct / 100.0)
    else:
        profit_target = entry_price * (1.0 - profit_threshold_pct / 100.0)

    last_idx = len(bars) - 1
    for idx in range(entry_idx, len(bars)):
        bar = bars.iloc[idx]
        bar_time = bar["timestamp"].time()

        if bar_time >= _EOD_EXIT:
            return 0, "EOD", idx
        if direction == 1 and float(bar["low"]) <= stop_price:
            return 0, "STOP", idx
        if direction == -1 and float(bar["high"]) >= stop_price:
            return 0, "STOP", idx
        if direction == 1 and float(bar["high"]) >= profit_target:
            return 1, "TARGET", idx
        if direction == -1 and float(bar["low"]) <= profit_target:
            return 1, "TARGET", idx
        if idx > entry_idx:
            sig = detect_setup(bar)
            if sig is not None and sig.direction != direction:
                return 0, "OPPOSITE_SIGNAL", idx

    return 0, "EOD", last_idx


def _build_samples_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    profit_threshold_normal_pct: float,
    profit_threshold_earnings_pct: float,
) -> list[dict]:
    samples: list[dict] = []
    n = len(df)
    i = 0
    while i < n:
        bar = df.iloc[i]
        if bar["timestamp"].time() >= _NO_NEW_TRADES_AT:
            i += 1
            continue
        signal = detect_setup(bar)
        if signal is None:
            i += 1
            continue
        entry_idx = i + 1
        if entry_idx >= n:
            i += 1
            continue
        entry_bar = df.iloc[entry_idx]
        if entry_bar["timestamp"].date() != bar["timestamp"].date():
            i = entry_idx
            continue
        entry_price = float(entry_bar["open"])
        is_earn = _is_earnings_bar(bar)
        threshold_pct = (
            profit_threshold_earnings_pct if is_earn else profit_threshold_normal_pct
        )
        label, exit_reason, exit_idx = _simulate_label(
            df, entry_idx, signal.direction, entry_price,
            signal.stop_price, threshold_pct,
        )
        snap = bar.to_dict()
        snap["symbol"] = symbol
        snap["setup_type"] = signal.setup_type
        snap["side"] = signal.direction
        snap["entry_price"] = round(entry_price, 4)
        snap["stop_price"] = signal.stop_price
        snap["target_price"] = signal.tp_price
        snap["expected_move_required_pct"] = threshold_pct
        snap["is_earnings"] = is_earn
        snap["label"] = label
        snap["label_exit_reason"] = exit_reason
        samples.append(snap)
        i = exit_idx + 1
    return samples


def _build_training_dataset(
    symbols: list[str],
    features_dir: Path,
    train_start: date,
    train_end: date,
    profit_normal: float,
    profit_earnings: float,
    log: logging.Logger,
) -> Optional[pd.DataFrame]:
    all_samples: list[dict] = []
    for symbol in symbols:
        feat_path = features_dir / f"{symbol}.parquet"
        if not feat_path.exists():
            continue
        try:
            df = _load_features(feat_path, train_start, train_end)
            if len(df) == 0:
                continue
            samples = _build_samples_for_symbol(
                symbol, df, profit_normal, profit_earnings
            )
            all_samples.extend(samples)
        except Exception as exc:
            log.warning(f"  {symbol}: sample build error — {exc}")
    if not all_samples:
        return None
    result = pd.DataFrame(all_samples)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result.sort_values("timestamp").reset_index(drop=True)


# ── Model training helpers (mirrors train_models.py) ──────────────────────────

def _time_split(
    df: pd.DataFrame, fit_ratio: float = 0.70, cal_ratio: float = 0.10
) -> tuple[pd.Series, pd.Series, pd.Series]:
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
    n_estimators: int, learning_rate: float, max_depth: int, seed: int = 42
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


def _train_model(
    X_fit: pd.DataFrame, y_fit: pd.Series,
    X_cal: pd.DataFrame, y_cal: pd.Series,
    n_estimators: int, learning_rate: float, max_depth: int,
) -> CalibratedClassifierCV:
    pipeline = _build_base_pipeline(n_estimators, learning_rate, max_depth)
    pipeline.fit(X_fit, y_fit)
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    else:
        calibrated = CalibratedClassifierCV(pipeline, cv="prefit", method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def _extract_feature_importances(
    model: CalibratedClassifierCV, feature_names: list[str]
) -> dict:
    try:
        inner = model.calibrated_classifiers_[0].estimator
        base_pipeline = getattr(inner, "estimator", inner)
        clf = base_pipeline.named_steps["clf"]
        ranked = sorted(
            zip(feature_names, clf.feature_importances_.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return dict(ranked)
    except Exception:
        return {}


def _evaluate_model(
    model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    feature_names: list[str],
    model_name: str,
    n_fit: int,
    n_cal: int,
    split_dates: dict,
) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
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
    try:
        roc_auc: Optional[float] = float(roc_auc_score(y_test, proba))
    except ValueError:
        roc_auc = None
    return {
        "model": model_name,
        "threshold": threshold,
        "split_dates": split_dates,
        "sample_counts": {
            "n_fit": n_fit, "n_cal": n_cal, "n_test": int(len(y_test))
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


def _train_window_models(
    df_train: pd.DataFrame,
    main_threshold: float,
    indep_threshold: float,
    log: logging.Logger,
) -> Optional[tuple]:
    """
    Train main_ai and independent_ai on df_train.
    Returns (main_model, main_features, main_report,
             indep_model, indep_features, indep_report)
    or None when the training window is too small to fit.
    """
    fit_mask, cal_mask, test_mask = _time_split(df_train)

    if fit_mask.sum() < _MIN_TRAIN_SAMPLES or cal_mask.sum() < _MIN_CAL_SAMPLES:
        log.warning(
            f"  Insufficient data: fit={fit_mask.sum()}, "
            f"cal={cal_mask.sum()} — skipping"
        )
        return None

    df_feat = engineer_features(df_train)
    df_feat = add_relative_volume(df_feat, fit_mask)
    y = df_feat["label"].astype(int)

    def _ts_range(mask: pd.Series) -> str:
        sub = df_train.loc[mask, "timestamp"]
        return f"{sub.min().date()} to {sub.max().date()}"

    split_dates = {
        "fit": _ts_range(fit_mask),
        "calibration": _ts_range(cal_mask),
        "test": _ts_range(test_mask),
    }

    # Main AI
    feat_main = [f for f in MAIN_AI_FEATURES if f in df_feat.columns]
    X_main = df_feat[feat_main]
    main_model = _train_model(
        X_main[fit_mask], y[fit_mask],
        X_main[cal_mask], y[cal_mask],
        n_estimators=200, learning_rate=0.05, max_depth=4,
    )
    main_report = _evaluate_model(
        main_model,
        X_main[test_mask], y[test_mask],
        threshold=main_threshold,
        feature_names=feat_main,
        model_name="main_ai",
        n_fit=int(fit_mask.sum()), n_cal=int(cal_mask.sum()),
        split_dates=split_dates,
    )

    # Independent AI
    feat_indep = [f for f in INDEPENDENT_AI_FEATURES if f in df_feat.columns]
    X_indep = df_feat[feat_indep]
    indep_model = _train_model(
        X_indep[fit_mask], y[fit_mask],
        X_indep[cal_mask], y[cal_mask],
        n_estimators=150, learning_rate=0.08, max_depth=3,
    )
    indep_report = _evaluate_model(
        indep_model,
        X_indep[test_mask], y[test_mask],
        threshold=indep_threshold,
        feature_names=feat_indep,
        model_name="independent_ai",
        n_fit=int(fit_mask.sum()), n_cal=int(cal_mask.sum()),
        split_dates=split_dates,
    )

    return main_model, feat_main, main_report, indep_model, feat_indep, indep_report


# ── AI-filtered simulation (mirrors run_ai_backtest.py) ───────────────────────

def _select_features(snap_df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=snap_df.index)
    for col in feature_list:
        result[col] = snap_df[col] if col in snap_df.columns else float("nan")
    return result


def _build_snapshot(
    bar: pd.Series,
    signal: SetupSignal,
    symbol: str,
    profit_threshold_pct: float,
) -> pd.DataFrame:
    snap = bar.to_dict()
    snap["symbol"] = symbol
    snap["setup_type"] = signal.setup_type
    snap["side"] = signal.direction
    snap["entry_price"] = signal.signal_close
    snap["stop_price"] = signal.stop_price
    snap["target_price"] = signal.tp_price
    snap["expected_move_required_pct"] = profit_threshold_pct
    return pd.DataFrame([snap])


def _score_setup(
    snap_df: pd.DataFrame,
    mean_vol: float,
    main_model: CalibratedClassifierCV,
    main_features: list[str],
    indep_model: CalibratedClassifierCV,
    indep_features: list[str],
) -> tuple[float, float]:
    snap_df = engineer_features(snap_df.copy())
    vol = float(snap_df["volume"].iloc[0])
    snap_df["volume_rel"] = min(vol / max(mean_vol, 1.0), 50.0)
    main_prob = float(
        main_model.predict_proba(_select_features(snap_df, main_features))[:, 1][0]
    )
    indep_prob = float(
        indep_model.predict_proba(_select_features(snap_df, indep_features))[:, 1][0]
    )
    return main_prob, indep_prob


def _simulate_symbol_ai_filtered(
    symbol: str,
    df: pd.DataFrame,
    mean_vol: float,
    main_model: CalibratedClassifierCV,
    main_features: list[str],
    main_threshold: float,
    indep_model: CalibratedClassifierCV,
    indep_features: list[str],
    indep_threshold: float,
    profit_threshold_pct: float,
    initial_account: float,
    risk_pct: float,
    max_stop_pct: float,
    min_position_pct: float,
    max_daily_loss_pct: float,
) -> tuple[list, list, list, list]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    account = initial_account
    taken: list = []
    rejected: list = []
    blocked_main: list = []
    blocked_indep: list = []

    open_trade: Optional[Trade] = None
    pending: Optional[SetupSignal] = None
    current_date = None
    daily_pnl = 0.0
    day_start_account = account
    day_loss_hit = False

    for _, bar in df.iterrows():
        bar_ts = bar["timestamp"]
        bar_date = bar_ts.date()

        if bar_date != current_date:
            current_date = bar_date
            daily_pnl = 0.0
            day_start_account = account
            day_loss_hit = False
            if pending is not None and pending.signal_time.date() != bar_date:
                pending = None

        if pending is not None and open_trade is None and not day_loss_hit:
            raw_entry = float(bar["open"])
            actual_entry = entry_price_after_slippage(raw_entry, pending.direction)
            stop = pending.stop_price
            tp = pending.tp_price
            reject_reason: Optional[str] = None

            if not check_stop_distance(actual_entry, stop, max_stop_pct):
                dist_pct = abs(actual_entry - stop) / actual_entry * 100
                reject_reason = f"stop_distance {dist_pct:.2f}% > max {max_stop_pct}%"
            else:
                shares = calc_shares(account, risk_pct, actual_entry, stop)
                if not check_min_position(shares, actual_entry, account, min_position_pct):
                    pos_val = shares * actual_entry
                    reject_reason = (
                        f"min_position not met: {pos_val:.0f} < "
                        f"{account * min_position_pct / 100:.0f} ({shares} shares)"
                    )

            if reject_reason:
                rejected.append(RejectedSignal(
                    symbol=symbol,
                    setup_type=pending.setup_type,
                    signal_time=pending.signal_time,
                    signal_close=pending.signal_close,
                    stop_price=stop,
                    tp_price=tp,
                    reject_reason=reject_reason,
                ))
            else:
                open_trade = Trade(
                    symbol=symbol,
                    setup_type=pending.setup_type,
                    direction=pending.direction,
                    entry_time=bar_ts,
                    raw_entry_price=raw_entry,
                    entry_price=round(actual_entry, 4),
                    stop_price=stop,
                    tp_price=tp,
                    shares=shares,
                    account_at_entry=round(account, 2),
                )
            pending = None

        if open_trade is not None:
            result = check_exits(open_trade, bar)
            if result is not None:
                raw_exit, reason = result
                actual_exit = exit_price_after_slippage(raw_exit, open_trade.direction)
                gross = (
                    open_trade.shares * open_trade.direction
                    * (actual_exit - open_trade.entry_price)
                )
                fees = calc_fees(open_trade.shares, open_trade.entry_price, actual_exit)
                slip = calc_slippage_cost(
                    open_trade.shares, open_trade.raw_entry_price, raw_exit
                )
                net = gross - fees
                open_trade.exit_time = bar_ts
                open_trade.raw_exit_price = raw_exit
                open_trade.exit_price = round(actual_exit, 4)
                open_trade.exit_reason = reason
                open_trade.pnl_gross = round(gross, 4)
                open_trade.fees = round(fees, 4)
                open_trade.slippage_cost = round(slip, 4)
                open_trade.pnl_net = round(net, 4)
                account += net
                daily_pnl += net
                taken.append(open_trade)
                open_trade = None
                if not check_daily_loss_ok(daily_pnl, day_start_account, max_daily_loss_pct):
                    day_loss_hit = True

        if (
            open_trade is None
            and pending is None
            and not day_loss_hit
            and bar_ts.time() < _NO_NEW_TRADES_AT
        ):
            signal = detect_setup(bar)
            if signal is None:
                continue

            try:
                snap_df = _build_snapshot(bar, signal, symbol, profit_threshold_pct)
                main_prob, indep_prob = _score_setup(
                    snap_df, mean_vol,
                    main_model, main_features,
                    indep_model, indep_features,
                )
            except Exception as exc:
                blocked_main.append({
                    "symbol": symbol,
                    "setup_type": signal.setup_type,
                    "signal_time": signal.signal_time,
                    "signal_close": signal.signal_close,
                    "stop_price": signal.stop_price,
                    "tp_price": signal.tp_price,
                    "main_ai_prob": float("nan"),
                    "independent_ai_prob": float("nan"),
                    "block_reason": f"scoring_error: {exc}",
                })
                continue

            if main_prob < main_threshold:
                blocked_main.append({
                    "symbol": symbol,
                    "setup_type": signal.setup_type,
                    "signal_time": signal.signal_time,
                    "signal_close": signal.signal_close,
                    "stop_price": signal.stop_price,
                    "tp_price": signal.tp_price,
                    "main_ai_prob": main_prob,
                    "independent_ai_prob": float("nan"),
                    "block_reason": "main_ai",
                })
            elif indep_prob < indep_threshold:
                blocked_indep.append({
                    "symbol": symbol,
                    "setup_type": signal.setup_type,
                    "signal_time": signal.signal_time,
                    "signal_close": signal.signal_close,
                    "stop_price": signal.stop_price,
                    "tp_price": signal.tp_price,
                    "main_ai_prob": main_prob,
                    "independent_ai_prob": indep_prob,
                    "block_reason": "independent_ai",
                })
            else:
                pending = signal

    return taken, rejected, blocked_main, blocked_indep


# ── Per-trade metrics ──────────────────────────────────────────────────────────

def _trade_metrics(trades: list) -> dict:
    if not trades:
        return {
            "trades": 0, "winners": 0, "losers": 0,
            "win_rate_pct": 0.0, "net_pnl": 0.0,
            "avg_pnl_per_trade": 0.0, "profit_factor": 0.0,
        }
    pnls = [t.pnl_net for t in trades if t.pnl_net is not None]
    total = len(pnls)
    winners = sum(1 for p in pnls if p > 0)
    net = sum(pnls)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(p for p in pnls if p < 0)
    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")
    return {
        "trades": total,
        "winners": winners,
        "losers": total - winners,
        "win_rate_pct": round(winners / total * 100, 2) if total else 0.0,
        "net_pnl": round(net, 2),
        "avg_pnl_per_trade": round(net / total, 2) if total else 0.0,
        "profit_factor": round(pf, 4) if math.isfinite(pf) else None,
    }


def _records_to_df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    if dataclasses.is_dataclass(records[0]):
        return pd.DataFrame([dataclasses.asdict(r) for r in records])
    return pd.DataFrame(records)


# ── Per-month save ─────────────────────────────────────────────────────────────

def _save_month_results(
    month_dir: Path,
    taken_ai: list,
    rejected_ai: list,
    blocked_main: list,
    blocked_indep: list,
    taken_rule: list,
    rejected_rule: list,
    main_model: CalibratedClassifierCV,
    main_features: list[str],
    main_threshold: float,
    main_report: dict,
    indep_model: CalibratedClassifierCV,
    indep_features: list[str],
    indep_threshold: float,
    indep_report: dict,
    test_start: date,
    test_end: date,
    train_start: date,
    train_end: date,
    n_train_samples: int,
) -> dict:
    models_dir = month_dir / "models"
    reports_dir = month_dir / "reports"
    trades_dir = month_dir / "trades"
    for d in (models_dir, reports_dir, trades_dir):
        d.mkdir(parents=True, exist_ok=True)

    joblib.dump(main_model, models_dir / "main_ai.pkl")
    with open(models_dir / "main_ai_meta.json", "w") as f:
        json.dump({"features": main_features, "threshold_normal": main_threshold}, f, indent=2)

    joblib.dump(indep_model, models_dir / "independent_ai.pkl")
    with open(models_dir / "independent_ai_meta.json", "w") as f:
        json.dump({"features": indep_features, "threshold_normal": indep_threshold}, f, indent=2)

    with open(reports_dir / "main_ai_report.json", "w") as f:
        json.dump(main_report, f, indent=2, default=str)
    with open(reports_dir / "independent_ai_report.json", "w") as f:
        json.dump(indep_report, f, indent=2, default=str)

    _records_to_df(taken_ai).to_parquet(trades_dir / "trades_taken_ai.parquet", index=False)
    _records_to_df(rejected_ai).to_parquet(trades_dir / "trades_rejected_ai.parquet", index=False)
    _records_to_df(blocked_main).to_parquet(trades_dir / "blocked_by_main_ai.parquet", index=False)
    _records_to_df(blocked_indep).to_parquet(
        trades_dir / "blocked_by_independent_ai.parquet", index=False
    )
    _records_to_df(taken_rule).to_parquet(trades_dir / "trades_taken_rule.parquet", index=False)
    _records_to_df(rejected_rule).to_parquet(
        trades_dir / "trades_rejected_rule.parquet", index=False
    )

    rule_m = _trade_metrics(taken_rule)
    ai_m = _trade_metrics(taken_ai)

    def _delta(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 4)

    summary = {
        "month": month_dir.name,
        "test_period": {"start": str(test_start), "end": str(test_end)},
        "train_period": {"start": str(train_start), "end": str(train_end)},
        "n_train_samples": n_train_samples,
        "thresholds": {"main_ai": main_threshold, "independent_ai": indep_threshold},
        "rule_only": {**rule_m, "rejected_by_risk": len(rejected_rule)},
        "ai_filtered": {
            **ai_m,
            "rejected_by_risk": len(rejected_ai),
            "blocked_by_main_ai": len(blocked_main),
            "blocked_by_independent_ai": len(blocked_indep),
            "total_setups_detected": (
                ai_m["trades"] + len(rejected_ai)
                + len(blocked_main) + len(blocked_indep)
            ),
        },
        "comparison": {
            "trade_reduction_pct": round(
                (1.0 - ai_m["trades"] / rule_m["trades"]) * 100, 2
            ) if rule_m["trades"] else None,
            "win_rate_delta_pct": _delta(ai_m["win_rate_pct"], rule_m["win_rate_pct"]),
            "net_pnl_delta": _delta(ai_m["net_pnl"], rule_m["net_pnl"]),
            "avg_pnl_delta": _delta(ai_m["avg_pnl_per_trade"], rule_m["avg_pnl_per_trade"]),
            "profit_factor_delta": _delta(ai_m["profit_factor"], rule_m["profit_factor"]),
        },
    }

    with open(reports_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ── Aggregation across all months ──────────────────────────────────────────────

def _aggregate_results(month_summaries: list[dict], log: logging.Logger) -> None:
    rows = []
    for s in month_summaries:
        rule = s["rule_only"]
        ai = s["ai_filtered"]
        comp = s["comparison"]
        rows.append({
            "month": s["month"],
            "test_start": s["test_period"]["start"],
            "test_end": s["test_period"]["end"],
            "n_train_samples": s.get("n_train_samples", 0),
            "rule_trades": rule["trades"],
            "rule_win_rate_pct": rule["win_rate_pct"],
            "rule_net_pnl": rule["net_pnl"],
            "rule_avg_pnl": rule["avg_pnl_per_trade"],
            "rule_profit_factor": rule["profit_factor"],
            "ai_trades": ai["trades"],
            "ai_win_rate_pct": ai["win_rate_pct"],
            "ai_net_pnl": ai["net_pnl"],
            "ai_avg_pnl": ai["avg_pnl_per_trade"],
            "ai_profit_factor": ai["profit_factor"],
            "blocked_main_ai": ai["blocked_by_main_ai"],
            "blocked_indep_ai": ai["blocked_by_independent_ai"],
            "trade_reduction_pct": comp["trade_reduction_pct"],
            "win_rate_delta_pct": comp["win_rate_delta_pct"],
            "net_pnl_delta": comp["net_pnl_delta"],
        })

    df_monthly = pd.DataFrame(rows)
    df_monthly.to_parquet(_WF_DIR / "monthly_performance.parquet", index=False)
    log.info(f"Saved monthly_performance.parquet ({len(df_monthly)} months)")

    total_rule_trades = int(df_monthly["rule_trades"].sum())
    total_ai_trades = int(df_monthly["ai_trades"].sum())
    total_rule_pnl = round(float(df_monthly["rule_net_pnl"].sum()), 2)
    total_ai_pnl = round(float(df_monthly["ai_net_pnl"].sum()), 2)
    n_months = len(df_monthly)

    def _weighted_wr(trades_col: str, wr_col: str) -> float:
        total = df_monthly[trades_col].sum()
        if total == 0:
            return 0.0
        return round(
            float((df_monthly[trades_col] * df_monthly[wr_col]).sum() / total), 2
        )

    aggregate = {
        "walk_forward_period": {
            "start": str(df_monthly["test_start"].min()),
            "end": str(df_monthly["test_end"].max()),
            "months": n_months,
        },
        "training_window_months": _TRAIN_WINDOW_MONTHS,
        "rule_only": {
            "total_trades": total_rule_trades,
            "total_net_pnl": total_rule_pnl,
            "avg_monthly_pnl": round(total_rule_pnl / n_months, 2),
            "weighted_win_rate_pct": _weighted_wr("rule_trades", "rule_win_rate_pct"),
            "profitable_months": int((df_monthly["rule_net_pnl"] > 0).sum()),
        },
        "ai_filtered": {
            "total_trades": total_ai_trades,
            "total_net_pnl": total_ai_pnl,
            "avg_monthly_pnl": round(total_ai_pnl / n_months, 2),
            "weighted_win_rate_pct": _weighted_wr("ai_trades", "ai_win_rate_pct"),
            "profitable_months": int((df_monthly["ai_net_pnl"] > 0).sum()),
            "total_blocked_main_ai": int(df_monthly["blocked_main_ai"].sum()),
            "total_blocked_indep_ai": int(df_monthly["blocked_indep_ai"].sum()),
        },
        "comparison": {
            "avg_trade_reduction_pct": round(
                float(df_monthly["trade_reduction_pct"].dropna().mean()), 2
            ) if total_rule_trades else None,
            "total_pnl_delta": round(total_ai_pnl - total_rule_pnl, 2),
            "avg_monthly_pnl_delta": round((total_ai_pnl - total_rule_pnl) / n_months, 2),
            "avg_win_rate_delta_pct": round(
                float(df_monthly["win_rate_delta_pct"].dropna().mean()), 2
            ),
        },
        "monthly_breakdown": month_summaries,
    }

    with open(_WF_DIR / "aggregate_report.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    log.info(f"Saved aggregate_report.json")

    # Console summary
    w = 72
    print(f"\n{'=' * w}")
    print("  Walk-Forward Backtest — Etappe 7 — Aggregate Summary")
    print(f"{'=' * w}")
    wfp = aggregate["walk_forward_period"]
    print(
        f"  Period  : {wfp['start']} → {wfp['end']}  ({wfp['months']} months)\n"
        f"  Training: {_TRAIN_WINDOW_MONTHS}-month rolling window"
    )
    print(f"{'─' * w}")
    print(f"  {'Metric':<36}  {'Rule-only':>13}  {'AI-filtered':>13}")
    print(f"  {'─' * 65}")

    def _row(label: str, rv, av) -> None:
        print(f"  {label:<36}  {str(rv):>13}  {str(av):>13}")

    r = aggregate["rule_only"]
    a = aggregate["ai_filtered"]
    c = aggregate["comparison"]

    _row("Total trades", r["total_trades"], a["total_trades"])
    _row("Total net P&L", f"{r['total_net_pnl']:,.2f}", f"{a['total_net_pnl']:,.2f}")
    _row("Avg monthly P&L", f"{r['avg_monthly_pnl']:,.2f}", f"{a['avg_monthly_pnl']:,.2f}")
    _row(
        "Weighted win rate %",
        f"{r['weighted_win_rate_pct']:.1f}%",
        f"{a['weighted_win_rate_pct']:.1f}%",
    )
    _row("Profitable months", r["profitable_months"], a["profitable_months"])
    _row("Total P&L delta", "", f"{c['total_pnl_delta']:+,.2f}")
    _row("Avg monthly P&L delta", "", f"{c['avg_monthly_pnl_delta']:+,.2f}")
    _row("Avg win rate delta %", "", f"{c['avg_win_rate_delta_pct']:+.2f}%")
    if c["avg_trade_reduction_pct"] is not None:
        _row("Avg trade reduction %", "", f"{c['avg_trade_reduction_pct']:.1f}%")

    print(f"\n  Monthly breakdown:")
    print(
        f"  {'Month':<10}  {'Rule P&L':>12}  {'AI P&L':>12}  "
        f"{'Delta':>10}  {'AI Win%':>8}  {'Trades':>6}"
    )
    print(f"  {'─' * 62}")
    for s in month_summaries:
        r_pnl = s["rule_only"]["net_pnl"]
        a_pnl = s["ai_filtered"]["net_pnl"]
        delta = round(a_pnl - r_pnl, 2)
        a_wr = s["ai_filtered"]["win_rate_pct"]
        a_tr = s["ai_filtered"]["trades"]
        print(
            f"  {s['month']:<10}  {r_pnl:>12,.2f}  {a_pnl:>12,.2f}  "
            f"{delta:>+10,.2f}  {a_wr:>7.1f}%  {a_tr:>6}"
        )
    print(f"{'=' * w}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest (Etappe 7)")
    parser.add_argument("--symbols", nargs="+", help="Symbols to process")
    parser.add_argument(
        "--start", help="First test month YYYY-MM (default: backtest.start_date)"
    )
    parser.add_argument(
        "--end", help="Last test month YYYY-MM (default: backtest.end_date)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun months that already have results"
    )
    args = parser.parse_args()

    cfg = _load_config()
    log = _setup_logging()

    ai_cfg = cfg.get("ai", {})
    main_threshold = float(ai_cfg.get("main_ai_threshold_normal", 0.70))
    indep_threshold = float(ai_cfg.get("independent_ai_threshold_normal", 0.60))

    risk_cfg = cfg["risk"]
    bt_cfg = cfg["backtest"]
    dataset_cfg = cfg.get("dataset", {})

    initial_account = float(bt_cfg["initial_account"])
    risk_pct = float(risk_cfg["risk_pct_per_trade"])
    max_stop_pct = float(risk_cfg["max_stop_distance_pct"])
    min_position_pct = float(risk_cfg["minimum_position_size_pct"])
    max_daily_loss_pct = float(risk_cfg["max_daily_loss_pct"])
    profit_normal = float(
        dataset_cfg.get(
            "profit_threshold_normal_pct",
            ai_cfg.get("min_expected_move_normal_pct", 0.20),
        )
    )
    profit_earnings = float(
        dataset_cfg.get(
            "profit_threshold_earnings_pct",
            ai_cfg.get("min_expected_move_earnings_pct", 0.30),
        )
    )

    bt_start = str(bt_cfg.get("start_date", "2023-05-01"))
    bt_end = str(bt_cfg.get("end_date", "2024-10-31"))
    start_ym = args.start or bt_start[:7]
    end_ym = args.end or bt_end[:7]

    features_dir = ROOT / cfg["data"]["features_5m_dir"]

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [p.stem for p in sorted(features_dir.glob("*.parquet"))]

    if not symbols:
        log.warning(f"No feature parquet files found in {features_dir}")
        return

    _WF_DIR.mkdir(parents=True, exist_ok=True)

    test_months = _generate_test_months(start_ym, end_ym)
    log.info(
        f"Walk-forward: {start_ym} → {end_ym} | "
        f"training={_TRAIN_WINDOW_MONTHS}m rolling | "
        f"{len(test_months)} test months | "
        f"{len(symbols)} symbol(s)"
    )

    month_summaries: list[dict] = []

    for test_start, test_end, train_start, train_end in test_months:
        month_str = test_start.strftime("%Y-%m")
        month_dir = _WF_DIR / month_str
        summary_path = month_dir / "reports" / "backtest_summary.json"

        if summary_path.exists() and not args.force:
            log.info(f"{month_str}: results exist — loading (use --force to rerun)")
            try:
                with open(summary_path) as f:
                    month_summaries.append(json.load(f))
            except Exception as exc:
                log.warning(f"{month_str}: could not load cached summary — {exc}")
            continue

        log.info(
            f"{month_str}: train [{train_start} → {train_end}] | "
            f"test [{test_start} → {test_end}]"
        )

        # 1. Build training dataset for the 6-month window
        df_train = _build_training_dataset(
            symbols, features_dir, train_start, train_end,
            profit_normal, profit_earnings, log,
        )
        if df_train is None or len(df_train) < _MIN_TRAIN_SAMPLES:
            n = len(df_train) if df_train is not None else 0
            log.warning(f"{month_str}: {n} training samples (min {_MIN_TRAIN_SAMPLES}) — skipping")
            continue

        log.info(
            f"{month_str}: {len(df_train)} training samples | "
            f"win rate={df_train['label'].mean():.1%} | "
            f"symbols={df_train['symbol'].nunique()}"
        )

        # 2. Train fresh models on this window
        train_result = _train_window_models(df_train, main_threshold, indep_threshold, log)
        if train_result is None:
            log.warning(f"{month_str}: training failed — skipping")
            continue

        main_model, main_features, main_report, indep_model, indep_features, indep_report = (
            train_result
        )
        log.info(
            f"{month_str}: main_ai ROC-AUC={main_report.get('roc_auc')} | "
            f"indep_ai ROC-AUC={indep_report.get('roc_auc')}"
        )

        # 3. Backtest the test month with fresh models
        all_taken_ai: list = []
        all_rejected_ai: list = []
        all_blocked_main: list = []
        all_blocked_indep: list = []
        all_taken_rule: list = []
        all_rejected_rule: list = []

        for symbol in symbols:
            feat_path = features_dir / f"{symbol}.parquet"
            if not feat_path.exists():
                continue
            try:
                df_test = _load_features(feat_path, test_start, test_end)
                if len(df_test) == 0:
                    continue

                mean_vol = float(df_test["volume"].mean())

                taken_ai, rej_ai, bl_main, bl_indep = _simulate_symbol_ai_filtered(
                    symbol=symbol,
                    df=df_test,
                    mean_vol=mean_vol,
                    main_model=main_model,
                    main_features=main_features,
                    main_threshold=main_threshold,
                    indep_model=indep_model,
                    indep_features=indep_features,
                    indep_threshold=indep_threshold,
                    profit_threshold_pct=profit_normal,
                    initial_account=initial_account,
                    risk_pct=risk_pct,
                    max_stop_pct=max_stop_pct,
                    min_position_pct=min_position_pct,
                    max_daily_loss_pct=max_daily_loss_pct,
                )
                taken_rule, rej_rule = simulate_symbol(
                    symbol=symbol,
                    df=df_test,
                    initial_account=initial_account,
                    risk_pct=risk_pct,
                    max_stop_pct=max_stop_pct,
                    min_position_pct=min_position_pct,
                    max_daily_loss_pct=max_daily_loss_pct,
                )

                all_taken_ai.extend(taken_ai)
                all_rejected_ai.extend(rej_ai)
                all_blocked_main.extend(bl_main)
                all_blocked_indep.extend(bl_indep)
                all_taken_rule.extend(taken_rule)
                all_rejected_rule.extend(rej_rule)

            except Exception as exc:
                log.error(f"{month_str} {symbol}: {exc}")

        log.info(
            f"{month_str}: rule={len(all_taken_rule)} | "
            f"ai={len(all_taken_ai)} | "
            f"blocked_main={len(all_blocked_main)} | "
            f"blocked_indep={len(all_blocked_indep)}"
        )

        # 4. Save per-month outputs
        summary = _save_month_results(
            month_dir=month_dir,
            taken_ai=all_taken_ai,
            rejected_ai=all_rejected_ai,
            blocked_main=all_blocked_main,
            blocked_indep=all_blocked_indep,
            taken_rule=all_taken_rule,
            rejected_rule=all_rejected_rule,
            main_model=main_model,
            main_features=main_features,
            main_threshold=main_threshold,
            main_report=main_report,
            indep_model=indep_model,
            indep_features=indep_features,
            indep_threshold=indep_threshold,
            indep_report=indep_report,
            test_start=test_start,
            test_end=test_end,
            train_start=train_start,
            train_end=train_end,
            n_train_samples=len(df_train),
        )
        month_summaries.append(summary)
        log.info(f"{month_str}: saved to {month_dir}")

    # 5. Aggregate all months
    if month_summaries:
        _aggregate_results(month_summaries, log)
        log.info(f"Walk-forward complete. Results in {_WF_DIR}")
    else:
        log.warning("No months completed — check data availability and date range")


if __name__ == "__main__":
    main()

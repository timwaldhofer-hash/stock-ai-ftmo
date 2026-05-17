#!/usr/bin/env python3
"""
AI-filtered backtest — Etappe 6.

Loads both trained AI models, scores every rule-detected setup, and takes
trades only when both models clear their probability thresholds:
  main_ai_probability    >= 0.70  (ai.main_ai_threshold_normal)
  independent_ai_prob    >= 0.60  (ai.independent_ai_threshold_normal)

Reads  : models/main_ai.pkl, models/main_ai_meta.json
         models/independent_ai.pkl, models/independent_ai_meta.json
         data/features/5m/<symbol>.parquet

Writes : data/backtests/ai_filtered/trades_taken.parquet
         data/backtests/ai_filtered/trades_rejected.parquet
         data/backtests/ai_filtered/blocked_by_main_ai.parquet
         data/backtests/ai_filtered/blocked_by_independent_ai.parquet
         data/backtests/ai_filtered/summary_report.json

Usage:
    python scripts/run_ai_backtest.py
    python scripts/run_ai_backtest.py --symbols AAPL MSFT
    python scripts/run_ai_backtest.py --start 2023-05-01 --end 2024-10-31
    python scripts/run_ai_backtest.py --force
"""

import argparse
import dataclasses
import json
import logging
import math
import sys
from datetime import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml

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
from src.models.features import engineer_features

TZ = "America/New_York"
_NO_NEW_TRADES_AT = time(15, 30)

_MODELS_DIR = ROOT / "models"
_OUT_DIR = ROOT / "data" / "backtests" / "ai_filtered"


# ── Config & logging ───────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("run_ai_backtest")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models() -> tuple:
    """
    Returns (main_model, main_features, main_threshold,
             indep_model, indep_features, indep_threshold).
    Feature lists and thresholds come from the meta JSON files saved by
    train_models.py so inference uses the exact same columns as training.
    """
    for fname in ("main_ai.pkl", "main_ai_meta.json",
                  "independent_ai.pkl", "independent_ai_meta.json"):
        p = _MODELS_DIR / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found: {p}\n"
                "Run  python scripts/train_models.py  first."
            )

    main_model = joblib.load(_MODELS_DIR / "main_ai.pkl")
    indep_model = joblib.load(_MODELS_DIR / "independent_ai.pkl")

    with open(_MODELS_DIR / "main_ai_meta.json") as f:
        main_meta = json.load(f)
    with open(_MODELS_DIR / "independent_ai_meta.json") as f:
        indep_meta = json.load(f)

    return (
        main_model,
        main_meta["features"],
        float(main_meta["threshold_normal"]),
        indep_model,
        indep_meta["features"],
        float(indep_meta["threshold_normal"]),
    )


# ── Feature loading ────────────────────────────────────────────────────────────

def load_features(
    path: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    else:
        df["timestamp"] = ts.dt.tz_convert(TZ)
    if start:
        df = df[df["timestamp"].dt.date >= pd.Timestamp(start).date()]
    if end:
        df = df[df["timestamp"].dt.date <= pd.Timestamp(end).date()]
    return df.sort_values("timestamp").reset_index(drop=True)


# ── AI scoring helpers ─────────────────────────────────────────────────────────

def _select_features(snap_df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """Return a DataFrame with exactly `feature_list` columns in order.
    Missing columns are filled with NaN — the model's SimpleImputer handles them.
    """
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
    """
    Build a single-row DataFrame that mirrors what build_training_dataset.py
    produced per detected setup.  The signal bar's indicators are taken as-is;
    setup geometry fields are filled from the SetupSignal.
    """
    snap = bar.to_dict()
    snap["symbol"] = symbol
    snap["setup_type"] = signal.setup_type
    snap["side"] = signal.direction
    # Use signal_close as entry_price proxy (avoids 1-bar lookahead at signal time)
    snap["entry_price"] = signal.signal_close
    snap["stop_price"] = signal.stop_price
    snap["target_price"] = signal.tp_price
    snap["expected_move_required_pct"] = profit_threshold_pct
    return pd.DataFrame([snap])


def _score_setup(
    snap_df: pd.DataFrame,
    mean_vol: float,
    main_model,
    main_features: list,
    indep_model,
    indep_features: list,
) -> tuple:
    """
    Run feature engineering and score with both models.
    Returns (main_prob, indep_prob).
    """
    snap_df = engineer_features(snap_df.copy())

    # volume_rel: volume / per-symbol mean volume (computed from backtest data)
    vol = float(snap_df["volume"].iloc[0])
    snap_df["volume_rel"] = min(vol / max(mean_vol, 1.0), 50.0)

    X_main = _select_features(snap_df, main_features)
    X_indep = _select_features(snap_df, indep_features)

    main_prob = float(main_model.predict_proba(X_main)[:, 1][0])
    indep_prob = float(indep_model.predict_proba(X_indep)[:, 1][0])
    return main_prob, indep_prob


# ── AI-filtered simulation ─────────────────────────────────────────────────────

def simulate_symbol_ai_filtered(
    symbol: str,
    df: pd.DataFrame,
    mean_vol: float,
    main_model,
    main_features: list,
    main_threshold: float,
    indep_model,
    indep_features: list,
    indep_threshold: float,
    profit_threshold_pct: float,
    initial_account: float,
    risk_pct: float,
    max_stop_pct: float,
    min_position_pct: float,
    max_daily_loss_pct: float,
) -> tuple:
    """
    Per-symbol backtest loop with AI gate at setup-detection step.

    Returns:
        taken          list[Trade]
        rejected       list[RejectedSignal]  (risk-rule rejections after AI pass)
        blocked_main   list[dict]            (setups blocked by main AI)
        blocked_indep  list[dict]            (setups that passed main AI but failed
                                              independent AI)
    """
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

        # ── 1. Day boundary ────────────────────────────────────────────────
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl = 0.0
            day_start_account = account
            day_loss_hit = False
            if pending is not None and pending.signal_time.date() != bar_date:
                pending = None

        # ── 2. Pending entry ───────────────────────────────────────────────
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

        # ── 3. Exit check ──────────────────────────────────────────────────
        if open_trade is not None:
            result = check_exits(open_trade, bar)
            if result is not None:
                raw_exit, reason = result
                actual_exit = exit_price_after_slippage(raw_exit, open_trade.direction)

                gross = (
                    open_trade.shares
                    * open_trade.direction
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

        # ── 4. Setup scan + AI gate ────────────────────────────────────────
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


# ── Reporting helpers ──────────────────────────────────────────────────────────

def _trade_metrics(trades: list) -> dict:
    if not trades:
        return {
            "trades": 0,
            "winners": 0,
            "losers": 0,
            "win_rate_pct": 0.0,
            "net_pnl": 0.0,
            "avg_pnl_per_trade": 0.0,
            "profit_factor": 0.0,
        }
    pnls = [t.pnl_net for t in trades if t.pnl_net is not None]
    total = len(pnls)
    winners = sum(1 for p in pnls if p > 0)
    losers = total - winners
    net = sum(pnls)
    avg = net / total if total else 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(p for p in pnls if p < 0)
    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")
    return {
        "trades": total,
        "winners": winners,
        "losers": losers,
        "win_rate_pct": round(winners / total * 100, 2) if total else 0.0,
        "net_pnl": round(net, 2),
        "avg_pnl_per_trade": round(avg, 2),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else None,
    }


def _records_to_df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    if dataclasses.is_dataclass(records[0]):
        return pd.DataFrame([dataclasses.asdict(r) for r in records])
    return pd.DataFrame(records)


def save_results(
    taken: list,
    rejected: list,
    blocked_main: list,
    blocked_indep: list,
    rule_taken: list,
    rule_rejected: list,
    out_dir: Path,
    main_threshold: float,
    indep_threshold: float,
    date_range: dict,
    symbols: list,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _records_to_df(taken).to_parquet(out_dir / "trades_taken.parquet", index=False)
    _records_to_df(rejected).to_parquet(out_dir / "trades_rejected.parquet", index=False)
    _records_to_df(blocked_main).to_parquet(out_dir / "blocked_by_main_ai.parquet", index=False)
    _records_to_df(blocked_indep).to_parquet(
        out_dir / "blocked_by_independent_ai.parquet", index=False
    )

    rule_m = _trade_metrics(rule_taken)
    ai_m = _trade_metrics(taken)

    def _delta(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 4)

    report = {
        "thresholds": {
            "main_ai": main_threshold,
            "independent_ai": indep_threshold,
        },
        "date_range": date_range,
        "symbols_processed": len(symbols),
        "rule_only": {
            **rule_m,
            "rejected_by_risk": len(rule_rejected),
        },
        "ai_filtered": {
            **ai_m,
            "rejected_by_risk": len(rejected),
            "blocked_by_main_ai": len(blocked_main),
            "blocked_by_independent_ai": len(blocked_indep),
            "total_setups_detected": (
                ai_m["trades"]
                + len(rejected)
                + len(blocked_main)
                + len(blocked_indep)
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

    with open(out_dir / "summary_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


def print_summary(
    taken: list,
    rejected: list,
    blocked_main: list,
    blocked_indep: list,
    rule_taken: list,
    rule_rejected: list,
    main_threshold: float,
    indep_threshold: float,
) -> None:
    rule_m = _trade_metrics(rule_taken)
    ai_m = _trade_metrics(taken)

    w = 58
    print(f"\n{'=' * w}")
    print("  AI-Filtered Backtest — Etappe 6 — Summary")
    print(f"{'=' * w}")
    print(f"  Thresholds : main_ai >= {main_threshold:.2f}  |  independent_ai >= {indep_threshold:.2f}")
    print(f"{'─' * w}")
    print(f"  {'Metric':<30s}  {'Rule-only':>10}  {'AI-filtered':>10}")
    print(f"  {'─' * 54}")

    def _row(label, rv, av, fmt="{:>10}"):
        rv_s = fmt.format(rv) if rv is not None else "       N/A"
        av_s = fmt.format(av) if av is not None else "       N/A"
        print(f"  {label:<30s}  {rv_s}  {av_s}")

    _row("Trades taken", rule_m["trades"], ai_m["trades"], "{:>10d}")
    _row("Rejected by risk rules", len(rule_rejected), len(rejected), "{:>10d}")
    _row("Blocked by main AI", "-", len(blocked_main), "{:>10}")
    _row("Blocked by independent AI", "-", len(blocked_indep), "{:>10}")
    _row("Win rate %", f"{rule_m['win_rate_pct']:.1f}%", f"{ai_m['win_rate_pct']:.1f}%")
    _row("Net P&L", f"{rule_m['net_pnl']:,.2f}", f"{ai_m['net_pnl']:,.2f}")
    _row("Avg P&L / trade", f"{rule_m['avg_pnl_per_trade']:,.2f}", f"{ai_m['avg_pnl_per_trade']:,.2f}")

    pf_r = f"{rule_m['profit_factor']:.3f}" if rule_m["profit_factor"] is not None else "  inf"
    pf_a = f"{ai_m['profit_factor']:.3f}" if ai_m["profit_factor"] is not None else "  inf"
    _row("Profit factor", pf_r, pf_a)

    if rule_m["trades"] and ai_m["trades"]:
        reduction = (1.0 - ai_m["trades"] / rule_m["trades"]) * 100
        print(f"\n  Trade reduction by AI filter : {reduction:.1f}%")

    if ai_m["trades"]:
        df = _records_to_df(taken)
        by_type = df.groupby("setup_type")["pnl_net"].agg(["count", "sum", "mean"])
        print(f"\n  {'Setup type':<28s}  {'#':>5}  {'Total net':>12}  {'Avg net':>10}")
        print(f"  {'─' * 57}")
        for stype, row in by_type.iterrows():
            print(
                f"  {stype:<28s}  {int(row['count']):>5}"
                f"  {row['sum']:>12,.2f}  {row['mean']:>10,.2f}"
            )

        exit_dist = df["exit_reason"].value_counts().to_dict()
        print(f"\n  Exit breakdown (AI-filtered): {exit_dist}")

    print(f"{'=' * w}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-filtered backtest (Etappe 6)"
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to backtest")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing result files"
    )
    args = parser.parse_args()

    cfg = load_config()
    log = setup_logging()

    # Guard
    if (_OUT_DIR / "trades_taken.parquet").exists() and not args.force:
        log.info("Results already exist — use --force to overwrite")
        return

    # Load models
    try:
        (
            main_model, main_features, main_threshold,
            indep_model, indep_features, indep_threshold,
        ) = load_models()
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info(
        f"Models loaded | main_ai: {len(main_features)} features, "
        f"threshold={main_threshold} | "
        f"independent_ai: {len(indep_features)} features, "
        f"threshold={indep_threshold}"
    )

    ai_cfg = cfg.get("ai", {})
    # Override thresholds from config if they differ from meta
    cfg_main_thr = float(ai_cfg.get("main_ai_threshold_normal", main_threshold))
    cfg_indep_thr = float(ai_cfg.get("independent_ai_threshold_normal", indep_threshold))
    if cfg_main_thr != main_threshold or cfg_indep_thr != indep_threshold:
        log.info(
            f"Threshold override from config: "
            f"main={cfg_main_thr} (meta={main_threshold}), "
            f"indep={cfg_indep_thr} (meta={indep_threshold})"
        )
        main_threshold = cfg_main_thr
        indep_threshold = cfg_indep_thr

    risk_cfg = cfg["risk"]
    bt_cfg = cfg["backtest"]
    dataset_cfg = cfg.get("dataset", {})

    initial_account = float(bt_cfg["initial_account"])
    risk_pct = float(risk_cfg["risk_pct_per_trade"])
    max_stop_pct = float(risk_cfg["max_stop_distance_pct"])
    min_position_pct = float(risk_cfg["minimum_position_size_pct"])
    max_daily_loss_pct = float(risk_cfg["max_daily_loss_pct"])
    profit_threshold_pct = float(
        dataset_cfg.get(
            "profit_threshold_normal_pct",
            ai_cfg.get("min_expected_move_normal_pct", 0.20),
        )
    )

    start = args.start or str(bt_cfg.get("start_date", ""))
    end = args.end or str(bt_cfg.get("end_date", ""))

    features_dir = ROOT / cfg["data"]["features_5m_dir"]

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [p.stem for p in sorted(features_dir.glob("*.parquet"))]

    if not symbols:
        log.warning(f"No feature parquet files found in {features_dir}")
        return

    log.info(
        f"Backtesting {len(symbols)} symbol(s) | "
        f"account={initial_account:,.0f} | "
        f"period={start or 'all'} -> {end or 'all'}"
    )

    all_taken_ai: list = []
    all_rejected_ai: list = []
    all_blocked_main: list = []
    all_blocked_indep: list = []
    all_taken_rule: list = []
    all_rejected_rule: list = []

    for symbol in symbols:
        feat_path = features_dir / f"{symbol}.parquet"
        if not feat_path.exists():
            log.warning(f"{symbol}: feature file not found — skipping")
            continue

        try:
            df = load_features(feat_path, start or None, end or None)

            # Per-symbol mean volume for volume_rel computation at inference time
            mean_vol = float(df["volume"].mean()) if len(df) > 0 else 1.0

            # AI-filtered simulation
            taken_ai, rejected_ai, blocked_main, blocked_indep = (
                simulate_symbol_ai_filtered(
                    symbol=symbol,
                    df=df,
                    mean_vol=mean_vol,
                    main_model=main_model,
                    main_features=main_features,
                    main_threshold=main_threshold,
                    indep_model=indep_model,
                    indep_features=indep_features,
                    indep_threshold=indep_threshold,
                    profit_threshold_pct=profit_threshold_pct,
                    initial_account=initial_account,
                    risk_pct=risk_pct,
                    max_stop_pct=max_stop_pct,
                    min_position_pct=min_position_pct,
                    max_daily_loss_pct=max_daily_loss_pct,
                )
            )

            # Rule-only simulation (for comparison)
            taken_rule, rejected_rule = simulate_symbol(
                symbol=symbol,
                df=df,
                initial_account=initial_account,
                risk_pct=risk_pct,
                max_stop_pct=max_stop_pct,
                min_position_pct=min_position_pct,
                max_daily_loss_pct=max_daily_loss_pct,
            )

            all_taken_ai.extend(taken_ai)
            all_rejected_ai.extend(rejected_ai)
            all_blocked_main.extend(blocked_main)
            all_blocked_indep.extend(blocked_indep)
            all_taken_rule.extend(taken_rule)
            all_rejected_rule.extend(rejected_rule)

            log.info(
                f"{symbol}: rule={len(taken_rule)} trades | "
                f"ai={len(taken_ai)} trades | "
                f"blocked_main={len(blocked_main)} | "
                f"blocked_indep={len(blocked_indep)}"
            )

        except Exception as e:
            log.error(f"{symbol}: error — {e}")

    save_results(
        taken=all_taken_ai,
        rejected=all_rejected_ai,
        blocked_main=all_blocked_main,
        blocked_indep=all_blocked_indep,
        rule_taken=all_taken_rule,
        rule_rejected=all_rejected_rule,
        out_dir=_OUT_DIR,
        main_threshold=main_threshold,
        indep_threshold=indep_threshold,
        date_range={"start": start or "all", "end": end or "all"},
        symbols=symbols,
    )
    log.info(f"Results saved to {_OUT_DIR}")

    print_summary(
        taken=all_taken_ai,
        rejected=all_rejected_ai,
        blocked_main=all_blocked_main,
        blocked_indep=all_blocked_indep,
        rule_taken=all_taken_rule,
        rule_rejected=all_rejected_rule,
        main_threshold=main_threshold,
        indep_threshold=indep_threshold,
    )


if __name__ == "__main__":
    main()

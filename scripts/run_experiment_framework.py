#!/usr/bin/env python3
"""
Strategy Experiment Framework.

Tests parameterised strategy variants against the walk-forward backtest
period WITHOUT modifying the core strategy or any cached outputs.

Experiment outputs: data/walk_forward/strategy_experiments/

Modes
  quick       8 hand-picked variants (AI threshold, move filter, veto, exits)
  full        180 variants — grid of AI × move × veto × exit configs
  exit_rr     12 Exit/RR-stage variants targeting avg-loss reduction and
              improved avg-win/avg-loss ratio via partial TP, ATR trailing,
              time-based stop tightening and min R:R filters
  exit_rr_full  Full grid (~200 variants) for the exit/RR stage

FTMO rules enforced
  • max daily drawdown  : 4 %
  • max total drawdown  : 9 %
  Variants violating these are marked ftmo_passed=False and scored 0.

Usage
  python scripts/run_experiment_framework.py --mode quick
  python scripts/run_experiment_framework.py --mode full
  python scripts/run_experiment_framework.py --mode exit_rr
  python scripts/run_experiment_framework.py --mode exit_rr_full
  python scripts/run_experiment_framework.py --mode exit_rr --force
  python scripts/run_experiment_framework.py --mode exit_rr --symbols AAPL MSFT
"""

import argparse
import json
import logging
import sys
from calendar import monthrange
from datetime import date, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtester.setup_detection import detect_setup
from src.experiments.loss_veto import (
    LOSS_VETO_FEATURES,
    add_veto_indicators,
    train_loss_veto_model,
)
from src.experiments.metrics import (
    compute_monthly_stats,
    compute_robustness_score,
    compute_stability_metrics,
)
from src.experiments.reporter import save_experiment_results
from src.experiments.sim_variant import batch_simulate_symbol
from src.experiments.variant_config import (
    EXIT_RR_FULL_VARIANTS,
    EXIT_RR_QUICK_VARIANTS,
    FULL_MODE_VARIANTS,
    QUICK_MODE_VARIANTS,
    VariantConfig,
)
from src.models.features import (
    MAIN_AI_FEATURES,
    add_relative_volume,
    engineer_features,
)

TZ = "America/New_York"
_WF_DIR = ROOT / "data" / "walk_forward"
_EXP_DIR = _WF_DIR / "strategy_experiments"
_TRAIN_WINDOW_MONTHS = 6
_MIN_TRAIN_SAMPLES = 50
_MIN_CAL_SAMPLES = 10
# Extra calendar days loaded before test window for indicator warmup
_WARMUP_DAYS = 30
_NO_NEW_TRADES_TRAIN = time(15, 30)   # same cutoff used when building train samples


# ── Config / logging ──────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("experiment_framework")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


# ── Date helpers ──────────────────────────────────────────────────────────────

def _add_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    last = monthrange(year, month)[1]
    return date(year, month, min(d.day, last))


def _discover_wf_months() -> List[str]:
    """Return sorted list of months that have a cached main_ai.pkl."""
    months = []
    if not _WF_DIR.exists():
        return months
    for p in sorted(_WF_DIR.iterdir()):
        if p.is_dir() and len(p.name) == 7 and p.name[4] == "-":
            if (p / "models" / "main_ai.pkl").exists():
                months.append(p.name)
    return months


# ── Feature loading ───────────────────────────────────────────────────────────

def _load_features(path: Path, start: date, end: date) -> pd.DataFrame:
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    else:
        df["timestamp"] = ts.dt.tz_convert(TZ)
    mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
    return df[mask].sort_values("timestamp").reset_index(drop=True)


def _load_features_with_warmup(
    path: Path, start: date, end: date, warmup_days: int = _WARMUP_DAYS
) -> pd.DataFrame:
    """Load with extra lookback so rolling/EWM indicators are fully initialised."""
    extended_start = start - timedelta(days=warmup_days)
    return _load_features(path, extended_start, end)


# ── Training dataset builder ──────────────────────────────────────────────────

def _is_earnings(bar: pd.Series) -> bool:
    return bool(bar.get("is_earnings", False))


def _simulate_label(
    bars: pd.DataFrame,
    entry_idx: int,
    direction: int,
    entry_price: float,
    stop_price: float,
    threshold_pct: float,
) -> tuple:
    eod = time(15, 50)
    if direction == 1:
        target = entry_price * (1.0 + threshold_pct / 100.0)
    else:
        target = entry_price * (1.0 - threshold_pct / 100.0)
    last = len(bars) - 1
    for idx in range(entry_idx, len(bars)):
        bar = bars.iloc[idx]
        bt = bar["timestamp"].time()
        if bt >= eod:
            return 0, "EOD", idx
        if direction == 1 and float(bar["low"]) <= stop_price:
            return 0, "STOP", idx
        if direction == -1 and float(bar["high"]) >= stop_price:
            return 0, "STOP", idx
        if direction == 1 and float(bar["high"]) >= target:
            return 1, "TARGET", idx
        if direction == -1 and float(bar["low"]) <= target:
            return 1, "TARGET", idx
    return 0, "EOD", last


def _build_samples_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    profit_normal_pct: float,
    profit_earnings_pct: float,
) -> List[dict]:
    samples: List[dict] = []
    n = len(df)
    i = 0
    while i < n:
        bar = df.iloc[i]
        if bar["timestamp"].time() >= _NO_NEW_TRADES_TRAIN:
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
        is_earn = _is_earnings(bar)
        thr = profit_earnings_pct if is_earn else profit_normal_pct
        label, exit_reason, exit_idx = _simulate_label(
            df, entry_idx, signal.direction, entry_price, signal.stop_price, thr
        )
        snap = bar.to_dict()
        snap.update({
            "symbol": symbol,
            "setup_type": signal.setup_type,
            "side": signal.direction,
            "entry_price": round(entry_price, 4),
            "stop_price": signal.stop_price,
            "target_price": signal.tp_price,
            "expected_move_required_pct": thr,
            "is_earnings": is_earn,
            "label": label,
            "label_exit_reason": exit_reason,
        })
        samples.append(snap)
        i = exit_idx + 1
    return samples


def _build_training_dataset(
    symbols: List[str],
    features_dir: Path,
    train_start: date,
    train_end: date,
    profit_normal: float,
    profit_earnings: float,
    log: logging.Logger,
) -> Optional[pd.DataFrame]:
    all_samples: List[dict] = []
    for symbol in symbols:
        feat_path = features_dir / f"{symbol}.parquet"
        if not feat_path.exists():
            continue
        try:
            df = _load_features(feat_path, train_start, train_end)
            if len(df) == 0:
                continue
            # Add ADX + realised vol BEFORE building samples so they appear in snapshots
            df = add_veto_indicators(df)
            samples = _build_samples_for_symbol(symbol, df, profit_normal, profit_earnings)
            all_samples.extend(samples)
        except Exception as exc:
            log.warning(f"  {symbol}: sample build error — {exc}")
    if not all_samples:
        return None
    result = pd.DataFrame(all_samples)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result.sort_values("timestamp").reset_index(drop=True)


# ── Month model helpers ───────────────────────────────────────────────────────

def _time_split(df: pd.DataFrame, fit_ratio: float = 0.70, cal_ratio: float = 0.10):
    n = len(df)
    fe = int(n * fit_ratio)
    ce = int(n * (fit_ratio + cal_ratio))
    fit = pd.Series(False, index=df.index)
    cal = pd.Series(False, index=df.index)
    fit.iloc[:fe] = True
    cal.iloc[fe:ce] = True
    return fit, cal


def _load_or_train_month_models(
    month_str: str,
    symbols: List[str],
    features_dir: Path,
    train_start: date,
    train_end: date,
    profit_normal: float,
    profit_earnings: float,
    force: bool,
    log: logging.Logger,
) -> Optional[tuple]:
    """
    Load cached main_ai model for this month + train/load loss_veto model.
    Returns (main_model, main_features, loss_veto_model) or None on failure.
    """
    month_dir = _WF_DIR / month_str
    exp_month_dir = _EXP_DIR / month_str
    exp_month_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cached main_ai ──────────────────────────────────────────────────
    main_pkl = month_dir / "models" / "main_ai.pkl"
    main_meta = month_dir / "models" / "main_ai_meta.json"
    if not main_pkl.exists():
        log.warning(f"  {month_str}: main_ai.pkl not found — skipping month")
        return None
    try:
        main_model = joblib.load(main_pkl)
        with open(main_meta) as f:
            meta = json.load(f)
        main_features = meta["features"]
    except Exception as exc:
        log.warning(f"  {month_str}: could not load main_ai model — {exc}")
        return None

    # ── Load or train loss_veto ──────────────────────────────────────────────
    veto_pkl = exp_month_dir / "loss_veto.pkl"
    veto_meta = exp_month_dir / "loss_veto_meta.json"

    if veto_pkl.exists() and not force:
        try:
            loss_veto_model = joblib.load(veto_pkl)
            log.info(f"  {month_str}: loaded cached loss_veto model")
            return main_model, main_features, loss_veto_model
        except Exception as exc:
            log.warning(f"  {month_str}: could not load cached loss_veto — retraining: {exc}")

    log.info(f"  {month_str}: building training data for loss_veto …")
    df_train = _build_training_dataset(
        symbols, features_dir, train_start, train_end,
        profit_normal, profit_earnings, log,
    )
    if df_train is None or len(df_train) < _MIN_TRAIN_SAMPLES:
        n = len(df_train) if df_train is not None else 0
        log.warning(f"  {month_str}: {n} training samples — skipping loss_veto")
        return main_model, main_features, None

    log.info(f"  {month_str}: {len(df_train)} training samples — training loss_veto …")
    try:
        df_feat = engineer_features(df_train)
        fit_mask, cal_mask = _time_split(df_feat)
        if fit_mask.sum() < _MIN_TRAIN_SAMPLES or cal_mask.sum() < _MIN_CAL_SAMPLES:
            log.warning(f"  {month_str}: insufficient data for loss_veto")
            return main_model, main_features, None

        df_feat = add_relative_volume(df_feat, fit_mask)
        loss_veto_model = train_loss_veto_model(df_feat, fit_mask, cal_mask)

        joblib.dump(loss_veto_model, veto_pkl)
        with open(veto_meta, "w") as f:
            json.dump({"features": LOSS_VETO_FEATURES, "train_period": {
                "start": str(train_start), "end": str(train_end),
            }}, f, indent=2)

        log.info(f"  {month_str}: loss_veto trained and cached")
        return main_model, main_features, loss_veto_model

    except Exception as exc:
        log.warning(f"  {month_str}: loss_veto training failed — {exc}")
        return main_model, main_features, None


# ── Load baseline monthly PnL ─────────────────────────────────────────────────

def _load_baseline_monthly_pnl() -> Dict[str, float]:
    mp = _WF_DIR / "monthly_performance.parquet"
    if not mp.exists():
        return {}
    try:
        df = pd.read_parquet(mp)
        return dict(zip(df["month"].astype(str), df["ai_net_pnl"].astype(float)))
    except Exception:
        return {}


# ── Per-variant result builder ────────────────────────────────────────────────

def _compute_variant_result(
    variant: VariantConfig,
    monthly_trades: Dict[str, list],
    monthly_dd: Dict[str, tuple],
    baseline_monthly_pnl: Dict[str, float],
    initial_account: float,
) -> Dict:
    """Assemble the full result dict for one variant."""
    monthly_stats_list = []
    for month_str, trades in sorted(monthly_trades.items()):
        stats = compute_monthly_stats(trades, month_str)
        monthly_stats_list.append(stats)

    max_daily_dd = max((dd[0] for dd in monthly_dd.values()), default=0.0)
    max_total_dd = max((dd[1] for dd in monthly_dd.values()), default=0.0)

    stability = compute_stability_metrics(
        monthly_stats_list,
        baseline_monthly_pnl=baseline_monthly_pnl,
        max_daily_dd_pct=max_daily_dd,
        max_total_dd_pct=max_total_dd,
        initial_account=initial_account,
    )
    score = compute_robustness_score(stability)

    # Overfitting penalty for reporting
    import math
    btm = stability.get("best_to_median_ratio")
    op = 0.0
    if btm and math.isfinite(btm) and btm > 3.0:
        op = min(0.3, (btm - 3.0) / 10.0)

    result = {
        "variant_name": variant.name,
        "description": variant.description,
        # Parameters
        "main_ai_threshold": variant.main_ai_threshold,
        "min_expected_move_pct": variant.min_expected_move_pct,
        "allow_mean_reversion": variant.allow_mean_reversion,
        "use_loss_veto": variant.use_loss_veto,
        "loss_veto_threshold": variant.loss_veto_threshold if variant.use_loss_veto else None,
        "atr_stop_multiplier": variant.atr_stop_multiplier,
        "breakeven_trigger_pct": variant.breakeven_trigger_pct,
        "trailing_stop_pct": variant.trailing_stop_pct,
        "max_holding_bars": variant.max_holding_bars,
        "momentum_loss_exit": variant.momentum_loss_exit,
        "min_reward_risk": variant.min_reward_risk,
        "partial_tp_fraction": variant.partial_tp_fraction,
        "partial_tp_target_pct": variant.partial_tp_target_pct if variant.partial_tp_fraction else None,
        "atr_trail_mult": variant.atr_trail_mult,
        "time_stop_bars": variant.time_stop_bars,
        "time_stop_factor": variant.time_stop_factor if variant.time_stop_bars else None,
        # Metrics
        **stability,
        "robustness_adjusted_score": score,
        "overfitting_penalty": round(op, 4),
        # Monthly breakdown (for JSON output)
        "monthly_breakdown": monthly_stats_list,
    }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy experiment framework")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "exit_rr", "exit_rr_full"],
        default="quick",
        help=(
            "quick (8 variants) | full (180-variant grid) | "
            "exit_rr (12 Exit/RR variants) | exit_rr_full (~200-variant Exit/RR grid)"
        ),
    )
    parser.add_argument("--symbols", nargs="+", help="Override symbol list")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if cached results exist",
    )
    args = parser.parse_args()

    cfg = _load_config()
    log = _setup_logging()

    ai_cfg = cfg.get("ai", {})
    risk_cfg = cfg["risk"]
    bt_cfg = cfg["backtest"]
    dataset_cfg = cfg.get("dataset", {})

    initial_account = float(bt_cfg["initial_account"])
    risk_pct = float(risk_cfg["risk_pct_per_trade"])
    max_stop_pct = float(risk_cfg["max_stop_distance_pct"])
    min_position_pct = float(risk_cfg["minimum_position_size_pct"])
    max_daily_loss_pct = float(risk_cfg["max_daily_loss_pct"])
    profit_normal = float(
        dataset_cfg.get("profit_threshold_normal_pct",
                        ai_cfg.get("min_expected_move_normal_pct", 0.20))
    )
    profit_earnings = float(
        dataset_cfg.get("profit_threshold_earnings_pct",
                        ai_cfg.get("min_expected_move_earnings_pct", 0.30))
    )

    features_dir = ROOT / cfg["data"]["features_5m_dir"]

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [p.stem for p in sorted(features_dir.glob("*.parquet"))]

    if not symbols:
        log.error(f"No feature files found in {features_dir}")
        return

    # Select variants
    _variant_map = {
        "quick": QUICK_MODE_VARIANTS,
        "full": FULL_MODE_VARIANTS,
        "exit_rr": EXIT_RR_QUICK_VARIANTS,
        "exit_rr_full": EXIT_RR_FULL_VARIANTS,
    }
    variants = _variant_map[args.mode]
    log.info(
        f"Mode: {args.mode.upper()} | {len(variants)} variants | "
        f"{len(symbols)} symbols"
    )

    # Discover walk-forward months
    wf_months = _discover_wf_months()
    if not wf_months:
        log.error("No walk-forward months found — run run_walk_forward.py first")
        return
    log.info(f"Found {len(wf_months)} walk-forward months: {wf_months[0]} -> {wf_months[-1]}")

    # Load baseline for comparison
    baseline_monthly_pnl = _load_baseline_monthly_pnl()
    log.info(f"Loaded baseline monthly PnL for {len(baseline_monthly_pnl)} months")

    # Check for cached experiment run
    cache_file = _EXP_DIR / "experiment_cache.json"
    cached_results: Dict[str, dict] = {}
    if cache_file.exists() and not args.force:
        try:
            with open(cache_file) as f:
                cached_results = json.load(f)
            log.info(f"Loaded {len(cached_results)} cached variant results")
        except Exception:
            cached_results = {}

    _EXP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Main loop ────────────────────────────────────────────────────────────
    # Per-variant accumulator: {variant_name: {month: [trades]}}
    variant_monthly_trades: Dict[str, Dict[str, list]] = {v.name: {} for v in variants}
    variant_monthly_dd: Dict[str, Dict[str, tuple]] = {v.name: {} for v in variants}

    for month_str in wf_months:
        # Parse dates
        year, mo = int(month_str[:4]), int(month_str[5:7])
        test_start = date(year, mo, 1)
        _, last_day = monthrange(year, mo)
        test_end = date(year, mo, last_day)
        train_end = test_start - timedelta(days=1)
        train_start = _add_months(test_start, -_TRAIN_WINDOW_MONTHS)

        log.info(
            f"\n{'-' * 60}\n"
            f"Month {month_str}: test [{test_start}->{test_end}]  "
            f"train [{train_start}->{train_end}]"
        )

        # Load / train models for this month
        model_result = _load_or_train_month_models(
            month_str, symbols, features_dir,
            train_start, train_end,
            profit_normal, profit_earnings,
            force=args.force, log=log,
        )
        if model_result is None:
            log.warning(f"  {month_str}: no models — skipping month")
            continue

        main_model, main_features, loss_veto_model = model_result

        if loss_veto_model is None:
            log.info(f"  {month_str}: no loss_veto model (variants using it will skip veto gate)")

        # Load test-month feature data per symbol (with warmup for indicators)
        symbol_test_data: Dict[str, pd.DataFrame] = {}
        symbol_mean_vol: Dict[str, float] = {}
        for symbol in symbols:
            feat_path = features_dir / f"{symbol}.parquet"
            if not feat_path.exists():
                continue
            try:
                df_full = _load_features_with_warmup(feat_path, test_start, test_end)
                if len(df_full) == 0:
                    continue
                # Add veto indicators on full range (needs rolling history)
                df_full = add_veto_indicators(df_full)
                # Slice to test window only for simulation
                df_test = df_full[df_full["timestamp"].dt.date >= test_start].copy()
                if len(df_test) == 0:
                    continue
                symbol_test_data[symbol] = df_test
                symbol_mean_vol[symbol] = float(df_test["volume"].mean())
            except Exception as exc:
                log.warning(f"  {month_str} {symbol}: feature load error — {exc}")

        if not symbol_test_data:
            log.warning(f"  {month_str}: no symbol data for test window — skipping")
            continue

        log.info(f"  {month_str}: loaded {len(symbol_test_data)} symbols for test window")

        # ── Run all variants for this month (batched per symbol) ─────────────
        if any(v.use_loss_veto for v in variants) and loss_veto_model is None:
            log.warning(f"  {month_str}: loss_veto variants will run without veto gate")

        # Initialise per-variant accumulators for this month
        month_trades: Dict[str, list] = {v.name: [] for v in variants}
        month_daily_dd: Dict[str, float] = {v.name: 0.0 for v in variants}
        month_total_dd: Dict[str, float] = {v.name: 0.0 for v in variants}

        for symbol, df_test in symbol_test_data.items():
            mean_vol = symbol_mean_vol[symbol]
            try:
                batch_results = batch_simulate_symbol(
                    symbol=symbol,
                    df=df_test,
                    mean_vol=mean_vol,
                    main_model=main_model,
                    main_features=main_features,
                    loss_veto_model=loss_veto_model,
                    variants=variants,
                    initial_account=initial_account,
                    risk_pct=risk_pct,
                    max_stop_pct=max_stop_pct,
                    min_position_pct=min_position_pct,
                    max_daily_loss_pct=max_daily_loss_pct,
                    profit_threshold_pct=profit_normal,
                )
                for vname, (trades, d_dd, t_dd) in batch_results.items():
                    month_trades[vname].extend(trades)
                    month_daily_dd[vname] = max(month_daily_dd[vname], d_dd)
                    month_total_dd[vname] = max(month_total_dd[vname], t_dd)
            except Exception as exc:
                log.error(f"  {month_str} {symbol}: batch simulation error — {exc}")

        for variant in variants:
            variant_monthly_trades[variant.name][month_str] = month_trades[variant.name]
            variant_monthly_dd[variant.name][month_str] = (
                month_daily_dd[variant.name], month_total_dd[variant.name]
            )

        # Progress summary for this month
        n_trades_per_variant = {
            v.name: len(variant_monthly_trades[v.name].get(month_str, []))
            for v in variants
        }
        log.info(
            f"  {month_str}: trades by variant — "
            + " | ".join(f"{k}={n}" for k, n in n_trades_per_variant.items())
        )

    # ── Assemble results ─────────────────────────────────────────────────────
    log.info(f"\n{'=' * 60}\nAssembling results …")
    results: List[dict] = []
    failed_variants: List[dict] = []

    for variant in variants:
        if variant.name in cached_results and not args.force:
            log.info(f"  {variant.name}: using cached result")
            results.append(cached_results[variant.name])
            continue

        monthly_trades = variant_monthly_trades[variant.name]
        monthly_dd = variant_monthly_dd[variant.name]

        total_trades = sum(len(t) for t in monthly_trades.values())
        if total_trades == 0:
            log.warning(f"  {variant.name}: 0 trades — marking as failed")
            failed_variants.append({
                "variant_name": variant.name,
                "fail_reason": "zero_trades",
                "description": variant.description,
            })
            continue

        result = _compute_variant_result(
            variant=variant,
            monthly_trades=monthly_trades,
            monthly_dd=monthly_dd,
            baseline_monthly_pnl=baseline_monthly_pnl,
            initial_account=initial_account,
        )

        if not result["ftmo_passed"]:
            failed_variants.append({
                "variant_name": variant.name,
                "fail_reason": "ftmo_violation",
                "daily_dd_max": result["daily_drawdown_max_pct"],
                "total_dd_max": result["total_drawdown_max_pct"],
                "description": variant.description,
            })
            # Still include in results so user can see the details
        results.append(result)

        log.info(
            f"  {variant.name}: score={result['robustness_adjusted_score']:.4f}  "
            f"ftmo={'PASS' if result['ftmo_passed'] else 'FAIL'}  "
            f"pnl={result['net_pnl']:,.0f}  trades={result['trade_count']}"
        )

    # Cache full results for subsequent runs
    cached_results.update({r["variant_name"]: r for r in results})
    try:
        # Strip monthly_breakdown from cache to keep file manageable
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "monthly_breakdown"}
                for k, v in cached_results.items()}
        with open(cache_file, "w") as f:
            json.dump(slim, f, default=lambda o: None)
    except Exception as exc:
        log.warning(f"Could not write cache: {exc}")

    # ── Save output files ────────────────────────────────────────────────────
    experiment_config = {
        "mode": args.mode,
        "symbols": symbols,
        "n_symbols": len(symbols),
        "wf_months": wf_months,
        "n_months": len(wf_months),
        "n_variants": len(variants),
        "initial_account": initial_account,
        "ftmo_daily_dd_limit_pct": 4.0,
        "ftmo_total_dd_limit_pct": 9.0,
        "default_no_new_trades_after": "14:30 NY",
        "loss_veto_new_indicators": ["adx_14", "realized_vol_10"],
        "loss_veto_indicator_rationale": {
            "adx_14": "Weak trend strength predicts choppy conditions unfavourable for trend setups",
            "realized_vol_10": "Extreme short-term volatility increases stop-out risk",
        },
        "variants": [
            {
                "name": v.name,
                "description": v.description,
                "main_ai_threshold": v.main_ai_threshold,
                "min_expected_move_pct": v.min_expected_move_pct,
                "use_loss_veto": v.use_loss_veto,
                "loss_veto_threshold": v.loss_veto_threshold if v.use_loss_veto else None,
                "atr_stop_multiplier": v.atr_stop_multiplier,
                "breakeven_trigger_pct": v.breakeven_trigger_pct,
                "trailing_stop_pct": v.trailing_stop_pct,
                "min_reward_risk": v.min_reward_risk,
                "partial_tp_fraction": v.partial_tp_fraction,
                "partial_tp_target_pct": v.partial_tp_target_pct if v.partial_tp_fraction else None,
                "atr_trail_mult": v.atr_trail_mult,
                "time_stop_bars": v.time_stop_bars,
                "time_stop_factor": v.time_stop_factor if v.time_stop_bars else None,
            }
            for v in variants
        ],
    }

    save_experiment_results(
        results=results,
        failed_variants=failed_variants,
        experiment_config=experiment_config,
        output_dir=_EXP_DIR,
    )

    # ── Console ranking ──────────────────────────────────────────────────────
    passed = [r for r in results if r.get("ftmo_passed")]
    ranked = sorted(passed, key=lambda r: r["robustness_adjusted_score"], reverse=True)

    w = 82
    print(f"\n{'=' * w}")
    print(f"  EXPERIMENT RESULTS -- {args.mode.upper()} MODE -- "
          f"{len(wf_months)} months | {len(results)} variants")
    print(f"{'=' * w}")
    print(f"  {'Rk':>3}  {'Variant':<28}  {'Score':>7}  {'NetPnL':>10}  "
          f"{'PF':>6}  {'Exp$':>7}  {'Tr':>5}  {'PM':>4}")
    print(f"  {'-' * (w - 2)}")

    for rank, r in enumerate(ranked[:20], 1):
        pf = r.get("profit_factor")
        pf_s = f"{pf:.3f}" if pf else "  N/A"
        print(
            f"  {rank:>3}  {r['variant_name']:<28}  "
            f"{r['robustness_adjusted_score']:>7.4f}  "
            f"{r.get('net_pnl', 0):>10,.2f}  {pf_s:>6}  "
            f"{r.get('expectancy', 0):>7.2f}  "
            f"{int(r.get('trade_count', 0)):>5}  "
            f"{int(r.get('profitable_months', 0)):>4}"
        )

    if failed_variants:
        print(f"\n  FAILED ({len(failed_variants)}): "
              + ", ".join(f["variant_name"] for f in failed_variants))

    print(f"\n  Full results: {_EXP_DIR}")
    print(f"{'=' * w}\n")


if __name__ == "__main__":
    main()

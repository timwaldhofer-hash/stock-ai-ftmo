"""
Variant-parameterised AI simulation for the experiment framework.

Two-phase approach for speed:
  Phase 1 (once per symbol-month): scan all bars for setups, score each with
    main_ai + loss_veto models, cache as a list of ScoredSignal.
  Phase 2 (once per variant): fast replay over bar arrays using cached scores.
    Model inference is skipped — variant thresholds are applied to cached probs.

Both phases use pre-extracted NumPy arrays for bar data to avoid pandas
iterrows() overhead (5-10× faster than iterrows).

Variant features beyond base walk-forward:
  • configurable main_ai threshold
  • optional loss-veto model gate
  • 14:30 ET no-new-trades cutoff (default)
  • mean-reversion filter
  • minimum expected-move filter
  • configurable ATR stop multiplier
  • minimum reward/risk filter
  • breakeven and trailing-stop exit logic
  • max-holding-bars exit
  • momentum-loss exit
  • partial take-profit with automatic breakeven on remainder
  • ATR-based trailing stop from favorable peak
  • time-based stop tightening after N bars with no progress
  • FTMO daily + total drawdown tracking
"""

import math
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.backtester.costs import (
    calc_fees,
    calc_slippage_cost,
    entry_price_after_slippage,
    exit_price_after_slippage,
)
from src.backtester.risk import (
    calc_shares,
    check_daily_loss_ok,
    check_min_position,
    check_stop_distance,
)
from src.backtester.setup_detection import detect_setup
from src.backtester.trade import SetupSignal
from src.experiments.loss_veto import LOSS_VETO_FEATURES, predict_loss_probability
from src.experiments.variant_config import VariantConfig
from src.models.features import MAIN_AI_FEATURES, engineer_features

_EOD_EXIT_H = 15
_EOD_EXIT_M = 50


# ── Scored signal (output of Phase 1) ─────────────────────────────────────────

@dataclass
class ScoredSignal:
    bar_idx: int           # position in the bars array
    signal: SetupSignal
    main_prob: float
    veto_prob: float
    # Derived stop with default ATR multiplier (1.5x) already in signal.stop_price


@dataclass
class VariantTrade:
    symbol: str
    setup_type: str
    direction: int
    entry_time: object     # pd.Timestamp
    raw_entry_price: float
    entry_price: float
    stop_price: float
    tp_price: float
    shares: int
    account_at_entry: float
    exit_time: object = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_gross: Optional[float] = None
    fees: Optional[float] = None
    slippage_cost: Optional[float] = None
    pnl_net: Optional[float] = None
    main_ai_prob: float = 0.0
    loss_veto_prob: float = 0.0
    # Partial TP fields (zero when unused)
    partial_tp_shares: int = 0
    partial_tp_exit_price: float = 0.0
    partial_tp_pnl_gross: float = 0.0
    partial_tp_pnl_net: float = 0.0


# ── Phase 1: precompute signals for one symbol ─────────────────────────────────

def _select_features(snap_df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=snap_df.index)
    for col in feature_list:
        result[col] = snap_df[col] if col in snap_df.columns else float("nan")
    return result


def _build_snapshot(
    bar_dict: dict,
    signal: SetupSignal,
    symbol: str,
    profit_threshold_pct: float,
) -> pd.DataFrame:
    snap = dict(bar_dict)
    snap["symbol"] = symbol
    snap["setup_type"] = signal.setup_type
    snap["side"] = signal.direction
    snap["entry_price"] = signal.signal_close
    snap["stop_price"] = signal.stop_price
    snap["target_price"] = signal.tp_price
    snap["expected_move_required_pct"] = profit_threshold_pct
    return pd.DataFrame([snap])


def precompute_signals(
    symbol: str,
    df: pd.DataFrame,
    mean_vol: float,
    main_model: CalibratedClassifierCV,
    main_features: List[str],
    loss_veto_model: Optional[CalibratedClassifierCV],
    profit_threshold_pct: float,
    no_new_trades_after: time = time(14, 30),
) -> List[ScoredSignal]:
    """
    Phase 1: scan all bars for setups, score each with AI models.
    Returns a list of ScoredSignal sorted by bar_idx.
    Uses a minimum main_ai score of 0.70 to avoid scoring obviously weak signals
    (all experiment variants use >= 0.80, so 0.70 is a safe lower bound).
    """
    _MIN_PRESCORING_THRESHOLD = 0.70
    scored: List[ScoredSignal] = []
    rows = list(df.itertuples(index=False))

    for i, row in enumerate(rows):
        bar_ts = row.timestamp
        if bar_ts.time() >= no_new_trades_after:
            continue

        # Convert to dict for detect_setup compatibility
        bar_dict = row._asdict()
        signal = detect_setup(bar_dict)
        if signal is None:
            continue

        # Score with main_ai
        try:
            snap_df = _build_snapshot(bar_dict, signal, symbol, profit_threshold_pct)
            snap_feat = engineer_features(snap_df.copy())
            snap_feat["volume_rel"] = min(float(row.volume) / max(mean_vol, 1.0), 50.0)

            main_prob = float(
                main_model.predict_proba(_select_features(snap_feat, main_features))[:, 1][0]
            )

            if main_prob < _MIN_PRESCORING_THRESHOLD:
                continue

            veto_prob = 0.0
            if loss_veto_model is not None:
                try:
                    veto_prob = predict_loss_probability(loss_veto_model, snap_feat)
                except Exception:
                    pass

        except Exception:
            continue

        scored.append(ScoredSignal(
            bar_idx=i,
            signal=signal,
            main_prob=main_prob,
            veto_prob=veto_prob,
        ))

    return scored


# ── Phase 2: fast replay for a specific variant ───────────────────────────────

def _recompute_stop(
    close: float, direction: int, atr: float, support: float, resistance: float,
    atr_mult: float,
) -> float:
    if direction == 1:
        return round(max(close - atr_mult * atr, support), 4)
    return round(min(close + atr_mult * atr, resistance), 4)


def replay_variant(
    symbol: str,
    df: pd.DataFrame,
    scored_signals: List[ScoredSignal],
    variant: VariantConfig,
    initial_account: float,
    risk_pct: float,
    max_stop_pct: float,
    min_position_pct: float,
    max_daily_loss_pct: float,
) -> Tuple[List[VariantTrade], float, float]:
    """
    Phase 2: replay pre-scored signals for a specific variant.
    Returns (trades, max_daily_dd_pct, max_total_dd_pct).
    """
    if not scored_signals:
        return [], 0.0, 0.0

    # Pre-extract NumPy arrays for fast access
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    timestamps = df["timestamp"].values   # numpy datetime64 / object array
    if hasattr(df["timestamp"].iloc[0], "date"):
        ts_list = df["timestamp"].tolist()   # list of pd.Timestamps for .time()/.date()
    else:
        ts_list = pd.to_datetime(df["timestamp"]).tolist()

    atrs = df["atr_14"].to_numpy(dtype=np.float64)
    supports = df["sr_support"].to_numpy(dtype=np.float64)
    resistances = df["sr_resistance"].to_numpy(dtype=np.float64)
    moms = df["mom_10"].to_numpy(dtype=np.float64)

    n = len(opens)

    # Build a fast lookup: bar_idx → ScoredSignal
    signal_map: Dict[int, ScoredSignal] = {}
    for ss in scored_signals:
        # Skip mean-reversion if variant disables it
        if not variant.allow_mean_reversion and "MEAN_REVERSION" in ss.signal.setup_type:
            continue

        # Apply variant ATR stop override BEFORE storing
        if variant.atr_stop_multiplier != 1.5:
            i = ss.bar_idx
            new_stop = _recompute_stop(
                closes[i], ss.signal.direction,
                atrs[i], supports[i], resistances[i],
                variant.atr_stop_multiplier,
            )
            sig = SetupSignal(
                setup_type=ss.signal.setup_type,
                direction=ss.signal.direction,
                signal_time=ss.signal.signal_time,
                signal_close=ss.signal.signal_close,
                stop_price=new_stop,
                tp_price=ss.signal.tp_price,
            )
            ss = ScoredSignal(ss.bar_idx, sig, ss.main_prob, ss.veto_prob)

        # Expected move filter
        entry_price = ss.signal.signal_close
        tp_dist_pct = abs(ss.signal.tp_price - entry_price) / entry_price * 100
        if tp_dist_pct < variant.min_expected_move_pct:
            continue

        # Reward/risk filter
        if variant.min_reward_risk is not None:
            stop_dist = abs(ss.signal.stop_price - entry_price)
            if stop_dist > 0 and tp_dist_pct / (stop_dist / entry_price * 100) < variant.min_reward_risk:
                continue

        # Threshold filters
        if ss.main_prob < variant.main_ai_threshold:
            continue
        if variant.use_loss_veto and ss.veto_prob > variant.loss_veto_threshold:
            continue

        signal_map[ss.bar_idx] = ss

    if not signal_map:
        return [], 0.0, 0.0

    sorted_signal_idxs = sorted(signal_map.keys())
    sig_ptr = 0   # pointer into sorted_signal_idxs

    # Simulation state
    account = initial_account
    peak_account = initial_account
    taken: List[VariantTrade] = []
    max_daily_dd_pct = 0.0
    max_total_dd_pct = 0.0

    open_trade: Optional[VariantTrade] = None
    pending_idx: int = -1    # bar_idx where pending signal fires (entry on next bar)
    pending_ss: Optional[ScoredSignal] = None

    current_date = None
    daily_pnl = 0.0
    day_start_account = account
    day_loss_hit = False

    # Trade exit state (mutable, reset on each new trade)
    current_stop = 0.0
    breakeven_hit = False
    peak_favorable_pct = 0.0
    bars_held = 0
    partial_tp_done = False
    time_stop_triggered = False

    for i in range(n):
        bar_ts = ts_list[i]
        bar_date = bar_ts.date()
        bar_time = bar_ts.time()

        # ── Day boundary ──────────────────────────────────────────────────
        if bar_date != current_date:
            if current_date is not None and daily_pnl < 0 and day_start_account > 0:
                dd = abs(daily_pnl) / day_start_account * 100
                if dd > max_daily_dd_pct:
                    max_daily_dd_pct = dd
            current_date = bar_date
            daily_pnl = 0.0
            day_start_account = account
            day_loss_hit = False
            if pending_ss is not None and pending_ss.signal.signal_time.date() != bar_date:
                pending_ss = None
                pending_idx = -1

        # ── Enter pending signal ──────────────────────────────────────────
        if pending_ss is not None and open_trade is None and not day_loss_hit:
            raw_entry = opens[i]
            direction = pending_ss.signal.direction
            actual_entry = entry_price_after_slippage(float(raw_entry), direction)
            stop = pending_ss.signal.stop_price
            tp = pending_ss.signal.tp_price

            reject = None
            if not check_stop_distance(actual_entry, stop, max_stop_pct):
                reject = "stop_distance"
            else:
                shares = calc_shares(account, risk_pct, actual_entry, stop)
                if not check_min_position(shares, actual_entry, account, min_position_pct):
                    reject = "min_position"

            if reject is None:
                open_trade = VariantTrade(
                    symbol=symbol,
                    setup_type=pending_ss.signal.setup_type,
                    direction=direction,
                    entry_time=bar_ts,
                    raw_entry_price=float(raw_entry),
                    entry_price=round(actual_entry, 4),
                    stop_price=stop,
                    tp_price=tp,
                    shares=shares,
                    account_at_entry=round(account, 2),
                    main_ai_prob=pending_ss.main_prob,
                    loss_veto_prob=pending_ss.veto_prob,
                )
                current_stop = stop
                breakeven_hit = False
                peak_favorable_pct = 0.0
                bars_held = 0
                partial_tp_done = False
                time_stop_triggered = False
            pending_ss = None
            pending_idx = -1

        # ── Exit check ────────────────────────────────────────────────────
        if open_trade is not None:
            direction = open_trade.direction
            entry = open_trade.entry_price
            tp = open_trade.tp_price
            high = float(highs[i])
            low = float(lows[i])
            open_p = float(opens[i])
            raw_exit: Optional[float] = None
            reason: Optional[str] = None

            # EOD
            if bar_time.hour > _EOD_EXIT_H or (bar_time.hour == _EOD_EXIT_H and bar_time.minute >= _EOD_EXIT_M):
                raw_exit, reason = open_p, "EOD"

            if reason is None:
                bars_held += 1

                # Max holding bars
                if variant.max_holding_bars is not None and bars_held >= variant.max_holding_bars:
                    raw_exit, reason = open_p, "MAX_BARS"

            if reason is None:
                # Update peak favorable excursion
                fav = (high - entry) / entry * 100 if direction == 1 else (entry - low) / entry * 100
                if fav > peak_favorable_pct:
                    peak_favorable_pct = fav

                # Breakeven
                if (
                    variant.breakeven_trigger_pct is not None
                    and not breakeven_hit
                    and peak_favorable_pct >= variant.breakeven_trigger_pct
                ):
                    breakeven_hit = True
                    if direction == 1:
                        current_stop = max(current_stop, entry)
                    else:
                        current_stop = min(current_stop, entry)

                # Partial TP: exit a fraction of the position at an interim target,
                # then auto-move stop to breakeven on the remaining shares.
                if (
                    variant.partial_tp_fraction is not None
                    and not partial_tp_done
                ):
                    raw_partial = entry + direction * abs(tp - entry) * variant.partial_tp_target_pct
                    triggered = (
                        (direction == 1 and high >= raw_partial)
                        or (direction == -1 and low <= raw_partial)
                    )
                    if triggered:
                        partial_shares = int(open_trade.shares * variant.partial_tp_fraction)
                        if partial_shares > 0:
                            actual_partial = exit_price_after_slippage(float(raw_partial), direction)
                            p_gross = partial_shares * direction * (actual_partial - entry)
                            p_fees = calc_fees(partial_shares, entry, actual_partial)
                            p_slip = calc_slippage_cost(partial_shares, open_trade.raw_entry_price, float(raw_partial))
                            p_net = p_gross - p_fees

                            open_trade.partial_tp_shares = partial_shares
                            open_trade.partial_tp_exit_price = round(actual_partial, 4)
                            open_trade.partial_tp_pnl_gross = round(p_gross, 4)
                            open_trade.partial_tp_pnl_net = round(p_net, 4)
                            open_trade.shares -= partial_shares

                            # Auto-move stop to breakeven after partial TP
                            if not breakeven_hit:
                                breakeven_hit = True
                                if direction == 1:
                                    current_stop = max(current_stop, entry)
                                else:
                                    current_stop = min(current_stop, entry)

                        partial_tp_done = True

                # Trailing stop (percentage-based)
                if variant.trailing_stop_pct is not None and peak_favorable_pct > 0:
                    if direction == 1:
                        trail = entry * (1.0 + peak_favorable_pct / 100.0) * (1.0 - variant.trailing_stop_pct / 100.0)
                        if trail > current_stop:
                            current_stop = trail
                    else:
                        trail = entry * (1.0 - peak_favorable_pct / 100.0) * (1.0 + variant.trailing_stop_pct / 100.0)
                        if trail < current_stop:
                            current_stop = trail

                # ATR-based trailing stop: trail N×ATR from the favorable peak price.
                if variant.atr_trail_mult is not None and peak_favorable_pct > 0:
                    atr_val = float(atrs[i])
                    if not math.isnan(atr_val) and atr_val > 0:
                        if direction == 1:
                            peak_price = entry * (1.0 + peak_favorable_pct / 100.0)
                            atr_trail = peak_price - variant.atr_trail_mult * atr_val
                            if atr_trail > current_stop:
                                current_stop = atr_trail
                        else:
                            peak_price = entry * (1.0 - peak_favorable_pct / 100.0)
                            atr_trail = peak_price + variant.atr_trail_mult * atr_val
                            if atr_trail < current_stop:
                                current_stop = atr_trail

                # Time-based stop tightening: after N bars held, compress the
                # remaining stop gap if the stop hasn't already been improved.
                if (
                    variant.time_stop_bars is not None
                    and bars_held >= variant.time_stop_bars
                    and not time_stop_triggered
                    and not breakeven_hit
                ):
                    time_stop_triggered = True
                    stop_gap = abs(entry - current_stop)
                    if stop_gap > 0:
                        new_gap = stop_gap * variant.time_stop_factor
                        if direction == 1:
                            tightened = entry - new_gap
                            if tightened > current_stop:
                                current_stop = tightened
                        else:
                            tightened = entry + new_gap
                            if tightened < current_stop:
                                current_stop = tightened

                # Stop hit
                if direction == 1 and low <= current_stop:
                    raw_exit, reason = current_stop, "STOP"
                elif direction == -1 and high >= current_stop:
                    raw_exit, reason = current_stop, "STOP"

                # TP hit
                if reason is None:
                    if direction == 1 and high >= tp:
                        raw_exit, reason = tp, "TP"
                    elif direction == -1 and low <= tp:
                        raw_exit, reason = tp, "TP"

                # Momentum-loss exit
                if reason is None and variant.momentum_loss_exit:
                    mom = float(moms[i])
                    if not math.isnan(mom):
                        if direction == 1 and mom < 0:
                            raw_exit, reason = open_p, "MOM_EXIT"
                        elif direction == -1 and mom > 0:
                            raw_exit, reason = open_p, "MOM_EXIT"

            if raw_exit is not None:
                actual_exit = exit_price_after_slippage(float(raw_exit), open_trade.direction)
                # open_trade.shares may already be reduced by partial TP
                rem_gross = open_trade.shares * open_trade.direction * (actual_exit - entry)
                rem_fees = calc_fees(open_trade.shares, entry, actual_exit)
                rem_slip = calc_slippage_cost(open_trade.shares, open_trade.raw_entry_price, float(raw_exit))
                rem_net = rem_gross - rem_fees

                total_gross = rem_gross + open_trade.partial_tp_pnl_gross
                total_fees = rem_fees + (open_trade.partial_tp_pnl_gross - open_trade.partial_tp_pnl_net)
                total_net = rem_net + open_trade.partial_tp_pnl_net

                open_trade.exit_time = bar_ts
                open_trade.exit_price = round(actual_exit, 4)
                open_trade.exit_reason = reason
                open_trade.pnl_gross = round(total_gross, 4)
                open_trade.fees = round(total_fees, 4)
                open_trade.slippage_cost = round(rem_slip, 4)
                open_trade.pnl_net = round(total_net, 4)

                account += net
                daily_pnl += net
                taken.append(open_trade)
                open_trade = None

                if account > peak_account:
                    peak_account = account
                total_dd = (peak_account - account) / peak_account * 100 if peak_account > 0 else 0.0
                if total_dd > max_total_dd_pct:
                    max_total_dd_pct = total_dd

                if not check_daily_loss_ok(daily_pnl, day_start_account, max_daily_loss_pct):
                    day_loss_hit = True

        # ── Setup scan ────────────────────────────────────────────────────
        if (
            open_trade is None
            and pending_ss is None
            and not day_loss_hit
            and bar_time < variant.no_new_trades_after
        ):
            # Advance signal pointer past bar_idx
            while sig_ptr < len(sorted_signal_idxs) and sorted_signal_idxs[sig_ptr] < i:
                sig_ptr += 1

            if sig_ptr < len(sorted_signal_idxs) and sorted_signal_idxs[sig_ptr] == i:
                pending_ss = signal_map[i]

    # Flush final daily DD
    if daily_pnl < 0 and day_start_account > 0:
        dd = abs(daily_pnl) / day_start_account * 100
        if dd > max_daily_dd_pct:
            max_daily_dd_pct = dd

    return taken, max_daily_dd_pct, max_total_dd_pct


# ── Convenience wrapper ───────────────────────────────────────────────────────

def simulate_symbol_variant(
    symbol: str,
    df: pd.DataFrame,
    mean_vol: float,
    main_model: CalibratedClassifierCV,
    main_features: List[str],
    loss_veto_model: Optional[CalibratedClassifierCV],
    variant: VariantConfig,
    initial_account: float,
    risk_pct: float,
    max_stop_pct: float,
    min_position_pct: float,
    max_daily_loss_pct: float,
) -> Tuple[List[VariantTrade], float, float]:
    """Single-variant wrapper (for API compatibility; prefer batch_simulate_symbol)."""
    scored = precompute_signals(
        symbol, df, mean_vol, main_model, main_features, loss_veto_model,
        profit_threshold_pct=variant.min_expected_move_pct,
        no_new_trades_after=variant.no_new_trades_after,
    )
    return replay_variant(
        symbol, df, scored, variant,
        initial_account, risk_pct, max_stop_pct, min_position_pct, max_daily_loss_pct,
    )


def batch_simulate_symbol(
    symbol: str,
    df: pd.DataFrame,
    mean_vol: float,
    main_model: CalibratedClassifierCV,
    main_features: List[str],
    loss_veto_model: Optional[CalibratedClassifierCV],
    variants: List[VariantConfig],
    initial_account: float,
    risk_pct: float,
    max_stop_pct: float,
    min_position_pct: float,
    max_daily_loss_pct: float,
    profit_threshold_pct: float = 0.20,
) -> Dict[str, Tuple[List[VariantTrade], float, float]]:
    """
    Run all variants for one symbol in a single pass.
    Phase 1 (signal scoring) is shared across all variants.
    Phase 2 (replay) is run once per variant.
    """
    # Use the least-restrictive time cutoff for precomputation
    # (14:30 is the default for all quick-mode variants)
    earliest_cutoff = min(v.no_new_trades_after for v in variants)

    scored = precompute_signals(
        symbol, df, mean_vol, main_model, main_features, loss_veto_model,
        profit_threshold_pct=profit_threshold_pct,
        no_new_trades_after=earliest_cutoff,
    )

    results: Dict[str, Tuple[List[VariantTrade], float, float]] = {}
    for variant in variants:
        results[variant.name] = replay_variant(
            symbol, df, scored, variant,
            initial_account, risk_pct, max_stop_pct, min_position_pct, max_daily_loss_pct,
        )
    return results

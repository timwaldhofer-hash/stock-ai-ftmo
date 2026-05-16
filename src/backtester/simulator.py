"""Per-symbol rule-based simulation loop.

State machine per bar (already-closed 5m candle)
-------------------------------------------------
1. Day boundary  — reset daily P&L, discard stale pending signals.
2. Pending entry — enter at this bar's open (signal fired on previous bar).
3. Exit check    — evaluate EOD / stop / TP against this bar's OHLC.
4. Setup scan    — if no open trade and time < 15:30, detect a new signal
                   (entry deferred to next bar's open).
"""

from datetime import time
from typing import Optional

import pandas as pd

from .costs import (
    calc_fees,
    calc_slippage_cost,
    entry_price_after_slippage,
    exit_price_after_slippage,
)
from .exits import check_exits
from .risk import (
    calc_shares,
    check_daily_loss_ok,
    check_min_position,
    check_stop_distance,
)
from .setup_detection import detect_setup
from .trade import RejectedSignal, SetupSignal, Trade

_NO_NEW_TRADES_AT = time(15, 30)
TZ = "America/New_York"


def simulate_symbol(
    symbol: str,
    df: pd.DataFrame,
    initial_account: float,
    risk_pct: float,
    max_stop_pct: float,
    min_position_pct: float,
    max_daily_loss_pct: float,
) -> tuple[list[Trade], list[RejectedSignal]]:
    """
    Run the rule-based backtest for one symbol.
    df must be sorted by timestamp (NY-tz aware) and contain feature columns.
    Returns (taken_trades, rejected_signals).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    account = initial_account
    taken: list[Trade] = []
    rejected: list[RejectedSignal] = []

    open_trade: Optional[Trade] = None
    pending: Optional[SetupSignal] = None

    current_date = None
    daily_pnl = 0.0
    day_start_account = account
    day_loss_hit = False

    for _, bar in df.iterrows():
        bar_ts = bar["timestamp"]
        bar_date = bar_ts.date()

        # ── 1. Day boundary reset ──────────────────────────────────────
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl = 0.0
            day_start_account = account
            day_loss_hit = False
            # Discard signals that didn't get filled before day end
            if pending is not None and pending.signal_time.date() != bar_date:
                pending = None

        # ── 2. Pending entry: enter at this bar's open ─────────────────
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

        # ── 3. Exit check ──────────────────────────────────────────────
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
                net = gross - fees  # slippage already embedded in actual prices

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

        # ── 4. Setup scan ──────────────────────────────────────────────
        if (
            open_trade is None
            and pending is None
            and not day_loss_hit
            and bar_ts.time() < _NO_NEW_TRADES_AT
        ):
            signal = detect_setup(bar)
            if signal is not None:
                pending = signal

    return taken, rejected

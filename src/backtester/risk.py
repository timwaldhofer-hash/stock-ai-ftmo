"""Risk checks and position sizing."""

import math


def check_stop_distance(entry: float, stop: float, max_pct: float) -> bool:
    """True when |entry - stop| / entry * 100 <= max_pct."""
    return abs(entry - stop) / entry * 100 <= max_pct


def calc_shares(account: float, risk_pct: float, entry: float, stop: float) -> int:
    """Risk-based sizing: floor(account * risk_pct% / stop_distance)."""
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return 0
    return math.floor(account * risk_pct / 100 / stop_distance)


def check_min_position(shares: int, entry: float, account: float, min_pct: float) -> bool:
    """True when position value >= min_pct % of account."""
    return shares > 0 and (shares * entry) >= (account * min_pct / 100)


def check_daily_loss_ok(daily_pnl: float, day_start_account: float, max_loss_pct: float) -> bool:
    """True when we have NOT yet breached the max daily loss limit."""
    return daily_pnl > -(day_start_account * max_loss_pct / 100)

"""Exit logic for open trades.

Priority per bar
----------------
1. EOD force-close  — bar time >= 15:50 NY
2. Stop hit         — bar low/high crosses stop price
3. TP hit           — bar high/low reaches TP price
"""

from datetime import time
from typing import Optional

import pandas as pd

from .trade import Trade

_CLOSE_ALL_AT = time(15, 50)


def check_exits(trade: Trade, bar: pd.Series) -> Optional[tuple[float, str]]:
    """
    Return (raw_exit_price, reason) when an exit triggers, else None.
    raw_exit_price is before slippage; the caller applies slippage.
    """
    bar_time = bar["timestamp"].time()

    # 1. EOD force-close
    if bar_time >= _CLOSE_ALL_AT:
        return float(bar["open"]), "EOD"

    # 2. Stop hit — fill at stop price (worst-case within bar)
    if trade.direction == 1 and bar["low"] <= trade.stop_price:
        return trade.stop_price, "STOP"
    if trade.direction == -1 and bar["high"] >= trade.stop_price:
        return trade.stop_price, "STOP"

    # 3. TP hit — fill at TP price
    if trade.direction == 1 and bar["high"] >= trade.tp_price:
        return trade.tp_price, "TP"
    if trade.direction == -1 and bar["low"] <= trade.tp_price:
        return trade.tp_price, "TP"

    return None

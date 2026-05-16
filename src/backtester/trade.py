from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SetupSignal:
    setup_type: str        # TREND_LONG | TREND_SHORT | MEAN_REVERSION_LONG | MEAN_REVERSION_SHORT
    direction: int         # +1 long, -1 short
    signal_time: pd.Timestamp
    signal_close: float
    stop_price: float
    tp_price: float        # placeholder from S/R or VWAP


@dataclass
class Trade:
    symbol: str
    setup_type: str
    direction: int
    entry_time: pd.Timestamp
    raw_entry_price: float     # bar open before slippage
    entry_price: float         # after slippage
    stop_price: float
    tp_price: float
    shares: int
    account_at_entry: float
    exit_time: Optional[pd.Timestamp] = None
    raw_exit_price: Optional[float] = None   # stop/TP/EOD open before slippage
    exit_price: Optional[float] = None       # after slippage
    exit_reason: Optional[str] = None        # STOP | TP | EOD | MAX_DAILY_LOSS
    pnl_gross: Optional[float] = None
    fees: Optional[float] = None
    slippage_cost: Optional[float] = None
    pnl_net: Optional[float] = None


@dataclass
class RejectedSignal:
    symbol: str
    setup_type: str
    signal_time: pd.Timestamp
    signal_close: float
    stop_price: float
    tp_price: float
    reject_reason: str

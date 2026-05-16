"""Rule-based setup detection on closed 5m bars.

Setups and conditions
---------------------
TREND_LONG
  close > vwap_1session, rsi_14 > 55, mom_10 > 0
  Stop : close - 1.5 * atr_14  (floored at sr_support)
  TP   : sr_resistance  (placeholder)

TREND_SHORT
  close < vwap_1session, rsi_14 < 45, mom_10 < 0
  Stop : close + 1.5 * atr_14  (capped at sr_resistance)
  TP   : sr_support  (placeholder)

MEAN_REVERSION_LONG
  rsi_14 < 35, close < vwap_1session, sr_dist_to_support <= 0.5 %
  Stop : sr_support - 1.0 * atr_14
  TP   : vwap_1session  (placeholder)

MEAN_REVERSION_SHORT
  rsi_14 > 65, close > vwap_1session, sr_dist_to_resistance <= 0.5 %
  Stop : sr_resistance + 1.0 * atr_14
  TP   : vwap_1session  (placeholder)
"""

from typing import Optional

import numpy as np
import pandas as pd

from .trade import SetupSignal

_REQUIRED = [
    "rsi_14", "mom_10", "vwap_1session", "atr_14",
    "sr_resistance", "sr_support",
    "sr_dist_to_resistance", "sr_dist_to_support",
]

_TREND_RSI_LONG = 55.0
_TREND_RSI_SHORT = 45.0
_MR_RSI_OVERSOLD = 35.0
_MR_RSI_OVERBOUGHT = 65.0
_MR_SR_PROXIMITY = 0.005   # 0.5 % of close
_TREND_ATR_MULT = 1.5
_MR_ATR_MULT = 1.0


def _any_nan(bar: pd.Series) -> bool:
    for col in _REQUIRED:
        v = bar.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return True
    return False


def detect_setup(bar: pd.Series) -> Optional[SetupSignal]:
    """
    Inspect one closed 5m bar; return the first matching SetupSignal or None.
    Priority: TREND_LONG > TREND_SHORT > MR_LONG > MR_SHORT.
    """
    if _any_nan(bar):
        return None

    close = float(bar["close"])
    rsi = float(bar["rsi_14"])
    mom = float(bar["mom_10"])
    vwap = float(bar["vwap_1session"])
    atr = float(bar["atr_14"])
    resistance = float(bar["sr_resistance"])
    support = float(bar["sr_support"])
    dist_res = float(bar["sr_dist_to_resistance"])
    dist_sup = float(bar["sr_dist_to_support"])
    ts = bar["timestamp"]

    # TREND_LONG
    if close > vwap and rsi > _TREND_RSI_LONG and mom > 0:
        stop = max(close - _TREND_ATR_MULT * atr, support)
        tp = resistance
        if stop < close and tp > close + 0.5 * atr:
            return SetupSignal("TREND_LONG", 1, ts, close, round(stop, 4), round(tp, 4))

    # TREND_SHORT
    if close < vwap and rsi < _TREND_RSI_SHORT and mom < 0:
        stop = min(close + _TREND_ATR_MULT * atr, resistance)
        tp = support
        if stop > close and tp < close - 0.5 * atr:
            return SetupSignal("TREND_SHORT", -1, ts, close, round(stop, 4), round(tp, 4))

    # MEAN_REVERSION_LONG
    if rsi < _MR_RSI_OVERSOLD and close < vwap and dist_sup <= _MR_SR_PROXIMITY:
        stop = support - _MR_ATR_MULT * atr
        tp = vwap
        if stop < close and tp > close:
            return SetupSignal("MEAN_REVERSION_LONG", 1, ts, close, round(stop, 4), round(tp, 4))

    # MEAN_REVERSION_SHORT
    if rsi > _MR_RSI_OVERBOUGHT and close > vwap and dist_res <= _MR_SR_PROXIMITY:
        stop = resistance + _MR_ATR_MULT * atr
        tp = vwap
        if stop > close and tp < close:
            return SetupSignal("MEAN_REVERSION_SHORT", -1, ts, close, round(stop, 4), round(tp, 4))

    return None

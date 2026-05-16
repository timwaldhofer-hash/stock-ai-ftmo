import numpy as np
import pandas as pd


def compute_vwap(df: pd.DataFrame, sessions: int = 1) -> pd.Series:
    """
    Volume-Weighted Average Price over `sessions` trading sessions.

    df must have: timestamp (tz-aware NY time), high, low, close, volume
    For sessions=1: standard intraday VWAP, resets each day.
    For sessions=2: VWAP resets at start of previous session, giving
                    continuity across two days.
    Returns a Series aligned to df's index.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tpv = tp * df["volume"]
    dates = df["timestamp"].dt.date
    unique_dates = sorted(dates.unique())

    result = pd.Series(np.nan, index=df.index, dtype=float)

    for i, date in enumerate(unique_dates):
        window_dates = unique_dates[max(0, i - sessions + 1) : i + 1]
        window_mask = dates.isin(window_dates)
        today_mask = dates == date

        cum_tpv = tpv[window_mask].cumsum()
        cum_vol = df["volume"][window_mask].cumsum()
        vwap_series = cum_tpv / cum_vol.replace(0, np.nan)

        result[today_mask] = vwap_series[today_mask].values

    return result

"""
Feature engineering for AI model training.

Computes all derived features from a train_samples snapshot DataFrame.
All features are scale-free ratios or bounded indicators to keep
the models stable across different price levels and symbols.
"""

import numpy as np
import pandas as pd

_TZ_NY = "America/New_York"
_SESSION_OPEN_MINUTES = 9 * 60 + 30   # 09:30
_SESSION_DURATION_MIN = 390            # 09:30 – 16:00

# ── Feature lists ──────────────────────────────────────────────────────────────

# Main AI: comprehensive signal-quality features including setup metadata
MAIN_AI_FEATURES = [
    # Momentum / trend indicators
    "rsi_14",
    "mom_10",
    # VWAP positioning
    "vwap_dist_pct",
    # Volatility
    "atr_pct",
    # Support / resistance proximity
    "sr_dist_to_resistance",
    "sr_dist_to_support",
    # Candle microstructure
    "candle_body_pct",
    "candle_upper_wick_pct",
    "candle_lower_wick_pct",
    "candle_range_atr",
    # Setup geometry
    "stop_dist_pct",
    "target_dist_pct",
    "expected_move_required_pct",
    # Direction
    "side",
    # Time of day (cyclical encoding)
    "time_sin",
    "time_cos",
    "hour_of_day",
    # Volume
    "volume_log",
    "volume_rel",
    # Setup type (one-hot)
    "setup_trend_long",
    "setup_trend_short",
    "setup_mean_reversion_long",
    "setup_mean_reversion_short",
]

# Independent AI: market microstructure subset — no setup-metadata features
INDEPENDENT_AI_FEATURES = [
    # Volume
    "volume_log",
    "volume_rel",
    # Price / candle structure
    "candle_body_pct",
    "candle_upper_wick_pct",
    "candle_lower_wick_pct",
    "candle_range_atr",
    # VWAP distance
    "vwap_dist_pct",
    # Momentum indicators
    "rsi_14",
    "mom_10",
    # Volatility
    "atr_pct",
    # Support / resistance distance
    "sr_dist_to_resistance",
    "sr_dist_to_support",
    # Time of day (cyclical)
    "time_sin",
    "time_cos",
]


# ── Derived-feature computation ────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived model features to a train_samples snapshot DataFrame.

    Only new columns are added; existing columns (rsi_14, mom_10, atr_14, …)
    are left untouched. Returns a copy.
    """
    df = df.copy()

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr = df["atr_14"].astype(float).replace(0.0, np.nan)

    # ── Candle microstructure ──────────────────────────────────────────────
    candle_max = pd.concat([close, open_], axis=1).max(axis=1)
    candle_min = pd.concat([close, open_], axis=1).min(axis=1)

    safe_close = close.replace(0.0, np.nan)
    safe_open = open_.replace(0.0, np.nan)

    df["candle_body_pct"] = (close - open_) / safe_open
    df["candle_upper_wick_pct"] = (high - candle_max) / safe_close
    df["candle_lower_wick_pct"] = (candle_min - low) / safe_close
    df["candle_range_atr"] = (high - low) / atr

    # ── VWAP distance (signed: positive = price above VWAP) ───────────────
    df["vwap_dist_pct"] = (close - df["vwap_1session"].astype(float)) / safe_close

    # ── ATR as fraction of price ───────────────────────────────────────────
    df["atr_pct"] = atr / safe_close

    # ── Setup geometry distances ───────────────────────────────────────────
    entry = df["entry_price"].astype(float).replace(0.0, np.nan)
    df["stop_dist_pct"] = (
        (df["entry_price"].astype(float) - df["stop_price"].astype(float)).abs() / entry
    )
    df["target_dist_pct"] = (
        (df["target_price"].astype(float) - df["entry_price"].astype(float)).abs() / entry
    )

    # ── Volume (log-scale to reduce right skew) ────────────────────────────
    vol = df["volume"].astype(float).clip(lower=1.0)
    df["volume_log"] = np.log1p(vol)

    # ── Time of day (cyclical, referenced to NYSE open 09:30 ET) ──────────
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(_TZ_NY)
    elif str(ts.dt.tz) != _TZ_NY:
        ts = ts.dt.tz_convert(_TZ_NY)

    minutes_since_open = (ts.dt.hour * 60 + ts.dt.minute) - _SESSION_OPEN_MINUTES
    minutes_clipped = minutes_since_open.clip(0, _SESSION_DURATION_MIN)
    phase = 2.0 * np.pi * minutes_clipped / _SESSION_DURATION_MIN

    df["time_sin"] = np.sin(phase)
    df["time_cos"] = np.cos(phase)
    df["hour_of_day"] = ts.dt.hour + ts.dt.minute / 60.0

    # ── Setup type one-hot ─────────────────────────────────────────────────
    st = df["setup_type"]
    df["setup_trend_long"] = (st == "TREND_LONG").astype(int)
    df["setup_trend_short"] = (st == "TREND_SHORT").astype(int)
    df["setup_mean_reversion_long"] = (st == "MEAN_REVERSION_LONG").astype(int)
    df["setup_mean_reversion_short"] = (st == "MEAN_REVERSION_SHORT").astype(int)

    return df


def add_relative_volume(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """
    Add volume_rel column: volume / per-symbol mean volume.

    The mean is computed only on rows selected by train_mask to avoid
    test-set information leaking into the feature.  Symbols absent from
    the training set fall back to relative volume = 1.0.
    """
    mean_vol_by_symbol = (
        df.loc[train_mask]
        .groupby("symbol")["volume"]
        .mean()
    )
    df = df.copy()
    df["volume_rel"] = (
        df["volume"].astype(float) / df["symbol"].map(mean_vol_by_symbol)
    ).fillna(1.0).clip(upper=50.0)
    return df

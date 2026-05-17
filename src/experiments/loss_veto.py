"""
Loss veto model — predicts P(trade is a loss) to block dangerous entries.

Two additional indicators chosen for the veto model:
  1. ADX (14): measures trend strength.  Weak trend (low ADX) is dangerous for
     trend-following setups; the model can learn to veto low-ADX entries.
  2. Realized volatility (10 bars): captures short-term vol regime.  Extreme
     volatility often leads to stop-outs even on directionally correct trades.

Training target is INVERTED relative to main_ai:  1 = loss, 0 = win.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None


# ── Feature list ──────────────────────────────────────────────────────────────

LOSS_VETO_FEATURES = [
    # Momentum / trend
    "rsi_14",
    "mom_10",
    # VWAP positioning
    "vwap_dist_pct",
    # Volatility
    "atr_pct",
    # S/R proximity
    "sr_dist_to_resistance",
    "sr_dist_to_support",
    # Candle structure
    "candle_body_pct",
    "candle_upper_wick_pct",
    "candle_lower_wick_pct",
    "candle_range_atr",
    # Setup geometry
    "stop_dist_pct",
    "target_dist_pct",
    # Direction + time
    "side",
    "time_sin",
    "time_cos",
    # Volume
    "volume_log",
    "volume_rel",
    # ── New indicators ────────────────────────────────────────────────────────
    "adx_14",           # trend strength — weak ADX = dangerous for trend trades
    "realized_vol_10",  # short-term realized vol regime
]


# ── New indicator computation ─────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX(14) using Wilder's exponential smoothing."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up = high - high.shift(1)
    dn = low.shift(1) - low
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)

    alpha = 1.0 / period
    atr_s = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean().clip(lower=1e-10)
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_s

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx.rename("adx_14")


def compute_realized_vol(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Rolling std of log-returns over `period` bars."""
    log_ret = np.log(
        df["close"].astype(float)
        / df["close"].astype(float).shift(1).replace(0.0, np.nan)
    )
    return log_ret.rolling(period, min_periods=max(1, period // 2)).std().rename("realized_vol_10")


def add_veto_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add adx_14 and realized_vol_10 to a time-series feature DataFrame.
    Must be called on the FULL series (not single-row snapshots) so that
    rolling/EWM windows are correctly initialised.
    Returns a copy.
    """
    df = df.copy()
    df["adx_14"] = compute_adx(df)
    df["realized_vol_10"] = compute_realized_vol(df)
    return df


# ── Model helpers ─────────────────────────────────────────────────────────────

def _select_veto_features(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in LOSS_VETO_FEATURES:
        result[col] = df[col] if col in df.columns else float("nan")
    return result


def train_loss_veto_model(
    df_feat: pd.DataFrame,
    fit_mask: pd.Series,
    cal_mask: pd.Series,
) -> CalibratedClassifierCV:
    """
    Train loss-veto classifier on df_feat (must already include engineered
    features + adx_14 + realized_vol_10).
    Target is INVERTED: 1 = loss, 0 = win.
    """
    y_loss = (1 - df_feat["label"].astype(int))
    X = _select_veto_features(df_feat)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])
    pipeline.fit(X[fit_mask], y_loss[fit_mask])

    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    else:
        calibrated = CalibratedClassifierCV(pipeline, cv="prefit", method="sigmoid")
    calibrated.fit(X[cal_mask], y_loss[cal_mask])
    return calibrated


def predict_loss_probability(
    model: CalibratedClassifierCV,
    snap_df: pd.DataFrame,
) -> float:
    """Return P(loss) for a single-row snapshot DataFrame."""
    X = _select_veto_features(snap_df)
    return float(model.predict_proba(X)[:, 1][0])

import pandas as pd


def compute_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Price momentum: close minus close `period` bars ago."""
    return close - close.shift(period)


def compute_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change as percentage over `period` bars."""
    return (close / close.shift(period) - 1) * 100

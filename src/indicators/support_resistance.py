import pandas as pd


def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Simple rolling support and resistance levels.

    Resistance: rolling max of high over `window` bars.
    Support:    rolling min of low  over `window` bars.
    Distance columns express gap as a fraction of close price.
    """
    resistance = df["high"].rolling(window, min_periods=1).max()
    support = df["low"].rolling(window, min_periods=1).min()

    return pd.DataFrame(
        {
            "sr_resistance": resistance,
            "sr_support": support,
            "sr_dist_to_resistance": (resistance - df["close"]) / df["close"],
            "sr_dist_to_support": (df["close"] - support) / df["close"],
        },
        index=df.index,
    )

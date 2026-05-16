#!/usr/bin/env python3
"""
Berechnet technische Indikatoren für alle 5m-Parquet-Dateien
und speichert Feature-DataFrames unter data/features/5m/.

Indikatoren:
  vwap_1session  — intraday VWAP (täglicher Reset)
  vwap_2session  — 2-Sitzungen VWAP (Reset am Beginn der Vortagessitzung)
  rsi_14         — RSI(14) nach Wilder
  mom_10         — Momentum (close - close[10])
  roc_10         — Rate of Change in % über 10 Bars
  atr_14         — ATR(14) nach Wilder
  sr_resistance  — Rolling-Hoch der letzten 20 Bars
  sr_support     — Rolling-Tief der letzten 20 Bars
  sr_dist_to_resistance / sr_dist_to_support — Abstand zu Levels (als Anteil von close)

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --symbols AAPL MSFT
    python scripts/build_features.py --force
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.atr import compute_atr
from src.indicators.momentum import compute_momentum, compute_roc
from src.indicators.rsi import compute_rsi
from src.indicators.support_resistance import compute_support_resistance
from src.indicators.vwap import compute_vwap

TZ = "America/New_York"


def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("build_features")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet alle Indikatoren für einen Symbol-DataFrame (5m-Bars).
    Erwartet Spalten: timestamp, open, high, low, close, volume.
    timestamp wird bei Bedarf nach NY-Zeit konvertiert.
    """
    df = df.copy()

    # timestamp in NY-Zeit sicherstellen
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    else:
        df["timestamp"] = ts.dt.tz_convert(TZ)

    # VWAP
    df["vwap_1session"] = compute_vwap(df, sessions=1)
    df["vwap_2session"] = compute_vwap(df, sessions=2)

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], period=14)

    # Momentum
    df["mom_10"] = compute_momentum(df["close"], period=10)
    df["roc_10"] = compute_roc(df["close"], period=10)

    # ATR
    df["atr_14"] = compute_atr(df, period=14)

    # Support / Resistance
    sr = compute_support_resistance(df, window=20)
    df = pd.concat([df, sr], axis=1)

    return df


def process_symbol(
    symbol: str,
    processed_dir: Path,
    features_dir: Path,
    force: bool,
    log: logging.Logger,
) -> bool:
    in_path = processed_dir / f"{symbol}.parquet"
    out_path = features_dir / f"{symbol}.parquet"

    if not in_path.exists():
        log.warning(f"{symbol}: 5m-Datei nicht gefunden -- {in_path}")
        return False

    if out_path.exists() and not force:
        log.info(f"{symbol}: Feature-Datei bereits vorhanden (--force zum Überschreiben)")
        return True

    try:
        df = pd.read_parquet(in_path)
        df_feat = build_features(df)

        features_dir.mkdir(parents=True, exist_ok=True)
        df_feat.to_parquet(out_path, index=False)
        log.info(
            f"{symbol}: {len(df):,} Bars → {len(df_feat.columns)} Spalten gespeichert"
        )
        return True

    except Exception as e:
        log.error(f"{symbol}: Fehler -- {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Technische Indikatoren auf 5m-Bars berechnen"
    )
    parser.add_argument("--symbols", nargs="+", help="Nur diese Symbole verarbeiten")
    parser.add_argument(
        "--force", action="store_true", help="Vorhandene Feature-Dateien überschreiben"
    )
    args = parser.parse_args()

    cfg = load_config()
    log = setup_logging()

    processed_dir = ROOT / cfg["data"]["processed_5m_dir"]
    features_dir = ROOT / cfg["data"]["features_5m_dir"]

    if args.symbols:
        symbols = args.symbols
    else:
        parquet_files = sorted(processed_dir.glob("*.parquet"))
        symbols = [p.stem for p in parquet_files]

    if not symbols:
        log.warning(f"Keine 5m-Parquet-Dateien in {processed_dir} gefunden")
        return

    log.info(f"{len(symbols)} Symbole werden verarbeitet")

    ok_count = 0
    err_count = 0
    for symbol in symbols:
        if process_symbol(symbol, processed_dir, features_dir, args.force, log):
            ok_count += 1
        else:
            err_count += 1

    log.info(f"Fertig: {ok_count} OK, {err_count} Fehler")


if __name__ == "__main__":
    main()

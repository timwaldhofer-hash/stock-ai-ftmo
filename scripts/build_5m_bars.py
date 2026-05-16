#!/usr/bin/env python3
"""
Baut 5-Minuten-OHLCV-Kerzen aus 1-Minuten-Parquet-Dateien.

Zeitregel laut Spec:
  09:30–09:34 NY → abgeschlossene 09:35-Kerze
  15:55–15:59 NY → abgeschlossene 16:00-Kerze

Usage:
    python scripts/build_5m_bars.py
    python scripts/build_5m_bars.py --symbols AAPL MSFT
    python scripts/build_5m_bars.py --force   # vorhandene 5m-Dateien überschreiben
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

TZ = "America/New_York"
# 1m-Filter: 09:30 bis 15:59 (15:59-Bar ist letzte reguläre Minute)
SESSION_1M_START = "09:30"
SESSION_1M_END = "15:59"
# 5m-Label-Range nach Aggregation: 09:35 (erste abgeschlossene Kerze) bis 16:00 (letzte)
SESSION_5M_START = "09:35"
SESSION_5M_END = "16:00"


def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("build_5m")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


def build_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Konvertiert 1m-Bars (UTC-Timestamps) zu 5m-Bars in NY-Zeit.
    Gibt leeres DataFrame zurück wenn keine Session-Daten vorhanden.
    """
    df = df_1m.copy()

    # UTC → America/New_York
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
    df.index = ts

    # Nur reguläre Session-Bars (09:30–15:59 NY)
    session = df.between_time(SESSION_1M_START, SESSION_1M_END)
    if session.empty:
        return pd.DataFrame()

    # Spalten sicherstellen
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in session.columns:
            raise ValueError(f"Spalte '{col}' fehlt in 1m-Daten")

    # 5m-Aggregation:
    #   closed='left'  → Bucket [09:30, 09:35) enthält Bars 09:30..09:34
    #   label='right'  → Bucket [09:30, 09:35) bekommt Label 09:35
    ohlcv = session[["open", "high", "low", "close", "volume"]].resample(
        "5min", closed="left", label="right"
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    # Leere Buckets entfernen (entstehen an Session-Grenzen oder bei Datenlücken)
    ohlcv = ohlcv.dropna(subset=["open", "close"])
    ohlcv = ohlcv[ohlcv["volume"] > 0]

    # Nur Bars innerhalb des gültigen 5m-Label-Fensters behalten
    ohlcv = ohlcv.between_time(SESSION_5M_START, SESSION_5M_END)

    if ohlcv.empty:
        return pd.DataFrame()

    ohlcv = ohlcv.reset_index().rename(columns={"index": "timestamp"})
    return ohlcv


def process_symbol(
    symbol: str,
    raw_dir: Path,
    out_dir: Path,
    force: bool,
    log: logging.Logger,
) -> bool:
    in_path = raw_dir / f"{symbol}.parquet"
    out_path = out_dir / f"{symbol}.parquet"

    if not in_path.exists():
        log.warning(f"{symbol}: 1m-Datei nicht gefunden -- {in_path}")
        return False

    if out_path.exists() and not force:
        log.info(f"{symbol}: 5m-Datei bereits vorhanden (--force zum Überschreiben)")
        return True

    try:
        df_1m = pd.read_parquet(in_path)
        df_5m = build_5m(df_1m)

        if df_5m.empty:
            log.warning(f"{symbol}: keine 5m-Bars nach Session-Filter")
            return False

        out_dir.mkdir(parents=True, exist_ok=True)
        df_5m.to_parquet(out_path, index=False)
        log.info(f"{symbol}: {len(df_1m):,} 1m -> {len(df_5m):,} 5m Bars")
        return True

    except Exception as e:
        log.error(f"{symbol}: Fehler -- {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="1m-Parquet zu 5m-Bars aggregieren")
    parser.add_argument("--symbols", nargs="+", help="Nur diese Symbole verarbeiten")
    parser.add_argument("--force", action="store_true", help="Vorhandene 5m-Dateien überschreiben")
    args = parser.parse_args()

    cfg = load_config()
    log = setup_logging()

    raw_dir = ROOT / cfg["data"]["raw_1m_dir"]
    out_dir = ROOT / cfg["data"]["processed_5m_dir"]

    if args.symbols:
        symbols = args.symbols
    else:
        parquet_files = sorted(raw_dir.glob("*.parquet"))
        symbols = [p.stem for p in parquet_files]

    if not symbols:
        log.warning(f"Keine 1m-Parquet-Dateien in {raw_dir} gefunden")
        return

    log.info(f"{len(symbols)} Symbole werden verarbeitet")

    ok_count = 0
    err_count = 0
    for symbol in symbols:
        if process_symbol(symbol, raw_dir, out_dir, args.force, log):
            ok_count += 1
        else:
            err_count += 1

    log.info(f"Fertig: {ok_count} OK, {err_count} Fehler")


if __name__ == "__main__":
    main()

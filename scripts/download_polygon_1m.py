#!/usr/bin/env python3
"""
Download 1-minute OHLCV bars from Polygon.io for all symbols in the universe.

Usage:
    python scripts/download_polygon_1m.py
    python scripts/download_polygon_1m.py --symbols AAPL MSFT   # nur einzelne Symbole
    python scripts/download_polygon_1m.py --start 2024-01-01    # abweichendes Startdatum
"""

import argparse
import calendar
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from polygon import RESTClient
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("download")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / "download_polygon_1m.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


def load_symbols(path: Path) -> list[str]:
    df = pd.read_csv(path)
    raw = df["symbol"].dropna().str.strip().tolist()
    # Entferne offensichtliche Tippfehler (Trailing-s bei Tickern über 4 Zeichen)
    cleaned = []
    for s in raw:
        if len(s) > 5:
            logging.getLogger("download").warning(
                f"Ungewoehnliches Symbol uebersprungen: '{s}' -- bitte symbol_universe.csv pruefen"
            )
        else:
            cleaned.append(s)
    return cleaned


def month_ranges(start: date, end: date):
    """Liefert (monat_start, monat_end)-Tupel für den Zeitraum start..end."""
    current = start.replace(day=1)
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        yield current, min(date(current.year, current.month, last_day), end)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def get_last_date(parquet_path: Path) -> date | None:
    """Gibt das letzte Datum einer existierenden Parquet-Datei zurück."""
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["timestamp"])
        if df.empty:
            return None
        ts = pd.to_datetime(df["timestamp"]).max()
        return ts.date() if hasattr(ts, "date") else None
    except Exception as e:
        logging.getLogger("download").warning(f"Konnte {parquet_path} nicht lesen: {e}")
        return None


def bars_to_dataframe(bars: list) -> pd.DataFrame:
    records = []
    for b in bars:
        records.append({
            "timestamp": b.timestamp,   # Millisekunden seit Epoch UTC
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "vwap": getattr(b, "vwap", None),
            "transactions": getattr(b, "transactions", None),
        })
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def download_symbol(
    client: RESTClient,
    symbol: str,
    start: date,
    end: date,
    out_path: Path,
    log: logging.Logger,
) -> bool:
    last_date = get_last_date(out_path)
    effective_start = start

    if last_date is not None:
        effective_start = last_date + timedelta(days=1)
        if effective_start > end:
            log.info(f"{symbol}: bereits aktuell (letzter Tag: {last_date})")
            return True

    log.info(f"{symbol}: lade {effective_start} -> {end}")
    new_bars: list = []
    failed_months: list = []

    for month_start, month_end in month_ranges(effective_start, end):
        try:
            bars = list(
                client.list_aggs(
                    symbol,
                    multiplier=1,
                    timespan="minute",
                    from_=month_start.isoformat(),
                    to=month_end.isoformat(),
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                )
            )
            if bars:
                new_bars.extend(bars)
                log.debug(f"  {symbol} {month_start}: {len(bars)} Bars")
            else:
                log.warning(f"  {symbol} {month_start}: keine Daten")
        except Exception as e:
            log.error(f"  {symbol} {month_start}: Download fehlgeschlagen -- {e}")
            failed_months.append(month_start)

    if not new_bars:
        if failed_months:
            log.error(f"{symbol}: alle Downloads fehlgeschlagen")
            return False
        log.warning(f"{symbol}: keine neuen Bars (möglicherweise kein Handel in diesem Zeitraum)")
        return True

    df_new = bars_to_dataframe(new_bars)

    # Mit vorhandenen Daten zusammenführen
    if out_path.exists() and last_date is not None:
        try:
            df_existing = pd.read_parquet(out_path)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            log.warning(f"{symbol}: Zusammenführen fehlgeschlagen, überschreibe Datei: {e}")
            df = df_new
    else:
        df = df_new

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"{symbol}: {len(df):,} Bars gespeichert -> {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Polygon 1m-Daten herunterladen")
    parser.add_argument("--symbols", nargs="+", help="Nur diese Symbole laden")
    parser.add_argument("--start", help="Startdatum überschreiben (YYYY-MM-DD)")
    parser.add_argument("--end", help="Enddatum überschreiben (YYYY-MM-DD)")
    args = parser.parse_args()

    cfg = load_config()
    log = setup_logging(ROOT / cfg["logging"]["log_dir"])

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key or api_key == "TODO":
        log.error("POLYGON_API_KEY fehlt in .env -- bitte eintragen")
        sys.exit(1)

    raw_dir = ROOT / cfg["data"]["raw_1m_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    symbols_path = ROOT / cfg["data"]["symbol_universe_file"]
    if not symbols_path.exists():
        log.error(f"Symbol-Universe nicht gefunden: {symbols_path}")
        sys.exit(1)

    all_symbols = load_symbols(symbols_path)
    symbols = args.symbols if args.symbols else all_symbols
    log.info(f"{len(symbols)} Symbole geladen")

    start = date.fromisoformat(args.start or cfg["download"]["start_date"])
    end = date.fromisoformat(args.end or cfg["download"]["end_date"])
    log.info(f"Zeitraum: {start} -> {end}")

    client = RESTClient(api_key)
    errors: list[str] = []

    for symbol in tqdm(symbols, desc="Symbole", unit="sym"):
        out_path = raw_dir / f"{symbol}.parquet"
        try:
            ok = download_symbol(client, symbol, start, end, out_path, log)
            if not ok:
                errors.append(symbol)
        except Exception as e:
            log.error(f"{symbol}: unerwarteter Fehler -- {e}")
            errors.append(symbol)

    log.info(f"Fertig. {len(symbols) - len(errors)}/{len(symbols)} Symbole erfolgreich.")
    if errors:
        log.warning(f"Fehlgeschlagen: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()

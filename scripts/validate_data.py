#!/usr/bin/env python3
"""
Validiert 1m- und 5m-Parquet-Daten.

Prüft:
  - Existenz der Dateien
  - Zeilenanzahl und Datum-Range pro Symbol
  - 5m-Bars nur innerhalb regulärer Session (09:35–16:00 NY)
  - Zeitstempel-Plausibilität
  - Offensichtliche Datenlücken (> 5 Handelstage ohne Daten)
  - Symbole im Universe ohne Daten

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --symbols AAPL MSFT
    python scripts/validate_data.py --verbose
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

TZ = "America/New_York"
SESSION_5M_START = pd.Timestamp("09:35").time()
SESSION_5M_END = pd.Timestamp("16:00").time()
MAX_GAP_DAYS = 5   # mehr als N Handelstage ohne Bars = Warnung


def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def load_universe(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df["symbol"].dropna().str.strip().tolist()


def read_timestamps(path: Path, tz: str = TZ) -> pd.Series:
    """Liest Timestamp-Spalte und konvertiert zu NY-Zeit."""
    df = pd.read_parquet(path, columns=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(tz)


def detect_gaps(ts: pd.Series, max_gap_days: int = MAX_GAP_DAYS) -> list[str]:
    """Gibt Liste von Datumslücken zurück, die > max_gap_days Handelstage überspringen."""
    dates = ts.dt.normalize().drop_duplicates().sort_values()
    if len(dates) < 2:
        return []
    gaps = []
    prev = None
    for d in dates:
        if prev is not None:
            # Handelstage zwischen prev und d (Mo–Fr)
            bdays = pd.bdate_range(prev, d, freq="B")
            gap = len(bdays) - 1  # -1 weil prev selbst dabei ist
            if gap > max_gap_days:
                gaps.append(
                    f"{prev.strftime('%Y-%m-%d')} -> {d.strftime('%Y-%m-%d')} ({gap} Handelstage)"
                )
        prev = d
    return gaps


def validate_1m(path: Path, verbose: bool) -> dict:
    result = {"symbol": path.stem, "ok": True, "issues": [], "warnings": []}
    try:
        df = pd.read_parquet(path)
        result["rows"] = len(df)

        ts = read_timestamps(path)
        result["start"] = ts.min().strftime("%Y-%m-%d")
        result["end"] = ts.max().strftime("%Y-%m-%d")

        if verbose:
            gaps = detect_gaps(ts)
            for g in gaps:
                result["warnings"].append(f"Lücke: {g}")

        # Basale Plausibilität
        if result["rows"] == 0:
            result["issues"].append("Datei ist leer")
            result["ok"] = False

    except Exception as e:
        result["rows"] = 0
        result["start"] = result["end"] = "?"
        result["issues"].append(f"Lesefehler: {e}")
        result["ok"] = False

    return result


def validate_5m(path: Path, verbose: bool) -> dict:
    result = {"symbol": path.stem, "ok": True, "issues": [], "warnings": []}
    try:
        df = pd.read_parquet(path)
        result["rows"] = len(df)

        ts = read_timestamps(path)
        result["start"] = ts.min().strftime("%Y-%m-%d")
        result["end"] = ts.max().strftime("%Y-%m-%d")

        if result["rows"] == 0:
            result["issues"].append("Datei ist leer")
            result["ok"] = False
            return result

        # Alle 5m-Bars müssen innerhalb 09:35–16:00 NY liegen
        times = ts.dt.time
        out_of_session = ((times < SESSION_5M_START) | (times > SESSION_5M_END)).sum()
        if out_of_session:
            result["issues"].append(f"{out_of_session} Bars ausserhalb Session (09:35-16:00)")
            result["ok"] = False

        # Intervall-Prüfung: Abstände innerhalb eines Tages sollten 5 Minuten sein
        sorted_ts = ts.sort_values()
        diffs = sorted_ts.diff().dropna()
        same_day_diffs = diffs[diffs < pd.Timedelta("1h")]   # Session-Übergang ausschließen
        wrong = (same_day_diffs != pd.Timedelta("5min")).sum()
        if wrong:
            result["warnings"].append(f"{wrong} Intervalle != 5 Minuten (Session-Grenzen erwartet)")

        # Wochentags-Prüfung: keine Bars am Wochenende
        weekdays = ts.dt.dayofweek
        weekend_bars = (weekdays >= 5).sum()
        if weekend_bars:
            result["issues"].append(f"{weekend_bars} Bars am Wochenende")
            result["ok"] = False

        if verbose:
            gaps = detect_gaps(ts)
            for g in gaps:
                result["warnings"].append(f"Lücke: {g}")

    except Exception as e:
        result["rows"] = 0
        result["start"] = result["end"] = "?"
        result["issues"].append(f"Lesefehler: {e}")
        result["ok"] = False

    return result


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_row(r: dict, show_warnings: bool = False):
    status = "OK" if r["ok"] else "FEHLER"
    issues = " | ".join(r.get("issues", []))
    warnings = " | ".join(r.get("warnings", []))

    line = (
        f"  {r['symbol']:<8s}"
        f"  {r.get('rows', 0):>9,} Bars"
        f"  {r.get('start', '?')} -> {r.get('end', '?')}"
        f"  [{status}]"
    )
    if issues:
        line += f"\n    !! {issues}"
    if show_warnings and warnings:
        line += f"\n    >> {warnings}"
    print(line)


def main():
    parser = argparse.ArgumentParser(description="Daten-Validierung für 1m- und 5m-Parquet")
    parser.add_argument("--symbols", nargs="+", help="Nur diese Symbole prüfen")
    parser.add_argument("--verbose", action="store_true", help="Datenlücken und Warnungen anzeigen")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = ROOT / cfg["data"]["raw_1m_dir"]
    proc_dir = ROOT / cfg["data"]["processed_5m_dir"]
    universe_path = ROOT / cfg["data"]["symbol_universe_file"]

    universe = load_universe(universe_path)

    if args.symbols:
        filter_set = set(args.symbols)
    else:
        filter_set = None

    # --- 1m-Daten ---
    print_section("1-Minuten-Daten (raw)")
    raw_files = sorted(raw_dir.glob("*.parquet"))
    if filter_set:
        raw_files = [f for f in raw_files if f.stem in filter_set]

    raw_results = []
    if not raw_files:
        print(f"  Keine 1m-Parquet-Dateien in {raw_dir}")
    else:
        for path in raw_files:
            r = validate_1m(path, args.verbose)
            raw_results.append(r)
            print_row(r, args.verbose)

    # --- 5m-Daten ---
    print_section("5-Minuten-Daten (processed)")
    proc_files = sorted(proc_dir.glob("*.parquet"))
    if filter_set:
        proc_files = [f for f in proc_files if f.stem in filter_set]

    proc_results = []
    if not proc_files:
        print(f"  Keine 5m-Parquet-Dateien in {proc_dir}")
    else:
        for path in proc_files:
            r = validate_5m(path, args.verbose)
            proc_results.append(r)
            print_row(r, args.verbose)

    # --- Abdeckungsvergleich ---
    raw_symbols = {r["symbol"] for r in raw_results}
    proc_symbols = {r["symbol"] for r in proc_results}
    universe_set = set(universe)

    print_section("Zusammenfassung")

    raw_ok = sum(1 for r in raw_results if r["ok"])
    proc_ok = sum(1 for r in proc_results if r["ok"])

    print(f"  1m-Dateien:  {len(raw_results)} gefunden, {raw_ok} OK, {len(raw_results) - raw_ok} Fehler")
    print(f"  5m-Dateien:  {len(proc_results)} gefunden, {proc_ok} OK, {len(proc_results) - proc_ok} Fehler")

    if universe_set:
        missing_1m = universe_set - raw_symbols
        missing_5m = raw_symbols - proc_symbols
        if missing_1m:
            print(f"\n  Universe-Symbole ohne 1m-Daten ({len(missing_1m)}): {sorted(missing_1m)}")
        if missing_5m:
            print(f"\n  1m-Symbole ohne 5m-Daten ({len(missing_5m)}): {sorted(missing_5m)}")

    total_ok = (len(raw_results) == raw_ok) and (len(proc_results) == proc_ok)
    print(f"\n  Status: {'ALLE CHECKS BESTANDEN' if total_ok else 'PROBLEME GEFUNDEN -- Details oben'}")
    print()


if __name__ == "__main__":
    main()

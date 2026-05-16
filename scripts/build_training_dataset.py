#!/usr/bin/env python3
"""
Build AI training dataset from 5m feature parquets and rule-based setup detection.

For every detected setup signal, records the feature snapshot at signal time
plus a forward-simulated binary label:

    label = 1  ->  trade reached +expected_move_required_pct% profit
                   before stop / EOD exit / opposite signal
    label = 0  ->  stop hit, EOD force-close, or opposite signal fired first

Stop is checked before profit target within each bar (conservative, matches
backtester priority — if both could have triggered on the same bar, stop wins).

Earnings setups use a higher profit threshold (0.30% vs 0.20% normal).
Earnings detection is a placeholder: add an 'is_earnings' boolean column to
the feature parquets, or join an external earnings calendar in _is_earnings_bar,
to activate it. Until then every setup uses the normal threshold.

Output:
    data/datasets/train_samples.parquet     -- all labeled samples
    data/datasets/unlabeled_samples.parquet -- setups skipped (no entry bar / overnight)

Usage:
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --symbols AAPL MSFT
    python scripts/build_training_dataset.py --start 2023-05-01 --end 2024-10-31
    python scripts/build_training_dataset.py --force
"""

import argparse
import logging
import sys
from datetime import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtester.setup_detection import detect_setup

TZ = "America/New_York"
_EOD_EXIT = time(15, 50)
_NO_NEW_TRADES_AT = time(15, 30)


def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("build_training_dataset")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


def load_features(
    path: Path,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC").dt.tz_convert(TZ)
    else:
        df["timestamp"] = ts.dt.tz_convert(TZ)
    if start:
        df = df[df["timestamp"].dt.date >= pd.Timestamp(start).date()]
    if end:
        df = df[df["timestamp"].dt.date <= pd.Timestamp(end).date()]
    return df.sort_values("timestamp").reset_index(drop=True)


def _is_earnings_bar(bar: pd.Series) -> bool:
    """
    Placeholder: returns True when this bar is near an earnings release.

    To activate: add an 'is_earnings' boolean column to the feature parquets
    (set True on bars within N days of the earnings date), or join an external
    earnings calendar DataFrame before calling this function and set a flag on
    the matching rows.
    """
    return bool(bar.get("is_earnings", False))


def _simulate_label(
    bars: pd.DataFrame,
    entry_idx: int,
    direction: int,
    entry_price: float,
    stop_price: float,
    profit_threshold_pct: float,
) -> tuple[int, str, int]:
    """
    Scan bars forward from entry_idx and return (label, exit_reason, exit_bar_idx).

    Exit priority per bar (matches backtester exits.py):
      1. EOD force-close (bar time >= 15:50)
      2. Stop hit  -> label 0
      3. Profit target reached  -> label 1
      4. Opposite signal fires  -> label 0
    """
    if direction == 1:
        profit_target = entry_price * (1.0 + profit_threshold_pct / 100.0)
    else:
        profit_target = entry_price * (1.0 - profit_threshold_pct / 100.0)

    last_idx = len(bars) - 1

    for idx in range(entry_idx, len(bars)):
        bar = bars.iloc[idx]
        bar_time = bar["timestamp"].time()

        # 1. EOD force-close
        if bar_time >= _EOD_EXIT:
            return 0, "EOD", idx

        # 2. Stop hit (checked before target — conservative)
        if direction == 1 and float(bar["low"]) <= stop_price:
            return 0, "STOP", idx
        if direction == -1 and float(bar["high"]) >= stop_price:
            return 0, "STOP", idx

        # 3. Profit target
        if direction == 1 and float(bar["high"]) >= profit_target:
            return 1, "TARGET", idx
        if direction == -1 and float(bar["low"]) <= profit_target:
            return 1, "TARGET", idx

        # 4. Opposite signal (skip the entry bar itself to avoid double-scan)
        if idx > entry_idx:
            sig = detect_setup(bar)
            if sig is not None and sig.direction != direction:
                return 0, "OPPOSITE_SIGNAL", idx

    return 0, "EOD", last_idx


def build_samples_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    profit_threshold_normal_pct: float,
    profit_threshold_earnings_pct: float,
) -> tuple[list[dict], list[dict]]:
    """
    Walk through all bars detecting setups and assigning forward-simulated labels.

    One sample is generated per detected setup. After each sample the bar
    cursor advances past the trade's exit bar so samples don't overlap.

    Returns (labeled_samples, unlabeled_samples).
    """
    samples: list[dict] = []
    unlabeled: list[dict] = []

    n = len(df)
    i = 0

    while i < n:
        bar = df.iloc[i]

        # No new setups after 15:30
        if bar["timestamp"].time() >= _NO_NEW_TRADES_AT:
            i += 1
            continue

        signal = detect_setup(bar)
        if signal is None:
            i += 1
            continue

        entry_idx = i + 1

        # No forward bar available (end of data)
        if entry_idx >= n:
            unlabeled.append({
                "symbol": symbol,
                "timestamp": bar["timestamp"],
                "setup_type": signal.setup_type,
                "reason": "no_entry_bar",
            })
            i += 1
            continue

        entry_bar = df.iloc[entry_idx]

        # Signal fired at end of day; entry would be on next session
        if entry_bar["timestamp"].date() != bar["timestamp"].date():
            unlabeled.append({
                "symbol": symbol,
                "timestamp": bar["timestamp"],
                "setup_type": signal.setup_type,
                "reason": "overnight_gap",
            })
            i = entry_idx
            continue

        entry_price = float(entry_bar["open"])

        is_earnings = _is_earnings_bar(bar)
        threshold_pct = (
            profit_threshold_earnings_pct if is_earnings
            else profit_threshold_normal_pct
        )

        label, exit_reason, exit_idx = _simulate_label(
            df,
            entry_idx,
            signal.direction,
            entry_price,
            signal.stop_price,
            threshold_pct,
        )

        # Feature snapshot: all indicator columns from the signal bar
        snapshot = bar.to_dict()

        # Metadata and label fields
        snapshot["symbol"] = symbol
        snapshot["setup_type"] = signal.setup_type
        snapshot["side"] = signal.direction
        snapshot["entry_price"] = round(entry_price, 4)
        snapshot["stop_price"] = signal.stop_price
        snapshot["target_price"] = signal.tp_price
        snapshot["expected_move_required_pct"] = threshold_pct
        snapshot["is_earnings"] = is_earnings
        snapshot["label"] = label
        snapshot["label_exit_reason"] = exit_reason

        samples.append(snapshot)

        # Advance past the exit bar so next scan starts fresh
        i = exit_idx + 1

    return samples, unlabeled


def _log_summary(log: logging.Logger, df: pd.DataFrame) -> None:
    total = len(df)
    wins = int((df["label"] == 1).sum())
    log.info(
        f"Total: {total} samples | "
        f"win rate: {wins}/{total} ({100 * wins / total:.1f}%)"
    )

    for stype, grp in df.groupby("setup_type"):
        w = int((grp["label"] == 1).sum())
        log.info(
            f"  {stype}: {len(grp)} samples, "
            f"{w}/{len(grp)} wins ({100 * w / len(grp):.1f}%)"
        )

    exit_dist = df["label_exit_reason"].value_counts().to_dict()
    log.info(f"Exit breakdown: {exit_dist}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build AI training dataset from 5m feature parquets"
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to process")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files"
    )
    args = parser.parse_args()

    cfg = load_config()
    log = setup_logging()

    features_dir = ROOT / cfg["data"]["features_5m_dir"]
    dataset_cfg = cfg.get("dataset", {})
    out_dir = ROOT / dataset_cfg.get("output_dir", "data/datasets")

    # Profit thresholds: dataset section overrides ai section defaults
    profit_normal = float(
        dataset_cfg.get(
            "profit_threshold_normal_pct",
            cfg["ai"]["min_expected_move_normal_pct"],
        )
    )
    profit_earnings = float(
        dataset_cfg.get(
            "profit_threshold_earnings_pct",
            cfg["ai"]["min_expected_move_earnings_pct"],
        )
    )

    bt_cfg = cfg.get("backtest", {})
    start = args.start or str(bt_cfg.get("start_date", ""))
    end = args.end or str(bt_cfg.get("end_date", ""))

    out_path = out_dir / "train_samples.parquet"
    if out_path.exists() and not args.force:
        log.info("Dataset already exists — use --force to overwrite")
        return

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [p.stem for p in sorted(features_dir.glob("*.parquet"))]

    if not symbols:
        log.warning(f"No feature parquet files found in {features_dir}")
        return

    log.info(
        f"Building dataset: {len(symbols)} symbol(s) | "
        f"normal_threshold={profit_normal}% | "
        f"earnings_threshold={profit_earnings}% | "
        f"period={start or 'all'} -> {end or 'all'}"
    )

    all_samples: list[dict] = []
    all_unlabeled: list[dict] = []

    for symbol in symbols:
        feat_path = features_dir / f"{symbol}.parquet"
        if not feat_path.exists():
            log.warning(f"{symbol}: feature file not found — skipping")
            continue

        try:
            df = load_features(feat_path, start or None, end or None)
            samples, unlabeled = build_samples_for_symbol(
                symbol, df, profit_normal, profit_earnings
            )
            all_samples.extend(samples)
            all_unlabeled.extend(unlabeled)

            if samples:
                wins = sum(1 for s in samples if s["label"] == 1)
                log.info(
                    f"{symbol}: {len(samples)} setups | "
                    f"{wins}/{len(samples)} wins "
                    f"({100 * wins / len(samples):.1f}%)"
                )
            else:
                log.info(f"{symbol}: 0 setups detected")

        except Exception as e:
            log.error(f"{symbol}: error — {e}")

    if not all_samples:
        log.warning("No samples generated — check feature files and date range")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(all_samples)
    df_out.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df_out)} samples -> {out_path}")
    _log_summary(log, df_out)

    if all_unlabeled:
        unlabeled_path = out_dir / "unlabeled_samples.parquet"
        pd.DataFrame(all_unlabeled).to_parquet(unlabeled_path, index=False)
        log.info(
            f"Saved {len(all_unlabeled)} unlabeled (skipped) setups "
            f"-> {unlabeled_path}"
        )


if __name__ == "__main__":
    main()

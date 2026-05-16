#!/usr/bin/env python3
"""
Rule-based backtest on 5m feature parquet files.

Reads   : data/features/5m/<symbol>.parquet
Writes  : data/backtests/rule_based/trades_taken.parquet
          data/backtests/rule_based/trades_rejected.parquet

Usage:
    python scripts/run_rule_backtest.py
    python scripts/run_rule_backtest.py --symbols AAPL MSFT
    python scripts/run_rule_backtest.py --force
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtester.reporting import print_summary, save_results
from src.backtester.simulator import simulate_symbol

TZ = "America/New_York"


def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("run_rule_backtest")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


def load_features(
    path: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
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
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based backtest on 5m feature data"
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to backtest")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing result files"
    )
    args = parser.parse_args()

    cfg = load_config()
    log = setup_logging()

    features_dir = ROOT / cfg["data"]["features_5m_dir"]
    out_dir = ROOT / cfg["backtest"]["rule_based_dir"]

    # Guard against re-running without --force
    if (out_dir / "trades_taken.parquet").exists() and not args.force:
        log.info("Results already exist — use --force to overwrite")
        return

    risk_cfg = cfg["risk"]
    bt_cfg = cfg["backtest"]

    initial_account: float = float(bt_cfg["initial_account"])
    risk_pct: float = float(risk_cfg["risk_pct_per_trade"])
    max_stop_pct: float = float(risk_cfg["max_stop_distance_pct"])
    min_position_pct: float = float(risk_cfg["minimum_position_size_pct"])
    max_daily_loss_pct: float = float(risk_cfg["max_daily_loss_pct"])

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [p.stem for p in sorted(features_dir.glob("*.parquet"))]

    if not symbols:
        log.warning(f"No feature parquet files found in {features_dir}")
        return

    log.info(
        f"Backtesting {len(symbols)} symbol(s) | "
        f"account={initial_account:,.0f} | risk={risk_pct}% | "
        f"max_stop={max_stop_pct}% | max_daily_loss={max_daily_loss_pct}%"
    )

    all_taken = []
    all_rejected = []

    for symbol in symbols:
        feat_path = features_dir / f"{symbol}.parquet"
        if not feat_path.exists():
            log.warning(f"{symbol}: feature file not found — skipping")
            continue

        try:
            df = load_features(feat_path, start=args.start, end=args.end)
            taken, rejected = simulate_symbol(
                symbol=symbol,
                df=df,
                initial_account=initial_account,
                risk_pct=risk_pct,
                max_stop_pct=max_stop_pct,
                min_position_pct=min_position_pct,
                max_daily_loss_pct=max_daily_loss_pct,
            )
            all_taken.extend(taken)
            all_rejected.extend(rejected)
            log.info(
                f"{symbol}: {len(taken)} trades taken, {len(rejected)} rejected"
            )
        except Exception as e:
            log.error(f"{symbol}: error — {e}")

    save_results(all_taken, all_rejected, out_dir)
    log.info(f"Results saved to {out_dir}")

    print_summary(all_taken, all_rejected)


if __name__ == "__main__":
    main()

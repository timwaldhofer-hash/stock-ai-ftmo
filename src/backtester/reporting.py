"""Save backtest results and print a summary."""

import dataclasses
from pathlib import Path

import pandas as pd

from .trade import RejectedSignal, Trade


def _records_to_df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([dataclasses.asdict(r) for r in records])


def save_results(
    taken: list[Trade],
    rejected: list[RejectedSignal],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _records_to_df(taken).to_parquet(out_dir / "trades_taken.parquet", index=False)
    _records_to_df(rejected).to_parquet(out_dir / "trades_rejected.parquet", index=False)


def print_summary(taken: list[Trade], rejected: list[RejectedSignal]) -> None:
    if not taken:
        print("  No trades taken.")
        return

    df = _records_to_df(taken)
    total = len(df)
    winners = (df["pnl_net"] > 0).sum()
    win_rate = winners / total * 100 if total else 0.0
    total_net = df["pnl_net"].sum()
    avg_net = df["pnl_net"].mean()
    gross_profit = df.loc[df["pnl_net"] > 0, "pnl_net"].sum()
    gross_loss = df.loc[df["pnl_net"] < 0, "pnl_net"].sum()
    profit_factor = (
        abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")
    )

    by_type = df.groupby("setup_type")["pnl_net"].agg(["count", "sum", "mean"])

    print(f"\n{'=' * 56}")
    print("  Rule-Based Backtest — Summary")
    print(f"{'=' * 56}")
    print(f"  Trades taken   : {total}")
    print(f"  Trades rejected: {len(rejected)}")
    print(f"  Win rate       : {win_rate:.1f}%  ({winners}W / {total - winners}L)")
    print(f"  Net P&L        : {total_net:,.2f}")
    print(f"  Avg net / trade: {avg_net:,.2f}")
    print(f"  Profit factor  : {profit_factor:.2f}")

    print(f"\n  {'Setup':<25s}  {'#':>5}  {'Total net':>12}  {'Avg net':>10}")
    print(f"  {'-' * 55}")
    for stype, row in by_type.iterrows():
        print(
            f"  {stype:<25s}  {int(row['count']):>5}"
            f"  {row['sum']:>12,.2f}  {row['mean']:>10,.2f}"
        )

    exit_dist = df["exit_reason"].value_counts().to_dict()
    print(f"\n  Exit breakdown: {exit_dist}")
    print(f"{'=' * 56}\n")

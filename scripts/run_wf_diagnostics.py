#!/usr/bin/env python3
"""
Walk-forward diagnostic reporting — Etappe 8.

Reads per-month trade outputs from data/walk_forward/YYYY-MM/trades/,
aggregates along multiple dimensions, and saves CSV/JSON reports to
data/walk_forward/diagnostics/.

Dimensions covered:
  setup_type, side, symbol, month, hour_of_day, exit_reason,
  AI probability buckets (blocked trades), main-AI-only vs main+independent.

Metrics per slice:
  trades, winners, losers, winrate_pct, avg_win, avg_loss,
  expectancy, profit_factor, net_pnl, total_fees, total_slippage,
  cost_impact, max_drawdown (where applicable).

Usage:
    python scripts/run_wf_diagnostics.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_WF_DIR = ROOT / "data" / "walk_forward"
_DIAG_DIR = _WF_DIR / "diagnostics"

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")

_PROB_BINS = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
_PROB_LABELS = [
    "<0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70",
    "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", ">=0.90",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_all_months() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan walk-forward month directories and concatenate trade parquets."""
    taken_chunks: list[pd.DataFrame] = []
    blocked_main_chunks: list[pd.DataFrame] = []
    blocked_indep_chunks: list[pd.DataFrame] = []
    rejected_chunks: list[pd.DataFrame] = []

    for month_dir in sorted(_WF_DIR.iterdir()):
        if not month_dir.is_dir() or not _MONTH_RE.match(month_dir.name):
            continue
        trades_dir = month_dir / "trades"
        if not trades_dir.exists():
            continue
        month = month_dir.name

        for fname, bucket in [
            ("trades_taken_ai.parquet", taken_chunks),
            ("blocked_by_main_ai.parquet", blocked_main_chunks),
            ("blocked_by_independent_ai.parquet", blocked_indep_chunks),
            ("trades_rejected_ai.parquet", rejected_chunks),
        ]:
            p = trades_dir / fname
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    df["wf_month"] = month
                    bucket.append(df)
                except Exception as exc:
                    print(f"  Warning: {month}/{fname}: {exc}", file=sys.stderr)

    def _cat(chunks: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    return _cat(taken_chunks), _cat(blocked_main_chunks), _cat(blocked_indep_chunks), _cat(rejected_chunks)


def _prepare_taken(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to the taken-trades DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            ts = pd.to_datetime(df[col])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            df[col] = ts.dt.tz_convert("America/New_York")
    df["month"] = df["entry_time"].dt.strftime("%Y-%m")
    df["hour_of_day"] = df["entry_time"].dt.hour
    df["side"] = df["direction"].astype(int).map({1: "Long", -1: "Short"})
    return df


# ── Metrics ────────────────────────────────────────────────────────────────────

def _max_drawdown(pnl: pd.Series) -> float:
    """Max peak-to-trough drawdown from a time-ordered PnL sequence."""
    if pnl.empty:
        return 0.0
    cum = pnl.cumsum()
    peak = cum.cummax()
    return round(float((cum - peak).min()), 2)


def _metrics(grp: pd.DataFrame) -> dict:
    """Compute trade metrics for a group; grp must be sorted by entry_time."""
    pnl = grp["pnl_net"].dropna()
    n = len(pnl)
    if n == 0:
        return {
            "trades": 0, "winners": 0, "losers": 0,
            "winrate_pct": None, "avg_win": None, "avg_loss": None,
            "expectancy": None, "profit_factor": None, "net_pnl": 0.0,
            "total_fees": 0.0, "total_slippage": 0.0, "cost_impact": 0.0,
            "max_drawdown": 0.0,
        }
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    wr = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    gw = float(wins.sum())
    gl = abs(float(losses.sum()))
    pf: float | None = round(gw / gl, 4) if gl > 0 else None
    fees = float(grp["fees"].sum()) if "fees" in grp.columns else 0.0
    slip = float(grp["slippage_cost"].sum()) if "slippage_cost" in grp.columns else 0.0
    sort_col = "entry_time" if "entry_time" in grp.columns else grp.columns[0]
    dd = _max_drawdown(grp.sort_values(sort_col)["pnl_net"])

    return {
        "trades": n,
        "winners": int(len(wins)),
        "losers": int(len(losses)),
        "winrate_pct": round(wr * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(wr * avg_win + (1 - wr) * avg_loss, 2),
        "profit_factor": pf,
        "net_pnl": round(float(pnl.sum()), 2),
        "total_fees": round(fees, 2),
        "total_slippage": round(slip, 2),
        "cost_impact": round(fees + slip, 2),
        "max_drawdown": dd,
    }


def _agg_by(df: pd.DataFrame, *group_cols: str) -> pd.DataFrame:
    """Group taken trades by one or more columns and compute metrics per slice."""
    if df.empty:
        return pd.DataFrame()
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        return pd.DataFrame()

    rows = []
    for key, grp in df.groupby(list(group_cols), sort=True):
        m = _metrics(grp)
        if len(group_cols) == 1:
            m[group_cols[0]] = key
        else:
            for col, val in zip(group_cols, key):
                m[col] = val
        rows.append(m)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    lead = list(group_cols)
    return out[lead + [c for c in out.columns if c not in lead]]


# ── AI probability bucket analysis ────────────────────────────────────────────

def _prob_buckets(blocked_main: pd.DataFrame, blocked_indep: pd.DataFrame) -> pd.DataFrame:
    """Blocked-setup counts grouped by AI probability bucket."""
    rows = []
    datasets = [
        ("main_ai", blocked_main, "main_ai_prob"),
        ("independent_ai", blocked_indep, "independent_ai_prob"),
    ]
    for label, df, col in datasets:
        if df.empty or col not in df.columns:
            continue
        probs = df[col].dropna()
        if probs.empty:
            continue
        bucketed = pd.cut(probs, bins=_PROB_BINS, labels=_PROB_LABELS, right=False)
        counts = bucketed.value_counts().reindex(_PROB_LABELS, fill_value=0)
        total = len(probs)
        for bucket, cnt in counts.items():
            rows.append({
                "filter": label,
                "prob_bucket": str(bucket),
                "count": int(cnt),
                "pct_of_blocked": round(cnt / total * 100, 2) if total else 0.0,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Main-AI-only vs main+independent comparison ────────────────────────────────

def _filter_comparison(taken: pd.DataFrame, blocked_indep: pd.DataFrame) -> dict:
    """Summarise filtering efficiency of adding independent AI on top of main AI."""
    n_taken = len(taken)
    n_blocked_indep = len(blocked_indep)
    n_passed_main = n_taken + n_blocked_indep

    out: dict = {
        "note": (
            "Trades blocked by independent AI were never executed, so their PnL "
            "cannot be measured. Only taken-trade metrics are computed."
        ),
        "total_passed_main_ai": n_passed_main,
        "additionally_blocked_by_independent_ai": n_blocked_indep,
        "additional_filter_pct": (
            round(n_blocked_indep / n_passed_main * 100, 2) if n_passed_main else None
        ),
        "final_taken_trades": n_taken,
    }

    if not taken.empty:
        out["taken_metrics"] = _metrics(taken)

    if not blocked_indep.empty and "independent_ai_prob" in blocked_indep.columns:
        probs = blocked_indep["independent_ai_prob"].dropna()
        if not probs.empty:
            out["blocked_by_indep_prob_stats"] = {
                "n": int(len(probs)),
                "mean": round(float(probs.mean()), 4),
                "p10": round(float(probs.quantile(0.10)), 4),
                "p25": round(float(probs.quantile(0.25)), 4),
                "p50": round(float(probs.quantile(0.50)), 4),
                "p75": round(float(probs.quantile(0.75)), 4),
                "p90": round(float(probs.quantile(0.90)), 4),
            }

    return out


# ── JSON serialisation ─────────────────────────────────────────────────────────

def _to_python(obj):
    """Recursively convert numpy scalars and NaN/Inf to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _save_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(_to_python(data), f, indent=2)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.replace({float("nan"): None}).to_csv(path, index=False)


# ── Console summary ────────────────────────────────────────────────────────────

def _print_summary(
    taken: pd.DataFrame,
    blocked_main: pd.DataFrame,
    blocked_indep: pd.DataFrame,
    rejected: pd.DataFrame,
) -> None:
    w = 72
    print(f"\n{'=' * w}")
    print("  Walk-Forward Diagnostics - Etappe 8")
    print(f"{'=' * w}")

    if taken.empty:
        print("  No taken trade data found.")
        print(f"{'=' * w}\n")
        return

    overall = _metrics(taken.sort_values("entry_time"))
    n_months = taken["month"].nunique() if "month" in taken.columns else "?"

    print(f"\n  Overall - {n_months} months | {overall['trades']:,} AI-filtered trades")
    print(f"  {'-' * 50}")
    print(f"  Win Rate      : {overall['winrate_pct']:.1f}%")
    print(f"  Net PnL       : ${overall['net_pnl']:>12,.2f}")
    print(f"  Expectancy    : ${overall['expectancy']:>8.2f}  per trade")
    pf_str = f"{overall['profit_factor']:.4f}" if overall["profit_factor"] else "N/A"
    print(f"  Profit Factor : {pf_str}")
    print(f"  Avg Win       : ${overall['avg_win']:>8.2f}")
    print(f"  Avg Loss      : ${overall['avg_loss']:>8.2f}")
    print(f"  Cost Impact   : ${overall['cost_impact']:>10,.2f}  (fees + slippage)")
    print(f"  Max Drawdown  : ${overall['max_drawdown']:>10,.2f}")

    n_taken = overall["trades"]
    n_blk_main = len(blocked_main)
    n_blk_indep = len(blocked_indep)
    n_rej = len(rejected)
    total_scored = n_blk_main + n_blk_indep + n_rej + n_taken

    print(f"\n  Filtering Funnel (cumulative across all months)")
    print(f"  {'-' * 50}")
    print(f"  Scored by main AI       : {total_scored:>8,}")
    pct = lambda n: f"{n/total_scored*100:.1f}%" if total_scored else "N/A"
    print(f"  Blocked by main AI      : {n_blk_main:>8,}  ({pct(n_blk_main)})")
    print(f"  Blocked by independent  : {n_blk_indep:>8,}  ({pct(n_blk_indep)})")
    print(f"  Rejected by risk rules  : {n_rej:>8,}  ({pct(n_rej)})")
    print(f"  Taken (both AI pass)    : {n_taken:>8,}  ({pct(n_taken)})")

    hdr = f"  {'Metric':<26} {'Trades':>6} {'WinRate':>8} {'Net PnL':>12} {'Expect':>8}"
    sep = f"  {'-' * 62}"

    print(f"\n  By Setup Type")
    print(hdr)
    print(sep)
    for st, grp in taken.groupby("setup_type", sort=True):
        m = _metrics(grp)
        print(
            f"  {st:<26} {m['trades']:>6,} {m['winrate_pct']:>7.1f}% "
            f"${m['net_pnl']:>11,.2f} {m['expectancy']:>8.2f}"
        )

    print(f"\n  By Side")
    print(hdr)
    print(sep)
    for side, grp in taken.groupby("side", sort=True):
        m = _metrics(grp)
        print(
            f"  {side:<26} {m['trades']:>6,} {m['winrate_pct']:>7.1f}% "
            f"${m['net_pnl']:>11,.2f} {m['expectancy']:>8.2f}"
        )

    print(f"\n  By Exit Reason")
    print(hdr)
    print(sep)
    for reason, grp in taken.groupby("exit_reason", sort=True):
        m = _metrics(grp)
        print(
            f"  {reason:<26} {m['trades']:>6,} {m['winrate_pct']:>7.1f}% "
            f"${m['net_pnl']:>11,.2f} {m['expectancy']:>8.2f}"
        )

    print(f"\n  By Hour of Day (top 8 by trade count)")
    print(f"  {'Hour':>6} {'Trades':>6} {'WinRate':>8} {'Net PnL':>12} {'Expect':>8}")
    print(f"  {'-' * 46}")
    hour_rows = [(hr, _metrics(grp)) for hr, grp in taken.groupby("hour_of_day")]
    for hr, m in sorted(hour_rows, key=lambda x: -x[1]["trades"])[:8]:
        print(
            f"  {hr:>6}  {m['trades']:>6,} {m['winrate_pct']:>7.1f}% "
            f"${m['net_pnl']:>11,.2f} {m['expectancy']:>8.2f}"
        )

    print(f"\n  Reports saved to: {_DIAG_DIR}")
    print(f"{'=' * w}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading walk-forward outputs from {_WF_DIR} ...")
    taken, blocked_main, blocked_indep, rejected = _load_all_months()

    if taken.empty:
        print("ERROR: No trades_taken_ai.parquet files found. Run run_walk_forward.py first.")
        sys.exit(1)

    taken = _prepare_taken(taken)
    n_months = taken["wf_month"].nunique()
    print(
        f"  Taken trades  : {len(taken):,} across {n_months} months\n"
        f"  Blocked main  : {len(blocked_main):,}\n"
        f"  Blocked indep : {len(blocked_indep):,}\n"
        f"  Rejected risk : {len(rejected):,}"
    )

    all_reports: dict = {}

    # ── Single-dimension aggregations ──────────────────────────────────────────
    single_dims = [
        ("setup_type", "by_setup_type.csv"),
        ("side",       "by_side.csv"),
        ("symbol",     "by_symbol.csv"),
        ("month",      "by_month.csv"),
        ("hour_of_day","by_hour_of_day.csv"),
        ("exit_reason","by_exit_reason.csv"),
    ]
    for col, fname in single_dims:
        df_agg = _agg_by(taken, col)
        if not df_agg.empty:
            _save_csv(df_agg, _DIAG_DIR / fname)
            all_reports[f"by_{col}"] = _to_python(
                df_agg.replace({float("nan"): None}).to_dict(orient="records")
            )
        print(f"  Saved {fname}")

    # ── Cross-tab: setup_type × side ───────────────────────────────────────────
    df_cross = _agg_by(taken, "setup_type", "side")
    if not df_cross.empty:
        _save_csv(df_cross, _DIAG_DIR / "by_setup_type_and_side.csv")
        all_reports["by_setup_type_and_side"] = _to_python(
            df_cross.replace({float("nan"): None}).to_dict(orient="records")
        )
    print("  Saved by_setup_type_and_side.csv")

    # ── AI probability buckets ─────────────────────────────────────────────────
    df_probs = _prob_buckets(blocked_main, blocked_indep)
    if not df_probs.empty:
        _save_csv(df_probs, _DIAG_DIR / "ai_prob_buckets.csv")
        all_reports["ai_prob_buckets"] = df_probs.to_dict(orient="records")
    print("  Saved ai_prob_buckets.csv")

    # ── Main-AI-only vs main+independent ──────────────────────────────────────
    filter_cmp = _filter_comparison(taken, blocked_indep)
    _save_json(_DIAG_DIR / "filter_comparison.json", filter_cmp)
    all_reports["filter_comparison"] = filter_cmp
    print("  Saved filter_comparison.json")

    # ── Overall metrics ────────────────────────────────────────────────────────
    all_reports["overall"] = _to_python(_metrics(taken.sort_values("entry_time")))
    all_reports["funnel"] = {
        "total_scored_by_main_ai": (
            len(blocked_main) + len(blocked_indep) + len(rejected) + len(taken)
        ),
        "blocked_by_main_ai": len(blocked_main),
        "blocked_by_independent_ai": len(blocked_indep),
        "rejected_by_risk_rules": len(rejected),
        "taken": len(taken),
    }

    # ── Master JSON ────────────────────────────────────────────────────────────
    _save_json(_DIAG_DIR / "diagnostics_summary.json", all_reports)
    print("  Saved diagnostics_summary.json")

    # ── Console summary ────────────────────────────────────────────────────────
    _print_summary(taken, blocked_main, blocked_indep, rejected)


if __name__ == "__main__":
    main()

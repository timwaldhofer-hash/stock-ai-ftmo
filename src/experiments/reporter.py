"""Output generation for experiment results."""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_OUT_DIR = Path("data/walk_forward/strategy_experiments")


def save_experiment_results(
    results: List[Dict],
    failed_variants: List[Dict],
    experiment_config: Dict,
    output_dir: Optional[Path] = None,
) -> None:
    """Write all required output files to output_dir."""
    if output_dir is None:
        output_dir = _OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("robustness_adjusted_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # ranked_results.csv / .json
    # Drop monthly_breakdown from flat CSV (save it separately)
    df_flat = df.drop(columns=["monthly_breakdown"], errors="ignore")
    df_flat.to_csv(output_dir / "ranked_results.csv", index=False)

    ranked_json = df.to_dict(orient="records")
    with open(output_dir / "ranked_results.json", "w") as f:
        json.dump(ranked_json, f, indent=2, default=_json_safe)

    # monthly_breakdown_by_variant.csv
    monthly_rows = []
    for r in results:
        for m in r.get("monthly_breakdown", []):
            row = {"variant_name": r["variant_name"], **m}
            monthly_rows.append(row)
    if monthly_rows:
        pd.DataFrame(monthly_rows).to_csv(
            output_dir / "monthly_breakdown_by_variant.csv", index=False
        )

    # failed_variants.csv
    if failed_variants:
        pd.DataFrame(failed_variants).to_csv(output_dir / "failed_variants.csv", index=False)
    else:
        pd.DataFrame(columns=["variant_name", "fail_reason"]).to_csv(
            output_dir / "failed_variants.csv", index=False
        )

    # robustness_metrics.csv
    rob_cols = [
        "variant_name", "rank", "robustness_adjusted_score", "overfitting_penalty",
        "ftmo_passed", "profitable_months", "total_months",
        "months_better_than_baseline", "best_to_median_ratio",
        "monthly_pnl_std", "max_drawdown_pct",
        "daily_drawdown_max_pct", "total_drawdown_max_pct",
        "profit_factor_stability", "expectancy_stability",
    ]
    df_flat[[c for c in rob_cols if c in df_flat.columns]].to_csv(
        output_dir / "robustness_metrics.csv", index=False
    )

    # experiment_config.json
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2, default=_json_safe)

    # top_20_summary.txt
    _write_top20(df, output_dir)

    print(f"\nResults saved to {output_dir}")
    print(f"  ranked_results.csv         ({len(df)} variants)")
    print(f"  monthly_breakdown_by_variant.csv")
    print(f"  failed_variants.csv        ({len(failed_variants)} failed)")
    print(f"  robustness_metrics.csv")
    print(f"  experiment_config.json")
    print(f"  top_20_summary.txt")


def _write_top20(df: pd.DataFrame, output_dir: Path) -> None:
    w = 82
    lines: List[str] = []
    lines.append("=" * w)
    lines.append("  EXPERIMENT FRAMEWORK  —  RANKED RESULTS  (TOP 20)")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Variants  : {len(df)} evaluated")
    lines.append("=" * w)

    hdr = (
        f"  {'Rk':>3}  {'Variant':<28}  {'Score':>7}  {'FTMO':>5}  "
        f"{'NetPnL':>10}  {'PF':>6}  {'Exp$':>7}  {'#Tr':>5}  {'PrMo':>4}"
    )
    lines.append(hdr)
    lines.append("  " + "─" * (w - 2))

    for _, row in df.head(20).iterrows():
        ftmo = "PASS" if row.get("ftmo_passed") else "FAIL"
        pf = row.get("profit_factor")
        pf_s = f"{pf:.3f}" if pf and math.isfinite(float(pf)) else "  N/A"
        lines.append(
            f"  {int(row['rank']):>3}  {str(row['variant_name']):<28}  "
            f"{row['robustness_adjusted_score']:>7.4f}  {ftmo:>5}  "
            f"{row.get('net_pnl', 0):>10,.2f}  {pf_s:>6}  "
            f"{row.get('expectancy', 0):>7.2f}  {int(row.get('trade_count', 0)):>5}  "
            f"{int(row.get('profitable_months', 0)):>4}"
        )

    lines.append("=" * w)
    lines.append("")
    lines.append("  COLUMNS")
    lines.append("    Score  = robustness_adjusted_score (higher is better, max 1.0)")
    lines.append("    FTMO   = daily DD <4% AND total DD <9%")
    lines.append("    NetPnL = cumulative net P&L across all test months ($)")
    lines.append("    PF     = profit factor")
    lines.append("    Exp$   = expectancy per trade ($)")
    lines.append("    #Tr    = total trades taken")
    lines.append("    PrMo   = profitable months out of total test months")
    lines.append("=" * w)
    lines.append("")

    lines.append("  DETAILED VIEW  —  TOP 5")
    lines.append("=" * w)

    for i, (_, row) in enumerate(df.head(5).iterrows()):
        n = i + 1
        pf = row.get("profit_factor")
        pf_s = f"{pf:.3f}" if pf and math.isfinite(float(pf)) else "N/A"
        wlr = row.get("avg_win_loss_ratio")
        wlr_s = f"{wlr:.3f}" if wlr else "N/A"
        btm = row.get("best_to_median_ratio")
        btm_s = f"{btm:.2f}x" if btm and math.isfinite(float(btm)) else "N/A"

        lines += [
            f"",
            f"  [{n}] {row['variant_name']}",
            f"      Description       : {row.get('description', '')}",
            f"      Robustness score  : {row['robustness_adjusted_score']:.4f}",
            f"      Overfitting pen.  : {row.get('overfitting_penalty', 0.0):.4f}",
            f"      FTMO compliance   : {'PASS' if row.get('ftmo_passed') else 'FAIL'}",
            f"      ─────────────────────────────────────────",
            f"      Net P&L           : ${row.get('net_pnl', 0):,.2f}",
            f"      Profit Factor     : {pf_s}",
            f"      Expectancy        : ${row.get('expectancy', 0):.2f} / trade",
            f"      Win Rate          : {row.get('win_rate_pct', 0):.1f}%",
            f"      Total Trades      : {int(row.get('trade_count', 0))}",
            f"      Avg Win           : ${row.get('avg_win', 0):.2f}",
            f"      Avg Loss          : ${row.get('avg_loss', 0):.2f}",
            f"      Win/Loss Ratio    : {wlr_s}",
            f"      Cost Impact       : ${row.get('cost_impact', 0):,.2f}",
            f"      ─────────────────────────────────────────",
            f"      Max DD (equity)   : {row.get('max_drawdown_pct', 0):.2f}%",
            f"      Daily DD max      : {row.get('daily_drawdown_max_pct', 0):.2f}%",
            f"      Total DD max      : {row.get('total_drawdown_max_pct', 0):.2f}%",
            f"      ─────────────────────────────────────────",
            f"      Profitable months : {row.get('profitable_months', 0)} / {row.get('total_months', 0)}",
            f"      Months > baseline : {row.get('months_better_than_baseline', 0)}",
            f"      Best month        : ${row.get('best_month', 0):.2f}",
            f"      Worst month       : ${row.get('worst_month', 0):.2f}",
            f"      Median month      : ${row.get('median_month', 0):.2f}",
            f"      Monthly PnL std   : ${row.get('monthly_pnl_std', 0):.2f}",
            f"      Best/Median ratio : {btm_s}",
            f"      PF stability (std): {row.get('profit_factor_stability', 'N/A')}",
            f"      Exp stability (std): {row.get('expectancy_stability', 'N/A')}",
        ]

    lines.append("")
    lines.append("=" * w)

    with open(output_dir / "top_20_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _json_safe(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
    raise TypeError(f"Not JSON serialisable: {type(obj)}")

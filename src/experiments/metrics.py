"""
Monthly stability metrics and robustness score for the experiment framework.

All EXPERIMENT_SPEC fields are computed here:
  monthly_net_pnl, monthly_expectancy, monthly_profit_factor,
  number_of_profitable_months, number_of_months_better_than_baseline,
  worst_month, best_month, median_month, standard_deviation_monthly_pnl,
  best_month_to_median_ratio, worst_month_drawdown, max_drawdown,
  average_win, average_loss, average_win_loss_ratio, trade_count,
  cost_impact, profit_factor_stability, expectancy_stability,
  robustness_adjusted_score, overfitting_penalty.
"""

import math
from typing import Dict, List, Optional

import numpy as np


_FTMO_DAILY_DD_PCT = 4.0
_FTMO_TOTAL_DD_PCT = 9.0


# ── Per-month statistics ──────────────────────────────────────────────────────

def compute_monthly_stats(trades: list, month: str) -> Dict:
    """Basic stats for a single month's trade list."""
    pnls = [t.pnl_net for t in trades if t.pnl_net is not None]
    total = len(pnls)

    if total == 0:
        return {
            "month": month,
            "trade_count": 0,
            "win_count": 0,
            "win_rate_pct": 0.0,
            "net_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_win_loss_ratio": None,
            "profit_factor": None,
            "expectancy": 0.0,
            "cost_impact": 0.0,
            "partial_tp_count": 0,
            "partial_tp_rate_pct": 0.0,
        }

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    win_count = len(winners)

    gp = sum(winners)
    gl = sum(losers)
    avg_win = gp / len(winners) if winners else 0.0
    avg_loss = gl / len(losers) if losers else 0.0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None
    wr = win_count / total
    pf = abs(gp / gl) if gl != 0 else (float("inf") if gp > 0 else None)
    expectancy = wr * avg_win + (1.0 - wr) * avg_loss
    cost_impact = sum(
        (t.fees or 0.0) + (t.slippage_cost or 0.0)
        for t in trades if t.pnl_net is not None
    )

    partial_tp_count = sum(
        1 for t in trades
        if t.pnl_net is not None and getattr(t, "partial_tp_shares", 0) > 0
    )

    return {
        "month": month,
        "trade_count": total,
        "win_count": win_count,
        "win_rate_pct": round(wr * 100, 2),
        "net_pnl": round(sum(pnls), 2),
        "avg_pnl": round(sum(pnls) / total, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_win_loss_ratio": round(wl_ratio, 4) if wl_ratio is not None else None,
        "profit_factor": (
            round(pf, 4) if pf is not None and math.isfinite(pf) else None
        ),
        "expectancy": round(expectancy, 2),
        "cost_impact": round(cost_impact, 2),
        "partial_tp_count": partial_tp_count,
        "partial_tp_rate_pct": round(partial_tp_count / total * 100, 2),
    }


# ── Aggregate stability metrics ───────────────────────────────────────────────

def compute_stability_metrics(
    monthly_stats: List[Dict],
    baseline_monthly_pnl: Optional[Dict[str, float]],
    max_daily_dd_pct: float,
    max_total_dd_pct: float,
    initial_account: float = 100_000.0,
) -> Dict:
    """
    Aggregate across months for one variant.
    baseline_monthly_pnl: {month_str: pnl} from original walk-forward.
    """
    if not monthly_stats:
        return _empty_metrics()

    total_months = len(monthly_stats)
    monthly_pnls = [m["net_pnl"] for m in monthly_stats]

    profitable_months = sum(1 for p in monthly_pnls if p > 0)
    worst_month = min(monthly_pnls)
    best_month = max(monthly_pnls)
    median_month = float(np.median(monthly_pnls))
    mean_month = float(np.mean(monthly_pnls))
    std_month = float(np.std(monthly_pnls, ddof=1)) if total_months > 1 else 0.0

    abs_median = abs(median_month)
    if abs_median > 1e-4:
        best_to_median = best_month / abs_median
    elif best_month > 0:
        best_to_median = float("inf")
    else:
        best_to_median = 0.0

    # Months better than baseline
    months_better = 0
    if baseline_monthly_pnl:
        for m in monthly_stats:
            base = baseline_monthly_pnl.get(m["month"], 0.0)
            if m["net_pnl"] > base:
                months_better += 1

    # Aggregate trade metrics
    all_trades = sum(m["trade_count"] for m in monthly_stats)
    all_wins = sum(m["win_count"] for m in monthly_stats)
    total_costs = sum(m["cost_impact"] for m in monthly_stats)

    total_gross_profit = sum(
        m["avg_win"] * m["win_count"]
        for m in monthly_stats
        if m["avg_win"] and m["win_count"] > 0
    )
    total_gross_loss = sum(
        abs(m["avg_loss"]) * (m["trade_count"] - m["win_count"])
        for m in monthly_stats
        if m["avg_loss"] and m["trade_count"] > 0
    )
    profit_factor = (
        total_gross_profit / total_gross_loss if total_gross_loss > 0 else None
    )

    wr = all_wins / all_trades if all_trades > 0 else 0.0
    valid_wins = [m["avg_win"] for m in monthly_stats if m.get("avg_win")]
    valid_losses = [m["avg_loss"] for m in monthly_stats if m.get("avg_loss")]
    agg_avg_win = float(np.mean(valid_wins)) if valid_wins else 0.0
    agg_avg_loss = float(np.mean(valid_losses)) if valid_losses else 0.0
    wl_ratio = abs(agg_avg_win / agg_avg_loss) if agg_avg_loss != 0 else None
    expectancy = wr * agg_avg_win + (1.0 - wr) * agg_avg_loss if all_trades > 0 else 0.0

    # FTMO compliance
    ftmo_passed = (max_daily_dd_pct < _FTMO_DAILY_DD_PCT) and (max_total_dd_pct < _FTMO_TOTAL_DD_PCT)

    # Equity-curve based max drawdown
    running = 0.0
    peak_running = 0.0
    max_dd_pct = 0.0
    for pnl in monthly_pnls:
        running += pnl
        peak_running = max(peak_running, running)
        dd = (peak_running - running) / initial_account * 100
        max_dd_pct = max(max_dd_pct, dd)

    # Profit-factor stability across months
    monthly_pf = [
        m["profit_factor"] for m in monthly_stats
        if m.get("profit_factor") is not None
    ]
    pf_stability = float(np.std(monthly_pf, ddof=1)) if len(monthly_pf) > 1 else None

    # Expectancy stability across months
    monthly_exp = [m["expectancy"] for m in monthly_stats]
    exp_stability = float(np.std(monthly_exp, ddof=1)) if len(monthly_exp) > 1 else None

    pf_rounded = (
        round(profit_factor, 4) if profit_factor and math.isfinite(profit_factor) else None
    )

    return {
        "total_months": total_months,
        "trade_count": all_trades,
        "net_pnl": round(sum(monthly_pnls), 2),
        "avg_monthly_pnl": round(mean_month, 2),
        "profit_factor": pf_rounded,
        "expectancy": round(expectancy, 2),
        "win_rate_pct": round(wr * 100, 2),
        "avg_win": round(agg_avg_win, 2),
        "avg_loss": round(agg_avg_loss, 2),
        "avg_win_loss_ratio": round(wl_ratio, 4) if wl_ratio is not None else None,
        "cost_impact": round(total_costs, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "daily_drawdown_max_pct": round(max_daily_dd_pct, 4),
        "total_drawdown_max_pct": round(max_total_dd_pct, 4),
        "ftmo_passed": ftmo_passed,
        "profitable_months": profitable_months,
        "months_better_than_baseline": months_better,
        "worst_month": round(worst_month, 2),
        "best_month": round(best_month, 2),
        "median_month": round(median_month, 2),
        "monthly_pnl_std": round(std_month, 2),
        "best_to_median_ratio": (
            round(best_to_median, 4) if math.isfinite(best_to_median) else None
        ),
        "profit_factor_stability": (
            round(pf_stability, 4) if pf_stability is not None else None
        ),
        "expectancy_stability": (
            round(exp_stability, 4) if exp_stability is not None else None
        ),
    }


def _empty_metrics() -> Dict:
    return {
        "total_months": 0, "trade_count": 0, "net_pnl": 0.0,
        "avg_monthly_pnl": 0.0, "profit_factor": None, "expectancy": 0.0,
        "win_rate_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "avg_win_loss_ratio": None, "cost_impact": 0.0,
        "max_drawdown_pct": 0.0, "daily_drawdown_max_pct": 0.0,
        "total_drawdown_max_pct": 0.0, "ftmo_passed": False,
        "profitable_months": 0, "months_better_than_baseline": 0,
        "worst_month": 0.0, "best_month": 0.0, "median_month": 0.0,
        "monthly_pnl_std": 0.0, "best_to_median_ratio": None,
        "profit_factor_stability": None, "expectancy_stability": None,
    }


# ── Robustness score ──────────────────────────────────────────────────────────

def compute_robustness_score(metrics: Dict) -> float:
    """
    Robustness-adjusted score in [0, 1].

    Weights:
      30 %  monthly stability (penalises outlier months + high variance)
      25 %  monthly consistency (fraction profitable)
      15 %  expectancy quality
      15 %  profit factor quality
      15 %  drawdown quality

    FTMO violation → 0.0.
    Overfitting penalty applied when best_month >> median_month.
    """
    if not metrics.get("ftmo_passed", False):
        return 0.0

    total_months = metrics.get("total_months", 0)
    if total_months == 0 or metrics.get("trade_count", 0) == 0:
        return 0.0

    # (1) Monthly stability
    best_to_median = metrics.get("best_to_median_ratio")
    if best_to_median is None or not math.isfinite(best_to_median):
        outlier_factor = 0.0
    elif best_to_median <= 3.0:
        outlier_factor = 1.0
    else:
        outlier_factor = max(0.0, 1.0 - (best_to_median - 3.0) / 3.0)

    mean_pnl = abs(metrics.get("avg_monthly_pnl", 0.0))
    std_pnl = metrics.get("monthly_pnl_std", 0.0)
    if mean_pnl > 1e-4:
        cv = std_pnl / mean_pnl
        cv_factor = max(0.0, 1.0 - max(0.0, cv - 1.0) / 2.0)
    else:
        cv_factor = 0.3

    stability_score = outlier_factor * cv_factor

    # (2) Monthly consistency
    profitable_ratio = metrics.get("profitable_months", 0) / total_months
    consistency_score = profitable_ratio

    # (3) Expectancy
    exp = metrics.get("expectancy", 0.0)
    if exp >= 200.0:
        exp_score = 1.0
    elif exp >= 0.0:
        exp_score = exp / 200.0
    else:
        exp_score = 0.0

    # (4) Profit factor
    pf = metrics.get("profit_factor") or 0.0
    if not math.isfinite(pf):
        pf = 2.0  # treat +inf as very good
    if pf >= 2.0:
        pf_score = 1.0
    elif pf >= 1.0:
        pf_score = pf - 1.0
    else:
        pf_score = 0.0

    # (5) Drawdown
    dd = metrics.get("max_drawdown_pct", 9.0)
    dd_score = max(0.0, 1.0 - dd / 9.0)

    raw = (
        0.30 * stability_score
        + 0.25 * consistency_score
        + 0.15 * exp_score
        + 0.15 * pf_score
        + 0.15 * dd_score
    )

    # Overfitting penalty: best month disproportionately dominates
    overfitting_penalty = 0.0
    if best_to_median is not None and math.isfinite(best_to_median) and best_to_median > 3.0:
        overfitting_penalty = min(0.3, (best_to_median - 3.0) / 10.0)

    return round(max(0.0, raw - overfitting_penalty), 4)

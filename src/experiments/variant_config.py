"""Experiment variant definitions for the strategy experiment framework."""

from dataclasses import dataclass, field
from datetime import time
from typing import List, Optional


@dataclass
class VariantConfig:
    name: str

    # ── AI thresholds ─────────────────────────────────────────────────────────
    main_ai_threshold: float = 0.80

    # ── Expected move filter ──────────────────────────────────────────────────
    min_expected_move_pct: float = 0.40

    # ── Setup filters ─────────────────────────────────────────────────────────
    allow_mean_reversion: bool = False
    no_new_trades_after: time = field(default_factory=lambda: time(14, 30))

    # ── Loss veto model ───────────────────────────────────────────────────────
    use_loss_veto: bool = False
    loss_veto_threshold: float = 0.30   # block trade if P(loss) > this

    # ── Exit logic ────────────────────────────────────────────────────────────
    atr_stop_multiplier: float = 1.5
    breakeven_trigger_pct: Optional[float] = None   # move stop to BE after X% gain
    trailing_stop_pct: Optional[float] = None        # trailing stop % from peak
    max_holding_bars: Optional[int] = None
    momentum_loss_exit: bool = False

    # ── Setup quality filters ─────────────────────────────────────────────────
    min_reward_risk: Optional[float] = None   # tp_dist / stop_dist minimum

    # ── Exit & Risk-Reward stage ──────────────────────────────────────────────
    # Partial TP: exit a fraction of the position at an interim target, then
    # auto-move stop to breakeven on the remainder.
    partial_tp_fraction: Optional[float] = None   # e.g. 0.50 → exit half at partial TP
    partial_tp_target_pct: float = 0.50           # 0–1: fraction of full TP distance

    # ATR-based trailing stop: trail N×ATR from the favorable price peak.
    # Applied alongside (or instead of) trailing_stop_pct.
    atr_trail_mult: Optional[float] = None

    # Time-based stop tightening: after N bars held without progress, compress
    # the remaining stop gap by time_stop_factor (e.g. 0.5 = halve the gap).
    # Triggers once per trade.
    time_stop_bars: Optional[int] = None
    time_stop_factor: float = 0.50

    description: str = ""


# ── Quick mode: 8 focused variants ────────────────────────────────────────────

QUICK_MODE_VARIANTS: List[VariantConfig] = [
    VariantConfig(
        name="QK01_baseline",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        description="Baseline: main_ai=0.80, move=0.40%, no veto, ATR 1.5x",
    ),
    VariantConfig(
        name="QK02_tighter_ai",
        main_ai_threshold=0.85,
        min_expected_move_pct=0.40,
        description="Tighter AI: main_ai=0.85, move=0.40%",
    ),
    VariantConfig(
        name="QK03_move_strict",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.50,
        description="Strict move: main_ai=0.80, move=0.50%",
    ),
    VariantConfig(
        name="QK04_veto_moderate",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        use_loss_veto=True,
        loss_veto_threshold=0.30,
        description="Loss veto: main_ai=0.80, veto<0.30",
    ),
    VariantConfig(
        name="QK05_veto_strict",
        main_ai_threshold=0.85,
        min_expected_move_pct=0.40,
        use_loss_veto=True,
        loss_veto_threshold=0.25,
        description="Strict veto: main_ai=0.85, veto<0.25",
    ),
    VariantConfig(
        name="QK06_breakeven",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        breakeven_trigger_pct=0.15,
        description="Breakeven at +0.15%: main_ai=0.80",
    ),
    VariantConfig(
        name="QK07_tighter_stop",
        main_ai_threshold=0.85,
        min_expected_move_pct=0.40,
        atr_stop_multiplier=1.2,
        description="Tighter stop: main_ai=0.85, ATR 1.2x",
    ),
    VariantConfig(
        name="QK08_combined",
        main_ai_threshold=0.85,
        min_expected_move_pct=0.40,
        use_loss_veto=True,
        loss_veto_threshold=0.30,
        breakeven_trigger_pct=0.20,
        min_reward_risk=1.2,
        description="Combined: main_ai=0.85, veto<0.30, BE+0.20%, RR>=1.2",
    ),
]


# ── Full mode: grid of all combinations ───────────────────────────────────────

def _full_mode_variants() -> List[VariantConfig]:
    variants: List[VariantConfig] = []

    thresholds = [0.80, 0.85, 0.90]
    min_moves = [0.30, 0.40, 0.50]
    veto_thresholds = [None, 0.25, 0.30, 0.35, 0.40]
    exit_configs = [
        {"breakeven_trigger_pct": None, "atr_stop_multiplier": 1.5},
        {"breakeven_trigger_pct": 0.15, "atr_stop_multiplier": 1.5},
        {"breakeven_trigger_pct": 0.20, "atr_stop_multiplier": 1.5},
        {"breakeven_trigger_pct": None, "atr_stop_multiplier": 1.2},
    ]

    idx = 1
    for thr in thresholds:
        for move in min_moves:
            for veto in veto_thresholds:
                for exit_cfg in exit_configs:
                    name = f"FM{idx:03d}_ai{int(thr*100)}_mv{int(move*100)}"
                    if veto is not None:
                        name += f"_veto{int(veto*100)}"
                    if exit_cfg["breakeven_trigger_pct"] is not None:
                        name += f"_be{int(exit_cfg['breakeven_trigger_pct']*100)}"
                    if exit_cfg["atr_stop_multiplier"] != 1.5:
                        name += f"_atr{int(exit_cfg['atr_stop_multiplier']*10)}"

                    desc = (
                        f"main_ai={thr}, move={move}%"
                        + (f", veto<{veto}" if veto else "")
                        + (f", BE+{exit_cfg['breakeven_trigger_pct']}%" if exit_cfg["breakeven_trigger_pct"] else "")
                        + (f", ATR {exit_cfg['atr_stop_multiplier']}x" if exit_cfg["atr_stop_multiplier"] != 1.5 else "")
                    )

                    variants.append(VariantConfig(
                        name=name,
                        main_ai_threshold=thr,
                        min_expected_move_pct=move,
                        use_loss_veto=(veto is not None),
                        loss_veto_threshold=veto if veto is not None else 0.30,
                        breakeven_trigger_pct=exit_cfg["breakeven_trigger_pct"],
                        atr_stop_multiplier=exit_cfg["atr_stop_multiplier"],
                        description=desc,
                    ))
                    idx += 1

    return variants


FULL_MODE_VARIANTS: List[VariantConfig] = _full_mode_variants()


# ── Exit & Risk-Reward stage: quick mode (12 focused variants) ─────────────────

EXIT_RR_QUICK_VARIANTS: List[VariantConfig] = [
    VariantConfig(
        name="ERR01_baseline",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        description="Exit/RR baseline: default exits, no special logic",
    ),
    VariantConfig(
        name="ERR02_partial50_half",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        partial_tp_fraction=0.50,
        partial_tp_target_pct=0.50,
        description="50% exit at halfway TP → auto-BE on remainder",
    ),
    VariantConfig(
        name="ERR03_partial50_third",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        partial_tp_fraction=0.50,
        partial_tp_target_pct=0.33,
        description="50% exit at 33% of TP distance → auto-BE on remainder",
    ),
    VariantConfig(
        name="ERR04_atr_trail_1x",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        atr_trail_mult=1.0,
        description="ATR 1.0× trailing stop from favorable peak",
    ),
    VariantConfig(
        name="ERR05_atr_trail_15x",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        atr_trail_mult=1.5,
        description="ATR 1.5× trailing stop from favorable peak",
    ),
    VariantConfig(
        name="ERR06_time_stop_8",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        time_stop_bars=8,
        time_stop_factor=0.50,
        description="Halve stop gap after 8 bars with no meaningful progress",
    ),
    VariantConfig(
        name="ERR07_time_stop_12",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        time_stop_bars=12,
        time_stop_factor=0.50,
        description="Halve stop gap after 12 bars with no meaningful progress",
    ),
    VariantConfig(
        name="ERR08_rr_150",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        min_reward_risk=1.5,
        description="Min R:R 1.5 entry filter — forces better TP/stop geometry",
    ),
    VariantConfig(
        name="ERR09_rr_200",
        main_ai_threshold=0.80,
        min_expected_move_pct=0.40,
        min_reward_risk=2.0,
        description="Min R:R 2.0 entry filter",
    ),
    VariantConfig(
        name="ERR10_partial50_rr150",
        main_ai_threshold=0.85,
        min_expected_move_pct=0.40,
        partial_tp_fraction=0.50,
        partial_tp_target_pct=0.50,
        min_reward_risk=1.5,
        description="50% partial TP at mid + min R:R 1.5 + ai=0.85",
    ),
    VariantConfig(
        name="ERR11_full_protect",
        main_ai_threshold=0.85,
        min_expected_move_pct=0.40,
        partial_tp_fraction=0.50,
        partial_tp_target_pct=0.50,
        atr_trail_mult=1.0,
        min_reward_risk=1.5,
        description="Full protection: partial TP + ATR trail 1× + min R:R 1.5",
    ),
    VariantConfig(
        name="ERR12_tight_atr_partial",
        main_ai_threshold=0.85,
        min_expected_move_pct=0.40,
        atr_stop_multiplier=1.2,
        partial_tp_fraction=0.50,
        partial_tp_target_pct=0.50,
        atr_trail_mult=1.0,
        description="Tight ATR 1.2× stop + partial TP at mid + ATR trail 1×",
    ),
]


# ── Exit & Risk-Reward stage: full grid ───────────────────────────────────────

def _exit_rr_full_variants() -> List[VariantConfig]:
    variants: List[VariantConfig] = []
    idx = 1

    ai_thresholds = [0.80, 0.85]
    atr_stops = [1.5, 1.2]

    # (fraction_to_exit, target_pct_of_tp_dist)
    partial_tp_opts = [
        None,
        (0.50, 0.33),
        (0.50, 0.50),
        (0.33, 0.50),
    ]

    atr_trail_opts = [None, 1.0, 1.5, 2.0]
    min_rr_opts = [None, 1.2, 1.5, 2.0]

    for ai in ai_thresholds:
        for atr_stop in atr_stops:
            for partial in partial_tp_opts:
                for trail in atr_trail_opts:
                    for min_rr in min_rr_opts:
                        name = f"ERR{idx:03d}_ai{int(ai * 100)}_s{int(atr_stop * 10)}"
                        desc_parts = [f"ai={ai}", f"atr={atr_stop}×"]

                        if partial is not None:
                            frac, tgt = partial
                            name += f"_p{int(frac * 100)}t{int(tgt * 100)}"
                            desc_parts.append(f"partTP={int(frac*100)}%@{int(tgt*100)}%")
                        if trail is not None:
                            name += f"_tr{int(trail * 10)}"
                            desc_parts.append(f"trail={trail}×ATR")
                        if min_rr is not None:
                            name += f"_rr{int(min_rr * 10)}"
                            desc_parts.append(f"minRR={min_rr}")

                        variants.append(VariantConfig(
                            name=name,
                            main_ai_threshold=ai,
                            atr_stop_multiplier=atr_stop,
                            partial_tp_fraction=partial[0] if partial else None,
                            partial_tp_target_pct=partial[1] if partial else 0.50,
                            atr_trail_mult=trail,
                            min_reward_risk=min_rr,
                            description=", ".join(desc_parts),
                        ))
                        idx += 1

    # Time-stop cross: bars × tightness × ai × partial
    time_stop_opts = [(6, 0.25), (8, 0.50), (8, 0.25), (12, 0.50), (12, 0.25)]
    partial_tp_ts = [None, (0.50, 0.50)]

    for bars, factor in time_stop_opts:
        for ai in ai_thresholds:
            for partial in partial_tp_ts:
                name = f"ERR{idx:03d}_ts{bars}f{int(factor * 100)}_ai{int(ai * 100)}"
                desc_parts = [f"ai={ai}", f"timeStop={bars}bars@{factor}×"]
                if partial is not None:
                    frac, tgt = partial
                    name += f"_p{int(frac * 100)}t{int(tgt * 100)}"
                    desc_parts.append(f"partTP={int(frac*100)}%@{int(tgt*100)}%")

                variants.append(VariantConfig(
                    name=name,
                    main_ai_threshold=ai,
                    time_stop_bars=bars,
                    time_stop_factor=factor,
                    partial_tp_fraction=partial[0] if partial else None,
                    partial_tp_target_pct=partial[1] if partial else 0.50,
                    description=", ".join(desc_parts),
                ))
                idx += 1

    return variants


EXIT_RR_FULL_VARIANTS: List[VariantConfig] = _exit_rr_full_variants()

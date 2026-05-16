"""Fee and slippage calculations.

All costs are applied per side:
  - Fee:      0.005 % of trade value (each leg)
  - Slippage: 3 bps of trade value (each leg)
"""

FEE_PCT = 0.005 / 100    # per leg
SLIPPAGE_BPS = 3         # basis points per leg


def entry_price_after_slippage(raw_price: float, direction: int) -> float:
    """Long buy: pay more.  Short sell: receive less."""
    return raw_price * (1 + direction * SLIPPAGE_BPS / 10_000)


def exit_price_after_slippage(raw_price: float, direction: int) -> float:
    """Long sell: receive less.  Short buy: pay more."""
    return raw_price * (1 - direction * SLIPPAGE_BPS / 10_000)


def calc_fees(shares: int, entry: float, exit_price: float) -> float:
    """Two-leg commission."""
    return shares * (entry + exit_price) * FEE_PCT


def calc_slippage_cost(shares: int, raw_entry: float, raw_exit: float) -> float:
    """Total dollar value of slippage on both legs (direction-independent)."""
    return shares * (raw_entry + raw_exit) * SLIPPAGE_BPS / 10_000

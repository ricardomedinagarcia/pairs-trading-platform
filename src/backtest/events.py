"""Event types for the event-driven backtester.

Each event represents one of the four phases of a trading bar:
1. MarketDataEvent: new price data arrives
2. SignalEvent: strategy decides a trading idea (direction + strength)
3. OrderEvent: portfolio decides to act on the signal (sized + risk-managed)
4. FillEvent: execution simulator returns the actual fill (price + costs)

The engine processes these in strict chronological order, ensuring the
strategy only sees data that would have been available in real time.

Why dataclasses: lightweight, immutable-by-default, free __repr__ and
__eq__. Suitable for value objects passed between components.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderDirection(Enum):
    """Direction of an order or position."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass(frozen=True)
class MarketDataEvent:
    """A new bar of price data is available.

    For pairs trading, this carries both legs' prices at the same timestamp.
    """
    timestamp: datetime
    ticker1: str
    price1: float       # adjusted close of ticker1
    ticker2: str
    price2: float       # adjusted close of ticker2


@dataclass(frozen=True)
class SignalEvent:
    """Strategy's view: should we be long the spread, short, or flat?

    z_score: how far the current spread is from its rolling mean, in
        rolling-std units. Strategy decides based on this.
    target_direction: where the strategy wants to be (LONG/SHORT/FLAT).
    """
    timestamp: datetime
    ticker1: str
    ticker2: str
    z_score: float
    target_direction: OrderDirection


@dataclass(frozen=True)
class OrderEvent:
    """Portfolio decision: trade these specific quantities.

    A positive quantity is a buy; negative is a sell. For pairs:
        - LONG spread = buy ticker1, sell ticker2 (hedge ratio determines ratio)
        - SHORT spread = sell ticker1, buy ticker2

    quantity1, quantity2 are signed shares for each leg.
    """
    timestamp: datetime
    ticker1: str
    quantity1: float
    ticker2: str
    quantity2: float


@dataclass(frozen=True)
class FillEvent:
    """Execution simulator's report: actual fill prices and costs.

    fill_price includes slippage; total commission is reported separately.
    """
    timestamp: datetime
    ticker1: str
    quantity1: float          # signed (positive=long, negative=short)
    fill_price1: float
    ticker2: str
    quantity2: float
    fill_price2: float
    commission: float         # total commission across both legs
    slippage_cost: float      # total slippage estimate (already in fill prices,
                              # reported separately for accounting/analysis)
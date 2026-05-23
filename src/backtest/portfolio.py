"""Portfolio: tracks cash, positions, and equity over the backtest.

Core accounting identity:
    equity[t] = cash[t] + sum_i (position[i] * price[i, t])

This identity must hold at every step. Tests verify it.

Position sizing approach: hedge-ratio (β) based. For a pair (t1, t2)
with hedge ratio β, "1 unit of spread" means (long 1 share t1) +
(short β shares t2). Notional exposure determines how many units to
hold. Default notional is $10,000 per pair position, configurable.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.backtest.events import FillEvent, OrderDirection, OrderEvent, SignalEvent


@dataclass
class PortfolioConfig:
    """Portfolio behavior parameters."""

    initial_capital: float = 100_000.0
    notional_per_pair: float = 10_000.0  # target dollar exposure on ticker1 leg
    allow_short_proceeds_as_cash: bool = True  # treat short sale proceeds as cash


@dataclass
class Portfolio:
    """Portfolio state and event handlers.

    Maintains running cash, positions, and equity time series. Generates
    OrderEvents in response to SignalEvents; applies FillEvents to update
    state.
    """

    config: PortfolioConfig = field(default_factory=PortfolioConfig)
    hedge_ratio: float = 1.0  # set per pair on initialization

    cash: float = field(init=False)
    positions: dict[str, float] = field(init=False, default_factory=dict)
    current_direction: OrderDirection = field(init=False, default=OrderDirection.FLAT)
    equity_history: list[tuple[datetime, float]] = field(init=False, default_factory=list)
    trades: list[FillEvent] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.config.initial_capital

    # ------------------------------------------------------------------
    # Equity accounting
    # ------------------------------------------------------------------

    def mark_to_market(
        self, timestamp: datetime, price1: float, price2: float, ticker1: str, ticker2: str
    ) -> float:
        """Compute current equity and append to history.

        equity = cash + position_value across all tickers.
        """
        position_value = (
            self.positions.get(ticker1, 0.0) * price1
            + self.positions.get(ticker2, 0.0) * price2
        )
        equity = self.cash + position_value
        self.equity_history.append((timestamp, equity))
        return equity

    # ------------------------------------------------------------------
    # Signal -> Order translation
    # ------------------------------------------------------------------

    def on_signal(
        self,
        signal: SignalEvent,
        price1: float,
        price2: float,
    ) -> OrderEvent | None:
        """Decide what to trade in response to a signal.

        Returns None if no trade is needed (already in target direction).

        Position sizing: target $notional_per_pair of exposure on ticker1.
        Compute shares of ticker1, then β-scale to get shares of ticker2.

        Direction interpretation for the SPREAD = ticker1 - β * ticker2:
            LONG spread  = buy ticker1, sell β shares of ticker2
            SHORT spread = sell ticker1, buy β shares of ticker2
            FLAT         = exit any existing position
        """
        if signal.target_direction == self.current_direction:
            return None  # already where we want to be

        # First, generate orders to close any existing position
        close_qty1 = -self.positions.get(signal.ticker1, 0.0)
        close_qty2 = -self.positions.get(signal.ticker2, 0.0)

        # Then, orders to open the new target position (if any)
        if signal.target_direction == OrderDirection.FLAT:
            open_qty1, open_qty2 = 0.0, 0.0
        else:
            sign = signal.target_direction.value  # +1 for LONG, -1 for SHORT
            shares_t1 = self.config.notional_per_pair / price1
            shares_t2 = self.hedge_ratio * shares_t1
            open_qty1 = sign * shares_t1
            open_qty2 = -sign * shares_t2  # opposite leg

        total_qty1 = close_qty1 + open_qty1
        total_qty2 = close_qty2 + open_qty2

        if total_qty1 == 0 and total_qty2 == 0:
            return None

        return OrderEvent(
            timestamp=signal.timestamp,
            ticker1=signal.ticker1,
            quantity1=total_qty1,
            ticker2=signal.ticker2,
            quantity2=total_qty2,
        )

    # ------------------------------------------------------------------
    # Fill application
    # ------------------------------------------------------------------

    def on_fill(self, fill: FillEvent, new_direction: OrderDirection) -> None:
        """Apply a fill: update positions, cash, and current direction."""
        # Update positions (signed quantities)
        self.positions[fill.ticker1] = (
            self.positions.get(fill.ticker1, 0.0) + fill.quantity1
        )
        self.positions[fill.ticker2] = (
            self.positions.get(fill.ticker2, 0.0) + fill.quantity2
        )

        # Cash flow: buys debit cash, sells credit cash
        # quantity * price gives signed cash flow (negative for buys)
        cash_flow = -(
            fill.quantity1 * fill.fill_price1 + fill.quantity2 * fill.fill_price2
        )
        self.cash += cash_flow
        self.cash -= fill.commission  # always a cost

        # Update direction state
        self.current_direction = new_direction

        # Record the trade
        self.trades.append(fill)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def equity_series(self) -> pd.Series:
        """Return equity history as a pandas Series indexed by timestamp."""
        if not self.equity_history:
            return pd.Series(dtype=float)
        timestamps, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="equity")

    def trade_log(self) -> pd.DataFrame:
        """Return all fills as a DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])
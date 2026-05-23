"""Event-driven backtest engine.

Loops over market data in chronological order, invoking the strategy,
portfolio, and execution components in sequence at each bar. Maintains
an equity time series that becomes the primary input to performance
analysis.

Order of operations within a single bar:
    1. Mark-to-market at the previous bar's positions (using new prices).
    2. Strategy sees the new bar -> may emit a signal.
    3. Portfolio translates signal to order.
    4. Execution simulator fills the order at the current bar's close.
    5. Portfolio applies the fill.

Step 1 happens BEFORE the trade so the equity series shows the
mark-to-market of yesterday's position at today's prices, before any
new action. This separates "P&L from holding" from "P&L impact of
trading," which makes performance attribution cleaner.

KNOWN SIMPLIFICATION: signal and execution use the same bar's close.
Real-world implementations would generate signals at close N and execute
at open N+1 to avoid the closing-auction price feasibility question.
This is a standard backtest simplification and is documented as a
limitation.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.backtest.events import MarketDataEvent, OrderDirection
from src.backtest.execution import ExecutionSimulator
from src.backtest.portfolio import Portfolio
from src.backtest.strategy import PairStrategy

@dataclass
class BacktestResult:
    """Final state and metrics from a backtest run."""

    equity: pd.Series
    trades: pd.DataFrame
    portfolio: Portfolio
    strategy: PairStrategy
    n_bars: int
    n_signals: int
    n_fills: int


@dataclass
class BacktestEngine:
    """Run a single-pair backtest from a price DataFrame.

    Inputs:
        prices: wide-format DataFrame with columns including ticker1, ticker2,
                indexed by date.
        strategy: PairStrategy instance (carries the cointegrating params).
        portfolio: Portfolio instance (carries the hedge ratio matching strategy).
        execution: ExecutionSimulator (transaction cost model).

    Run via .run() -> BacktestResult.
    """

    prices: pd.DataFrame
    strategy: PairStrategy
    portfolio: Portfolio
    execution: ExecutionSimulator

    n_signals: int = field(init=False, default=0)
    n_fills: int = field(init=False, default=0)

    def run(self) -> BacktestResult:
        """Execute the event loop bar by bar."""
        # Validate inputs
        if self.strategy.ticker1 not in self.prices.columns:
            raise ValueError(f"Price data missing column: {self.strategy.ticker1}")
        if self.strategy.ticker2 not in self.prices.columns:
            raise ValueError(f"Price data missing column: {self.strategy.ticker2}")

        # Sanity-check hedge ratio matches between strategy and portfolio
        if abs(self.strategy.hedge_ratio - self.portfolio.hedge_ratio) > 1e-9:
            raise ValueError(
                f"Strategy hedge ratio ({self.strategy.hedge_ratio}) and "
                f"portfolio hedge ratio ({self.portfolio.hedge_ratio}) don't match"
            )

        t1, t2 = self.strategy.ticker1, self.strategy.ticker2
        df = self.prices[[t1, t2]].dropna().sort_index()
        n_bars = len(df)

        for idx, (timestamp, row) in enumerate(df.iterrows()):
            price1 = float(row[t1])
            price2 = float(row[t2])
            ts = self._coerce_timestamp(timestamp)

            # Step 1: mark to market at new prices BEFORE acting
            self.portfolio.mark_to_market(ts, price1, price2, t1, t2)

            # Step 2: strategy decides
            signal = self.strategy.on_market_data(ts, price1, price2)
            if signal is None:
                continue
            self.n_signals += 1

            # Step 3: portfolio sizes the order
            order = self.portfolio.on_signal(signal, price1, price2)
            if order is None:
                continue

            # Step 4: execution simulator fills
            fill = self.execution.fill_order(order, price1, price2)

            # Step 5: portfolio applies the fill
            self.portfolio.on_fill(fill, new_direction=signal.target_direction)
            self.n_fills += 1

        return BacktestResult(
            equity=self.portfolio.equity_series(),
            trades=self.portfolio.trade_log(),
            portfolio=self.portfolio,
            strategy=self.strategy,
            n_bars=n_bars,
            n_signals=self.n_signals,
            n_fills=self.n_fills,
        )

    @staticmethod
    def _coerce_timestamp(ts) -> datetime:
        """Normalize a pandas Timestamp or datetime.date to datetime."""
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        if isinstance(ts, datetime):
            return ts
        # date object
        return datetime.combine(ts, datetime.min.time())
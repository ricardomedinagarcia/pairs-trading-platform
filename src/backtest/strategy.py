"""Mean-reversion strategy for pairs trading.

The strategy maintains a rolling estimate of the spread's mean and standard
deviation, computes the current z-score, and emits LONG/SHORT/FLAT signals
based on standard thresholds.

CRITICAL: All rolling statistics use shift(1) before .rolling() to ensure
the value at time t depends only on data through t-1. Without this, the
strategy uses 'present' information at decision time, creating one-bar
look-ahead bias that silently inflates backtests.

Strategy state is rebuilt from scratch on each new MarketDataEvent so that
the engine can replay events in chronological order without history
accumulating implicitly. The strategy is a pure function of historical
prices through (and including) the current bar.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.events import OrderDirection, SignalEvent


@dataclass
class StrategyConfig:
    """Strategy parameters."""

    lookback: int = 60          # bars used for rolling z-score
    entry_z: float = 2.0        # |z| > entry_z triggers entry
    exit_z: float = 0.5         # |z| < exit_z triggers exit
    min_history: int = 60       # minimum bars before generating signals


@dataclass
class PairStrategy:
    """Mean-reversion strategy for one cointegrated pair.

    Maintains the hedge ratio (β) and intercept (α) from the cointegration
    regression as fixed parameters — these were estimated on training data
    and frozen for the backtest. Rolling z-score statistics are computed
    bar-by-bar.
    """

    ticker1: str
    ticker2: str
    hedge_ratio: float           # β from cointegration regression
    intercept: float             # α from cointegration regression
    config: StrategyConfig

    # Internal state: price history
    _prices1: list[float] = None  # type: ignore[assignment]
    _prices2: list[float] = None  # type: ignore[assignment]
    _timestamps: list = None  # type: ignore[assignment]
    _current_direction: OrderDirection = OrderDirection.FLAT

    def __post_init__(self) -> None:
        self._prices1 = []
        self._prices2 = []
        self._timestamps = []

    def on_market_data(
        self, timestamp, price1: float, price2: float
    ) -> SignalEvent | None:
        """Process a new bar and possibly emit a signal.

        Returns:
            A SignalEvent if the target direction should change, else None.
        """
        self._timestamps.append(timestamp)
        self._prices1.append(price1)
        self._prices2.append(price2)

        if len(self._prices1) < self.config.min_history + 1:
            # Not enough history to compute a z-score with shift(1)
            return None

        z = self._compute_z_score()
        if z is None:
            return None

        target = self._z_to_direction(z)
        if target == self._current_direction:
            return None  # no change; no signal

        signal = SignalEvent(
            timestamp=timestamp,
            ticker1=self.ticker1,
            ticker2=self.ticker2,
            z_score=z,
            target_direction=target,
        )
        self._current_direction = target
        return signal

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_z_score(self) -> float | None:
        """Z-score of the current spread vs rolling mean/std using ONLY
        prior bars (shift-by-one to prevent look-ahead bias)."""
        p1 = np.array(self._prices1, dtype=float)
        p2 = np.array(self._prices2, dtype=float)
        spread = np.log(p1) - self.hedge_ratio * np.log(p2) - self.intercept

        # Current spread is at index -1. Use bars [-lookback-1 : -1] for stats
        # (excludes the current bar — that's the shift(1) discipline).
        window = spread[-self.config.lookback - 1 : -1]
        if len(window) < self.config.lookback:
            return None

        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        if sigma <= 0:
            return None

        z = (spread[-1] - mu) / sigma
        return float(z)

    def _z_to_direction(self, z: float) -> OrderDirection:
        """Translate z-score and current direction into a target direction.

        Hysteresis: once in a position, we hold until z reverts past
        exit_z (not all the way to zero), to avoid whipsaws.
        """
        cfg = self.config

        if self._current_direction == OrderDirection.FLAT:
            if z > cfg.entry_z:
                return OrderDirection.SHORT  # spread too high; bet on reversion
            if z < -cfg.entry_z:
                return OrderDirection.LONG   # spread too low; bet on reversion up
            return OrderDirection.FLAT

        if self._current_direction == OrderDirection.LONG:
            # Exit long if z has risen back toward zero
            if z > -cfg.exit_z:
                return OrderDirection.FLAT
            return OrderDirection.LONG

        if self._current_direction == OrderDirection.SHORT:
            # Exit short if z has fallen back toward zero
            if z < cfg.exit_z:
                return OrderDirection.FLAT
            return OrderDirection.SHORT

        return OrderDirection.FLAT
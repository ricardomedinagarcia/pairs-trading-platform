"""Execution simulator: turns orders into fills with realistic costs.

Models three sources of friction:
1. Commission: fixed cost per share (e.g., $0.005/share for Interactive Brokers)
2. Half-spread: assumes we cross the bid-ask spread on every trade
3. Slippage: market impact + adverse selection, modeled as % of trade value

These are conservative defaults; production code would model these
empirically from intraday data. For daily-bar backtesting they're
reasonable approximations.
"""

from dataclasses import dataclass

from src.backtest.events import FillEvent, OrderEvent


@dataclass
class ExecutionCosts:
    """Transaction cost parameters.

    All slippage components combine multiplicatively on the trade direction
    so that buys are filled at slightly higher prices than the reference,
    and sells at slightly lower.
    """
    commission_per_share: float = 0.005    # $0.005/share, IB-style
    half_spread_bps: float = 5.0           # 5 basis points = 0.05% per side
    market_impact_bps: float = 2.0         # 2 basis points additional slippage

    @property
    def total_slippage_bps(self) -> float:
        """Total one-way slippage in basis points (1 bp = 0.01%)."""
        return self.half_spread_bps + self.market_impact_bps


class ExecutionSimulator:
    """Fills orders at reference prices adjusted for slippage and commission.

    Reference prices for backtest are typically the bar's close. Real systems
    would use bid/ask quotes or next-bar opens to avoid look-ahead at the
    current bar's close. This is a simplification.
    """

    def __init__(self, costs: ExecutionCosts | None = None) -> None:
        self.costs = costs or ExecutionCosts()

    def fill_order(
        self,
        order: OrderEvent,
        reference_price1: float,
        reference_price2: float,
    ) -> FillEvent:
        """Convert an order to a fill at slipped prices.

        Args:
            order: The order to fill.
            reference_price1: Reference price for ticker1 (typically bar close).
            reference_price2: Reference price for ticker2.

        Returns:
            FillEvent with actual fill prices, total commission, and slippage cost.
        """
        slip_factor = self.costs.total_slippage_bps / 10_000  # bps to fraction

        # Buys fill above reference; sells fill below.
        fill_price1 = self._apply_slippage(reference_price1, order.quantity1, slip_factor)
        fill_price2 = self._apply_slippage(reference_price2, order.quantity2, slip_factor)

        commission = (
            abs(order.quantity1) * self.costs.commission_per_share
            + abs(order.quantity2) * self.costs.commission_per_share
        )

        # Slippage cost = (fill_price - reference_price) * quantity, summed
        slippage_cost = (
            (fill_price1 - reference_price1) * order.quantity1
            + (fill_price2 - reference_price2) * order.quantity2
        )

        return FillEvent(
            timestamp=order.timestamp,
            ticker1=order.ticker1,
            quantity1=order.quantity1,
            fill_price1=fill_price1,
            ticker2=order.ticker2,
            quantity2=order.quantity2,
            fill_price2=fill_price2,
            commission=commission,
            slippage_cost=slippage_cost,
        )

    @staticmethod
    def _apply_slippage(reference: float, quantity: float, slip_factor: float) -> float:
        """Adjust price against the trade direction.

        A buy (quantity > 0) fills at reference * (1 + slip).
        A sell (quantity < 0) fills at reference * (1 - slip).
        A zero quantity returns the reference unchanged.
        """
        if quantity > 0:
            return reference * (1.0 + slip_factor)
        elif quantity < 0:
            return reference * (1.0 - slip_factor)
        return reference
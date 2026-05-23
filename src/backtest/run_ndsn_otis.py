"""Run the backtest on NDSN/OTIS — our FDR-validated cointegrated pair.

Loads prices from DuckDB, instantiates strategy/portfolio/execution with
parameters from the screening (hedge ratio 0.649, intercept 2.573), and
runs the event-driven engine over the full 2015-2024 window.

This is the in-sample backtest. Results are diagnostic only -- the
strategy parameters and the pair selection both came from the same data,
so any positive performance here is partly a function of selection bias.
The honest test will be out-of-sample (Phase 3c.2 onward).
"""

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.execution import ExecutionCosts, ExecutionSimulator
from src.backtest.portfolio import Portfolio, PortfolioConfig
from src.backtest.strategy import PairStrategy, StrategyConfig
from src.data.storage import get_connection, load_prices


# Parameters from screening_v2 output for NDSN/OTIS
HEDGE_RATIO = 0.649471
INTERCEPT = 2.572748
TICKER1 = "NDSN"
TICKER2 = "OTIS"


def main() -> None:
    print(f"Backtesting {TICKER1}/{TICKER2} over 2015-2024 (in-sample, full window)")
    print(f"Cointegration parameters: β={HEDGE_RATIO}, α={INTERCEPT}")
    print()

    # Load prices
    conn = get_connection("data/market.duckdb")
    long_df = load_prices(conn, tickers=[TICKER1, TICKER2], start="2015-01-01")
    long_df["date"] = pd.to_datetime(long_df["date"])
    conn.close()

    # Pivot to wide format for the engine
    wide = long_df.pivot(index="date", columns="ticker", values="adj_close")
    print(f"Loaded {len(wide)} bars from {wide.index.min().date()} to {wide.index.max().date()}")
    print(f"NDSN obs: {wide[TICKER1].notna().sum()}, OTIS obs: {wide[TICKER2].notna().sum()}")
    print()

    # Instantiate components
    strategy = PairStrategy(
        ticker1=TICKER1,
        ticker2=TICKER2,
        hedge_ratio=HEDGE_RATIO,
        intercept=INTERCEPT,
        config=StrategyConfig(lookback=60, entry_z=2.0, exit_z=0.5),
    )
    portfolio = Portfolio(
        config=PortfolioConfig(initial_capital=100_000, notional_per_pair=10_000),
        hedge_ratio=HEDGE_RATIO,
    )
    execution = ExecutionSimulator(ExecutionCosts())

    engine = BacktestEngine(
        prices=wide,
        strategy=strategy,
        portfolio=portfolio,
        execution=execution,
    )

    result = engine.run()

    # Summary
    print(f"{'='*60}")
    print(f"Backtest complete")
    print(f"{'='*60}")
    print(f"Bars processed:   {result.n_bars}")
    print(f"Signals emitted:  {result.n_signals}")
    print(f"Fills executed:   {result.n_fills}")
    print(f"Initial equity:   ${100_000:,.2f}")
    print(f"Final equity:     ${result.equity.iloc[-1]:,.2f}")
    print(f"Total return:     {(result.equity.iloc[-1] / 100_000 - 1) * 100:+.2f}%")
    print()

    # Quick performance stats
    returns = result.equity.pct_change().dropna()
    if len(returns) > 0:
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * (252 ** 0.5)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        rolling_max = result.equity.cummax()
        drawdown = (result.equity - rolling_max) / rolling_max
        max_dd = drawdown.min()
        print(f"Annualized return: {ann_return*100:+.2f}%")
        print(f"Annualized vol:    {ann_vol*100:.2f}%")
        print(f"Sharpe ratio:      {sharpe:.2f}")
        print(f"Max drawdown:      {max_dd*100:.2f}%")

    # Save the equity series and trade log
    result.equity.to_csv("data/backtest_ndsn_otis_equity.csv", header=True)
    result.trades.to_csv("data/backtest_ndsn_otis_trades.csv", index=False)
    print()
    print("Saved: data/backtest_ndsn_otis_equity.csv")
    print("Saved: data/backtest_ndsn_otis_trades.csv")


if __name__ == "__main__":
    main()
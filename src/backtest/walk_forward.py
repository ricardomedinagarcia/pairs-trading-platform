"""Walk-forward analysis: rolling re-estimation with continuous out-of-sample
evaluation.

The train/test split in Phase 3c.2 produced one out-of-sample verdict.
Walk-forward produces many: it re-estimates the cointegration parameters
on each trailing window, applies them to the next test window, then slides
forward. The concatenated test-window returns form a continuous out-of-
sample equity curve.

Window mechanics:
    | training (24mo) | test (6mo) |
                              | training (24mo, shifted) | test (6mo) |
                                                      | training (24mo) | test |
                                                                       ...

At every test window, parameters were fit on data that ended BEFORE the
test window started. No look-ahead. Real trading desks operate this way:
periodic re-estimation on the trailing window.

Output:
- Per-step parameter estimates (track β drift over time)
- Per-step out-of-sample metrics (Sharpe, win rate, return)
- Concatenated out-of-sample equity curve
- Aggregate out-of-sample Sharpe
"""

from dataclasses import dataclass, field

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.execution import ExecutionCosts, ExecutionSimulator
from src.backtest.portfolio import Portfolio, PortfolioConfig
from src.backtest.strategy import PairStrategy, StrategyConfig
from src.stats.cointegration import engle_granger_test


@dataclass
class WalkForwardStep:
    """One iteration of the walk-forward loop."""

    step_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    test_equity_start: float    # starting equity for this test window
    test_equity_end: float
    test_return: float           # period return (not annualized)
    test_sharpe: float           # period Sharpe annualized
    test_n_trades: int
    test_equity_series: pd.Series  # the equity curve for this window


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward output."""

    steps: list[WalkForwardStep] = field(default_factory=list)
    ticker1: str = ""
    ticker2: str = ""
    train_window_months: int = 24
    test_window_months: int = 6

    def stitched_equity(self) -> pd.Series:
        """Concatenate all test-window equity curves into one continuous series.

        Each step's curve is rebased so the stitching is continuous: the
        starting equity of step N equals the ending equity of step N-1.
        """
        if not self.steps:
            return pd.Series(dtype=float)

        pieces = []
        running_equity = self.steps[0].test_equity_start
        for step in self.steps:
            # Rebase this step's equity to start from running_equity
            equity = step.test_equity_series.copy()
            initial_equity = equity.iloc[0]
            if initial_equity == 0:
                continue
            scale = running_equity / initial_equity
            equity = equity * scale
            pieces.append(equity)
            running_equity = equity.iloc[-1]

        return pd.concat(pieces)

    def parameter_history(self) -> pd.DataFrame:
        """Return per-step parameters as a DataFrame for plotting."""
        rows = []
        for s in self.steps:
            rows.append({
                "step": s.step_index,
                "train_end": s.train_end,
                "test_start": s.test_start,
                "test_end": s.test_end,
                "hedge_ratio": s.hedge_ratio,
                "intercept": s.intercept,
                "adf_statistic": s.adf_statistic,
                "test_return": s.test_return,
                "test_sharpe": s.test_sharpe,
                "test_n_trades": s.test_n_trades,
            })
        return pd.DataFrame(rows)


def run_walk_forward(
    prices: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    train_window_months: int = 24,
    test_window_months: int = 6,
    initial_capital: float = 100_000.0,
    strategy_config: StrategyConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
    execution_costs: ExecutionCosts | None = None,
) -> WalkForwardResult:
    """Run walk-forward analysis on a pair.

    Args:
        prices: Wide-format DataFrame with both ticker columns.
        ticker1, ticker2: Pair tickers.
        train_window_months: Trailing window for parameter estimation.
        test_window_months: Forward window for out-of-sample testing.
        initial_capital: Capital for the first test step; subsequent steps
            inherit the running equity from the previous step.

    Returns:
        WalkForwardResult with per-step metrics and a stitched equity curve.
    """
    strategy_config = strategy_config or StrategyConfig()
    portfolio_config_template = portfolio_config or PortfolioConfig(initial_capital=initial_capital)
    execution_costs = execution_costs or ExecutionCosts()

    # Restrict to dates where both tickers have data
    df = prices[[ticker1, ticker2]].dropna().sort_index()
    if len(df) == 0:
        raise ValueError("No overlapping price history between the two tickers")

    first_date = df.index.min()
    last_date = df.index.max()

    result = WalkForwardResult(
        ticker1=ticker1,
        ticker2=ticker2,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
    )

    # Walk forward through time
    step_idx = 0
    current_train_end = first_date + pd.DateOffset(months=train_window_months)
    running_equity = initial_capital

    while current_train_end + pd.DateOffset(months=test_window_months) <= last_date + pd.DateOffset(days=1):
        train_start = current_train_end - pd.DateOffset(months=train_window_months)
        train_end = current_train_end
        test_start = current_train_end
        test_end = current_train_end + pd.DateOffset(months=test_window_months)

        # Slice
        train_slice = df.loc[(df.index >= train_start) & (df.index < train_end)]
        test_slice = df.loc[(df.index >= test_start) & (df.index < test_end)]

        if len(train_slice) < 100 or len(test_slice) < 30:
            # Not enough data this step; skip but advance
            current_train_end = current_train_end + pd.DateOffset(months=test_window_months)
            continue

        # Fit parameters on train slice
        try:
            coint = engle_granger_test(
                train_slice[ticker1],
                train_slice[ticker2],
                ticker1=ticker1,
                ticker2=ticker2,
            )
        except Exception as e:
            print(f"  Step {step_idx}: cointegration regression failed ({e}); skipping")
            current_train_end = current_train_end + pd.DateOffset(months=test_window_months)
            continue

        hedge_ratio = coint.hedge_ratio
        intercept = coint.intercept
        adf_stat = coint.adf_statistic

        # Build strategy/portfolio for the test slice
        test_portfolio_cfg = PortfolioConfig(
            initial_capital=running_equity,
            notional_per_pair=portfolio_config_template.notional_per_pair,
        )
        strategy = PairStrategy(
            ticker1=ticker1, ticker2=ticker2,
            hedge_ratio=hedge_ratio, intercept=intercept,
            config=strategy_config,
        )
        portfolio = Portfolio(config=test_portfolio_cfg, hedge_ratio=hedge_ratio)

        # WARMUP: strategy needs lookback bars before it can signal. Prepend
        # the trailing lookback days from the train slice so the strategy
        # has sufficient price history when test_slice begins.
        warmup_bars = strategy_config.lookback + 5
        warmup_slice = train_slice.tail(warmup_bars)
        combined = pd.concat([warmup_slice, test_slice])

        engine = BacktestEngine(
            prices=combined,
            strategy=strategy,
            portfolio=portfolio,
            execution=ExecutionSimulator(execution_costs),
        )
        bt_result = engine.run()

        # Slice the equity history to only the test window (drop warmup bars)
        test_equity = bt_result.equity.loc[bt_result.equity.index >= test_start]
        if len(test_equity) < 2:
            current_train_end = current_train_end + pd.DateOffset(months=test_window_months)
            continue

        test_start_eq = test_equity.iloc[0]
        test_end_eq = test_equity.iloc[-1]
        test_return = (test_end_eq / test_start_eq) - 1.0

        returns = test_equity.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            test_sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
        else:
            test_sharpe = 0.0

        # Count fills that occurred during the test window only
        n_test_trades = sum(
            1 for fill in bt_result.portfolio.trades
            if fill.timestamp >= test_start
        )

        step = WalkForwardStep(
            step_index=step_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            adf_statistic=adf_stat,
            test_equity_start=test_start_eq,
            test_equity_end=test_end_eq,
            test_return=test_return,
            test_sharpe=test_sharpe,
            test_n_trades=n_test_trades,
            test_equity_series=test_equity,
        )
        result.steps.append(step)

        running_equity = test_end_eq  # carry forward
        step_idx += 1
        current_train_end = current_train_end + pd.DateOffset(months=test_window_months)

    return result


def summarize_walk_forward(result: WalkForwardResult) -> None:
    """Print per-step and aggregate metrics."""
    if not result.steps:
        print("No walk-forward steps completed (insufficient data).")
        return

    print(f"\n{'='*78}")
    print(f"Walk-Forward Analysis: {result.ticker1}/{result.ticker2}")
    print(f"{'='*78}")
    print(f"Training window: {result.train_window_months} months")
    print(f"Test window:     {result.test_window_months} months")
    print(f"Total steps:     {len(result.steps)}")

    print(f"\n{'Step':<5} {'Test Period':<25} {'β':>7} {'ADF':>7} {'Trades':>7} {'Return':>9} {'Sharpe':>8}")
    print("-" * 78)
    for s in result.steps:
        period = f"{s.test_start.date()} to {s.test_end.date()}"
        print(
            f"{s.step_index:<5} {period:<25} "
            f"{s.hedge_ratio:>7.3f} {s.adf_statistic:>7.2f} "
            f"{s.test_n_trades:>7} {s.test_return*100:>+8.2f}% {s.test_sharpe:>8.2f}"
        )

    # Aggregate stats
    stitched = result.stitched_equity()
    if len(stitched) > 1:
        returns = stitched.pct_change().dropna()
        agg_total_return = (stitched.iloc[-1] / stitched.iloc[0]) - 1.0
        agg_ann_return = (1 + returns.mean()) ** 252 - 1
        agg_ann_vol = returns.std() * (252 ** 0.5)
        agg_sharpe = agg_ann_return / agg_ann_vol if agg_ann_vol > 0 else 0.0
        rolling_max = stitched.cummax()
        max_dd = ((stitched - rolling_max) / rolling_max).min()

        print(f"\n{'='*78}")
        print("Aggregate (stitched) out-of-sample metrics:")
        print(f"{'='*78}")
        print(f"Total return:            {agg_total_return*100:+.2f}%")
        print(f"Annualized return:       {agg_ann_return*100:+.2f}%")
        print(f"Annualized volatility:   {agg_ann_vol*100:.2f}%")
        print(f"Aggregate Sharpe:        {agg_sharpe:.2f}")
        print(f"Max drawdown:            {max_dd*100:.2f}%")
        print(f"Total trades (all OOS):  {sum(s.test_n_trades for s in result.steps)}")

    # Parameter drift
    params = result.parameter_history()
    if len(params) > 1:
        print(f"\nParameter drift across steps:")
        print(f"  β range: [{params['hedge_ratio'].min():.3f}, {params['hedge_ratio'].max():.3f}]")
        print(f"  α range: [{params['intercept'].min():.3f}, {params['intercept'].max():.3f}]")
        print(f"  ADF range: [{params['adf_statistic'].min():.2f}, {params['adf_statistic'].max():.2f}]")


if __name__ == "__main__":
    from src.data.storage import get_connection, load_prices

    TICKER1, TICKER2 = "NDSN", "OTIS"

    conn = get_connection("data/market.duckdb")
    long_df = load_prices(conn, tickers=[TICKER1, TICKER2], start="2015-01-01")
    long_df["date"] = pd.to_datetime(long_df["date"])
    conn.close()
    wide = long_df.pivot(index="date", columns="ticker", values="adj_close")

    print(f"Running walk-forward on {TICKER1}/{TICKER2}")
    result = run_walk_forward(
        wide, TICKER1, TICKER2,
        train_window_months=24,
        test_window_months=6,
    )
    summarize_walk_forward(result)

    # Save outputs
    result.parameter_history().to_csv("data/walkforward_steps.csv", index=False)
    result.stitched_equity().to_csv("data/walkforward_equity.csv", header=True)
    print("\nSaved: data/walkforward_steps.csv, data/walkforward_equity.csv")
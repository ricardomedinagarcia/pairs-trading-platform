"""Train/test split for honest out-of-sample evaluation.

The Phase 3c.1 backtest used the full window for both parameter estimation
AND performance evaluation, which inflates the apparent edge. This module
splits the data by date, fits parameters on the train portion only, then
applies them frozen to the test portion. The test-period Sharpe is the
honest performance estimate.

Workflow:
    1. split_prices(...)       -> (train_df, test_df)
    2. fit_pair_parameters(train_df, t1, t2) -> hedge_ratio, intercept
    3. run_split_backtest(...) -> {train: BacktestResult, test: BacktestResult}
    4. summarize_split(...)    -> printed comparison

Key methodological choice: parameters are estimated ONLY on the train
window. The test window then uses these frozen parameters, simulating
the experience of a researcher who fit a model in (say) 2023 and is
now evaluating its 2024 performance.
"""

from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.execution import ExecutionCosts, ExecutionSimulator
from src.backtest.portfolio import Portfolio, PortfolioConfig
from src.backtest.strategy import PairStrategy, StrategyConfig
from src.stats.cointegration import engle_granger_test

@dataclass
class SplitResult:
    """Container for train and test backtest outputs side-by-side."""

    train_result: BacktestResult
    test_result: BacktestResult
    hedge_ratio: float
    intercept: float
    cointegration_adf: float       # ADF stat on train residuals
    split_date: pd.Timestamp
    n_train_bars: int
    n_test_bars: int


def split_prices(
    prices: pd.DataFrame,
    split_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a wide-format price DataFrame chronologically.

    Args:
        prices: Wide-format DataFrame indexed by date.
        split_date: Cutoff. Train = dates strictly before split_date,
            test = dates on or after split_date.

    Returns:
        (train_df, test_df) tuple.
    """
    cutoff = pd.Timestamp(split_date)
    train = prices.loc[prices.index < cutoff]
    test = prices.loc[prices.index >= cutoff]
    return train, test


def fit_pair_parameters(
    train_prices: pd.DataFrame,
    ticker1: str,
    ticker2: str,
) -> tuple[float, float, float]:
    """Fit cointegration parameters on train data only.

    Args:
        train_prices: Wide-format prices containing ticker1 and ticker2 columns.
        ticker1, ticker2: Names of the two legs.

    Returns:
        (hedge_ratio, intercept, adf_statistic_on_residuals)
    """
    p1 = train_prices[ticker1].dropna()
    p2 = train_prices[ticker2].dropna()
    result = engle_granger_test(p1, p2, ticker1=ticker1, ticker2=ticker2)
    return result.hedge_ratio, result.intercept, result.adf_statistic


def run_split_backtest(
    prices: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    split_date: str | pd.Timestamp,
    strategy_config: StrategyConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
    execution_costs: ExecutionCosts | None = None,
) -> SplitResult:
    """Run train backtest, refit-frozen test backtest, return both.

    Args:
        prices: Wide-format DataFrame with ticker1 and ticker2 columns.
        ticker1, ticker2: Pair tickers.
        split_date: Cutoff between train and test.
        strategy_config: If None, uses sensible defaults (lookback=60, etc).
        portfolio_config: If None, defaults to $100k initial, $10k notional.
        execution_costs: If None, defaults to IB-style commissions + bps slippage.

    Returns:
        SplitResult with both backtest outputs and the fitted parameters.
    """
    strategy_config = strategy_config or StrategyConfig()
    portfolio_config = portfolio_config or PortfolioConfig()
    execution_costs = execution_costs or ExecutionCosts()

    # 1. Split
    train_df, test_df = split_prices(prices, split_date)
    if len(train_df) < 200:
        raise ValueError(f"Train window too short: {len(train_df)} bars")
    if len(test_df) < 100:
        raise ValueError(f"Test window too short: {len(test_df)} bars")

    # 2. Fit parameters on train ONLY
    hedge_ratio, intercept, adf_stat = fit_pair_parameters(train_df, ticker1, ticker2)

    # 3. Run train backtest (for reference / in-sample comparison)
    train_strategy = PairStrategy(
        ticker1=ticker1, ticker2=ticker2,
        hedge_ratio=hedge_ratio, intercept=intercept,
        config=strategy_config,
    )
    train_portfolio = Portfolio(config=portfolio_config, hedge_ratio=hedge_ratio)
    train_engine = BacktestEngine(
        prices=train_df,
        strategy=train_strategy,
        portfolio=train_portfolio,
        execution=ExecutionSimulator(execution_costs),
    )
    train_result = train_engine.run()

    # 4. Run test backtest with FROZEN parameters
    test_strategy = PairStrategy(
        ticker1=ticker1, ticker2=ticker2,
        hedge_ratio=hedge_ratio, intercept=intercept,
        config=strategy_config,
    )
    test_portfolio = Portfolio(config=portfolio_config, hedge_ratio=hedge_ratio)
    test_engine = BacktestEngine(
        prices=test_df,
        strategy=test_strategy,
        portfolio=test_portfolio,
        execution=ExecutionSimulator(execution_costs),
    )
    test_result = test_engine.run()

    return SplitResult(
        train_result=train_result,
        test_result=test_result,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        cointegration_adf=adf_stat,
        split_date=pd.Timestamp(split_date),
        n_train_bars=len(train_df),
        n_test_bars=len(test_df),
    )


def _compute_metrics(result: BacktestResult, initial_capital: float) -> dict:
    """Compute Sharpe, total return, max drawdown, win rate from a result."""
    equity = result.equity
    if len(equity) < 2:
        return {"total_return": 0.0, "annualized_return": 0.0,
                "annualized_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "n_trades": 0}

    returns = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / initial_capital) - 1
    ann_return = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": result.n_fills,
    }


def summarize_split(split: SplitResult, initial_capital: float = 100_000) -> None:
    """Print side-by-side train and test metrics."""
    train_m = _compute_metrics(split.train_result, initial_capital)
    test_m = _compute_metrics(split.test_result, initial_capital)

    print(f"\n{'='*64}")
    print(f"Train/Test Split Backtest")
    print(f"{'='*64}")
    print(f"Pair:               {split.train_result.strategy.ticker1}/{split.train_result.strategy.ticker2}")
    print(f"Hedge ratio (β):    {split.hedge_ratio:.4f}")
    print(f"Intercept (α):      {split.intercept:.4f}")
    print(f"Train-period ADF:   {split.cointegration_adf:.3f}")
    print(f"Split date:         {split.split_date.date()}")
    print(f"Train bars:         {split.n_train_bars}")
    print(f"Test bars:          {split.n_test_bars}")

    print(f"\n{'Metric':<25} {'Train':>15} {'Test':>15}")
    print(f"{'-'*55}")
    rows = [
        ("Total return", f"{train_m['total_return']*100:+.2f}%", f"{test_m['total_return']*100:+.2f}%"),
        ("Annualized return", f"{train_m['annualized_return']*100:+.2f}%", f"{test_m['annualized_return']*100:+.2f}%"),
        ("Annualized volatility", f"{train_m['annualized_vol']*100:.2f}%", f"{test_m['annualized_vol']*100:.2f}%"),
        ("Sharpe ratio", f"{train_m['sharpe']:.2f}", f"{test_m['sharpe']:.2f}"),
        ("Max drawdown", f"{train_m['max_drawdown']*100:.2f}%", f"{test_m['max_drawdown']*100:.2f}%"),
        ("Number of trades", f"{train_m['n_trades']}", f"{test_m['n_trades']}"),
    ]
    for label, train_val, test_val in rows:
        print(f"{label:<25} {train_val:>15} {test_val:>15}")

    sharpe_gap = train_m["sharpe"] - test_m["sharpe"]
    print(f"\n{'='*64}")
    print(f"Overfitting gap (train Sharpe - test Sharpe): {sharpe_gap:+.2f}")
    if test_m["sharpe"] <= 0:
        print("  Out-of-sample performance is NOT POSITIVE.")
        print("  Strategy may be overfitting or relationship has broken.")
    elif sharpe_gap > 0.5:
        print("  Significant overfitting gap: strategy partially relies on selection bias.")
    elif sharpe_gap > 0:
        print("  Modest overfitting gap: edge appears partially robust.")
    else:
        print("  Test Sharpe >= train Sharpe (rare; either unlucky train period or robust edge).")


if __name__ == "__main__":
    from src.data.storage import get_connection, load_prices

    TICKER1, TICKER2 = "NDSN", "OTIS"
    # Split at ~70/30 of the available OTIS history (post-April 2020)
    SPLIT_DATE = "2023-06-01"

    conn = get_connection("data/market.duckdb")
    long_df = load_prices(conn, tickers=[TICKER1, TICKER2], start="2015-01-01")
    long_df["date"] = pd.to_datetime(long_df["date"])
    conn.close()
    wide = long_df.pivot(index="date", columns="ticker", values="adj_close")

    print(f"Loaded {len(wide)} bars for {TICKER1}/{TICKER2}")

    split = run_split_backtest(wide, TICKER1, TICKER2, SPLIT_DATE)
    summarize_split(split)

    # Save outputs
    split.train_result.equity.to_csv("data/split_train_equity.csv", header=True)
    split.test_result.equity.to_csv("data/split_test_equity.csv", header=True)
    split.train_result.trades.to_csv("data/split_train_trades.csv", index=False)
    split.test_result.trades.to_csv("data/split_test_trades.csv", index=False)
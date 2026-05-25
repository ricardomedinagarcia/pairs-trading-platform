"""Run walk-forward analysis across all 5 candidate pairs and aggregate
results into a portfolio-level view.

Per-pair walk-forwards use each pair's full available history (longer
pairs get more steps). The portfolio aggregation restricts to a common
window where all pairs have data, ensuring fair comparison.

Output:
- Per-pair tabulated summary
- Side-by-side per-step Sharpes
- Portfolio (equal-weight) aggregate Sharpe over the common window
"""

from dataclasses import dataclass

import pandas as pd

from src.backtest.walk_forward import (
    WalkForwardResult,
    run_walk_forward,
)
from src.data.storage import get_connection, load_prices


# Five candidate pairs from Phase 3b screening
CANDIDATES = [
    ("NDSN", "OTIS"),   # FDR-validated (only one)
    ("CARR", "TT"),     # HVAC manufacturers
    ("MA", "V"),        # Card networks
    ("EOG", "FANG"),    # Permian E&P
    ("TRGP", "WMB"),    # Natural gas midstream
]


@dataclass
class PairSummary:
    """One pair's walk-forward outcome."""

    ticker1: str
    ticker2: str
    n_steps: int
    n_trades_total: int
    agg_sharpe: float
    agg_return: float
    agg_max_dd: float
    median_step_sharpe: float
    n_positive_steps: int


def _summarize_pair(result: WalkForwardResult) -> PairSummary:
    """Reduce a WalkForwardResult to summary metrics."""
    stitched = result.stitched_equity()
    if len(stitched) < 2:
        return PairSummary(
            ticker1=result.ticker1, ticker2=result.ticker2,
            n_steps=0, n_trades_total=0, agg_sharpe=0.0, agg_return=0.0,
            agg_max_dd=0.0, median_step_sharpe=0.0, n_positive_steps=0,
        )

    returns = stitched.pct_change().dropna()
    if returns.std() > 0:
        ann_ret = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * (252 ** 0.5)
        agg_sharpe = ann_ret / ann_vol
    else:
        agg_sharpe = 0.0

    rolling_max = stitched.cummax()
    drawdown = (stitched - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    total_return = (stitched.iloc[-1] / stitched.iloc[0]) - 1
    step_sharpes = [s.test_sharpe for s in result.steps]
    median_sharpe = float(pd.Series(step_sharpes).median()) if step_sharpes else 0.0
    n_positive = sum(1 for s in step_sharpes if s > 0)
    n_trades = sum(s.test_n_trades for s in result.steps)

    return PairSummary(
        ticker1=result.ticker1, ticker2=result.ticker2,
        n_steps=len(result.steps),
        n_trades_total=n_trades,
        agg_sharpe=agg_sharpe,
        agg_return=total_return,
        agg_max_dd=max_dd,
        median_step_sharpe=median_sharpe,
        n_positive_steps=n_positive,
    )


def run_all_pairs() -> dict[tuple[str, str], WalkForwardResult]:
    """Run walk-forward on every candidate pair."""
    conn = get_connection("data/market.duckdb")
    all_tickers = sorted({t for pair in CANDIDATES for t in pair})
    long_df = load_prices(conn, tickers=all_tickers, start="2015-01-01")
    long_df["date"] = pd.to_datetime(long_df["date"])
    conn.close()

    wide = long_df.pivot(index="date", columns="ticker", values="adj_close")

    results: dict[tuple[str, str], WalkForwardResult] = {}
    for t1, t2 in CANDIDATES:
        print(f"\n{'='*60}")
        print(f"Walk-forward: {t1}/{t2}")
        print(f"{'='*60}")
        r = run_walk_forward(
            wide, t1, t2,
            train_window_months=24,
            test_window_months=6,
        )
        results[(t1, t2)] = r
        print(f"  Steps completed: {len(r.steps)}")
        if r.steps:
            step_sharpes = [s.test_sharpe for s in r.steps]
            print(f"  Per-step Sharpes: {[f'{s:.2f}' for s in step_sharpes]}")

    return results


def build_per_pair_summary(results: dict) -> pd.DataFrame:
    """Tabulate per-pair walk-forward summaries."""
    rows = [_summarize_pair(r).__dict__ for r in results.values()]
    df = pd.DataFrame(rows)
    df = df.sort_values("agg_sharpe", ascending=False).reset_index(drop=True)
    return df


def build_portfolio_equity(
    results: dict,
    common_start: str = "2022-04-01",
    initial_capital_per_pair: float = 20_000.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Equal-weight portfolio: $20k per pair, summed equity curve.

    Restricted to the common window where all pairs have walk-forward data.
    Reports the summed portfolio equity and the per-pair contribution.

    Args:
        results: Dict of pair -> WalkForwardResult.
        common_start: Earliest date to include in the portfolio aggregation.
        initial_capital_per_pair: Each pair gets this dollar allocation.

    Returns:
        (portfolio_equity, per_pair_equity_df) tuple.
    """
    cutoff = pd.Timestamp(common_start)

    pair_equities: dict[str, pd.Series] = {}
    for (t1, t2), result in results.items():
        stitched = result.stitched_equity()
        if len(stitched) == 0:
            continue
        # Restrict to common window and rebase to initial_capital_per_pair
        post = stitched.loc[stitched.index >= cutoff]
        if len(post) < 2:
            continue
        rebased = post / post.iloc[0] * initial_capital_per_pair
        pair_equities[f"{t1}/{t2}"] = rebased

    if not pair_equities:
        return pd.Series(dtype=float), pd.DataFrame()

    # Align on common date index and sum
    per_pair_df = pd.DataFrame(pair_equities)
    per_pair_df = per_pair_df.ffill().bfill()  # fill missing days within window
    portfolio = per_pair_df.sum(axis=1)
    return portfolio, per_pair_df


def summarize_portfolio(portfolio: pd.Series, initial_total: float) -> None:
    """Print portfolio-level metrics."""
    if len(portfolio) < 2:
        print("Insufficient portfolio data.")
        return

    returns = portfolio.pct_change().dropna()
    total_return = (portfolio.iloc[-1] / initial_total) - 1
    if returns.std() > 0:
        ann_ret = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * (252 ** 0.5)
        sharpe = ann_ret / ann_vol
    else:
        ann_ret = 0.0
        ann_vol = 0.0
        sharpe = 0.0

    rolling_max = portfolio.cummax()
    max_dd = ((portfolio - rolling_max) / rolling_max).min()

    print(f"\n{'='*60}")
    print(f"Portfolio (equal-weight, common-window aggregation)")
    print(f"{'='*60}")
    print(f"Window: {portfolio.index.min().date()} to {portfolio.index.max().date()}")
    print(f"Initial:           ${initial_total:,.2f}")
    print(f"Final:             ${portfolio.iloc[-1]:,.2f}")
    print(f"Total return:      {total_return*100:+.2f}%")
    print(f"Annualized return: {ann_ret*100:+.2f}%")
    print(f"Annualized vol:    {ann_vol*100:.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max drawdown:      {max_dd*100:.2f}%")


if __name__ == "__main__":
    print("Running walk-forward across all 5 candidate pairs...")
    print("This will take 2-3 minutes.\n")

    results = run_all_pairs()

    # Per-pair summary table
    summary = build_per_pair_summary(results)
    print(f"\n{'='*78}")
    print("Per-Pair Walk-Forward Summary (sorted by aggregate Sharpe)")
    print(f"{'='*78}")
    pretty_cols = ["ticker1", "ticker2", "n_steps", "n_trades_total",
                   "agg_sharpe", "agg_return", "agg_max_dd",
                   "median_step_sharpe", "n_positive_steps"]
    print(summary[pretty_cols].to_string(index=False, float_format="%.3f"))

    # Portfolio aggregation
    initial_capital_per_pair = 20_000.0
    n_pairs = len(results)
    initial_total = initial_capital_per_pair * n_pairs
    portfolio, per_pair_df = build_portfolio_equity(
        results,
        common_start="2022-04-01",
        initial_capital_per_pair=initial_capital_per_pair,
    )
    summarize_portfolio(portfolio, initial_total=initial_total)

    # Per-pair Sharpe contribution over the common window
    if len(per_pair_df) > 1:
        print(f"\n{'='*60}")
        print("Per-pair contribution (over common window)")
        print(f"{'='*60}")
        for col in per_pair_df.columns:
            series = per_pair_df[col]
            returns = series.pct_change().dropna()
            if returns.std() > 0:
                ann_ret = (1 + returns.mean()) ** 252 - 1
                ann_vol = returns.std() * (252 ** 0.5)
                sh = ann_ret / ann_vol
            else:
                sh = 0.0
            final_ret = (series.iloc[-1] / series.iloc[0]) - 1
            print(f"  {col:<12} Sharpe={sh:+.2f}  Return={final_ret*100:+.2f}%")

    # Save outputs
    summary.to_csv("data/multi_pair_summary.csv", index=False)
    if not per_pair_df.empty:
        per_pair_df.to_csv("data/multi_pair_per_pair_equity.csv")
        portfolio.to_frame("portfolio_equity").to_csv("data/multi_pair_portfolio_equity.csv")
    print("\nSaved: data/multi_pair_summary.csv")
    print("       data/multi_pair_per_pair_equity.csv")
    print("       data/multi_pair_portfolio_equity.csv")
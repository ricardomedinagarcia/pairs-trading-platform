"""Cointegration screening engine. 

Applies the Engle-Granger test pairwise across a universe of tickers,
ranks results, and identifies candidates that pass at conventional
significance levels.

Caveat: this performs many simultaneous tests, so naive 5% significance
on individual pairs does NOT mean 5% false positive rate overall.
With 45 pairs tested at 5%, expect ~2.25 false positives by chance.
Multiple-testing correction (Bonferroni, FDR) is discussed in the
output analysis, not applied here -- the unadjusted results are kept
for transparency and downstream analysis.
"""

from dataclasses import dataclass 
from itertools import combinations

import pandas as pd

from src.stats.cointegration import engle_granger_test


@dataclass
class PairScreenResult:
    """Single pair's screening result."""

    ticker1: str
    ticker2: str
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    n_obs: int
    n_lags: int
    is_cointegrated_5pct: bool
    is_cointegrated_1pct: bool

def screen_all_pairs(
    prices: pd.DataFrame,
    use_log: bool = True,
) -> pd.DataFrame:
    """Apply Engle-Granger to every unique pair in the universe.

    Args:
        prices: Long-format DataFrame with columns ticker, date, adj_close
            (other columns ignored). Must contain at least 2 tickers.
        use_log: Pass through to engle_granger_test.

    Returns:
        DataFrame with one row per pair, sorted by ADF statistic ascending
        (most negative first = strongest cointegration evidence).
    """
    if "ticker" not in prices.columns or "adj_close" not in prices.columns:
        raise ValueError("prices must have 'ticker' and 'adj_close' columns")

    # Pivot to wide format: index=date, columns=ticker, values=adj_close
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    tickers = sorted(wide.columns.tolist())

    if len(tickers) < 2:
        raise ValueError(f"Need at least 2 tickers, got {len(tickers)}")

    results: list[PairScreenResult] = []
    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    print(f"Screening {n_pairs} pairs from {len(tickers)} tickers...")

    for t1, t2 in combinations(tickers, 2):
        p1 = wide[t1].dropna()
        p2 = wide[t2].dropna()
        try:
            r = engle_granger_test(p1, p2, ticker1=t1, ticker2=t2, use_log=use_log)
        except Exception as e:
            print(f"  WARNING: {t1}/{t2} failed: {e}")
            continue

        results.append(
            PairScreenResult(
                ticker1=t1,
                ticker2=t2,
                hedge_ratio=r.hedge_ratio,
                intercept=r.intercept,
                adf_statistic=r.adf_statistic,
                n_obs=r.n_obs,
                n_lags=r.n_lags,
                is_cointegrated_5pct=r.is_cointegrated_5pct,
                is_cointegrated_1pct=r.is_cointegrated_1pct,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in results])
    df = df.sort_values("adf_statistic").reset_index(drop=True)
    return df


def summarize_screening(df: pd.DataFrame) -> None:
    """Print a human-readable summary of screening results."""
    n_total = len(df)
    n_5pct = df["is_cointegrated_5pct"].sum()
    n_1pct = df["is_cointegrated_1pct"].sum()

    print(f"\n{'='*60}")
    print(f"Screening Summary")
    print(f"{'='*60}")
    print(f"Total pairs tested:       {n_total}")
    print(f"Cointegrated at 5%:       {n_5pct}  ({100*n_5pct/n_total:.1f}%)")
    print(f"Cointegrated at 1%:       {n_1pct}  ({100*n_1pct/n_total:.1f}%)")
    print(f"\nBonferroni-adjusted 5% threshold: {0.05/n_total:.4f}")
    print(f"(Pairs passing this stricter bar are robust to multiple testing.)")

    print(f"\n{'='*60}")
    print(f"Top 10 by ADF statistic (most negative = strongest)")
    print(f"{'='*60}")
    print(df.head(10).to_string(index=False))

    if n_5pct > 0:
        print(f"\n{'='*60}")
        print(f"Pairs passing 5% significance:")
        print(f"{'='*60}")
        passing = df[df["is_cointegrated_5pct"]]
        print(passing.to_string(index=False))


if __name__ == "__main__":
    from src.data.storage import get_connection, load_prices

    conn = get_connection("data/market.duckdb")
    prices = load_prices(conn, start="2015-01-01")
    prices["date"] = pd.to_datetime(prices["date"])

    results = screen_all_pairs(prices)
    summarize_screening(results)

    # Save for downstream use
    results.to_csv("data/cointegration_screening.csv", index=False)
    print(f"\nResults saved to data/cointegration_screening.csv")

    conn.close()
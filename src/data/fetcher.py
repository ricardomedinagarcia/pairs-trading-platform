"""Fetch historical equity data from Yahoo Finance.

This module is the project's data ingestion layer. It pulls daily OHLCV data
for a list of tickers and returns a clean long-format DataFrame ready for
storage in the DuckDB layer.

Design decisions:
- Retry with exponential backoff on transient failures (Yahoo can be flaky).
- Skip bad tickers gracefully rather than failing the whole batch.
- Return long-format (one row per ticker-date) to match the database schema.
- Include both raw close and adjusted close — adjusted for returns/cointegration,
  raw for understanding actual execution prices.
"""

import time
from datetime import date
from typing import Sequence

import pandas as pd
import yfinance as yf


def fetch_daily_prices(
    tickers: Sequence[str],
    start: str | date,
    end: str | date,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for a list of tickers.

    Args:
        tickers: Iterable of ticker symbols, e.g. ["AAPL", "MSFT"].
        start: Start date (inclusive), as "YYYY-MM-DD" string or date object.
        end: End date (exclusive), same format.
        max_retries: Number of retry attempts per ticker on failure.

    Returns:
        Long-format DataFrame with columns:
            ticker, date, open, high, low, close, adj_close, volume

        Empty DataFrame if no tickers returned data.
    """
    frames = []
    for ticker in tickers:
        df = _fetch_one_with_retry(ticker, start, end, max_retries)
        if df is not None and not df.empty:
            frames.append(df)
        else:
            print(f"WARNING: no data for {ticker}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _fetch_one_with_retry(
    ticker: str,
    start: str | date,
    end: str | date,
    max_retries: int,
) -> pd.DataFrame | None:
    """Fetch a single ticker with exponential backoff on failure.

    Returns None if all retries fail or the ticker has no data.
    """
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,  # we want both close and adj_close
            )

            if raw.empty:
                return None

            # yfinance returns a multi-index in newer versions; flatten it
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = pd.DataFrame({
                "ticker": ticker,
                "date": raw.index.date,
                "open": raw["Open"].values,
                "high": raw["High"].values,
                "low": raw["Low"].values,
                "close": raw["Close"].values,
                "adj_close": raw["Adj Close"].values,
                "volume": raw["Volume"].values.astype("int64"),
            })
            return df

        except Exception as e:
            wait = 2 ** attempt
            print(f"Attempt {attempt + 1} failed for {ticker}: {e}. Retrying in {wait}s")
            time.sleep(wait)

    return None


if __name__ == "__main__":
    # Quick smoke test — run with `python -m src.data.fetcher` from project root
    tickers = ["KO", "PEP", "XOM", "CVX"]
    df = fetch_daily_prices(tickers, start="2015-01-01", end="2024-12-31")

    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    print(f"Tickers fetched: {df['ticker'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
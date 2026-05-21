"""Rate-limit-aware batch fetcher for large ticker universes.

Yfinance has informal rate limits that aren't documented but are real:
hammering it with 700+ rapid requests will start returning empty results
or 429s. This module handles the universe-scale download with:

- Controlled delay between requests
- Per-ticker retries (delegated to fetcher._fetch_one_with_retry)
- Progress logging
- Checkpoint: skip tickers that already have data in the DB
- Graceful handling of unrecoverable tickers (delisted, acquired)
"""

import time
from datetime import date
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd

from src.data.fetcher import _fetch_one_with_retry
from src.data.storage import get_connection, initialize_schema, upsert_prices


def batch_fetch(
    tickers: Sequence[str],
    start: str | date,
    end: str | date,
    db_path: Path | str = "data/market.duckdb",
    delay_seconds: float = 0.3,
    skip_existing: bool = True,
    progress_every: int = 25,
) -> dict:
    """Fetch daily prices for many tickers and store in DuckDB.

    Args:
        tickers: Universe of ticker symbols to fetch.
        start, end: Date range (inclusive start, exclusive end).
        db_path: Path to the DuckDB file.
        delay_seconds: Sleep between requests. 0.3s = ~200/minute pace,
            comfortably under yfinance's informal rate limits.
        skip_existing: If True, skip tickers that already have rows in the
            prices table (resumable downloads).
        progress_every: Print progress every N tickers.

    Returns:
        dict with keys: succeeded, failed, skipped, total_rows.
    """
    conn = get_connection(db_path)
    initialize_schema(conn)

    if skip_existing:
        existing = set(
            conn.execute("SELECT DISTINCT ticker FROM prices").fetchnumpy()["ticker"].tolist()
        )
        print(f"Found {len(existing)} tickers already in database; will skip those.")
    else:
        existing = set()

    n_total = len(tickers)
    succeeded = []
    failed = []
    skipped = []
    total_rows = 0
    start_time = time.time()

    for i, ticker in enumerate(tickers, start=1):
        if ticker in existing:
            skipped.append(ticker)
            continue

        df = _fetch_one_with_retry(ticker, start, end, max_retries=3)
        if df is None or df.empty:
            failed.append(ticker)
        else:
            n = upsert_prices(conn, df)
            succeeded.append(ticker)
            total_rows += n

        if i % progress_every == 0 or i == n_total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{n_total}] ok={len(succeeded)} "
                f"failed={len(failed)} skipped={len(skipped)} "
                f"rows={total_rows:,} elapsed={elapsed:.0f}s "
                f"eta={eta:.0f}s"
            )

        time.sleep(delay_seconds)

    conn.close()

    print(f"\n{'='*60}")
    print(f"Batch fetch complete in {time.time() - start_time:.0f}s")
    print(f"Succeeded: {len(succeeded)} tickers")
    print(f"Failed:    {len(failed)} tickers")
    print(f"Skipped:   {len(skipped)} tickers")
    print(f"Total rows: {total_rows:,}")
    if failed:
        print(f"\nFailed tickers (first 20): {failed[:20]}")
    print(f"{'='*60}")

    return {
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "total_rows": total_rows,
    }


if __name__ == "__main__":
    from src.data.universe import (
        fetch_current_constituents,
        get_ever_member_tickers,
    )
    from src.data.storage import upsert_securities

    print("Building S&P 500 ever-member universe for 2015-2024...")
    tickers = get_ever_member_tickers("2015-01-01", "2024-12-31")
    print(f"Universe size: {len(tickers)} tickers")

    # Populate securities metadata for current constituents (sectors)
    print("\nPopulating securities metadata...")
    current = fetch_current_constituents()
    current["is_current_sp500"] = True
    current["first_added_date"] = pd.to_datetime(
        current.get("date_added"), errors="coerce"
    ).dt.date
    conn = get_connection("data/market.duckdb")
    initialize_schema(conn)
    n_securities = upsert_securities(conn, current)
    conn.close()
    print(f"  Upserted {n_securities} securities with sector metadata.")

    print("\nStarting batch price fetch (this will take 30-45 minutes)...")
    result = batch_fetch(tickers, start="2015-01-01", end="2024-12-31")
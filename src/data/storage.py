"""DuckDB storage layer for market data.

This module owns all interactions with the project's DuckDB database. The
fetcher returns DataFrames; this layer persists them, queries them back, and
manages schema.

Design decisions:
- DuckDB chosen over Postgres/SQLite: columnar OLAP engine optimized for
  analytical queries on price data; zero-config single-file database.
- Long-format prices table (one row per ticker-date) for clean joins and
  efficient filtering.
- INSERT OR REPLACE for upserts so we can safely re-run ingestion without
  duplicate-key errors.
- Schema separated from data access functions for clarity.
"""

from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_DB_PATH = Path("data/market.duckdb")


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating the data directory if needed.

    Caller is responsible for closing the connection (use as context manager
    or call .close() explicitly).
    """
    db_path = Path(db_path)  # coerce strings to Path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def initialize_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't exist. Safe to run repeatedly (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS securities (
            ticker      VARCHAR PRIMARY KEY,
            name        VARCHAR,
            sector      VARCHAR,
            added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker      VARCHAR NOT NULL,
            date        DATE    NOT NULL,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            adj_close   DOUBLE,
            volume      BIGINT,
            PRIMARY KEY (ticker, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS corporate_actions (
            ticker      VARCHAR NOT NULL,
            date        DATE    NOT NULL,
            action_type VARCHAR NOT NULL,  -- 'split' or 'dividend'
            value       DOUBLE  NOT NULL,
            PRIMARY KEY (ticker, date, action_type)
        )
    """)


def upsert_prices(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert price data, replacing existing rows on (ticker, date) conflict.

    Args:
        conn: Active DuckDB connection.
        df: Long-format DataFrame matching the prices schema columns.

    Returns:
        Number of rows inserted/replaced.
    """
    if df.empty:
        return 0

    conn.register("incoming", df)
    conn.execute("""
        INSERT OR REPLACE INTO prices
        SELECT ticker, date, open, high, low, close, adj_close, volume
        FROM incoming
    """)
    conn.unregister("incoming")
    return len(df)


def load_prices(
    conn: duckdb.DuckDBPyConnection,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load price data with optional filters.

    Args:
        conn: Active DuckDB connection.
        tickers: List of tickers to load (None = all tickers).
        start: Earliest date inclusive (None = no lower bound).
        end: Latest date inclusive (None = no upper bound).

    Returns:
        Long-format DataFrame ordered by ticker, date.
    """
    query = "SELECT * FROM prices WHERE 1=1"
    params: list = []

    if tickers:
        placeholders = ", ".join("?" for _ in tickers)
        query += f" AND ticker IN ({placeholders})"
        params.extend(tickers)
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    query += " ORDER BY ticker, date"
    return conn.execute(query, params).df()


if __name__ == "__main__":
    # Smoke test — run with `python -m src.data.storage` from project root
    from src.data.fetcher import fetch_daily_prices

    tickers = ["KO", "PEP", "XOM", "CVX", "JPM", "BAC", "JNJ", "PFE", "AAPL", "MSFT"]
    df = fetch_daily_prices(tickers, start="2015-01-01", end="2024-12-31")

    conn = get_connection()
    initialize_schema(conn)
    rows = upsert_prices(conn, df)
    print(f"Inserted/updated {rows} rows")

    loaded = load_prices(conn, tickers=["KO", "PEP"], start="2024-01-01")
    print(f"\nLoaded {len(loaded)} KO/PEP rows for 2024:")
    print(loaded.head())

    conn.close()
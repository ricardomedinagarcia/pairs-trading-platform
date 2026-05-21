"""S&P 500 universe construction with point-in-time membership.

Source: Wikipedia's "List of S&P 500 companies" page, which maintains
both current constituents and a historical change log going back decades.

This module produces:
1. The current S&P 500 constituent list (ticker, name, sector)
2. A historical change log (date, added_ticker, removed_ticker, reason)
3. The "ever-member set": every ticker that has been in the S&P 500
   during a given time window. This is the universe we'll pull data for.

Survivorship bias note: yfinance has spotty coverage for delisted/acquired
tickers. Some historical members will be unrecoverable. This is a real
data quality limitation; we log warnings and continue.
"""

from dataclasses import dataclass
from datetime import date
from io import StringIO

import pandas as pd
import requests

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# A real browser User-Agent is required; Wikipedia rejects default Python agents.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def _fetch_wikipedia_tables() -> list[pd.DataFrame]:
    """Fetch and parse all tables from the S&P 500 Wikipedia page.

    Uses requests with a browser User-Agent because Wikipedia rejects
    the default Python urllib agent with 403 Forbidden.
    """
    response = requests.get(WIKIPEDIA_SP500_URL, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))

@dataclass
class SP500Constituent:
    """A current S&P 500 member."""

    ticker: str
    name: str
    sector: str
    sub_industry: str
    date_added: str | None  # ISO date string when available


@dataclass
class SP500Change:
    """A historical S&P 500 addition or removal event."""

    date: str  # ISO date string
    added_ticker: str | None
    added_name: str | None
    removed_ticker: str | None
    removed_name: str | None
    reason: str | None


def fetch_current_constituents() -> pd.DataFrame:
    """Scrape current S&P 500 constituents from Wikipedia.

    Returns:
        DataFrame with columns: ticker, name, sector, sub_industry,
        date_added (where available).
    """
    tables = _fetch_wikipedia_tables()
    # The first table on the page is the current constituents
    df = tables[0]

    # Wikipedia column names occasionally drift; normalize defensively
    column_map = {
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "Date added": "date_added",
        "Date first added": "date_added",  # alternative column name
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Some tickers use dots (e.g., BRK.B); yfinance wants dashes (BRK-B)
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)

    keep_cols = [c for c in ["ticker", "name", "sector", "sub_industry", "date_added"] if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def fetch_historical_changes() -> pd.DataFrame:
    """Scrape historical S&P 500 changes (additions/removals) from Wikipedia.

    Returns:
        DataFrame with columns: date, added_ticker, added_name,
        removed_ticker, removed_name, reason.
    """
    tables = _fetch_wikipedia_tables()
    # The second table is the change log
    df = tables[1]

    # The change log uses a multi-level column header on Wikipedia
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten: ('Added', 'Ticker') -> 'added_ticker', etc.
        new_cols = []
        for col in df.columns:
            level0 = str(col[0]).lower().strip()
            level1 = str(col[1]).lower().strip()
            if "date" in level0:
                new_cols.append("date")
            elif "added" in level0:
                new_cols.append(f"added_{level1.replace(' ', '_')}")
            elif "removed" in level0:
                new_cols.append(f"removed_{level1.replace(' ', '_')}")
            elif "reason" in level0:
                new_cols.append("reason")
            else:
                new_cols.append(f"{level0}_{level1}")
        df.columns = new_cols

    # Apply same ticker normalization (. -> -)
    for col in ["added_ticker", "removed_ticker"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(".", "-", regex=False)
            df.loc[df[col] == "nan", col] = None

    # Parse the date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df.reset_index(drop=True)


def get_ever_member_tickers(
    start: str | date,
    end: str | date,
) -> list[str]:
    """Get every ticker that was in the S&P 500 at any point in [start, end].

    Combines current constituents with historical changes to produce the
    "ever-member set" -- our universe for data pulls.

    Args:
        start: Start of the window (ISO date string or date object).
        end: End of the window.

    Returns:
        Sorted list of unique tickers.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    current = fetch_current_constituents()
    changes = fetch_historical_changes()

    # Start with everyone who's currently a member
    tickers: set[str] = set(current["ticker"].dropna().tolist())

    # Add every ticker that was added OR removed during the window
    # (added: they joined during our window, so we want them)
    # (removed: they were a member at some point during the window before being removed)
    if "added_ticker" in changes.columns:
        in_window_added = changes[
            (changes["date"] >= start_dt) & (changes["date"] <= end_dt) &
            changes["added_ticker"].notna()
        ]
        tickers.update(in_window_added["added_ticker"].tolist())

    if "removed_ticker" in changes.columns:
        # A ticker removed during the window was a member at the start of the window
        in_window_removed = changes[
            (changes["date"] >= start_dt) & (changes["date"] <= end_dt) &
            changes["removed_ticker"].notna()
        ]
        tickers.update(in_window_removed["removed_ticker"].tolist())

        # Also: tickers removed AFTER our window's end but BEFORE today
        # were members during our entire window
        post_window_removed = changes[
            (changes["date"] > end_dt) &
            changes["removed_ticker"].notna()
        ]
        tickers.update(post_window_removed["removed_ticker"].tolist())

    # Clean: drop empty strings, "nan", None
    tickers = {t for t in tickers if t and t != "nan" and t != "None"}
    return sorted(tickers)


if __name__ == "__main__":
    # Smoke test
    print("Fetching current constituents...")
    current = fetch_current_constituents()
    print(f"  Current S&P 500: {len(current)} tickers")
    print(current.head())

    print("\nFetching historical changes...")
    changes = fetch_historical_changes()
    print(f"  Total change events: {len(changes)}")
    print(changes.head())

    print("\nBuilding ever-member set for 2015-2024...")
    tickers = get_ever_member_tickers("2015-01-01", "2024-12-31")
    print(f"  Universe size: {len(tickers)} tickers")
    print(f"  Sample: {tickers[:20]}")
# Research Journal

# 2026-05-11 Project setup 

Set up development environment from scratch"
- Python 3.12.13 via Homebrew (Note: Had to debug PATH issues around macOS system python)
- Git configured with personal access token auth for GitHub
- Project structure: src/{data,stats,backtest,analysis}, tests/, notebooks/
- Virtual env active, all dependencies installed
- Initial commit pushed to github.com/ricardomedinagarcia/pairs-trading-platform

Next: Build data fetcher (yfinanace -> DuckDB) for 10 canidate tickers, including the classic pairs KO/PEP and XOM/CVX


## 2026-05-11 - Storage Layer 

Built 'src/data/storage.py' :
- DuckD chosen for colummar OLAP performance (vs Postgres/SQLite)
- Three tables: Securities, prices, corporate_actions
- Composite PK on (ticker, date) for prices prevents duplicates 
- INSERT OR REPLACE upsert pattern for idempotent re-ingestion 
- Parameterized queries throughout (SQL injection hygenie)

Populated database with 10 tickers x 10 years = ~25k rows.
File: data/market.duckdb (~X MB, gitignored).

Next: exploration notebok - visualize condidate pairs (KO/PEP, XOM/CVX)
and start building intuition before formalizing cointegration tests
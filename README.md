# Systematic Pairs Trading Research Platform

A research framework for identifying and backtesting mean-reverting equity pair
strategies. Built to rigorously test stat-arb hypotheses with point-in-time
correctness, realistic execution modeling, and honest evaluation.

## Project Status
**Phase 1: Data Infrastructure** (in progress)

## Methodology
- Cointegration testing: Engle-Granger and Johansen, implemented from first principles
- Backtesting: event-driven, with realistic transaction costs and position sizing
- Evaluation: deflated Sharpe ratio, walk-forward analysis, bootstrapped confidence intervals

## Structure
- `src/data/` — data ingestion and storage
- `src/stats/` — cointegration tests and statistical methods
- `src/backtest/` — event-driven backtesting engine
- `src/analysis/` — performance metrics and visualization
- `notebooks/` — exploratory research
- `tests/` — unit tests

## Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
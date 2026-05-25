# Systematic Pairs Trading Research Platform

End-to-end statistical arbitrage research pipeline applied to the S&P 500
universe over 2015-2024. Implements survivorship-bias-corrected universe
construction, FDR-controlled cointegration screening, an event-driven
backtester with look-ahead-bias prevention, and López de Prado-style
deflated performance metrics.

**Headline finding:** Of 1,384 within-sub-industry equity pairs tested,
one (NDSN/OTIS) survived 5% FDR multiple-testing correction. After
walk-forward out-of-sample evaluation and adjustment for selection bias
across all 1,384 trials, no single pair's edge survives strict statistical
deflation. Mastercard/Visa is the most structurally defensible candidate:
its 90% bootstrap confidence interval ([+0.39, +2.36] annualized Sharpe)
excludes zero, consistent with the durability of cointegration between
two card networks with identical business models. The aggregate result
is consistent with the academic literature on stat-arb capacity exhaustion:
classical pairs trading on daily equity data has been substantially
arbitraged away except in structurally-tight relationships.

The project is intentionally built end-to-end from first principles where
it teaches something (ADF and Engle-Granger tests are hand-implemented
and validated against `statsmodels` to ~0.01 numerical tolerance), and
delegated to standard libraries where it doesn't (MacKinnon p-value lookup,
bootstrap resampling). Every stage of the pipeline is documented in
[`journal.md`](journal.md), which records bugs found, design decisions
made, and findings interpreted as the project progressed.

---

## Why this project exists

Most pairs-trading projects on GitHub report inflated Sharpe ratios because
they:

- Use survivorship-biased ticker universes (current S&P 500 only)
- Backtest on the same data used to estimate parameters
- Skip out-of-sample evaluation entirely
- Don't account for multiple-testing burden when screening many pairs
- Use vectorized backtesters that silently introduce look-ahead bias

Each of these failures inflates apparent edge by 0.3-1.0 Sharpe units.
A "Sharpe 2" project where all four issues are present typically becomes
"Sharpe 0.3" when corrected — or negative.

The goal of this project was to build a research platform where these
inflation sources are *structurally prevented*, and then honestly report
whatever the corrected results turned out to be.

---

## Key results

### Pipeline summary

| Stage                       | Result                                          |
|-----------------------------|-------------------------------------------------|
| Universe (S&P 500 ever-members, 2015-2024) | 740 tickers, 618 with usable yfinance history |
| Sector-level screening      | 13,029 pairs tested, 1,051 pass naive 5%, 0 pass BH-FDR 10% |
| Sub-industry-level screening | 1,384 pairs tested, **1 passes BH-FDR 5%** (NDSN/OTIS) |
| Top 5 candidates by ADF stat | NDSN/OTIS, CARR/TT, MA/V, EOG/FANG, TRGP/WMB |

### Walk-forward out-of-sample performance

Each pair tested with 24-month rolling training windows and 6-month
out-of-sample test windows. Parameters re-estimated at each step.

| Pair       | Steps | OOS Sharpe | PSR  | DSR (N=1384) | 90% Bootstrap CI    |
|------------|-------|------------|------|--------------|---------------------|
| MA/V       | 15    | +0.55      | 0.99 | 0.00         | [+0.39, +2.36]      |
| NDSN/OTIS  | 5     | +0.52      | 0.79 | 0.00         | [-0.53, +1.58]      |
| EOG/FANG   | 15    | +0.45      | n/a  | 0.00         | n/a                 |
| TRGP/WMB   | 15    | -0.21      | 0.26 | 0.00         | [-1.46, +0.62]      |
| CARR/TT    | 5     | -0.37      | 0.28 | 0.00         | [-1.41, +0.73]      |

PSR = Probabilistic Sharpe Ratio = P(true SR > 0 | observed series).
DSR = Deflated Sharpe Ratio = P(true SR > expected max under N trials).
DSR ≈ 0 across all pairs reflects the selection bias of having screened
1,384 candidates.

### Equal-weight portfolio (common 2022-2024 window)

- Annualized Sharpe: **-0.30**
- 90% Bootstrap CI: [-1.34, +0.79]
- Total return: -1.10% over 30 months

Two of five pairs had positive individual Sharpes, but equal capital
allocation pulled the portfolio negative. Real portfolio construction
requires risk-budgeting and correlation-aware sizing, which were out
of scope here.

### Strategy decay

Per-step Sharpes show a clear temporal decay pattern in most pairs.
EOG/FANG Sharpes by step: [+2.71, +0.68, +2.01, +0.93, +2.12, +0.42,
-1.15, +1.53, +1.21, +1.74, +0.66, -0.14, +0.42, -0.72, -2.45]. Strong
in 2017-2020, intermittent through 2022, mostly negative by 2024.
MA/V is the exception: strongest steps are recent (+2.94 and +3.55
in late windows), consistent with structural rather than incidental
cointegration.

---

## Methodology

### Universe construction (`src/data/universe.py`)

The S&P 500 universe is constructed as the **ever-member set** for
2015-2024: every ticker that was an S&P 500 constituent at any point
during the window. Sourced from Wikipedia's current-constituents table
joined with the historical change log. This corrects for survivorship
bias — naive backtests using only current constituents silently exclude
every company that was acquired or removed during the window (Allergan,
Celgene, First Republic, Twitter, etc.), inflating returns.

### Cointegration testing (`src/stats/`)

The Augmented Dickey-Fuller test (`stationarity.py`) is implemented
from first principles: explicit OLS via `(X'X)^-1 X'y`, automatic AIC
lag selection bounded by Schwert's rule with fixed-sample comparison
across candidate lags. Validated against `statsmodels.tsa.stattools.adfuller`
within 0.01 numerical tolerance on both stationary and random-walk
test cases.

The Engle-Granger cointegration test (`cointegration.py`) layers OLS
hedge-ratio estimation on top of ADF. Uses MacKinnon (2010) critical
values for the cointegration test (stricter than standard ADF because
residuals come from an estimated regression).

A subtle bug was caught during development: the initial lag-selection
implementation used inconsistent sample sizes across candidate lag
counts, biasing AIC selection and producing test statistics 2x too
small in magnitude. Fixed by constraining all candidate lags to the
sample size required by max_lags during selection, then refitting
at the chosen lag with full sample for the final statistic.

### Screening with FDR control (`src/stats/screening_v2.py`)

Within-sector pairwise screening across the 618-ticker universe with
sufficient history. Two granularities:

- **Sector-level** (13,029 pairs): GICS sector grouping. Looser
  economic constraint.
- **Sub-industry-level** (1,384 pairs): GICS sub-industry grouping.
  Tighter economic constraint; reduces multiple-testing burden by ~9x.

Multiple-testing correction via Benjamini-Hochberg FDR rather than
Bonferroni. With thousands of tests, Bonferroni is too conservative;
BH controls the expected proportion of false discoveries (e.g., at
10% FDR, at most 10% of declared "discoveries" are expected to be
false positives). Standard in modern genomics and quant research.

P-values for individual ADF statistics are computed via
`statsmodels.tsa.adfvalues.mackinnonp`, the canonical implementation
of MacKinnon (1996) response surface coefficients.

Multiprocessing parallelizes the 1,384 cointegration regressions
across CPU cores via `Pool` with an initializer that loads the shared
price dictionary once per worker (avoiding repickling).

### Event-driven backtester (`src/backtest/`)

Five-module architecture matching production trading systems:

- `events.py`: immutable dataclasses for MarketData/Signal/Order/Fill
- `execution.py`: simulator with commission ($0.005/share), half-spread
  (5 bps), and market impact (2 bps) slippage models
- `portfolio.py`: hedge-ratio-based position sizing, strict cash +
  positions accounting (cash + Σposition*price = equity at every step)
- `strategy.py`: mean-reversion on rolling z-scored spread, with
  shift(1) discipline in the z-score computation to prevent
  look-ahead bias
- `engine.py`: chronological bar-by-bar event loop

The look-ahead prevention is structural rather than convention-based:
the strategy's rolling mean and standard deviation at time t are
computed from bars [t-lookback, t-1] inclusive, *excluding* the
current bar. The current bar contributes its value only to the z-score
numerator (where it's the new data point being scored), never to the
denominator (where it would be using future-information).

### Evaluation methodologies (`src/backtest/`)

The project applies four progressively-more-honest evaluation methods,
each documented in the journal with results:

- **In-sample full-window** (`run_ndsn_otis.py`): baseline diagnostic.
  Sharpe 0.59 on NDSN/OTIS, inflated by parameter estimation on the
  same data being evaluated.
- **Train/test split** (`train_test.py`): parameters fit on train,
  evaluated frozen on test. Sharpe drops to -0.10 on NDSN/OTIS,
  exposing parameter overfitting.
- **Walk-forward** (`walk_forward.py`): rolling re-estimation every
  6 months with 24-month training windows. Aggregate Sharpe recovers
  to +0.52 on NDSN/OTIS, showing the strategy *does* generalize when
  parameters are kept fresh.
- **Multi-pair portfolio** (`run_multi_pair.py`): all 5 candidates
  in equal-weight portfolio. Reveals strategy decay across most pairs
  and the limits of naive portfolio construction.

### Adjusted metrics (`src/backtest/metrics.py`)

Three López de Prado adjustments:

- **Probabilistic Sharpe Ratio (PSR)**: P(true_SR > 0) given observed
  returns, adjusted for skewness and excess kurtosis. Reflects how
  confident we should be that the strategy has any edge at all.
- **Deflated Sharpe Ratio (DSR)**: P(true_SR > expected max under N
  random trials). Reflects how much of the apparent edge can be
  explained purely by selection bias from screening N candidates.
- **Bootstrap CI**: non-parametric confidence interval on Sharpe via
  return-series resampling. Robust to non-normality.

---

## Technical Highlights

- Built an end-to-end statistical arbitrage research pipeline in Python.
- Implemented ADF and Engle-Granger tests from first principles.
- Corrected for survivorship bias using an S&P 500 ever-member universe.
- Applied Benjamini-Hochberg FDR control across 1,384 candidate pairs.
- Designed an event-driven backtester with explicit look-ahead-bias prevention.
- Evaluated performance using walk-forward testing, PSR, DSR, and bootstrap confidence intervals.

## Repository Structure

```text
pairs-trading/
├── src/
│   ├── data/
│   │   ├── universe.py        # Wikipedia-based S&P 500 ever-member set
│   │   ├── fetcher.py         # yfinance ingestion with retry
│   │   ├── batch_fetcher.py   # Rate-limited universe-scale download
│   │   └── storage.py         # DuckDB schema and access layer
│   ├── stats/
│   │   ├── stationarity.py    # Hand-built ADF, validated against statsmodels
│   │   ├── cointegration.py   # Engle-Granger two-step test
│   │   ├── screening.py       # Sector-level pairwise screen
│   │   └── screening_v2.py    # Sub-industry-level screen with FDR control
│   └── backtest/
│       ├── events.py          # Event dataclasses
│       ├── execution.py       # Slippage and commission models
│       ├── portfolio.py       # Position tracking and hedge-ratio sizing
│       ├── strategy.py        # Mean-reversion on z-scored spread
│       ├── engine.py          # Bar-by-bar event loop
│       ├── train_test.py      # Chronological train/test split
│       ├── walk_forward.py    # Rolling re-estimation
│       ├── run_multi_pair.py  # Multi-pair walk-forward portfolio
│       └── metrics.py         # PSR, DSR, bootstrap CI
├── tests/                     # Unit tests
├── notebooks/                 # Exploratory analysis and visualization
│   ├── 01_data_exploration.ipynb
│   ├── 02_backtest_ndsn_otis.ipynb
│   ├── 03_train_test_ndsn_otis.ipynb
│   ├── 04_walkforward_ndsn_otis.ipynb
│   └── 05_multi_pair_results.ipynb
├── journal.md                 # Research journal
└── README.md
```


The journal in `journal.md` is the project's substantive record. Every
phase has a corresponding entry documenting motivation, methodology,
findings, and limitations. Bugs caught and fixed during development
are written up with their symptoms, diagnoses, and resolutions.

---

## Reproducing the results

```bash
# Environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build the universe and download price data (~15 minutes)
python -m src.data.batch_fetcher

# Run the sub-industry screen (~3 minutes)
python -m src.stats.screening_v2

# Backtests
python -m src.backtest.run_ndsn_otis        # in-sample
python -m src.backtest.train_test           # train/test split
python -m src.backtest.walk_forward         # walk-forward
python -m src.backtest.run_multi_pair       # multi-pair portfolio

# Adjusted metrics
python -m src.backtest.metrics
```

Some yfinance failures are expected (~15-20% of historical S&P 500
members have unrecoverable price history on the free feed; with a
paid feed like CRSP these would be available).

---

## What's intentionally not in scope

- **Point-in-time S&P 500 membership**: when screening, all eligible
  tickers from the ever-member set are considered, regardless of
  whether they were S&P 500 members on the specific dates in question.
  Full point-in-time correctness is a Phase 4 refinement.
- **Risk overlays**: no time stops, drawdown stops, or position limits.
  The March 2023 10-sigma event on NDSN/OTIS could have produced large
  losses in adverse circumstances. Production systems require these.
- **Higher-frequency strategies**: daily-bar only. Intraday and
  microstructure-frequency stat arb is a different problem.
- **Alternative strategies**: only Engle-Granger pairs trading is
  implemented. Johansen multi-asset cointegration and Kalman-filter
  dynamic hedge ratios are well-documented in the literature and
  would be reasonable extensions.

---

## What I would do differently

Three things, in order of marginal impact:

1. **Larger universe**: 1,384 sub-industry pairs is barely above the
   threshold where FDR control can find anything. A wider universe
   (e.g., the Russell 3000) would provide both more candidates and
   stronger evidence in any survivor.
2. **Risk overlays**: time stops and drawdown stops would have cut
   the worst losses (e.g., the -100-day SHORT position in the in-sample
   NDSN/OTIS backtest) and likely improved out-of-sample Sharpe.
3. **Portfolio construction**: replace equal-weight allocation with
   volatility-targeted sizing and dynamic capital reallocation based
   on trailing per-pair edge.

---

## References

- López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality.
- López de Prado, M. (2018). Advances in Financial Machine Learning,
  Chapters 7, 11, and 14.
- MacKinnon, J. G. (1996, 2010). Critical Values for Cointegration Tests.
- Engle, R. F. and Granger, C. W. J. (1987). Co-Integration and Error
  Correction: Representation, Estimation, and Testing.
- Pole, A. (2007). Statistical Arbitrage: Algorithmic Trading Insights
  and Techniques.
- Chan, E. (2009). Algorithmic Trading: Winning Strategies and Their
  Rationale.
- Harris, L. (2002). Trading and Exchanges: Market Microstructure for
  Practitioners.
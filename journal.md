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

## 2026-05-11 - Visual Exploration 

KO/PEP: both -doubled over 10 years , clearly co-move, both crashed together in March 2020. 
Pep outperformed KI from -2018 onward in absolute terms, with a noticeable PEP weakness in mid-2023 to end of period - unclear if mean-reverting or structural break

XOM/CVX: Clear regime structure. Pre-2020 stable co-movement at similar levels; 2020 oil crash drove both down (XOM more), recovery saw CVX break out strongly in 2022 energy bull while XOM lagged. Post-2022 both range-bound but at noticeable different levels than pre-pandemic.

Implication: any analysis pooling all 10 years assumes constant parameters that visibly don't hold. Rolling-window or regime-aware estimation will be necessary 

## 2026-05-11 - Rolling Correlation Analysis 

Surprising finding: rolling correlation paints opposite picture from price plots.

KO/PEP (visually clean long-run co-movement) has choppy short-run correlation swinging from 0.25-0.95 over the course of 10 years. Notable correlation crash in late 2024 coinciding with PEP underperformance - possible structural break. 

XOM/CVX (visually messy with clear regime shifts) actually has STRONGER
short-run correlation, especially post-2019 where it stays ~0.8-0.9.
Hypothesis: post-COVID oil markets became more macro-driven, reducing
idiosyncratic noise and increasing co-movement.

Key insight: long-run co-movement (price plots) and short-run sync
(correlation plots) are different things. Pairs trading wants both:
cointegration AND reasonable correlation. Phase 2 will test cointegration
specifically — which can hold even when correlation is choppy.

Practical implication: correlation thresholds as entry gates would create
constant in/out trading. Need smarter logic (z-score on cointegration
spread, not rolling correlation).

## 2026-05-11 — Log spread analysis (Phase 1 conclusion)

Two pairs, two failure modes of the naive log spread:

KO/PEP: visually noisy but plausibly mean-reverting around -0.957.
However, clear regime structure — spread lives above mean 2015-2020,
below mean 2020-2024, currently breaking back up. A static-mean strategy
would have lost during the regime transition in 2020 and would be losing
right now in late 2024.

XOM/CVX: NOT mean-reverting. Drifts -0.05 -> -0.80 -> -0.30 over 10 years.
Classic visual signature of a non-stationary series. Static thresholds
would generate false signals for years at a time. The series wanders;
shocks don't decay.

Implications for Phase 2:
1. Need proper hedge ratio from cointegration regression, not 1:1 assumption
2. Need formal stationarity test (Augmented Dickey-Fuller) to reject pairs
   like XOM/CVX before risking capital on them
3. Need rolling re-estimation to handle regime shifts like KO/PEP 2020 break

Phase 1 complete. The naive analysis revealed exactly the right problems
that cointegration testing is designed to solve. Time to formalize.

## 2026-05-13 — Augmented Dickey-Fuller from first principles

Built `src/stats/stationarity.py` and `tests/test_stationarity.py`. Hand-
implemented the ADF regression via explicit OLS, with automatic AIC/BIC
lag selection bounded by Schwert's rule (max_lags = floor(12 * (T/100)^0.25)).

**Validated against statsmodels** within 0.01 on both stationary AR(0.5)
and random walk series. Both qualitative checks (stationary rejects null,
random walk does not) and quantitative comparisons pass.

### Subtle bug encountered and fixed

Initial implementation refit each candidate lag count using all available
observations for that lag, then picked the AIC minimum. This was wrong:
AIC depends on sample size, so comparing AIC values computed on different
sample sizes (979 obs at lags=20 vs 999 obs at lags=0) is invalid. The
log-likelihood scales with n, but the 2k penalty doesn't, so larger
samples systematically produce lower AIC for any model — making the
selector prefer the smallest lag count regardless of fit quality. Or in
my case, the opposite: spurious preference for high lag counts because
the residual variance fell faster than the penalty rose.

Symptom: test statistic was -7.48 on data where statsmodels gave -16.20.
Both rejected the null at 5%, but the magnitude was off by 2x — the kind
of "qualitatively right, quantitatively wrong" failure that's the most
dangerous in quant research.

Fix: during lag selection, constrain all candidate lag counts to the
sample size required by max_lags. After picking the best lag, refit at
that lag using the full sample available, for the final reported test
statistic. This matches statsmodels' procedure.

### Key technical insight

ADF t-statistic does NOT follow the standard t-distribution under the
null. Under H_0 of unit root, the regressor y_{t-1} is non-stationary,
violating standard regression inference assumptions. The distribution
was derived by Dickey and Fuller via simulation; MacKinnon (2010)
provides modern asymptotic critical values: -3.43 at 1%, -2.86 at 5%,
-2.57 at 10% (with constant, no trend).

### Lesson for the project

Code that gives plausible-but-wrong answers is the most dangerous failure
mode. Defense: validate every nontrivial numerical implementation against
a reference within tight tolerances. The test
`test_matches_statsmodels_stationary` is what caught this bug. Without it,
the project would have proceeded with a subtly broken cointegration
screener.

### Next

Implement Engle-Granger cointegration in `src/stats/cointegration.py`:
1. OLS regression of log(P1) on log(P2) to estimate hedge ratio
2. ADF on the residuals (using this module)
3. Compare against Engle-Granger critical values (-3.37 at 5% for k=2),
   not standard ADF critical values, because residuals come from an
   estimated regression

## 2026-05-20 — Engle-Granger cointegration

Built `src/stats/cointegration.py` and `tests/test_cointegration.py`. The
implementation is short because it reuses the hand-built ADF from yesterday.

Procedure: OLS regression of log(P1) on log(P2) for hedge ratio β,
then ADF on residuals using Engle-Granger critical values (stricter
than standard ADF because β is estimated rather than known).

Critical values used (MacKinnon 2010, k=2, constant, no trend):
- 1%: -3.96
- 5%: -3.37
- 10%: -3.07

These differ from standard ADF (-3.43, -2.86, -2.57) because the
residuals have less variation than a truly random series — they were
constructed by minimizing fit error.

Validation: matches statsmodels.tsa.stattools.coint within 0.5 on
both synthetic cointegrated and independent random walk pairs.
Looser tolerance than ADF because statsmodels has subtle internal
differences in lag selection within its embedded ADF.

Results on real data:
- KO/PEP: [paste from your smoke test]
- XOM/CVX: [paste from your smoke test]

Predicted before running: KO/PEP marginal cointegration, XOM/CVX not
cointegrated (based on visual analysis from the exploration notebook).
Actual: [your observations]

Next: screening engine — apply this to all C(10,2) = 45 pairs in the
current universe, rank by ADF statistic, identify candidates that
pass at 5% significance.


## 2026-05-21

Applied Engle-Granger screening to all C(10,2) = 45 pairs in the universe. 

**Results:** 
- 1 pair passed at 5% (KO/PEP, ADF=-3.42, threshold=-3.37)
- 0 pairs passed at 1%
- 0 pairs passed Bonferroni-adjusted threshold (0.05/45 = 0.0011)

**Interpretation:** Under the null hyptothesis of no cointegration anywhere in the universe, the expected 
number of false postives at 5% is 0.05 * 45 = 2.25. We obsered 1 rejection. This is fewer than expected
under the null. Statistically, the data proves esstially no evidence that ANY pair in this universe 
is truly cointegrated. 

KO/PEP, while passing the individual test, cannot be statistically
distinguished from a false positive given the multiple-testing burden.

**This is an honest, important finding -- not a failure.** Cointegration
in random pairs of large-cap US equities is rare. Three reasons:
1. Company-specific factors (M&A, capital allocation, business mix
   shifts) cause structural drift even within sectors
2. ETF and index flows synchronize correlation without tethering levels
3. True cointegration is a strong statistical claim; even slow drift
   breaks it

**Runner-ups are revealing:**
- AAPL/KO (ADF -3.27, β=3.01): no economic story; almost certainly
  spurious. Hedge ratio correctly captures AAPL's higher volatility.
- JNJ/MSFT (ADF -3.25, β=0.29): no economic story; same skepticism.
- JNJ/PEP, JNJ/KO: consumer staples adjacencies; weak economic story.
- XOM/CVX did not appear in top 10 -- confirms visual intuition from
  exploration notebook that the spread drifts.

**Hedge ratios carry real information even when pairs aren't cointegrated.**
The β values correctly reflect relative volatility scaling. This is a
subtle distinction: hedge ratios are well-defined OLS coefficients
regardless of stationarity. Cointegration tests whether the resulting
spread is tradeable.

**Implications for Phase 3:**
- 10-ticker universe is too small to find robust trading candidates
- Backtesting only KO/PEP would suffer from selection bias (we found
  the pair by screening; whatever Sharpe ratio we get is inflated)
- Strategy: build the backtester on synthetic cointegrated data first
  (where we know the right answer), validate it works correctly, THEN
  expand the universe and apply to real candidates

This sequencing also disentangles two different research problems
(backtester correctness vs. signal discovery) which is much cleaner
methodology.

**Next:** Phase 3 begins with synthetic data generation and the
event-driven backtester architecture in src/backtest/.

## 2026-05-21 — Phase 3a: universe expansion

Switched from 10-ticker hand-picked universe to full S&P 500 ever-member set
(2015-2024) for proper screening at scale.

Universe construction (`src/data/universe.py`):
- Source: Wikipedia "List of S&P 500 companies" (current + change log)
- Required custom User-Agent header (Wikipedia blocks default Python UA)
- Combined current 503 constituents with 240 historical members removed
  during our window -> 740 ticker ever-member set
- This is the textbook fix for survivorship bias in stat arb screening

Batch fetcher (`src/data/batch_fetcher.py`):
- Rate-limit aware (0.3s delay between requests, configurable)
- Resumable via skip_existing check against DB
- Per-ticker retry with exponential backoff
- Expected runtime: 30-45 min for 740 tickers

Expanded screening (`src/stats/screening_v2.py`):
- Sector prefiltering: only within-sector pairs
- Reduces ~273k all-pairs to ~13k within-sector pairs
- Economically defensible: same-sector pairs share macro drivers
- Parallelized via multiprocessing.Pool (8 workers on M-series Mac)
- FDR control via Benjamini-Hochberg, not Bonferroni
  - Bonferroni at 13k tests too strict (alpha = 3.8e-6)
  - BH controls expected false discovery RATE, not family-wise error
  - Standard in genomics, A/B testing, modern quant research

Key methodological talking points for interviews:
1. Survivorship bias correction via ever-member set
2. Multiple testing: why FDR > Bonferroni at scale
3. Sector prefiltering as economic vs statistical gate
4. Multiprocessing for embarrassingly parallel screening workloads

Next (once fetch completes): run screening_v2, examine results, identify
top candidates for backtester development.

I constructed an ever-member set of 740 tickers spanning 2015-2024 by combining current S&P 500 constituents with the Wikipedia change log. 
Of these, 18% had unrecoverable history on yfinance — companies like Allergan, Celgene, and First Republic Bank that were acquired or delisted. 
These specific cases illustrate why survivorship bias matters: a naive backtest using only currently-available tickers would silently exclude 
every M&A target and every failed company, producing inflated results. With access to a paid feed like CRSP, the missing data would be recoverable; 
with yfinance, we accept partial recovery and document the limitation.

## 2026-05-21 — Phase 3a: universe expansion + large-scale screening

### Universe construction

Switched from 10-ticker hand-picked universe to S&P 500 ever-member set
2015-2024 to address survivorship bias and enable scale-appropriate
screening.

- Source: Wikipedia "List of S&P 500 companies" (current page + change log)
- Required custom User-Agent (Wikipedia rejects default Python urllib)
- 740 unique ticker ever-member set: 503 current + 240 historically-removed
- 618 tickers (83.5%) successfully fetched via yfinance
- 122 failures (AGN, ATVI, CELG, FRC, RTN, SIVB, TWTR, MON, etc.) — yfinance
  has known gaps in delisted ticker history. With a paid feed (CRSP,
  Compustat) these would be recoverable. Documented limitation.
- Final database: 1.47M rows, 12-minute total runtime

### Screening v2: sector-restricted with FDR control

13,029 within-sector pairs tested. Multiprocessing on 8 cores brought
total runtime to ~3 minutes.

Results:
- Naive 5% (unadjusted): 1,051 pairs (8.1%, vs 5% expected under null)
- Naive 1% (unadjusted): 219 pairs (1.7%, vs 1% expected under null)
- BH-FDR at 10%: **0 pairs**
- BH-FDR at 5%: 0 pairs

### Honest interpretation of the FDR=0 finding

For the rank-1 pair (NDSN/OTIS, p=1.7e-5) to pass BH-FDR at 10%, it
needs p < 7.7e-6 (= 1/13029 * 0.10). It misses by a factor of 2.
No single pair in 13k+ is overwhelmingly strong enough to survive.

This is a real finding, not a methodological failure:
1. Modern equity markets have substantially less cointegration than
   1990s/2000s literature suggests
2. ETF and index flows synchronize equities into common factors,
   creating correlation without level-tethering
3. Multiple-testing burden at scale is severe; you need either much
   smaller universes or much stronger signals to survive FDR

Professional stat arb funds have moved to: higher frequencies
(intraday/microstructure), alternative data, ML on weak signal
combinations. Classical daily-frequency cointegration on equity pairs
is a teaching example more than a current production strategy.

### Numerics bugs caught and fixed during development

Three bugs caught by inspecting output, none would have raised an
exception:

1. **ADF lag selection** (Phase 2): inconsistent sample sizes across
   AIC candidates produced wrong-by-2x test statistics. Fixed via
   fixed-sample lag selection matching statsmodels behavior.

2. **Linear interpolation p-values** (Phase 3a, first attempt): capped
   minimum p at 0.005 due to interpolation grid limits. Produced FDR=0
   even when 219 pairs passed naive 1%.

3. **Inverted cubic polynomial** (Phase 3a, second attempt): hand-rolled
   MacKinnon response surface had sign convention backwards. Output
   showed strongly-cointegrated pairs with p≈0.99 and positive-ADF
   non-cointegrated pairs with p≈10^-5. Fixed by delegating to
   statsmodels.tsa.adfvalues.mackinnonp.

Each bug was qualitatively-right, quantitatively-wrong — the most
dangerous failure mode in quant research. Caught by reference
validation and by inspecting whether results matched theoretical
expectations.

### Top 20 by ADF statistic — economically interpretable

Several pairs have plausible economic stories:
- WEC/XEL: regulated multi-state utilities (classic pair)
- NTRS/TFC: specialty/regional banks
- NDSN/OTIS: industrial manufacturers
- AVGO/ORCL: mature large-cap tech (weaker story but maybe shared
  enterprise-spend exposure)

Others lack economic stories and are likely false positives despite
low p-values:
- ABBV/CI, ICE/WTW, BDX/MRNA

### Phase 3b/3c plan

- Phase 3b: re-screen at sub-industry level. Fewer tests (~1500-3000)
  improves FDR power. Combine survivors with top-ranked
  economically-sensible pairs for backtester input.

- Phase 3c: event-driven backtester. Will explicitly disclose selection
  bias for non-FDR-surviving pairs. Out-of-sample walk-forward
  validation is the real test

## 2026-05-22 — Phase 3b: sub-industry rerun finds FDR survivor

Added `group_by` parameter to screening_v2 with sub_industry as default.

### Numbers
- 1,384 pairs tested (vs 13,029 at sector level)
- 123 pass naive 5% (vs 69 expected under null — modest excess)
- 27 pass naive 1% (vs 14 expected — meaningful excess)
- **1 pair passes BH-FDR at 5% and 10%: NDSN/OTIS** (p = 1.7e-5,
  threshold = 7.2e-5 at FDR-10%)

### Why sub-industry granularity worked
Reduced multiple-testing burden by factor of 9. NDSN/OTIS p-value
unchanged (same data, same test) but now passes the less stringent
1.4k-test FDR threshold of 7.2e-5 vs 13k-test threshold of 7.7e-6.

Sub-industry is also more economically defensible than sector. Within
"Financials" you can pair an investment bank with an insurance company
and ask cointegration tests to flag the relationship — but there's no
economic mechanism that should produce such cointegration. Within
"Investment Banking & Brokerage," cointegration has a coherent story.

### Candidate selection for backtester

Tier 1 (FDR-validated):
- NDSN/OTIS — industrial machinery, p = 1.7e-5

Tier 2 (strong economic story + low p-value, selection bias acknowledged):
- CARR/TT — HVAC manufacturers, p = 0.0002
- MA/V — payment card networks, p = 0.001
- EOG/FANG — Permian Basin shale E&P, p = 0.001
- TRGP/WMB — natural gas midstream, p = 0.004

Five total. Documented selection bias for Tier 2: these pairs were
chosen because they ranked high AND have coherent economic stories.
Out-of-sample walk-forward analysis is the real test.

### Excluded with notes
- JNJ/VTRS (negative hedge ratio — likely artifact, not true coint)
- FTV/OTIS (only 1203 obs — Otis spun off 2020, insufficient history)
- GOOG/GOOGL (statistically cointegrated but spread too tight to trade
  profitably; included only as a methodological sanity check)

### Next: Phase 3c — event-driven backtester
Will build in src/backtest/. Architecture: portfolio state, signal
generator, execution engine, P&L tracker. Critical: avoid look-ahead
bias, model transaction costs realistically, use walk-forward
out-of-sample validation.

## 2026-05-22 — Phase 3c.1: event-driven backtester on NDSN/OTIS

Built the full event-driven backtester architecture (events, execution,
portfolio, strategy, engine) and ran in-sample on NDSN/OTIS. Code in
`src/backtest/`, analysis in `notebooks/02_backtest_ndsn_otis.ipynb`.

### Architecture
Five modules following production patterns:
- events.py: immutable dataclasses for MarketData/Signal/Order/Fill
- execution.py: commission + half-spread + market impact slippage
- portfolio.py: hedge-ratio sizing, strict cash+positions accounting
- strategy.py: rolling z-score mean-reversion with shift(1) discipline
- engine.py: chronological bar-by-bar event loop

Verified accounting identity holds exactly at every step (initial
capital - costs = final equity to the penny in the smoke test).

### Results (in-sample, April 2020 - Dec 2024)
- 1204 bars, 45 fills, 22 round-trips
- Total return +3.81%, annualized +0.80%, ann. vol 1.34%, Sharpe 0.59
- Max drawdown -1.50%
- Win rate 77.3%, mean winner $351, mean loser $284
- Total costs $17 (negligible at $10k notional)

### Interpretation

These numbers look small but the shape is right. Pairs trading is
dollar-neutral by construction — net market exposure ~0, so vol is
inherently low. Sharpe 0.59 with 1.34% vol is comparable to SPY's
Sharpe with 16% vol over the same period. The strategy extracts a
small uncorrelated edge.

The whole point of stat arb is running many such strategies in
parallel: 50 uncorrelated pairs each at Sharpe 0.6 → portfolio Sharpe
~4.2 by sqrt(N) scaling. Single-pair results aren't impressive in
isolation; the methodology is.

### Notable observation: 10-sigma event in March 2023

Z-score reached ~-10 briefly. Under normal distribution this is
mathematically "impossible" (10^-23). In reality it means the 60-day
rolling vol estimate broke down — local volatility spike, regime
change, or sector rotation. Strategy traded through it and exited
profitably, but this would NOT be safe in production without
additional overlays:
- Time stops (exit positions held > 60 days)
- Drawdown stops (exit if MtM loss > 2% of capital)
- Regime detection (don't trade if spread vol exceeds X over baseline)

### Specific failure case worth noting

Round-trip #5: SHORT entry 2021-09-07, held 100 days, lost $316.
Largest loser, longest holding period. Classic regime-break failure.
A 60-day time stop would have cut this earlier.

### Honest selection-bias acknowledgment

In-sample backtest using:
- A pair selected because it passed FDR screening on this same data
- Cointegration parameters (β, α) estimated on this same data
- Strategy parameters (lookback, entry/exit z) not tuned but also not
  tested out-of-sample

Sharpe 0.59 is likely inflated by these biases. The honest test is
out-of-sample, which Phase 3c.2 (train/test split) and Phase 3c.3
(walk-forward) will perform.

### Next session: Phase 3c.2

Standard 70/30 train/test split:
- Train 2020-04 to 2023-04 (~3 years)
- Test 2023-04 to 2024-12 (~1.7 years)
- Estimate β, α, and strategy parameters on train
- Apply frozen parameters to test
- Compare in-sample vs out-of-sample Sharpe to quantify the
  selection/overfitting penalty


## 2026-05-23 — Phase 3c.2: train/test split reveals overfitting

### Setup
- Pair: NDSN/OTIS (FDR-validated from screening_v2 sub-industry rerun)
- Train: pre-2023-06-01 (effectively April 2020 - June 2023 due to OTIS
  listing date; ~790 tradeable bars)
- Test: 2023-06-01 - 2024-12-30 (~398 bars)
- Parameters: re-estimated β=0.7036, α=2.3489 on train only
- Train-period ADF on residuals: -5.946 (stronger than full-window -5.516)

### Results

| Metric              | Train      | Test       |
|---------------------|------------|------------|
| Total return        | +4.16%     | -0.17%     |
| Annualized return   | +1.29%     | -0.10%     |
| Annualized vol      | 1.45%      | 1.02%      |
| Sharpe ratio        | **0.89**   | **-0.10**  |
| Max drawdown        | -1.42%     | -0.95%     |
| Trades              | 29         | 13         |

**Overfitting gap: +0.99 Sharpe units.** The entire apparent edge
disappears out-of-sample.

### Interpretation

Three non-mutually-exclusive explanations:

1. **Selection bias.** NDSN/OTIS was chosen because it survived a
   1384-test FDR screen — explicitly selected as the most cointegrated
   in the data we had. The training Sharpe is *conditional on this
   selection*; the test period does not benefit from it.

2. **Decaying cointegration.** The train-period ADF (-5.946) is stronger
   than the full-window ADF (-5.516). The relationship was strongest
   early and has weakened over time. NDSN and OTIS share industrial
   machinery exposure but differ in growth trajectories, customers,
   and capital structures. Slow divergence breaks cointegration.

3. **2024 regime shift.** Industrial sector dynamics may have shifted
   in some way. Harder to defend without economic context.

The cumulative explanation is probably #1 + some #2. Hard to separate
without more data; not necessary to separate for the actionable
conclusion.

### The 10-sigma event matters

The March 2023 z-score spike to -10 (noted in Phase 3c.1 journal)
falls *inside the train window*. That single regime break produced
a large mean-reverting profit when the spread snapped back. Without
that event, train Sharpe would likely be much lower. This is a clean
example of why backtest research must be skeptical of large gains
from rare events — they don't replicate.

### Methodological win: this is the project working as designed

Most undergraduate quant projects stop at the in-sample backtest and
report inflated Sharpes. The infrastructure built in Phases 1-3 —
hand-built validated cointegration test, survivorship-corrected
universe, sub-industry FDR screening, event-driven backtester with
look-ahead prevention — exists precisely so that a finding like this
*can* surface. The negative test Sharpe is not a project failure;
it's an honest research finding.

The story for interviews:
"I built a complete pairs-trading research pipeline including
survivorship-corrected universe construction, sub-industry-level
cointegration screening with FDR control, and an event-driven
backtester with proper out-of-sample evaluation. The headline finding
was that my best FDR-validated candidate pair, NDSN/OTIS, produced
a Sharpe of 0.89 in-sample but -0.10 out-of-sample — an overfitting
gap of 0.99 Sharpe units. This is the kind of finding that's
*invisible* to projects that don't separate train and test data,
and motivated the walk-forward analysis in the next phase."

### Minor reporting issue noted

The `n_train_bars=2117` in the summary counts raw days in the train
window, including pre-OTIS-listing days. The backtester correctly
drops these via `.dropna()` before running, so the actual number of
*tradeable* train bars is ~790. The reported summary should be fixed
to reflect tradeable bars rather than raw window bars. Minor cleanup,
not affecting any numerical result.

### Next: Phase 3c.3 (walk-forward analysis)

Train/test gave one verdict. Walk-forward gives many: re-estimate
parameters every N months on the trailing window, apply to the next
M months, slide forward. Produces a time series of out-of-sample
performance that reveals whether the strategy fails everywhere or
just in certain regimes.

The setup:
- Initial training window: 24 months
- Re-estimation frequency: every 6 months
- Test window: 6 months (until next re-estimation)
- Walk through the data measuring each out-of-sample window separately
- Compare aggregated out-of-sample Sharpe to the train/test result

## 2026-05-24 — Phase 3c.3: walk-forward reveals regime-dependent edge

Walk-forward analysis with 24-month training window, 6-month re-estimation
step. 5 walk-forward steps spanning April 2022 - September 2024.

### Headline result
Aggregate stitched out-of-sample Sharpe: **0.52** (vs train/test result
of -0.10 from Phase 3c.2). The strategy DOES generalize, but only when
parameters are re-estimated periodically. Frozen parameters from a
single fit don't work.

### Layered findings

**1. Per-step performance is wildly regime-dependent:**
- Step 0 (Mar-Sep 2022): Sharpe +0.37 (β=0.735, ADF=-4.82)
- Step 1 (Sep 2022-Mar 2023): Sharpe +2.31 (β=0.693, ADF=-2.76)
- Step 2 (Mar-Sep 2023): Sharpe -0.71 (β=0.544, ADF=-1.96)
- Step 3 (Sep 2023-Mar 2024): Sharpe -0.45 (β=0.517, ADF=-2.48)
- Step 4 (Mar-Sep 2024): Sharpe +0.99 (β=0.584, ADF=-2.51)

**2. Step 1 dominates aggregate Sharpe.** The 10-sigma z-score event
in March 2023 produced one big winner. Median per-step Sharpe is
+0.37; mean is +0.50; weighted aggregate is +0.52. Remove step 1 and
performance is near zero.

**3. Cointegration strength decayed structurally.** ADF statistic on
trailing residuals fell from -4.82 (step 0, strongly significant) to
-1.96 (step 2, not significant at any conventional level) and stayed
weak. The relationship was breaking down.

### The most important conceptual takeaway

There's a non-monotonic relationship between cointegration test
strength and forward performance:
- Step 1: weak ADF (-2.76) but strongest forward Sharpe (+2.31)
- Step 4: similarly weak ADF (-2.51) but positive Sharpe (+0.99)
- Steps 2-3: weak ADF, negative Sharpe

Standard ADF gating (only trade if trailing ADF < -3.37) would have
skipped steps 1-4, missing both the big winner AND the losing periods.
This is a clean reminder that statistical significance of a
cointegrating relationship does not predict forward profitability
in any simple way -- the relationship between cointegration strength
and trading edge is subtle.

### What this changes about the project narrative

Yesterday I framed the train/test result as "the strategy fails out-
of-sample." Today's walk-forward complicates that: the strategy
*can* be made to work out-of-sample if parameters are kept fresh
through rolling re-estimation. The honest summary for interviews is:

"With static parameters, the strategy did not generalize. With
walk-forward re-estimation, the aggregate out-of-sample Sharpe was
0.52, but performance was highly regime-dependent. A single 6-month
window dominated aggregate results, driven by a 10-sigma mean-
reversion event. The strategy has intermittent edge, not steady edge."

### Parameter drift quantified
- β range: [0.517, 0.735] — substantial variation
- α range: [2.216, 3.166]
- ADF range: [-4.82, -1.96]

### Methodological wins
1. Walk-forward uncovered regime dependence invisible to train/test
2. Concatenated stitched equity provides honest realistic experience
3. Each step's pre-test ADF is information available in real time;
   could power a regime filter in production

### Next steps (Phase 3c.4 onward)
- Extend to the other 4 candidate pairs (CARR/TT, MA/V, EOG/FANG,
  TRGP/WMB) under same walk-forward methodology
- Aggregate across pairs (portfolio-level Sharpe)
- Phase 3d: deflated Sharpe ratio adjustment, bootstrapped confidence
  intervals on the aggregate

## 2026-05-24 — Phase 3c.4: multi-pair walk-forward reveals strategy decay

Ran walk-forward on all 5 candidate pairs, then equal-weight portfolio
aggregation over the common 2022-2024 window.

### Per-pair aggregate Sharpes (full available history)

| Pair       | Steps | Trades | Agg Sharpe | Total Return | Median Step Sharpe |
|------------|-------|--------|------------|--------------|---------------------|
| MA/V       | 15    | 83     | +0.55      | +3.8%        | +0.03               |
| NDSN/OTIS  | 5     | 26     | +0.52      | +1.6%        | +0.37               |
| EOG/FANG   | 15    | 78     | +0.45      | +4.9%        | +0.69               |
| TRGP/WMB   | 15    | 69     | -0.21      | -4.7%        | +0.16               |
| CARR/TT    | 5     | 24     | -0.37      | -6.6%        | +0.23               |

### Common-window (2022-04 to 2024-09) per-pair Sharpes

| Pair       | Common Sharpe | Common Return |
|------------|---------------|---------------|
| MA/V       | +1.39         | +2.58%        |
| NDSN/OTIS  | +0.52         | +1.64%        |
| CARR/TT    | -0.37         | -6.57%        |
| TRGP/WMB   | -0.42         | -1.49%        |
| EOG/FANG   | -0.52         | -1.68%        |

### Portfolio (equal-weight, common window)
- Sharpe: **-0.30**
- Return: -1.10%
- Two pairs positive, three negative; equal-weighting drags portfolio
  into the red despite MA/V's strong +1.39

### Three layered findings

**1. Edge decay is systematic.** Most pairs show declining per-step
Sharpe over time. EOG/FANG had Sharpes of [2.71, 0.68, 2.01, 0.93, 2.12]
in its first 5 steps (2017-2019) and [-0.14, 0.42, -0.72, -2.45] in
its last 4 steps (2023-2024). TRGP/WMB shows the same pattern. This
is consistent with academic literature on stat arb capacity exhaustion --
the strategy was very profitable in the 1990s/2000s, became crowded in
the 2010s, and has now been largely arbitraged away.

**2. Structural tightness predicts durability.** MA/V is the only pair
showing strongest Sharpes in recent steps (+2.94 and +3.55 in late
steps). Mastercard and Visa are essentially the same business: card
networks with identical revenue models, regulatory environments, and
customer bases. The tighter the structural economic link, the more
durable cointegration over time.

**3. Equal-weight portfolio construction is naive and dangerous.**
Three losing pairs with -0.37 to -0.52 Sharpe drag down the portfolio
even though MA/V (+1.39) and NDSN/OTIS (+0.52) had positive edge.
$20k allocated to a losing pair costs the same in dollar terms as
$20k allocated to a winning pair. Real funds use risk-budgeting,
volatility-targeted sizing, and dynamic capital reallocation.

### The "same data, different conclusion" issue

EOG/FANG aggregate Sharpe over full 15 steps: +0.45. Over the common
2022-2024 5 steps: -0.52. Same strategy, same code, totally different
conclusion depending on which window you look at. This is a real
methodological caution -- backtest conclusions are time-window
dependent, and reporting a single number obscures regime variation.

### Specific observation: CARR/TT failure despite strong economic story

CARR and TT are both major HVAC manufacturers. They share customers,
regulatory environment, commodity exposure, and end markets. The
economic story for cointegration is excellent. Yet the strategy fails
on this pair (-0.37 aggregate, -6.57% in common window). Lesson:
economic similarity does not guarantee statistical tradability. The
specific dynamics of the spread (volatility, mean-reversion speed,
regime stability) matter as much as the underlying business links.

### What I would say in an interview

"I ran walk-forward analysis on five FDR-screened candidate pairs and
found that aggregate out-of-sample Sharpe ranged from +0.55 to -0.37.
Most pairs showed edge decay over time -- strong in 2015-2020, weak
in 2022-2024. Mastercard/Visa was the exception, with strongest
performance in recent windows due to the structural similarity of the
two card networks. An equal-weight portfolio of all five pairs produced
a negative aggregate Sharpe (-0.30) despite two pairs having positive
individual edge -- a clean illustration of why portfolio construction
and risk-budgeting matter as much as signal generation in stat arb."

### Next: Phase 3d

Need rigorous statistical adjustment for the selection bias and
multiple-testing burden. Specifically:
- Deflated Sharpe ratio (Lopez de Prado) on each pair
- Bootstrap confidence intervals on aggregate Sharpe
- Probabilistic Sharpe ratio
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
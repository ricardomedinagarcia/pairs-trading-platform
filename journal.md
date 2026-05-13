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

## 2026-05-12 — Augmented Dickey-Fuller from first principles

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
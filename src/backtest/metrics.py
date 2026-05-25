"""Adjusted performance metrics for backtest evaluation.

This module addresses the central critique of any screened backtest:
"You ran N tests and reported the best result; how do we know it's not
just the luckiest of N coin flips?"

Three complementary tools:

1. Deflated Sharpe Ratio (López de Prado, 2014). Adjusts an observed
   Sharpe downward based on the number of independent trials run.
   A "deflated Sharpe > 0" is a much stronger statement than a raw
   "Sharpe > 0" when many candidates were screened.

2. Probabilistic Sharpe Ratio. Returns P(true Sharpe > 0 | observed
   Sharpe), accounting for sample size and return moments. The
   probability gives a cleaner answer than "is Sharpe statistically
   significant" because it directly addresses the question of edge
   existence.

3. Bootstrap confidence intervals. Empirical CI on Sharpe via
   resampling. Doesn't assume normality, so robust to skewed/
   leptokurtic return distributions typical in finance.

References:
- López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality"
- López de Prado (2018), "Advances in Financial Machine Learning",
  Chapter 14
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SharpeAnalysis:
    """Bundle of Sharpe-related diagnostics for a return series."""

    observed_sharpe: float           # raw annualized Sharpe
    n_observations: int
    skewness: float
    excess_kurtosis: float
    probabilistic_sharpe: float       # P(true Sharpe > sr_benchmark)
    deflated_sharpe: float            # P(true Sharpe > 0 | N trials)
    bootstrap_mean: float             # bootstrap mean Sharpe
    bootstrap_ci_lower: float         # bootstrap CI lower bound (5th pct)
    bootstrap_ci_upper: float         # bootstrap CI upper bound (95th pct)
    n_trials_assumed: int             # how many independent trials we deflated against

    def __repr__(self) -> str:
        return (
            f"SharpeAnalysis(observed={self.observed_sharpe:.3f}, "
            f"PSR={self.probabilistic_sharpe:.3f}, "
            f"DSR={self.deflated_sharpe:.3f}, "
            f"95%CI=[{self.bootstrap_ci_lower:.2f}, {self.bootstrap_ci_upper:.2f}])"
        )


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Standard annualized Sharpe ratio.

    Returns 0.0 if the series has no variation (avoids division by zero).
    """
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sharpe_benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probabilistic Sharpe Ratio (PSR).

    P(true_Sharpe > sharpe_benchmark | observed return series).
    Accounts for skewness and kurtosis of the returns.

    Returns a probability in [0, 1]. A PSR > 0.95 is conventionally
    interpreted as "we're 95% confident the strategy has true Sharpe
    above the benchmark."

    Formula (López de Prado, 2014, Eq. 2):
        PSR = Phi((SR - SR*) * sqrt(n-1) / sqrt(1 - skew*SR + 0.25*(kurt-3)*SR^2))
    where Phi is the standard normal CDF.
    """
    returns = returns.dropna()
    n = len(returns)
    if n < 5 or returns.std() == 0:
        return 0.5  # uninformative

    sr_obs = annualized_sharpe(returns, periods_per_year)
    # Convert annualized benchmark and observed to per-period for the variance formula
    sr_obs_period = sr_obs / np.sqrt(periods_per_year)
    sr_bench_period = sharpe_benchmark / np.sqrt(periods_per_year)

    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))  # raw (Pearson) kurtosis

    denom_var = 1.0 - skew * sr_obs_period + (kurt - 1.0) / 4.0 * sr_obs_period ** 2
    if denom_var <= 0:
        # Numerical edge case; fall back to normal approximation
        denom_var = 1.0

    z = (sr_obs_period - sr_bench_period) * np.sqrt(n - 1) / np.sqrt(denom_var)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    estimated_sharpe_std: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe Ratio (DSR), López de Prado (2014).

    P(true_Sharpe > 0 | observed Sharpe, n_trials).
    Higher n_trials makes deflation stricter -- the more candidates
    that were screened, the higher the bar to claim a "real" edge.

    Args:
        returns: Out-of-sample return series of the candidate strategy.
        n_trials: Number of independent candidates that were considered.
            For our project, n_trials ≈ 1384 (the sub-industry screen).
        estimated_sharpe_std: Std deviation of Sharpes across trials.
            If None, uses a default approximation suitable when the trial
            Sharpes are roughly N(0,1)-distributed under the null.

    Formula: deflate the benchmark Sharpe from 0 to the expected maximum
    Sharpe under the null across n_trials, then compute PSR against that
    higher benchmark.
    """
    returns = returns.dropna()
    if len(returns) < 5:
        return 0.5

    if estimated_sharpe_std is None:
        # Conservative default: under the null, Sharpe is approximately
        # N(0, 1/sqrt(periods_per_year)). For PSR calc, we work in
        # annualized units, so std is approximately 1 if we set this
        # parameter for the annualized benchmark. We use a heuristic:
        # std of annualized Sharpe under the null ≈ 1 for reasonable T.
        estimated_sharpe_std = 1.0

    # Expected maximum of n_trials draws from N(0, estimated_sharpe_std).
    # Approximated by López de Prado (2014, Eq. 4):
    # E[max] ≈ estimated_sharpe_std * ((1 - euler) * Phi^-1(1 - 1/N) + euler * Phi^-1(1 - 1/(N*e)))
    euler_mascheroni = 0.5772156649
    e_const = np.e
    if n_trials < 2:
        expected_max = 0.0
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e_const))
        expected_max = estimated_sharpe_std * (
            (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2
        )

    return probabilistic_sharpe_ratio(
        returns, sharpe_benchmark=expected_max, periods_per_year=periods_per_year
    )


def bootstrap_sharpe(
    returns: pd.Series,
    n_resamples: int = 5_000,
    confidence: float = 0.90,
    periods_per_year: int = 252,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval on Sharpe ratio.

    Resamples the return series with replacement n_resamples times,
    computes Sharpe on each resample, returns (mean, lower, upper)
    where (lower, upper) is the (1-confidence)/2 to 1-(1-confidence)/2
    quantile interval.

    Default confidence=0.90 gives the 5th and 95th percentiles.
    """
    returns = returns.dropna()
    if len(returns) < 10:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    n = len(returns)
    sharpes = np.empty(n_resamples)
    arr = returns.values
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        std = sample.std()
        sharpes[i] = (sample.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q
    return (
        float(sharpes.mean()),
        float(np.quantile(sharpes, lo_q)),
        float(np.quantile(sharpes, hi_q)),
    )


def full_sharpe_analysis(
    returns: pd.Series,
    n_trials: int,
    periods_per_year: int = 252,
    n_bootstrap: int = 5_000,
    confidence: float = 0.90,
    seed: int = 42,
) -> SharpeAnalysis:
    """One-call analysis bundling all three diagnostics."""
    returns = returns.dropna()
    n = len(returns)
    sr = annualized_sharpe(returns, periods_per_year)
    psr = probabilistic_sharpe_ratio(returns, sharpe_benchmark=0.0,
                                      periods_per_year=periods_per_year)
    dsr = deflated_sharpe_ratio(returns, n_trials=n_trials,
                                 periods_per_year=periods_per_year)
    mean_sh, lo, hi = bootstrap_sharpe(returns, n_resamples=n_bootstrap,
                                        confidence=confidence,
                                        periods_per_year=periods_per_year,
                                        seed=seed)

    skew = float(stats.skew(returns, bias=False)) if n >= 3 else 0.0
    kurt = float(stats.kurtosis(returns, fisher=True, bias=False)) if n >= 4 else 0.0

    return SharpeAnalysis(
        observed_sharpe=sr,
        n_observations=n,
        skewness=skew,
        excess_kurtosis=kurt,
        probabilistic_sharpe=psr,
        deflated_sharpe=dsr,
        bootstrap_mean=mean_sh,
        bootstrap_ci_lower=lo,
        bootstrap_ci_upper=hi,
        n_trials_assumed=n_trials,
    )


if __name__ == "__main__":
    """Apply the metrics to the saved NDSN/OTIS walk-forward equity and
    to the multi-pair portfolio equity."""

    print("="*70)
    print("Adjusted Sharpe Analysis")
    print("="*70)

    # NDSN/OTIS walk-forward
    print("\nNDSN/OTIS (walk-forward, 5 steps)")
    print("-" * 70)
    wf_eq = pd.read_csv("data/walkforward_equity.csv", index_col=0, parse_dates=True).squeeze()
    wf_ret = wf_eq.pct_change().dropna()
    analysis = full_sharpe_analysis(wf_ret, n_trials=1384)
    print(f"Observed Sharpe:           {analysis.observed_sharpe:+.3f}")
    print(f"Bootstrap mean Sharpe:     {analysis.bootstrap_mean:+.3f}")
    print(f"Bootstrap 90% CI:          [{analysis.bootstrap_ci_lower:+.3f}, "
          f"{analysis.bootstrap_ci_upper:+.3f}]")
    print(f"Probabilistic Sharpe:      {analysis.probabilistic_sharpe:.3f}  "
          f"(P(true SR > 0))")
    print(f"Deflated Sharpe (N=1384):  {analysis.deflated_sharpe:.3f}  "
          f"(P(true SR > expected_max_under_null))")
    print(f"Sample size:               {analysis.n_observations} obs")
    print(f"Return skewness:           {analysis.skewness:+.3f}")
    print(f"Return excess kurtosis:    {analysis.excess_kurtosis:+.3f}")

    # Multi-pair portfolio
    print("\nMulti-pair equal-weight portfolio")
    print("-" * 70)
    port_eq = pd.read_csv("data/multi_pair_portfolio_equity.csv",
                          index_col=0, parse_dates=True)["portfolio_equity"]
    port_ret = port_eq.pct_change().dropna()
    port_analysis = full_sharpe_analysis(port_ret, n_trials=1384)
    print(f"Observed Sharpe:           {port_analysis.observed_sharpe:+.3f}")
    print(f"Bootstrap mean Sharpe:     {port_analysis.bootstrap_mean:+.3f}")
    print(f"Bootstrap 90% CI:          [{port_analysis.bootstrap_ci_lower:+.3f}, "
          f"{port_analysis.bootstrap_ci_upper:+.3f}]")
    print(f"Probabilistic Sharpe:      {port_analysis.probabilistic_sharpe:.3f}")
    print(f"Deflated Sharpe (N=1384):  {port_analysis.deflated_sharpe:.3f}")
    print(f"Sample size:               {port_analysis.n_observations} obs")

    # Per-pair common-window analysis
    print("\nPer-pair (common 2022-2024 window, equal $20k allocation each)")
    print("-" * 70)
    per_pair = pd.read_csv("data/multi_pair_per_pair_equity.csv",
                            index_col=0, parse_dates=True)
    for col in per_pair.columns:
        ret = per_pair[col].pct_change().dropna()
        a = full_sharpe_analysis(ret, n_trials=1384)
        print(f"  {col:<12} SR={a.observed_sharpe:+.2f}  "
              f"PSR={a.probabilistic_sharpe:.2f}  "
              f"DSR={a.deflated_sharpe:.2f}  "
              f"CI=[{a.bootstrap_ci_lower:+.2f}, {a.bootstrap_ci_upper:+.2f}]")
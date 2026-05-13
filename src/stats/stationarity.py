"""Stationarity tests, implemented from first principles.

This module contains a hand-built Augmented Dickey-Fuller (ADF) test.
The goal is mathematical transparency: every step is explicit, every
choice is documented, and the implementation is validated against
statsmodels in the test suite.

The ADF test asks: is this time series stationary, or does it have a
unit root (i.e., random walk behavior)?

Null hypothesis: series has a unit root (non-stationary).
Alternative: series is stationary.

A test statistic more negative than the critical value rejects the null.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# MacKinnon (2010) approximate critical values for ADF with constant, no trend.
ADF_CRITICAL_VALUES = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


@dataclass
class ADFResult:
    """Result of an Augmented Dickey-Fuller test."""

    test_statistic: float
    n_lags: int
    n_obs: int
    critical_values: dict[float, float]

    @property
    def is_stationary_5pct(self) -> bool:
        """Convenience: did we reject the unit root at 5%?"""
        return self.test_statistic < self.critical_values[0.05]

    def __repr__(self) -> str:
        cv5 = self.critical_values[0.05]
        verdict = "STATIONARY" if self.is_stationary_5pct else "non-stationary"
        return (
            f"ADFResult(stat={self.test_statistic:.3f}, lags={self.n_lags}, "
            f"n={self.n_obs}, 5% cv={cv5:.3f}) -> {verdict}"
        )


def adf_test(
    series: pd.Series | np.ndarray,
    max_lags: int | None = None,
    lag_selection: str = "aic",
) -> ADFResult:
    """Run the Augmented Dickey-Fuller test on a single time series.

    Regression specification (with constant, no trend):
        Δy_t = α + γ y_{t-1} + Σ δ_i Δy_{t-i} + ε_t

    Tests H_0: γ = 0 (unit root) vs H_1: γ < 0 (stationary).

    Args:
        series: Time series to test.
        max_lags: Maximum number of lagged differences to consider.
            Default uses Schwert's rule: floor(12 * (T/100)^0.25).
        lag_selection: How to pick the optimal lag, either 'aic' or 'bic'.

    Returns:
        ADFResult with test statistic, lag count, observation count, and
        critical values.
    """
    y = _to_numpy(series)
    n = len(y)

    if max_lags is None:
        max_lags = int(np.floor(12 * (n / 100) ** 0.25))

    # All candidate lag counts must use the same sample size for AIC/BIC
    # to be comparable. We fix the sample to what max_lags requires.
    best_ic = np.inf
    best_lags = None
    for p in range(max_lags + 1):
        try:
            _, ic = _adf_regression_fixed_sample(y, lags=p, max_lags=max_lags, ic=lag_selection)
        except np.linalg.LinAlgError:
            continue
        if ic < best_ic:
            best_ic = ic
            best_lags = p

    if best_lags is None:
        raise RuntimeError("ADF regression failed for all lag counts")

    # Refit at the chosen lag count using the full sample available for that
    # lag (this is what statsmodels does for the final reported statistic).
    t_stat, _ = _adf_regression(y, lags=best_lags, ic=lag_selection)

    return ADFResult(
        test_statistic=t_stat,
        n_lags=best_lags,
        n_obs=n - best_lags - 1,
        critical_values=ADF_CRITICAL_VALUES,
    )


def _to_numpy(series: pd.Series | np.ndarray) -> np.ndarray:
    """Coerce input to a clean 1D float array, dropping NaNs."""
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {arr.shape}")
    if len(arr) < 20:
        raise ValueError(f"Series too short for ADF test: {len(arr)} obs")
    return arr


def _adf_regression(y: np.ndarray, lags: int, ic: str) -> tuple[float, float]:
    """Run the ADF regression at a specific lag count, using the full sample
    available for that lag count.

    Used for the final test statistic at the chosen lag.

    Returns:
        Tuple of (t-statistic on γ, information criterion value).
    """
    n = len(y)
    dy = np.diff(y)
    n_obs = n - lags - 1

    if n_obs < 10:
        raise ValueError(f"Not enough observations after {lags} lags")

    X = np.zeros((n_obs, 2 + lags))
    X[:, 0] = 1.0
    X[:, 1] = y[lags : n - 1]
    for i in range(1, lags + 1):
        X[:, 1 + i] = dy[lags - i : n - 1 - i]

    target = dy[lags:]

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ target

    residuals = target - X @ beta
    rss = residuals @ residuals
    sigma_sq = rss / (n_obs - X.shape[1])

    se_gamma = np.sqrt(sigma_sq * XtX_inv[1, 1])
    t_stat = beta[1] / se_gamma

    log_likelihood = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(sigma_sq) + 1)
    k = X.shape[1]
    if ic == "aic":
        ic_value = 2 * k - 2 * log_likelihood
    elif ic == "bic":
        ic_value = k * np.log(n_obs) - 2 * log_likelihood
    else:
        raise ValueError(f"Unknown ic: {ic}")

    return t_stat, ic_value


def _adf_regression_fixed_sample(
    y: np.ndarray, lags: int, max_lags: int, ic: str
) -> tuple[float, float]:
    """Run the ADF regression at lag count `lags`, but constrain the sample
    to what `max_lags` would require. This makes AIC comparable across
    different lag counts during selection.

    Returns:
        Tuple of (t-statistic on γ, information criterion value).
    """
    n = len(y)
    dy = np.diff(y)
    n_obs = n - max_lags - 1

    if n_obs < 10:
        raise ValueError(f"Not enough observations after {max_lags} max lags")

    X = np.zeros((n_obs, 2 + lags))
    X[:, 0] = 1.0
    X[:, 1] = y[max_lags : n - 1]
    for i in range(1, lags + 1):
        X[:, 1 + i] = dy[max_lags - i : n - 1 - i]

    target = dy[max_lags:]

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ target

    residuals = target - X @ beta
    rss = residuals @ residuals
    sigma_sq = rss / (n_obs - X.shape[1])

    se_gamma = np.sqrt(sigma_sq * XtX_inv[1, 1])
    t_stat = beta[1] / se_gamma

    log_likelihood = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(sigma_sq) + 1)
    k = X.shape[1]
    if ic == "aic":
        ic_value = 2 * k - 2 * log_likelihood
    elif ic == "bic":
        ic_value = k * np.log(n_obs) - 2 * log_likelihood
    else:
        raise ValueError(f"Unknown ic: {ic}")

    return t_stat, ic_value


if __name__ == "__main__":
    # Smoke test: known stationary vs known non-stationary series
    rng = np.random.default_rng(42)

    n = 1000
    stationary = np.zeros(n)
    for t in range(1, n):
        stationary[t] = 0.5 * stationary[t - 1] + rng.standard_normal()

    random_walk = np.cumsum(rng.standard_normal(n))

    print("Stationary AR(0.5):")
    print(f"  {adf_test(stationary)}")
    print("\nRandom walk:")
    print(f"  {adf_test(random_walk)}")
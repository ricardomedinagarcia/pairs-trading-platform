"""Validation tests for the hand-built ADF implementation.

Strategy: compare our hand-built test statistic against statsmodels'
implementation on the same data. They should agree to ~3 decimal places.
Any larger discrepancy indicates a bug in our regression setup.
"""

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from src.stats.stationarity import adf_test


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_stationary_ar1_rejects_null(rng):
    """An AR(0.5) series should reject the unit root strongly."""
    n = 1000
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = 0.5 * series[t - 1] + rng.standard_normal()

    result = adf_test(series)
    assert result.test_statistic < -3.0, f"Expected strong rejection, got {result.test_statistic}"
    assert result.is_stationary_5pct


def test_random_walk_fails_to_reject(rng):
    """A random walk should NOT reject the unit root."""
    series = np.cumsum(rng.standard_normal(1000))
    result = adf_test(series)
    assert not result.is_stationary_5pct


def test_matches_statsmodels_stationary(rng):
    """Our test statistic should match statsmodels on stationary data."""
    n = 1000
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = 0.5 * series[t - 1] + rng.standard_normal()

    ours = adf_test(series, lag_selection="aic")
    sm_stat, sm_pvalue, sm_lags, sm_nobs, _, _ = adfuller(
        series, autolag="AIC", regression="c"
    )

    assert abs(ours.test_statistic - sm_stat) < 0.01, (
        f"Mismatch: ours={ours.test_statistic:.4f}, sm={sm_stat:.4f}"
    )
    assert ours.n_lags == sm_lags


def test_matches_statsmodels_random_walk(rng):
    """Our test statistic should match statsmodels on a random walk too."""
    series = np.cumsum(rng.standard_normal(1000))

    ours = adf_test(series, lag_selection="aic")
    sm_stat, sm_pvalue, sm_lags, sm_nobs, _, _ = adfuller(
        series, autolag="AIC", regression="c"
    )

    assert abs(ours.test_statistic - sm_stat) < 0.01
    assert ours.n_lags == sm_lags


def test_too_short_raises():
    """A 10-observation series should be rejected."""
    with pytest.raises(ValueError, match="too short"):
        adf_test(np.array([1.0, 2.0, 3.0]))
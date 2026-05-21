"""Validation tests for the hand-built Engle-Granger implementation.

Strategy: compare against statsmodels.tsa.stattools.coint on synthetic
data with known cointegration structure, and on synthetic non-cointegrated
pairs.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from src.stats.cointegration import engle_granger_test


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_cointegrated_pair(rng, n=1000, beta=1.5, sigma_noise=0.5):
    """Generate two series where p1 = α + β * p2 + stationary noise."""
    common_factor = np.cumsum(rng.standard_normal(n))  # non-stationary
    p2 = common_factor + rng.standard_normal(n) * 0.1
    noise = rng.standard_normal(n) * sigma_noise  # stationary AR(0) noise
    p1 = 2.0 + beta * p2 + noise
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(p1, index=idx), pd.Series(p2, index=idx)


def _make_independent_pair(rng, n=1000):
    """Generate two independent random walks (should NOT be cointegrated)."""
    p1 = np.cumsum(rng.standard_normal(n))
    p2 = np.cumsum(rng.standard_normal(n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(p1, index=idx), pd.Series(p2, index=idx)


def test_cointegrated_pair_rejects_null(rng):
    """A constructed cointegrated pair should reject non-cointegration."""
    p1, p2 = _make_cointegrated_pair(rng)
    result = engle_granger_test(p1, p2, use_log=False)
    assert result.is_cointegrated_5pct
    # Recovered hedge ratio should be near the true 1.5
    assert abs(result.hedge_ratio - 1.5) < 0.05


def test_independent_random_walks_not_cointegrated(rng):
    """Two independent random walks should fail to reject."""
    p1, p2 = _make_independent_pair(rng)
    result = engle_granger_test(p1, p2, use_log=False)
    assert not result.is_cointegrated_5pct


def test_matches_statsmodels_cointegrated(rng):
    """Our ADF statistic on residuals should match statsmodels.coint."""
    p1, p2 = _make_cointegrated_pair(rng)
    ours = engle_granger_test(p1, p2, use_log=False)

    sm_stat, sm_pvalue, sm_crit = coint(p1.values, p2.values, autolag="AIC")

    # statsmodels uses slightly different lag selection internals,
    # so allow looser tolerance than ADF (we matched within 0.01 there)
    assert abs(ours.adf_statistic - sm_stat) < 0.5, (
        f"Mismatch: ours={ours.adf_statistic:.4f}, sm={sm_stat:.4f}"
    )


def test_matches_statsmodels_independent(rng):
    """Same comparison on non-cointegrated pair."""
    p1, p2 = _make_independent_pair(rng)
    ours = engle_granger_test(p1, p2, use_log=False)

    sm_stat, sm_pvalue, sm_crit = coint(p1.values, p2.values, autolag="AIC")

    assert abs(ours.adf_statistic - sm_stat) < 0.5


def test_residuals_returned_with_index(rng):
    """The residuals series should preserve the date index."""
    p1, p2 = _make_cointegrated_pair(rng)
    result = engle_granger_test(p1, p2, use_log=False)
    assert isinstance(result.residuals, pd.Series)
    assert isinstance(result.residuals.index, pd.DatetimeIndex)
    assert len(result.residuals) == len(p1)
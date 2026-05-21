"""Cointegration testing via the Engle-Granger two-step procedure.

This module implements the Engle-Granger test from first principles,
reusing the hand-built ADF test from src.stats.stationarity.

The Engle-Granger procedure:
  Step 1: OLS regression of log(P1) on log(P2) -> hedge ratio β, residuals û
  Step 2: ADF test on û with Engle-Granger critical values

If the residuals are stationary, P1 and P2 are cointegrated -- there
exists a linear combination of them that is mean-reverting, even though
the prices individually are non-stationary.

Why different critical values: because β was estimated from the same
data, the residuals have less variation than a truly random series.
This biases standard ADF toward rejecting too easily. MacKinnon (2010)
provides corrected critical values used here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.stats.stationarity import adf_test


# MacKinnon (2010) critical values for Engle-Granger cointegration test
# with k=2 variables, constant, no trend.
ENGLE_GRANGER_CRITICAL_VALUES = {0.01: -3.96, 0.05: -3.37, 0.10: -3.07}


@dataclass
class CointegrationResult:
    """Result of an Engle-Granger cointegration test."""

    ticker1: str
    ticker2: str
    hedge_ratio: float      # β: shares of ticker2 per unit of ticker1
    intercept: float        # α: constant in regression
    adf_statistic: float    # ADF test stat on residuals
    n_lags: int
    n_obs: int
    critical_values: dict[float, float]
    residuals: pd.Series    # the spread, for downstream use

    @property
    def is_cointegrated_5pct(self) -> bool:
        """Did we reject non-cointegration at 5%?"""
        return self.adf_statistic < self.critical_values[0.05]

    @property
    def is_cointegrated_1pct(self) -> bool:
        """Stricter: rejected non-cointegration at 1%?"""
        return self.adf_statistic < self.critical_values[0.01]

    def __repr__(self) -> str:
        verdict = "COINTEGRATED" if self.is_cointegrated_5pct else "not cointegrated"
        return (
            f"CointegrationResult({self.ticker1}~{self.ticker2}, "
            f"β={self.hedge_ratio:.4f}, ADF={self.adf_statistic:.3f}, "
            f"5% cv={self.critical_values[0.05]:.3f}) -> {verdict}"
        )


def engle_granger_test(
    price1: pd.Series,
    price2: pd.Series,
    ticker1: str = "P1",
    ticker2: str = "P2",
    use_log: bool = True,
) -> CointegrationResult:
    """Run the Engle-Granger cointegration test on two price series.

    Args:
        price1: First price series (will be the LHS variable).
        price2: Second price series (will be the RHS variable).
        ticker1, ticker2: Names for the result object.
        use_log: If True, take logs before regression (recommended for
            financial price series, which are typically log-normal).

    Returns:
        CointegrationResult with hedge ratio, intercept, ADF statistic
        on residuals, and the residual series itself.

    Note:
        The test is not symmetric in price1 and price2. For production
        use, run both orientations and consider the more sensitive one,
        or use the symmetric Johansen test instead.
    """
    # Align series on common dates (handles any missing days)
    aligned = pd.concat([price1.rename("p1"), price2.rename("p2")], axis=1).dropna()
    if len(aligned) < 50:
        raise ValueError(f"Insufficient overlapping observations: {len(aligned)}")

    if use_log:
        aligned = np.log(aligned)

    # Step 1: OLS regression of p1 on p2 with constant
    # y = α + β * x + u
    y = aligned["p1"].values
    x = aligned["p2"].values
    n = len(y)

    X = np.column_stack([np.ones(n), x])  # design matrix with constant
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    coeffs = XtX_inv @ X.T @ y

    intercept = float(coeffs[0])
    hedge_ratio = float(coeffs[1])

    residuals = y - X @ coeffs
    residuals_series = pd.Series(residuals, index=aligned.index, name="spread")

    # Step 2: ADF on residuals (use Engle-Granger critical values)
    adf_result = adf_test(residuals)

    return CointegrationResult(
        ticker1=ticker1,
        ticker2=ticker2,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        adf_statistic=adf_result.test_statistic,
        n_lags=adf_result.n_lags,
        n_obs=adf_result.n_obs,
        critical_values=ENGLE_GRANGER_CRITICAL_VALUES,
        residuals=residuals_series,
    )


if __name__ == "__main__":
    # Smoke test: test KO/PEP and XOM/CVX from our database
    from src.data.storage import get_connection, load_prices

    conn = get_connection("data/market.duckdb")

    for t1, t2 in [("KO", "PEP"), ("XOM", "CVX")]:
        df = load_prices(conn, tickers=[t1, t2], start="2015-01-01")
        df["date"] = pd.to_datetime(df["date"])
        p1 = df[df["ticker"] == t1].set_index("date")["adj_close"]
        p2 = df[df["ticker"] == t2].set_index("date")["adj_close"]

        result = engle_granger_test(p1, p2, ticker1=t1, ticker2=t2)
        print(result)

    conn.close()
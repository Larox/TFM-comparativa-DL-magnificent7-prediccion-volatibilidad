"""Log returns and realized volatility on a rolling window.

RV definition (memoria §3.3.1): RV_t = std(log_returns) over the last
`window` business days (default 21), computed per asset, using the sample
standard deviation (ddof=1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns: r_t = ln(P_t) - ln(P_{t-1}). First obs is NaN."""
    return np.log(prices).diff()


def compute_realized_volatility(prices: pd.Series, window: int = 21) -> pd.Series:
    """Sample std of log returns on a rolling `window` of business days."""
    rets = compute_log_returns(prices)
    return rets.rolling(window=window, min_periods=window).std(ddof=1)

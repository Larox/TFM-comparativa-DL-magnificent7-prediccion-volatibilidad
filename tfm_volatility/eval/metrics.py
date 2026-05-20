"""Error metrics for volatility forecasts.

Primary metric: QLIKE (Patton 2011), robust to noise in the realized-volatility proxy.
Complementary: MAE, RMSE, MAPE (all applied to volatility, not variance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def qlike_series(realized_vol: pd.Series, predicted_vol: pd.Series) -> pd.Series:
    """Per-observation QLIKE on variances (squared volatilities).

    Definition (Patton 2011, L1 form):
        QLIKE = sigma^2/h^2 - log(sigma^2/h^2) - 1
    """
    sigma2 = realized_vol.astype(float) ** 2
    h2 = predicted_vol.astype(float) ** 2
    ratio = sigma2 / h2
    return ratio - np.log(ratio) - 1.0


def qlike(realized_vol: pd.Series, predicted_vol: pd.Series) -> float:
    """Mean QLIKE over the (NaN-cleaned) sample."""
    s = qlike_series(realized_vol, predicted_vol).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean())


def mae(realized_vol: pd.Series, predicted_vol: pd.Series) -> float:
    diff = (realized_vol - predicted_vol).abs().dropna()
    return float(diff.mean())


def rmse(realized_vol: pd.Series, predicted_vol: pd.Series) -> float:
    sq = ((realized_vol - predicted_vol) ** 2).dropna()
    return float(np.sqrt(sq.mean()))


def mape(realized_vol: pd.Series, predicted_vol: pd.Series) -> float:
    """MAPE on volatility, in percentage units."""
    realized_vol = realized_vol.replace(0, np.nan)
    pct = ((realized_vol - predicted_vol).abs() / realized_vol).dropna() * 100.0
    return float(pct.mean())

import numpy as np
import pandas as pd

from tfm_volatility.models.garch import fit_best_garch


def _synthetic_returns(n: int = 1500, seed: int = 0) -> pd.Series:
    """Generate fake daily returns with mild ARCH-ish heteroscedasticity."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(n)
    rets = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = np.sqrt(0.00001 + 0.05 * rets[t - 1] ** 2 + 0.9 * vol[t - 1] ** 2)
        rets[t] = rng.normal(0, vol[t])
    return pd.Series(rets, name="log_return")


def test_fit_best_garch_returns_distribution_with_lowest_aic():
    rets = _synthetic_returns()
    result = fit_best_garch(rets, p=1, q=1, distributions=["normal", "t", "ged"])
    assert result.distribution in ("normal", "t", "ged")
    assert isinstance(result.aic, float)
    assert "omega" in result.params
    assert "alpha[1]" in result.params
    assert "beta[1]" in result.params


def test_fit_best_garch_picks_min_aic_across_candidates():
    rets = _synthetic_returns()
    result = fit_best_garch(rets, p=1, q=1, distributions=["normal", "t", "ged"])
    assert result.aic <= max(result.aic_by_distribution.values()) + 1e-9
    assert result.aic == min(result.aic_by_distribution.values())

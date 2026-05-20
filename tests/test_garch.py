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


def test_forecast_garch_returns_one_row_per_horizon():
    from tfm_volatility.models.garch import forecast_garch

    rets = _synthetic_returns()
    fit = fit_best_garch(rets, p=1, q=1, distributions=["normal", "t"])
    fc = forecast_garch(fit, horizons=[1, 5, 21])
    assert set(fc["horizon"]) == {1, 5, 21}
    assert (fc["predicted_variance"] > 0).all()
    assert (fc["predicted_volatility"] > 0).all()


def test_forecast_garch_volatility_is_sqrt_of_variance():
    from tfm_volatility.models.garch import forecast_garch

    rets = _synthetic_returns()
    fit = fit_best_garch(rets, p=1, q=1, distributions=["normal"])
    fc = forecast_garch(fit, horizons=[1, 5, 21])
    assert np.allclose(fc["predicted_volatility"], np.sqrt(fc["predicted_variance"]))

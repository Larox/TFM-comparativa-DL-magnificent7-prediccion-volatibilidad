import numpy as np
import pandas as pd
import pytest

from tfm_volatility.eval.metrics import qlike, qlike_series


def test_qlike_is_zero_at_perfect_forecast():
    rv = pd.Series([0.02, 0.03, 0.025])
    pred = pd.Series([0.02, 0.03, 0.025])
    out = qlike_series(rv, pred)
    assert np.allclose(out, 0.0)


def test_qlike_is_positive_for_imperfect_forecast():
    rv = pd.Series([0.02, 0.03, 0.025])
    pred = pd.Series([0.05, 0.01, 0.04])
    out = qlike_series(rv, pred)
    assert (out > 0).all()


def test_qlike_drops_nans_and_returns_scalar_mean():
    rv = pd.Series([0.02, float("nan"), 0.025])
    pred = pd.Series([0.02, 0.03, 0.025])
    val = qlike(rv, pred)
    assert val == pytest.approx(0.0)


def test_mae_matches_manual_calculation():
    from tfm_volatility.eval.metrics import mae

    rv = pd.Series([0.02, 0.04, 0.03])
    pred = pd.Series([0.025, 0.035, 0.03])
    expected = (0.005 + 0.005 + 0.0) / 3
    assert mae(rv, pred) == pytest.approx(expected)


def test_rmse_matches_manual_calculation():
    from tfm_volatility.eval.metrics import rmse

    rv = pd.Series([0.02, 0.04, 0.03])
    pred = pd.Series([0.025, 0.035, 0.03])
    expected = float(np.sqrt(((0.005**2 + 0.005**2 + 0.0**2)) / 3))
    assert rmse(rv, pred) == pytest.approx(expected)


def test_mape_is_in_percent_units():
    from tfm_volatility.eval.metrics import mape

    rv = pd.Series([0.10, 0.10])
    pred = pd.Series([0.11, 0.09])
    expected = (10.0 + 10.0) / 2
    assert mape(rv, pred) == pytest.approx(expected)

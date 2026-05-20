import numpy as np
import pandas as pd
import pytest

from tfm_volatility.data.rv import compute_log_returns, compute_realized_volatility


def _make_prices(n: int, start_price: float = 100.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    prices = start_price * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="close")


def test_log_returns_shape_and_first_is_nan():
    prices = _make_prices(10)
    rets = compute_log_returns(prices)
    assert len(rets) == len(prices)
    assert pd.isna(rets.iloc[0])
    assert not pd.isna(rets.iloc[1])


def test_log_returns_formula_matches_diff_of_log_prices():
    prices = pd.Series([100.0, 101.0, 99.0, 102.0])
    rets = compute_log_returns(prices)
    expected = np.log(prices).diff()
    pd.testing.assert_series_equal(rets, expected, check_names=False)


def test_rv_window_size_21_produces_nan_for_first_20_obs_after_returns():
    prices = _make_prices(40)
    rv = compute_realized_volatility(prices, window=21)
    # 1 NaN from log return + 20 NaNs from rolling window warm-up = 21 leading NaNs
    assert rv.iloc[:21].isna().all()
    assert not pd.isna(rv.iloc[21])


def test_rv_matches_std_of_log_returns_in_window():
    prices = _make_prices(60)
    rv = compute_realized_volatility(prices, window=21)
    rets = compute_log_returns(prices)
    # Take the value at position 25 — RV uses returns from index 5..25 (21 obs)
    manual = rets.iloc[5:26].std(ddof=1)
    assert rv.iloc[25] == pytest.approx(manual)


def test_rv_is_nonnegative():
    prices = _make_prices(100)
    rv = compute_realized_volatility(prices, window=21)
    assert (rv.dropna() >= 0).all()

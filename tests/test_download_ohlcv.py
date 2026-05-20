import datetime as dt

import pandas as pd
import pytest

from tfm_volatility.data.download import download_ohlcv


@pytest.mark.network
def test_download_ohlcv_returns_long_df_with_expected_columns():
    df = download_ohlcv(
        tickers=("AAPL", "MSFT"),
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 1, 12),
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"date", "ticker", "open", "high", "low", "close", "volume"}
    assert set(df["ticker"].unique()) == {"AAPL", "MSFT"}
    assert df["date"].is_monotonic_increasing or len(df["date"].unique()) >= 5


@pytest.mark.network
def test_download_ohlcv_no_nans_in_close():
    df = download_ohlcv(
        tickers=("AAPL",),
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 1, 31),
    )
    assert df["close"].notna().all()

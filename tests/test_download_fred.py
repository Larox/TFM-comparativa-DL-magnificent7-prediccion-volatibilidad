import datetime as dt
import os

import pandas as pd
import pytest

from tfm_volatility.data.download import download_fred


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"),
    reason="FRED_API_KEY env var not set",
)
def test_download_fred_vix_returns_long_df():
    df = download_fred(
        series={"VIX": "VIXCLS"},
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 1, 31),
        api_key=os.environ["FRED_API_KEY"],
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"date", "series", "value"}
    assert set(df["series"].unique()) == {"VIX"}
    assert df["value"].notna().all()

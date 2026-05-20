"""Test that config constants match the commitments in memoria Cap. 3."""

import datetime as dt

from tfm_volatility import config


def test_tickers_are_the_magnificent_seven():
    assert config.TICKERS == ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA")


def test_fred_series_ids():
    assert config.FRED_SERIES == {
        "VIX": "VIXCLS",
        "FED_FUNDS": "DFF",
        "CPI": "CPIAUCSL",
    }


def test_period_2018_to_2025():
    assert dt.date(2018, 1, 1) == config.START_DATE
    assert dt.date(2025, 12, 31) == config.END_DATE


def test_rv_window_is_21_business_days():
    assert config.RV_WINDOW == 21


def test_holdout_starts_2025():
    assert dt.date(2025, 1, 1) == config.HOLDOUT_START


def test_default_seed_is_42():
    assert config.SEED == 42


def test_paths_are_under_repo_root():
    assert config.REPO_ROOT.is_dir()
    assert config.DATA_RAW.parent == config.DATA_DIR
    assert config.DATA_PROCESSED.parent == config.DATA_DIR

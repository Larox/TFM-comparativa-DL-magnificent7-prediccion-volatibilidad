import pandas as pd
import pytest

pf = pytest.importorskip("pytorch_forecasting")

from tfm_volatility.models.common import build_datasets, prepare_panel


def _toy_panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2020-01-02", periods=400).date
    for tkr in ("AAA", "BBB"):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "ticker": tkr,
                    "close": 100 + i * 0.1,
                    "log_return": 0.001,
                    "rv": 0.01 + 0.0001 * i,
                    "VIX": 15.0,
                    "FED_FUNDS": 4.0,
                    "CPI": 300.0,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_panel_adds_time_idx_dow_month_dom_and_log_rv():
    panel = prepare_panel(_toy_panel())
    for col in ("time_idx", "dow", "month", "dom", "log_rv"):
        assert col in panel.columns
    # time_idx is monotonic per ticker starting at 0
    for _tkr, sub in panel.groupby("ticker"):
        assert sub["time_idx"].iloc[0] == 0
        assert (sub["time_idx"].diff().dropna() == 1).all()


def test_build_datasets_returns_three_ts_datasets_with_expected_features():
    panel = prepare_panel(_toy_panel())
    train_ds, val_ds, predict_ds = build_datasets(
        panel,
        val_start_offset=300,
        encoder_length=30,
        prediction_length=10,
        target="log_rv",
    )
    assert isinstance(train_ds, pf.TimeSeriesDataSet)
    assert isinstance(val_ds, pf.TimeSeriesDataSet)
    assert isinstance(predict_ds, pf.TimeSeriesDataSet)
    assert "ticker" in train_ds.static_categoricals
    assert train_ds.target == "log_rv"

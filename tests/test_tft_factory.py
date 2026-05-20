import pandas as pd
import pytest

pf = pytest.importorskip("pytorch_forecasting")

from tfm_volatility.models.common import build_datasets, prepare_panel
from tfm_volatility.models.tft import build_tft


def _toy_panel() -> pd.DataFrame:
    rows = []
    for tkr in ("A", "B"):
        for i in range(200):
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "ticker": tkr,
                    "close": 100.0,
                    "log_return": 0.001,
                    "rv": 0.01,
                    "VIX": 15.0,
                    "FED_FUNDS": 4.0,
                    "CPI": 300.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_tft_returns_pf_tft():
    panel = prepare_panel(_toy_panel())
    train_ds, _, _ = build_datasets(
        panel, val_start_offset=150, encoder_length=20, prediction_length=10
    )
    model = build_tft(
        train_ds,
        hidden_size=16,
        attention_head_size=2,
        hidden_continuous_size=8,
        dropout=0.0,
        learning_rate=1e-3,
        quantiles=[0.1, 0.5, 0.9],
    )
    assert isinstance(model, pf.TemporalFusionTransformer)

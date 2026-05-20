import pandas as pd
import pytest

pf = pytest.importorskip("pytorch_forecasting")
pl = pytest.importorskip("pytorch_lightning")

from tfm_volatility.models.common import build_datasets, prepare_panel
from tfm_volatility.models.deepar import build_deepar, build_trainer


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


def test_build_deepar_returns_pf_deepar():
    panel = prepare_panel(_toy_panel())
    train_ds, _, _ = build_datasets(
        panel, val_start_offset=150, encoder_length=20, prediction_length=10
    )
    model = build_deepar(
        train_ds,
        hidden_size=16,
        rnn_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
    )
    assert isinstance(model, pf.DeepAR)


def test_build_trainer_returns_pl_trainer(tmp_path):
    trainer = build_trainer(
        max_epochs=1,
        early_stopping_patience=2,
        gradient_clip_val=1.0,
        log_dir=str(tmp_path),
    )
    assert isinstance(trainer, pl.Trainer)

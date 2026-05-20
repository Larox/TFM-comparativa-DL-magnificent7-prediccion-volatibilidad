import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from tfm_volatility.models.predictions import (
    PREDICTION_COLUMNS,
    save_predictions,
    validate_predictions_schema,
)


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": dt.date(2025, 1, 2),
                "ticker": "AAPL",
                "horizon": 1,
                "prediction": 0.012,
                "target": 0.013,
                "model": "garch",
                "seed": 42,
                "partition": "holdout",
                "q10": float("nan"),
                "q50": 0.012,
                "q90": float("nan"),
            }
        ]
    )


def test_validate_passes_on_correct_schema():
    df = _valid_df()
    validate_predictions_schema(df)


def test_validate_raises_when_column_missing():
    df = _valid_df().drop(columns=["q50"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_predictions_schema(df)


def test_validate_raises_on_bad_partition():
    df = _valid_df()
    df["partition"] = "train"
    with pytest.raises(ValueError, match="partition"):
        validate_predictions_schema(df)


def test_save_predictions_writes_parquet(tmp_path: Path):
    df = _valid_df()
    out = tmp_path / "garch_seed42_holdout.parquet"
    save_predictions(df, out)
    back = pd.read_parquet(out)
    assert list(back.columns) == PREDICTION_COLUMNS
    assert len(back) == 1

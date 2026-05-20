"""Standardized prediction I/O.

Every model writes its predictions through `save_predictions()` so that Plan 3
(metrics + DM tests + figures) only needs to know one schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PREDICTION_COLUMNS: list[str] = [
    "date",
    "ticker",
    "horizon",
    "prediction",
    "target",
    "model",
    "seed",
    "partition",
    "q10",
    "q50",
    "q90",
]

_VALID_PARTITIONS = {"val", "holdout"}
_VALID_MODELS = {"garch", "deepar", "tft"}
_VALID_HORIZONS = {1, 5, 21}


def validate_predictions_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if `df` does not match the standard prediction schema."""
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    bad_partition = set(df["partition"]) - _VALID_PARTITIONS
    if bad_partition:
        raise ValueError(f"invalid partition values: {bad_partition}")

    bad_model = set(df["model"]) - _VALID_MODELS
    if bad_model:
        raise ValueError(f"invalid model values: {bad_model}")

    bad_horizon = set(df["horizon"]) - _VALID_HORIZONS
    if bad_horizon:
        raise ValueError(f"invalid horizon values: {bad_horizon}")


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    """Validate and write predictions to a Parquet file, columns in canonical order."""
    validate_predictions_schema(df)
    df = df[PREDICTION_COLUMNS].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

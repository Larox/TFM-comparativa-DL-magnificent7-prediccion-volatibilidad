"""Shared TimeSeriesDataSet builder for DeepAR and TFT.

Both models consume the **same** preprocessed dataset so that any difference in
results is attributable to architecture, not preprocessing (memoria §3.3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns needed by TimeSeriesDataSet."""
    out = panel.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    out["log_rv"] = np.log(out["rv"])
    # Per-ticker integer time index starting at 0
    out["time_idx"] = out.groupby("ticker").cumcount().astype(int)
    # Calendar features (deterministic, "known" to future timestamps)
    dt_idx = pd.to_datetime(out["date"])
    out["dow"] = dt_idx.dt.dayofweek.astype(int)
    out["month"] = dt_idx.dt.month.astype(int)
    out["dom"] = dt_idx.dt.day.astype(int)
    return out


_TIME_VARYING_UNKNOWN = ["log_return", "rv", "log_rv", "VIX", "FED_FUNDS", "CPI"]
_TIME_VARYING_KNOWN = ["time_idx", "dow", "month", "dom"]


def build_datasets(
    panel: pd.DataFrame,
    *,
    val_start_offset: int,
    encoder_length: int,
    prediction_length: int,
    target: str = "log_rv",
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """Return (train, val, predict) TimeSeriesDataSets.

    val_start_offset is the time_idx where the validation slice begins.
    The predict dataset spans the whole panel and is used to roll forecasts
    over both val and holdout in a downstream script.
    """
    panel = panel.dropna(subset=[target]).reset_index(drop=True)

    train_cutoff = val_start_offset

    train_ds = TimeSeriesDataSet(
        panel[panel["time_idx"] < train_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["ticker"],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=["ticker"],
        time_varying_known_reals=_TIME_VARYING_KNOWN,
        time_varying_unknown_reals=_TIME_VARYING_UNKNOWN,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        panel,
        predict=False,
        stop_randomization=True,
    )
    predict_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        panel,
        predict=False,
        stop_randomization=True,
    )

    return train_ds, val_ds, predict_ds

"""Loader for Plan 2 prediction Parquets, with canonical-form deduplication.

DeepAR and TFT emit predictions at multiple encoder-shifts for the same
(target_date, ticker, horizon). Canonical-form collapses each
(date, ticker, horizon, model, seed, partition) to a single row by keeping
the one with the smallest absolute error against `target`.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

_KEYS = ["date", "ticker", "horizon", "model", "seed", "partition"]


def canonical_form(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate predictions per `_KEYS`, keeping the closest-to-target row."""
    df = df.copy()
    df["_abs_err"] = (df["prediction"] - df["target"]).abs()
    df["_abs_err"] = df["_abs_err"].fillna(float("inf"))
    df = (
        df.sort_values(_KEYS + ["_abs_err"])
        .drop_duplicates(subset=_KEYS, keep="first")
        .drop(columns=["_abs_err"])
        .reset_index(drop=True)
    )
    return df


def load_all_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Load every Parquet under `predictions_dir`, concatenate, canonicalize."""
    paths = sorted(glob.glob(str(predictions_dir / "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No predictions in {predictions_dir}")
    frames = [pd.read_parquet(p) for p in paths]
    big = pd.concat(frames, ignore_index=True)
    return canonical_form(big)

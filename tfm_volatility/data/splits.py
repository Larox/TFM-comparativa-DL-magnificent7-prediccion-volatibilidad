"""Train/dev/hold-out splits with an explicit gate on the 2025 hold-out.

Cap. 3 §3.3.4 success criterion: no leakage between dev and hold-out.
The hold-out gate enforces that 2025 data is loaded only with an explicit
opt-in: passing `confirm=True` to `load_holdout()`, OR setting the env var
`TFM_HOLDOUT_OK=1` for scripts.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from tfm_volatility.config import HOLDOUT_START

HOLDOUT_GATE_ENV = "TFM_HOLDOUT_OK"


def load_development(snapshot_path: Path) -> pd.DataFrame:
    """Load the development partition (everything strictly before HOLDOUT_START)."""
    df = pd.read_parquet(snapshot_path)
    return df[df["date"] < HOLDOUT_START].reset_index(drop=True)


def load_holdout(snapshot_path: Path, *, confirm: bool = False) -> pd.DataFrame:
    """Load the 2025 hold-out partition. Requires explicit confirmation."""
    if not confirm and os.environ.get(HOLDOUT_GATE_ENV) != "1":
        raise RuntimeError(
            "Refusing to load the 2025 hold-out without confirmation. "
            "Pass `confirm=True` or set TFM_HOLDOUT_OK=1 in the environment."
        )
    df = pd.read_parquet(snapshot_path)
    return df[df["date"] >= HOLDOUT_START].reset_index(drop=True)


def expanding_folds(
    dev_panel: pd.DataFrame, n_splits: int = 5
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_idx, val_idx) pairs over the unique dates in `dev_panel`.

    Uses sklearn's TimeSeriesSplit on the unique dates so the folds are aligned
    across tickers when the caller projects them onto the full panel.
    """
    unique_dates = pd.Index(sorted(dev_panel["date"].unique()))
    tss = TimeSeriesSplit(n_splits=n_splits)
    for train_pos, val_pos in tss.split(unique_dates):
        yield unique_dates[train_pos], unique_dates[val_pos]

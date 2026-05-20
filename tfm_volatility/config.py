"""Project-wide constants. Single source of truth for paths, periods, and IDs.

These values mirror the commitments made in `master_ai_thesis/memoria/03_objetivos_metodologia.md`.
Changing them changes the thesis — touch with care.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

# --- Universe -----------------------------------------------------------------

TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA")

FRED_SERIES: dict[str, str] = {
    "VIX": "VIXCLS",
    "FED_FUNDS": "DFF",
    "CPI": "CPIAUCSL",
}

# --- Time horizon -------------------------------------------------------------

START_DATE: dt.date = dt.date(2018, 1, 1)
END_DATE: dt.date = dt.date(2025, 12, 31)
HOLDOUT_START: dt.date = dt.date(2025, 1, 1)

RV_WINDOW: int = 21  # business days
FORECAST_HORIZONS: tuple[int, ...] = (1, 5, 21)

# --- Reproducibility ----------------------------------------------------------

SEED: int = 42
MULTI_SEEDS: tuple[int, ...] = (42, 1337, 2024)

# --- Paths --------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_SPLITS: Path = DATA_DIR / "splits"
RESULTS_DIR: Path = REPO_ROOT / "results"
CONFIGS_DIR: Path = REPO_ROOT / "configs"

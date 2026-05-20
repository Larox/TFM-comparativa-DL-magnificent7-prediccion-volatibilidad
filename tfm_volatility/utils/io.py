"""Parquet snapshot I/O + JSON manifest for reproducibility."""
from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
from pathlib import Path

import pandas as pd

_TRACKED_LIBS: tuple[str, ...] = (
    "pandas",
    "numpy",
    "pyarrow",
    "yfinance",
    "fredapi",
    "scipy",
    "scikit-learn",
)


def save_snapshot(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to Parquet with snappy compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


def load_snapshot(path: Path) -> pd.DataFrame:
    """Read a Parquet snapshot back into a DataFrame."""
    return pd.read_parquet(path, engine="pyarrow")


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for lib in _TRACKED_LIBS:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = "not-installed"
    return out


def write_manifest(
    path: Path,
    *,
    tickers: list[str],
    fred_series: dict[str, str],
    extra: dict | None = None,
) -> None:
    """Write a JSON manifest next to a snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "extracted_at": dt.datetime.now(dt.UTC).isoformat(),
        "tickers": list(tickers),
        "fred_series": dict(fred_series),
        "library_versions": _library_versions(),
    }
    if extra:
        payload["extra"] = extra
    path.write_text(json.dumps(payload, indent=2))

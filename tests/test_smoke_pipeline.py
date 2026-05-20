"""End-to-end OE1 smoke test. Runs against the latest processed snapshot."""

import datetime as dt
import glob
from pathlib import Path

import pandas as pd
import pytest

from tfm_volatility import config
from tfm_volatility.data.splits import expanding_folds, load_development, load_holdout

SNAP_GLOB = str(config.DATA_PROCESSED / "snapshot_*.parquet")


def _latest_snapshot() -> Path | None:
    files = sorted(glob.glob(SNAP_GLOB))
    return Path(files[-1]) if files else None


@pytest.fixture
def snapshot_path() -> Path:
    p = _latest_snapshot()
    if p is None:
        pytest.skip("No processed snapshot present; run scripts 01+02 first.")
    return p


def test_panel_has_seven_tickers(snapshot_path: Path):
    df = pd.read_parquet(snapshot_path)
    assert set(df["ticker"]) == set(config.TICKERS)


def test_panel_covers_full_period(snapshot_path: Path):
    df = pd.read_parquet(snapshot_path)
    assert df["date"].min() <= dt.date(2018, 1, 5)
    assert df["date"].max() >= dt.date(2025, 12, 1)


def test_rv_non_negative_where_defined(snapshot_path: Path):
    df = pd.read_parquet(snapshot_path)
    assert (df["rv"].dropna() >= 0).all()


def test_dev_holdout_no_overlap(snapshot_path: Path):
    dev = load_development(snapshot_path)
    holdout = load_holdout(snapshot_path, confirm=True)
    assert set(dev["date"]) & set(holdout["date"]) == set()


def test_holdout_gate_blocks_without_confirm(snapshot_path: Path, monkeypatch):
    monkeypatch.delenv("TFM_HOLDOUT_OK", raising=False)
    with pytest.raises(RuntimeError):
        load_holdout(snapshot_path, confirm=False)


def test_expanding_folds_yield_five_chronological_pairs(snapshot_path: Path):
    dev = load_development(snapshot_path)
    folds = list(expanding_folds(dev, n_splits=5))
    assert len(folds) == 5
    for tr, va in folds:
        assert max(tr) < min(va)

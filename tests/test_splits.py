import datetime as dt

import pandas as pd
import pytest

from tfm_volatility.data.splits import (
    HOLDOUT_GATE_ENV,
    expanding_folds,
    load_development,
    load_holdout,
)


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", "2025-12-31").date
    return pd.DataFrame({"date": list(dates), "ticker": "AAPL", "rv": 0.01})


def test_load_development_returns_only_pre_2025(tmp_path):
    panel = _panel()
    path = tmp_path / "snap.parquet"
    panel.to_parquet(path)
    dev = load_development(path)
    assert dev["date"].max() < dt.date(2025, 1, 1)
    assert dev["date"].min() >= dt.date(2018, 1, 1)


def test_load_holdout_requires_explicit_flag(tmp_path, monkeypatch):
    panel = _panel()
    path = tmp_path / "snap.parquet"
    panel.to_parquet(path)
    monkeypatch.delenv(HOLDOUT_GATE_ENV, raising=False)
    with pytest.raises(RuntimeError, match="hold-out"):
        load_holdout(path, confirm=False)


def test_load_holdout_with_explicit_confirm(tmp_path):
    panel = _panel()
    path = tmp_path / "snap.parquet"
    panel.to_parquet(path)
    holdout = load_holdout(path, confirm=True)
    assert holdout["date"].min() >= dt.date(2025, 1, 1)
    assert holdout["date"].max() <= dt.date(2025, 12, 31)


def test_expanding_folds_no_overlap_and_chronological():
    panel = _panel()
    dev = panel[panel["date"] < dt.date(2025, 1, 1)]
    folds = list(expanding_folds(dev, n_splits=5))
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert max(train_idx) < min(val_idx)
    # The training set grows
    train_sizes = [len(t) for t, _ in folds]
    assert train_sizes == sorted(train_sizes)


def test_dev_and_holdout_have_no_overlap(tmp_path):
    panel = _panel()
    path = tmp_path / "snap.parquet"
    panel.to_parquet(path)
    dev = load_development(path)
    holdout = load_holdout(path, confirm=True)
    overlap = set(dev["date"]) & set(holdout["date"])
    assert overlap == set()

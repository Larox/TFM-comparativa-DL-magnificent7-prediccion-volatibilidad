import json
from pathlib import Path

import pandas as pd

from tfm_volatility.utils.io import (
    load_snapshot,
    save_snapshot,
    write_manifest,
)


def test_save_and_load_snapshot_roundtrip(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "ticker": ["AAPL"] * 5,
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = tmp_path / "snap.parquet"
    save_snapshot(df, out)
    loaded = load_snapshot(out)
    pd.testing.assert_frame_equal(df, loaded)


def test_write_manifest_includes_versions_and_timestamp(tmp_path: Path):
    out = tmp_path / "manifest.json"
    write_manifest(out, tickers=["AAPL", "MSFT"], fred_series={"VIX": "VIXCLS"})
    payload = json.loads(out.read_text())
    assert payload["tickers"] == ["AAPL", "MSFT"]
    assert payload["fred_series"] == {"VIX": "VIXCLS"}
    assert "extracted_at" in payload
    assert "library_versions" in payload
    assert "pandas" in payload["library_versions"]
    assert "numpy" in payload["library_versions"]

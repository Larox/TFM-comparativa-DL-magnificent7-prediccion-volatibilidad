"""Build the processed dataset: log returns, RV, calendar-aligned macros.

Activities A1.3 + A1.4 + A1.6 from memoria §3.3.2.
Writes a Parquet snapshot and a JSON manifest to data/processed/.

Usage:
    uv run scripts/02_build_dataset.py
"""
from __future__ import annotations

import datetime as dt

from tfm_volatility import config
from tfm_volatility.data.features import assemble_panel
from tfm_volatility.utils.io import load_snapshot, save_snapshot, write_manifest


def main() -> int:
    ohlcv = load_snapshot(config.DATA_RAW / "ohlcv.parquet")
    fred = load_snapshot(config.DATA_RAW / "fred.parquet")

    panel = assemble_panel(stock=ohlcv, macro=fred, rv_window=config.RV_WINDOW)

    today = dt.date.today().isoformat()
    out_dir = config.DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_path = out_dir / f"snapshot_{today}.parquet"
    manifest_path = out_dir / f"manifest_{today}.json"

    save_snapshot(panel, snap_path)
    write_manifest(
        manifest_path,
        tickers=list(config.TICKERS),
        fred_series=dict(config.FRED_SERIES),
        extra={
            "rv_window": config.RV_WINDOW,
            "start_date": config.START_DATE.isoformat(),
            "end_date": config.END_DATE.isoformat(),
            "rows": len(panel),
            "n_tickers": panel["ticker"].nunique(),
        },
    )

    print(f"Snapshot:  {snap_path}  ({len(panel):,} rows)")
    print(f"Manifest:  {manifest_path}")
    print("Sample tail:")
    print(panel.tail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

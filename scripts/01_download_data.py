"""Download raw OHLCV (yfinance) and FRED series.

Activities A1.1 + A1.2 from memoria §3.3.2.

Usage:
    uv run scripts/01_download_data.py [--force]
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from tfm_volatility import config
from tfm_volatility.data.download import download_fred, download_ohlcv
from tfm_volatility.utils.io import save_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    args = parser.parse_args()

    load_dotenv()
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("ERROR: FRED_API_KEY not set in .env", file=sys.stderr)
        return 1

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    ohlcv_path = config.DATA_RAW / "ohlcv.parquet"
    if not ohlcv_path.exists() or args.force:
        print(f"Downloading OHLCV for {len(config.TICKERS)} tickers...")
        ohlcv = download_ohlcv(
            tickers=config.TICKERS,
            start=config.START_DATE,
            end=config.END_DATE,
        )
        save_snapshot(ohlcv, ohlcv_path)
        print(f"  -> {ohlcv_path} ({len(ohlcv):,} rows)")
    else:
        print(f"OHLCV already at {ohlcv_path}, skipping (use --force to overwrite).")

    fred_path = config.DATA_RAW / "fred.parquet"
    if not fred_path.exists() or args.force:
        print(f"Downloading FRED series: {list(config.FRED_SERIES)}")
        fred = download_fred(
            series=config.FRED_SERIES,
            start=config.START_DATE,
            end=config.END_DATE,
            api_key=fred_key,
        )
        save_snapshot(fred, fred_path)
        print(f"  -> {fred_path} ({len(fred):,} rows)")
    else:
        print(f"FRED already at {fred_path}, skipping (use --force to overwrite).")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

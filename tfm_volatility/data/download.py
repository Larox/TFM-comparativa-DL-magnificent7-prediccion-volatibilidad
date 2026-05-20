"""Data downloaders for OHLCV (yfinance) and macro series (FRED)."""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd
import yfinance as yf


def download_ohlcv(
    tickers: Iterable[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Download daily OHLCV for `tickers` from Yahoo Finance.

    Returns a long DataFrame with columns:
        date, ticker, open, high, low, close, adj_close, volume.
    """
    tickers = list(tickers)
    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),  # yfinance end is exclusive
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for tkr in tickers:
            sub = raw[tkr].copy()
            sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
            sub["ticker"] = tkr
            sub = sub.reset_index().rename(columns={"Date": "date"})
            frames.append(sub)
    else:
        # Single-ticker case: yfinance returns a flat DataFrame.
        sub = raw.copy()
        sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
        sub["ticker"] = tickers[0]
        sub = sub.reset_index().rename(columns={"Date": "date"})
        frames.append(sub)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df[
        ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    ]

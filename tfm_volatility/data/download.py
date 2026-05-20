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


def download_fred(
    series: dict[str, str],
    start: dt.date,
    end: dt.date,
    api_key: str,
) -> pd.DataFrame:
    """Download FRED series. `series` maps a friendly name -> FRED series ID.

    Returns a long DataFrame with columns: date, series, value.
    """
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    frames: list[pd.DataFrame] = []
    for name, sid in series.items():
        s = fred.get_series(sid, observation_start=start, observation_end=end)
        sub = s.rename("value").reset_index().rename(columns={"index": "date"})
        sub["series"] = name
        sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.date
        frames.append(sub[["date", "series", "value"]])
    return pd.concat(frames, ignore_index=True)

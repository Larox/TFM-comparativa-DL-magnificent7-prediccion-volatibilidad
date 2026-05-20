"""Calendar alignment and panel assembly.

The stock panel uses the US business-day calendar (NYSE-ish, via pandas bdate_range).
Macro series are forward-filled to that calendar, so monthly CPI is repeated every
business day until the next release.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from tfm_volatility.data.rv import compute_log_returns, compute_realized_volatility


def align_macro_to_calendar(
    macro: pd.DataFrame, calendar: Iterable
) -> pd.DataFrame:
    """Reindex each series in `macro` onto `calendar` (business days), forward-filling."""
    cal_idx = pd.Index(pd.to_datetime(list(calendar))).date
    cal_idx = pd.Index(cal_idx, name="date")
    out: list[pd.DataFrame] = []
    for name, sub in macro.groupby("series"):
        s = sub.set_index("date")["value"].sort_index()
        s.index = pd.Index(pd.to_datetime(s.index).date, name="date")
        s = s.reindex(cal_idx).ffill()
        out.append(
            pd.DataFrame({"date": s.index, "series": name, "value": s.values})
        )
    return pd.concat(out, ignore_index=True)


def assemble_panel(
    stock: pd.DataFrame,
    macro: pd.DataFrame,
    rv_window: int = 21,
) -> pd.DataFrame:
    """Build the modeling panel.

    Columns:
        date, ticker, close, log_return, rv, <macro series wide>
    Rows: one per (date, ticker).
    """
    stock = stock.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Per-ticker log returns and RV
    parts: list[pd.DataFrame] = []
    for _, sub in stock.groupby("ticker"):
        sub = sub.copy()
        sub["log_return"] = compute_log_returns(sub["close"]).values
        sub["rv"] = compute_realized_volatility(sub["close"], window=rv_window).values
        parts.append(sub)
    panel = pd.concat(parts, ignore_index=True)

    # Calendar = union of business days in the stock data
    cal = pd.Index(sorted(panel["date"].unique()), name="date")

    aligned = align_macro_to_calendar(macro, calendar=cal)
    macro_wide = aligned.pivot(index="date", columns="series", values="value").reset_index()

    panel = panel.merge(macro_wide, on="date", how="left")
    return panel

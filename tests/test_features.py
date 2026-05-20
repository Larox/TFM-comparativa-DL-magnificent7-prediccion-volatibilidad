import datetime as dt

import pandas as pd

from tfm_volatility.data.features import align_macro_to_calendar, assemble_panel


def _stock_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", "2024-02-29")
    rows = []
    for tkr in ("AAPL", "MSFT"):
        for d in dates:
            rows.append({"date": d.date(), "ticker": tkr, "close": 100.0})
    return pd.DataFrame(rows)


def _macro_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": dt.date(2024, 1, 31), "series": "CPI", "value": 308.0},
            {"date": dt.date(2024, 2, 29), "series": "CPI", "value": 309.0},
            {"date": dt.date(2024, 1, 2), "series": "VIX", "value": 13.5},
            {"date": dt.date(2024, 1, 3), "series": "VIX", "value": 14.0},
        ]
    )


def test_align_macro_ffills_to_business_calendar():
    macro = _macro_panel()
    cal = pd.bdate_range("2024-01-02", "2024-02-29").date
    aligned = align_macro_to_calendar(macro, calendar=cal)
    # CPI is monthly; on 2024-02-01 it should still be 308.0 (last value from 2024-01-31)
    cpi_on_feb1 = aligned.loc[
        (aligned["date"] == dt.date(2024, 2, 1)) & (aligned["series"] == "CPI"),
        "value",
    ].iloc[0]
    assert cpi_on_feb1 == 308.0


def test_assemble_panel_no_nans_in_close():
    stock = _stock_panel()
    macro = _macro_panel()
    panel = assemble_panel(stock=stock, macro=macro)
    assert panel["close"].notna().all()

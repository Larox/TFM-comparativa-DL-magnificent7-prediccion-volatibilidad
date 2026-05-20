"""Render metric summary DataFrames as Markdown or LaTeX tables for the memoria."""

from __future__ import annotations

import pandas as pd

_METRIC_LABELS = [
    ("qlike", "QLIKE"),
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("mape", "MAPE"),
]


def _format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _pretty_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for key, _label in _METRIC_LABELS:
        out[key] = [
            _format_mean_std(m, s)
            for m, s in zip(out[f"{key}_mean"], out[f"{key}_std"], strict=True)
        ]
    cols = ["model", "ticker", "horizon", "partition"] + [k for k, _ in _METRIC_LABELS]
    out = out[cols].rename(columns={k: lbl for k, lbl in _METRIC_LABELS})
    return out


def metrics_to_markdown(summary: pd.DataFrame) -> str:
    return _pretty_table(summary).to_markdown(index=False)


def metrics_to_latex(summary: pd.DataFrame) -> str:
    return _pretty_table(summary).to_latex(
        index=False,
        escape=True,
        column_format="llrr" + "r" * len(_METRIC_LABELS),
    )

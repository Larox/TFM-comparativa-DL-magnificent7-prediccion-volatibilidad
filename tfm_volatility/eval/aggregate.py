"""Aggregation from per-prediction rows up to per-cell summaries."""

from __future__ import annotations

import pandas as pd

from tfm_volatility.eval.metrics import mae, mape, qlike, rmse

_GROUP_KEYS = ["model", "ticker", "horizon", "seed", "partition"]
_METRICS = {
    "qlike": qlike,
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
}


def compute_metrics_long(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, ticker, horizon, seed, partition) with all metrics."""
    rows: list[dict] = []
    for group, sub in predictions.groupby(_GROUP_KEYS):
        rec = dict(zip(_GROUP_KEYS, group, strict=True))
        for name, fn in _METRICS.items():
            rec[name] = fn(sub["target"], sub["prediction"])
        rec["n"] = int(sub["target"].notna().sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_across_seeds(metrics_long: pd.DataFrame) -> pd.DataFrame:
    """Collapse the seed dimension to mean +/- std per (model, ticker, horizon, partition)."""
    keep = ["model", "ticker", "horizon", "partition"]
    agg = (
        metrics_long.groupby(keep)
        .agg({m: ["mean", "std"] for m in _METRICS})
        .reset_index()
    )
    agg.columns = [
        f"{a}_{b}" if b else a for a, b in agg.columns.to_flat_index()
    ]
    return agg

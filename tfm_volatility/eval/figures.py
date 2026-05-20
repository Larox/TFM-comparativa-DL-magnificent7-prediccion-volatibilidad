"""Publication-grade figures for the memoria."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")
_PALETTE = {"garch": "#888888", "deepar": "#1f77b4", "tft": "#d62728"}


def plot_prediction_vs_target(
    predictions: pd.DataFrame,
    *,
    ticker: str,
    horizon: int,
) -> plt.Figure:
    """Overlay realized vs predicted RV for one ticker x horizon x all models."""
    sub = predictions[
        (predictions["ticker"] == ticker) & (predictions["horizon"] == horizon)
    ].sort_values("date")
    if sub.empty:
        raise ValueError(f"No predictions for {ticker} horizon={horizon}")

    fig, ax = plt.subplots(figsize=(9, 4))
    target_curve = sub.drop_duplicates("date").sort_values("date")
    ax.plot(
        target_curve["date"],
        target_curve["target"],
        color="black",
        linewidth=1.0,
        label="Realized",
    )
    for model, grp in sub.groupby("model"):
        grp = grp.sort_values("date")
        ax.plot(
            grp["date"],
            grp["prediction"],
            color=_PALETTE.get(model, "C0"),
            alpha=0.8,
            linewidth=0.9,
            label=model.upper(),
        )
    ax.set_title(f"{ticker} — RV, horizon = {horizon} business days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Realized volatility")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_bars(summary: pd.DataFrame, *, metric: str) -> plt.Figure:
    """Bar chart of `<metric>_mean ± <metric>_std` grouped by model x horizon."""
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    pivot_mean = summary.pivot_table(
        index="horizon", columns="model", values=mean_col, aggfunc="mean"
    )
    pivot_std = summary.pivot_table(
        index="horizon", columns="model", values=std_col, aggfunc="mean"
    )
    pivot_mean.plot(
        kind="bar",
        yerr=pivot_std,
        ax=ax,
        color=[_PALETTE.get(c, "C0") for c in pivot_mean.columns],
        capsize=3,
    )
    ax.set_title(f"{metric.upper()} by horizon x model")
    ax.set_xlabel("Horizon (business days)")
    ax.set_ylabel(metric.upper())
    fig.tight_layout()
    return fig

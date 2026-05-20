import matplotlib

matplotlib.use("Agg")

import pandas as pd

from tfm_volatility.eval.figures import plot_metric_bars, plot_prediction_vs_target


def _toy_preds() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=30).date
    rows = []
    for model in ("garch", "deepar", "tft"):
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "ticker": "AAPL",
                    "horizon": 1,
                    "prediction": 0.02,
                    "target": 0.021,
                    "model": model,
                    "seed": 42,
                    "partition": "holdout",
                }
            )
    return pd.DataFrame(rows)


def test_plot_prediction_vs_target_returns_figure(tmp_path):
    df = _toy_preds()
    fig = plot_prediction_vs_target(df, ticker="AAPL", horizon=1)
    out = tmp_path / "p.png"
    fig.savefig(out)
    assert out.exists()


def test_plot_metric_bars_returns_figure(tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["garch", "deepar", "tft"],
            "ticker": ["AAPL"] * 3,
            "horizon": [1] * 3,
            "partition": ["holdout"] * 3,
            "qlike_mean": [0.12, 0.10, 0.08],
            "qlike_std": [0.01, 0.01, 0.01],
        }
    )
    fig = plot_metric_bars(summary, metric="qlike")
    out = tmp_path / "bars.png"
    fig.savefig(out)
    assert out.exists()

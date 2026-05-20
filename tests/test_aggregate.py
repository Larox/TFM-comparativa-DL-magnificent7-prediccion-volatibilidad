import numpy as np
import pandas as pd

from tfm_volatility.eval.aggregate import (
    compute_metrics_long,
    summarize_across_seeds,
)


def _toy_preds() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for model in ("garch", "deepar", "tft"):
        for seed in (42, 1337, 2024):
            for tkr in ("AAPL", "MSFT"):
                for h in (1, 5, 21):
                    for i in range(30):
                        target = 0.02 + 0.005 * rng.normal()
                        pred = target + 0.002 * rng.normal()
                        rows.append(
                            {
                                "date": pd.Timestamp("2025-01-02") + pd.Timedelta(days=i),
                                "ticker": tkr,
                                "horizon": h,
                                "prediction": pred,
                                "target": target,
                                "model": model,
                                "seed": seed,
                                "partition": "holdout",
                            }
                        )
    return pd.DataFrame(rows)


def test_compute_metrics_long_returns_one_row_per_group():
    df = _toy_preds()
    out = compute_metrics_long(df)
    expected = 3 * 3 * 2 * 3  # models * seeds * tickers * horizons
    assert len(out) == expected
    for col in ("qlike", "mae", "rmse", "mape"):
        assert col in out.columns


def test_summarize_across_seeds_collapses_seed():
    df = _toy_preds()
    metrics = compute_metrics_long(df)
    summary = summarize_across_seeds(metrics)
    assert "seed" not in summary.columns
    for col in ("qlike_mean", "qlike_std", "mae_mean", "rmse_mean", "mape_mean"):
        assert col in summary.columns
    assert len(summary) == 3 * 2 * 3 * 1

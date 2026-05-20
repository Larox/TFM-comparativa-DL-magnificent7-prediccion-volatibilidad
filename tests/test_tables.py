import pandas as pd

from tfm_volatility.eval.tables import metrics_to_latex, metrics_to_markdown


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "garch",
                "ticker": "AAPL",
                "horizon": 1,
                "partition": "holdout",
                "qlike_mean": 0.12,
                "qlike_std": 0.01,
                "mae_mean": 0.005,
                "mae_std": 0.0005,
                "rmse_mean": 0.007,
                "rmse_std": 0.0008,
                "mape_mean": 25.0,
                "mape_std": 1.0,
            },
            {
                "model": "tft",
                "ticker": "AAPL",
                "horizon": 1,
                "partition": "holdout",
                "qlike_mean": 0.09,
                "qlike_std": 0.012,
                "mae_mean": 0.004,
                "mae_std": 0.0004,
                "rmse_mean": 0.006,
                "rmse_std": 0.0007,
                "mape_mean": 21.0,
                "mape_std": 1.2,
            },
        ]
    )


def test_metrics_to_markdown_includes_all_metrics():
    md = metrics_to_markdown(_summary())
    assert "QLIKE" in md
    assert "MAE" in md
    assert "RMSE" in md
    assert "MAPE" in md
    assert "garch" in md.lower()
    assert "tft" in md.lower()


def test_metrics_to_latex_returns_tabular_block():
    tex = metrics_to_latex(_summary())
    assert "\\begin{tabular}" in tex
    assert "\\end{tabular}" in tex

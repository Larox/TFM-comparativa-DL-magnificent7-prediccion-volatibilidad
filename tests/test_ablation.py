import pandas as pd

from tfm_volatility.interpret.ablation import lofo_qlike_delta


def test_lofo_qlike_delta_returns_one_row_per_feature():
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-02", periods=10, freq="B"),
            "ticker": "AAPL",
            "rv": 0.02,
            "log_rv": -3.9,
            "VIX": 15.0,
            "FED_FUNDS": 4.0,
            "CPI": 300.0,
        }
    )

    def fake_predict(p, ablate_feature: str | None = None):
        pred = p["rv"].copy() + 0.001
        if ablate_feature == "VIX":
            pred = pred + 0.01  # ablating VIX makes things worse
        return pd.DataFrame({"prediction": pred, "target": p["rv"]})

    out = lofo_qlike_delta(
        panel,
        predict_fn=fake_predict,
        features=["VIX", "FED_FUNDS", "CPI"],
    )
    assert set(out["feature"]) == {"VIX", "FED_FUNDS", "CPI"}
    vix_delta = out.loc[out["feature"] == "VIX", "qlike_delta"].iloc[0]
    others = out.loc[out["feature"] != "VIX", "qlike_delta"]
    assert vix_delta > others.max()

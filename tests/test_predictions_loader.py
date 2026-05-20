import pandas as pd
import pytest

from tfm_volatility.eval.loader import canonical_form, load_all_predictions


def _toy(model: str, seed: int, n_dup: int = 1) -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2025-01-02", periods=3, freq="B").date:
        for tkr in ("AAPL", "MSFT"):
            for h in (1, 5, 21):
                for _dup in range(n_dup):
                    rows.append(
                        {
                            "date": d,
                            "ticker": tkr,
                            "horizon": h,
                            "prediction": 0.02,
                            "target": 0.021,
                            "model": model,
                            "seed": seed,
                            "partition": "holdout",
                            "q10": float("nan"),
                            "q50": 0.02,
                            "q90": float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def test_canonical_form_dedupes_per_key():
    df = _toy("deepar", 42, n_dup=3)
    canon = canonical_form(df)
    keys = ["date", "ticker", "horizon", "model", "seed", "partition"]
    assert canon.duplicated(keys).sum() == 0


def test_canonical_form_keeps_lowest_abs_error_per_key():
    df = _toy("deepar", 42, n_dup=1)
    extra = df.iloc[[0]].copy()
    extra["prediction"] = 1.0  # huge error
    df = pd.concat([df, extra], ignore_index=True)
    canon = canonical_form(df)
    chosen = canon[
        (canon["date"] == df["date"].iloc[0])
        & (canon["ticker"] == "AAPL")
        & (canon["horizon"] == 1)
    ]
    assert len(chosen) == 1
    assert chosen["prediction"].iloc[0] == pytest.approx(0.02)


def test_load_all_predictions_returns_concatenated_canonical_df(tmp_path):
    for model in ("garch", "deepar", "tft"):
        for seed in (42, 1337):
            for partition in ("val", "holdout"):
                df = _toy(model, seed)
                df["model"] = model
                df["seed"] = seed
                df["partition"] = partition
                out = tmp_path / f"{model}_seed{seed}_{partition}.parquet"
                df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
    big = load_all_predictions(tmp_path)
    assert set(big["model"]) == {"garch", "deepar", "tft"}
    assert set(big["seed"]) == {42, 1337}
    assert set(big["partition"]) == {"val", "holdout"}

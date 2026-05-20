"""Diebold-Mariano test with HLN finite-sample correction.

References:
- Diebold & Mariano (1995), J. Bus. Econ. Stat.
- Harvey, Leybourne & Newbold (1997), Int. J. Forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _autocov(x: np.ndarray, k: int) -> float:
    n = len(x)
    if k >= n:
        return 0.0
    x = x - x.mean()
    return float((x[: n - k] * x[k:]).sum() / n)


def diebold_mariano(
    target: pd.Series,
    pred_a: pd.Series,
    pred_b: pd.Series,
    *,
    horizon: int,
) -> dict:
    """Pairwise DM test (squared error loss) with HLN correction.

    Returns a dict with `dm_stat`, `dm_stat_hln`, `p_value`, `n`, `correction_factor`.
    `p_value` uses the HLN-corrected stat against Student's t with n-1 df.
    Sign convention: negative DM means model A has lower mean loss than B.
    """
    df = pd.concat([target, pred_a, pred_b], axis=1).dropna()
    df.columns = ["target", "a", "b"]
    if len(df) < 5:
        raise ValueError(f"Too few overlapping observations: {len(df)}")

    e_a = (df["target"] - df["a"]).to_numpy()
    e_b = (df["target"] - df["b"]).to_numpy()
    d = e_a**2 - e_b**2

    n = len(d)
    d_bar = float(d.mean())
    # Newey-West-style long-run variance of d_bar: gamma_0 + 2 sum_{k=1..h-1} gamma_k
    gamma_0 = _autocov(d, 0)
    var = gamma_0 + 2.0 * sum(_autocov(d, k) for k in range(1, horizon))
    var_d_bar = var / n
    dm_stat = 0.0 if var_d_bar <= 0 else d_bar / np.sqrt(var_d_bar)

    correction = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    correction = max(correction, 0.0)
    factor = float(np.sqrt(correction))
    dm_stat_hln = dm_stat * factor

    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_hln), df=n - 1))

    return {
        "dm_stat": float(dm_stat),
        "dm_stat_hln": float(dm_stat_hln),
        "p_value": float(p_value),
        "n": int(n),
        "correction_factor": factor,
    }


def run_dm_batch(
    predictions: pd.DataFrame,
    pairs: list[tuple[str, str]],
    *,
    seed_aggregation: str = "first",
) -> pd.DataFrame:
    """Run DM test for every (pair, horizon, scope) where scope is each ticker + 'pooled'.

    `seed_aggregation`:
        - 'first' keeps the lowest seed only.
        - 'mean' averages predictions across seeds before testing.
    """
    if seed_aggregation == "first":
        first_seed = sorted(predictions["seed"].unique())[0]
        preds = predictions[predictions["seed"] == first_seed].copy()
    elif seed_aggregation == "mean":
        keys = ["date", "ticker", "horizon", "model", "partition"]
        preds = predictions.groupby(keys, as_index=False).agg(
            {"prediction": "mean", "target": "mean"}
        )
    else:
        raise ValueError(f"Unknown seed_aggregation: {seed_aggregation}")

    tickers = sorted(preds["ticker"].unique())
    horizons = sorted(preds["horizon"].unique())
    rows: list[dict] = []

    for model_a, model_b in pairs:
        for h in horizons:
            sub = preds[preds["horizon"] == h]
            wide = sub.pivot_table(
                index=["date", "ticker"],
                columns="model",
                values=["prediction", "target"],
            ).reset_index()
            if (
                model_a not in wide["prediction"].columns
                or model_b not in wide["prediction"].columns
            ):
                continue
            for scope in tickers + ["pooled"]:
                slice_ = wide if scope == "pooled" else wide[wide["ticker"] == scope]
                if len(slice_) < 5:
                    continue
                try:
                    res = diebold_mariano(
                        slice_["target"][model_a],
                        slice_["prediction"][model_a],
                        slice_["prediction"][model_b],
                        horizon=int(h),
                    )
                except ValueError:
                    continue
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "horizon": int(h),
                        "scope": scope,
                        **res,
                    }
                )
    return pd.DataFrame(rows)

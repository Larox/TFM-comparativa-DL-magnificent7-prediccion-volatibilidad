"""Leave-One-Feature-Out ablation utility.

Given a `predict_fn(panel, ablate_feature=...)` callable, return the QLIKE delta
caused by zero-ablating each feature. Positive delta means the feature helps
the model (removing it increases the error).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from tfm_volatility.eval.metrics import qlike


def lofo_qlike_delta(
    panel: pd.DataFrame,
    *,
    predict_fn: Callable[..., pd.DataFrame],
    features: list[str],
) -> pd.DataFrame:
    """For each feature, compute QLIKE(ablated) - QLIKE(baseline). Positive => feature helps."""
    base = predict_fn(panel)
    base_qlike = qlike(base["target"], base["prediction"])

    rows: list[dict] = []
    for feat in features:
        out = predict_fn(panel, ablate_feature=feat)
        q = qlike(out["target"], out["prediction"])
        rows.append(
            {
                "feature": feat,
                "qlike_baseline": base_qlike,
                "qlike_ablated": q,
                "qlike_delta": q - base_qlike,
            }
        )
    return pd.DataFrame(rows)

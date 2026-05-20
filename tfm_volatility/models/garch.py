"""GARCH(p, q) fitting with AIC-based distribution selection.

Memoria Cap. 3 A2.1: "Comparar la distribución gaussiana, t-Student y GED para las
innovaciones con el criterio de Akaike y retener la distribución que tenga el AIC
más bajo (Bollerslev, 1986; Nelson, 1991)."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from arch import arch_model


@dataclass
class GarchFit:
    distribution: str
    aic: float
    params: dict[str, float]
    aic_by_distribution: dict[str, float]
    fitted: Any  # underlying arch result object


def fit_best_garch(
    returns: pd.Series,
    *,
    p: int = 1,
    q: int = 1,
    distributions: list[str] | None = None,
    mean: str = "Constant",
    rescale: bool = True,
) -> GarchFit:
    """Fit GARCH(p, q) with each candidate distribution and return the best by AIC."""
    if distributions is None:
        distributions = ["normal", "t", "ged"]

    series = returns.dropna()
    results: dict[str, Any] = {}
    aics: dict[str, float] = {}
    for dist in distributions:
        am = arch_model(
            series,
            mean=mean,
            vol="GARCH",
            p=p,
            q=q,
            dist=dist,
            rescale=rescale,
        )
        res = am.fit(disp="off", show_warning=False)
        results[dist] = res
        aics[dist] = float(res.aic)

    best_dist = min(aics, key=aics.get)
    best_res = results[best_dist]
    return GarchFit(
        distribution=best_dist,
        aic=aics[best_dist],
        params={k: float(v) for k, v in best_res.params.items()},
        aic_by_distribution=aics,
        fitted=best_res,
    )

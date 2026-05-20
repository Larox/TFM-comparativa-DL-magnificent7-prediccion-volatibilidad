"""Fit GARCH(1,1) per ticker and generate multi-horizon predictions.

Activities A2.1 + A2.2 from memoria §3.3.2.

Usage:
    uv run scripts/03_fit_garch.py --seed 42
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tfm_volatility import config
from tfm_volatility.data.splits import HOLDOUT_START
from tfm_volatility.models.garch import fit_best_garch, forecast_garch
from tfm_volatility.models.predictions import save_predictions
from tfm_volatility.utils.seeds import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("garch")

REFIT_INTERVAL_DAYS = 21


def _latest_snapshot() -> Path:
    files = sorted(glob.glob(str(config.DATA_PROCESSED / "snapshot_*.parquet")))
    if not files:
        raise FileNotFoundError("No processed snapshot. Run scripts 01 + 02 first.")
    return Path(files[-1])


def _val_split_date(dates: np.ndarray) -> dt.date:
    """Validation = last 20% of the development period."""
    dev_dates = [d for d in dates if d < HOLDOUT_START]
    if not dev_dates:
        raise RuntimeError("No development dates available.")
    return dev_dates[int(0.8 * len(dev_dates))]


def _fit_and_forecast_for_ticker(
    panel_ticker: pd.DataFrame,
    cfg: dict,
    val_split: dt.date,
    seed: int,
) -> pd.DataFrame:
    panel_ticker = panel_ticker.sort_values("date").reset_index(drop=True)
    horizons = list(config.FORECAST_HORIZONS)
    rows: list[dict] = []

    forecast_origins = panel_ticker.index[panel_ticker["date"] >= val_split].tolist()
    last_refit_idx = -10_000
    current_fit = None

    for origin_idx in forecast_origins:
        origin_date = panel_ticker.loc[origin_idx, "date"]
        if origin_idx - last_refit_idx >= REFIT_INTERVAL_DAYS or current_fit is None:
            train = panel_ticker.iloc[:origin_idx]
            current_fit = fit_best_garch(
                train["log_return"],
                p=cfg["p"],
                q=cfg["q"],
                distributions=cfg["distributions"],
                mean=cfg["mean"],
                rescale=cfg["rescale"],
            )
            last_refit_idx = origin_idx
            log.info(
                "Refit at %s (idx=%d): chose %s (AIC=%.2f)",
                origin_date,
                origin_idx,
                current_fit.distribution,
                current_fit.aic,
            )

        fc = forecast_garch(current_fit, horizons=horizons)
        partition = "holdout" if origin_date >= HOLDOUT_START else "val"
        for _, fc_row in fc.iterrows():
            h = int(fc_row["horizon"])
            target_idx = origin_idx + h
            if target_idx < len(panel_ticker):
                target_rv = float(panel_ticker.loc[target_idx, "rv"])
            else:
                target_rv = float("nan")
            rows.append(
                {
                    "date": origin_date,
                    "ticker": panel_ticker["ticker"].iloc[0],
                    "horizon": h,
                    "prediction": float(fc_row["predicted_volatility"]),
                    "target": target_rv,
                    "model": "garch",
                    "seed": seed,
                    "partition": partition,
                    "q10": float("nan"),
                    "q50": float(fc_row["predicted_volatility"]),
                    "q90": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--config", type=Path, default=config.CONFIGS_DIR / "garch.yaml"
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    cfg = yaml.safe_load(args.config.read_text())

    snap = _latest_snapshot()
    log.info("Loading snapshot: %s", snap)
    panel = (
        pd.read_parquet(snap).sort_values(["ticker", "date"]).reset_index(drop=True)
    )
    val_split = _val_split_date(np.array(sorted(panel["date"].unique())))
    log.info("Validation begins at %s", val_split)

    all_preds: list[pd.DataFrame] = []
    for tkr in config.TICKERS:
        sub = panel[panel["ticker"] == tkr].reset_index(drop=True)
        log.info("Ticker %s — %d rows", tkr, len(sub))
        preds = _fit_and_forecast_for_ticker(sub, cfg, val_split, args.seed)
        all_preds.append(preds)
    out_df = pd.concat(all_preds, ignore_index=True)

    for partition in ("val", "holdout"):
        slice_df = out_df[out_df["partition"] == partition]
        out = (
            config.RESULTS_DIR
            / "predictions"
            / f"garch_seed{args.seed}_{partition}.parquet"
        )
        save_predictions(slice_df, out)
        log.info("Wrote %s (%d rows)", out, len(slice_df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

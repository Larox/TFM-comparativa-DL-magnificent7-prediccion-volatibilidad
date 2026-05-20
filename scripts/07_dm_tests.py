"""Diebold-Mariano pairwise tests (TFT vs GARCH, TFT vs DeepAR, DeepAR vs GARCH).

Activities A3.4 + A3.5 from memoria §3.3.2.

Usage:
    uv run scripts/07_dm_tests.py [--partition holdout] [--seed-aggregation mean]
"""

from __future__ import annotations

import argparse
import logging

from tfm_volatility import config
from tfm_volatility.eval.dm_test import run_dm_batch
from tfm_volatility.eval.loader import load_all_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("dm")

_PAIRS = [("tft", "garch"), ("tft", "deepar"), ("deepar", "garch")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition", choices=["val", "holdout"], default="holdout"
    )
    parser.add_argument(
        "--seed-aggregation",
        choices=["first", "mean"],
        default="mean",
        help="Average predictions across seeds before testing, or just use the lowest seed.",
    )
    args = parser.parse_args()

    preds = load_all_predictions(config.RESULTS_DIR / "predictions")
    preds = preds[preds["partition"] == args.partition]
    log.info("Running DM batch on %d rows", len(preds))

    results = run_dm_batch(
        preds, pairs=_PAIRS, seed_aggregation=args.seed_aggregation
    )

    out_dir = config.RESULTS_DIR / "dm_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"dm_{args.partition}_{args.seed_aggregation}.parquet"
    out_csv = out_dir / f"dm_{args.partition}_{args.seed_aggregation}.csv"
    results.to_parquet(out_parquet, index=False)
    results.to_csv(out_csv, index=False)
    log.info("Wrote %s and %s", out_parquet, out_csv)
    print(results.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

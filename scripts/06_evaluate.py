"""Compute QLIKE/MAE/RMSE/MAPE per (model, ticker, horizon, seed) and summarize.

Activities A3.1 + A3.2 + A3.3 from memoria §3.3.2.

Usage:
    uv run scripts/06_evaluate.py [--partition val|holdout|both]
"""

from __future__ import annotations

import argparse
import logging

from tfm_volatility import config
from tfm_volatility.eval.aggregate import compute_metrics_long, summarize_across_seeds
from tfm_volatility.eval.loader import load_all_predictions
from tfm_volatility.eval.tables import metrics_to_latex, metrics_to_markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("eval")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        choices=["val", "holdout", "both"],
        default="holdout",
    )
    args = parser.parse_args()

    preds = load_all_predictions(config.RESULTS_DIR / "predictions")
    if args.partition != "both":
        preds = preds[preds["partition"] == args.partition]
    log.info("Loaded %d prediction rows", len(preds))

    metrics_long = compute_metrics_long(preds)
    summary = summarize_across_seeds(metrics_long)

    out_dir = config.RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_long.to_parquet(out_dir / "metrics_long.parquet", index=False)
    summary.to_parquet(out_dir / "metrics_summary.parquet", index=False)
    (out_dir / "metrics_summary.md").write_text(metrics_to_markdown(summary))
    (out_dir / "metrics_summary.tex").write_text(metrics_to_latex(summary))
    log.info("Wrote metrics tables to %s", out_dir)
    print(metrics_to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

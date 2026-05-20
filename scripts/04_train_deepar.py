"""Train DeepAR and dump predictions for val and holdout.

Activities A2.3 + A2.4 from memoria §3.3.2.

Usage:
    uv run scripts/04_train_deepar.py --seed 42 [--max-epochs 50]
"""

from __future__ import annotations

import argparse
import glob
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Silence the (harmless) sklearn warning that fires once per batch when
# pytorch-forecasting passes numpy arrays into a StandardScaler that was fitted
# with a named DataFrame. Issue #1421 of pytorch-forecasting, no numerical impact.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from tfm_volatility import config
from tfm_volatility.data.splits import HOLDOUT_START
from tfm_volatility.models.common import build_datasets, prepare_panel
from tfm_volatility.models.deepar import build_deepar, build_trainer, pick_device
from tfm_volatility.models.predictions import save_predictions
from tfm_volatility.utils.seeds import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("deepar")


def _latest_snapshot() -> Path:
    files = sorted(glob.glob(str(config.DATA_PROCESSED / "snapshot_*.parquet")))
    if not files:
        raise FileNotFoundError("No processed snapshot. Run scripts 01 + 02 first.")
    return Path(files[-1])


def _val_split_offset(panel: pd.DataFrame) -> int:
    dev_times = panel.loc[panel["date"] < HOLDOUT_START, "time_idx"]
    return int(0.8 * dev_times.max()) if len(dev_times) else 0


def _emit_predictions(
    model,
    predict_ds,
    panel: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Run model.predict on predict_ds and align outputs to (date, ticker, horizon)."""
    raw = model.predict(predict_ds, mode="raw", return_index=True, return_x=False)
    pred_tensor = raw.output["prediction"] if hasattr(raw.output, "__getitem__") else raw.output
    # pred_tensor shape can be (n_samples, prediction_length) or
    # (n_samples, prediction_length, n_quantiles_or_samples). Reduce the trailing dim.
    if hasattr(pred_tensor, "ndim") and pred_tensor.ndim > 2:
        pred_arr = pred_tensor.mean(dim=-1).cpu().numpy()
    else:
        pred_arr = pred_tensor.cpu().numpy()
    index_df = raw.index  # columns: ticker, time_idx (decoder start)

    horizons = list(config.FORECAST_HORIZONS)
    rows: list[dict] = []
    lookup = panel.set_index(["ticker", "time_idx"])[["date", "rv"]]

    for i in range(len(index_df)):
        tkr = index_df.iloc[i]["ticker"]
        start_time_idx = int(index_df.iloc[i]["time_idx"])
        for h in horizons:
            t_h = start_time_idx + (h - 1)
            try:
                date_t, rv_target = lookup.loc[(tkr, t_h)]
            except KeyError:
                continue
            log_pred = float(pred_arr[i, h - 1])
            pred = float(np.exp(log_pred))
            partition = "holdout" if date_t >= HOLDOUT_START else "val"
            rows.append(
                {
                    "date": date_t,
                    "ticker": tkr,
                    "horizon": h,
                    "prediction": pred,
                    "target": float(rv_target),
                    "model": "deepar",
                    "seed": seed,
                    "partition": partition,
                    "q10": float("nan"),
                    "q50": pred,
                    "q90": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--config", type=Path, default=config.CONFIGS_DIR / "deepar.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override config max_epochs")
    args = parser.parse_args()

    set_global_seed(args.seed)
    cfg = yaml.safe_load(args.config.read_text())
    max_epochs = args.max_epochs if args.max_epochs is not None else cfg["max_epochs"]

    snap = _latest_snapshot()
    log.info("Loading snapshot: %s", snap)
    raw_panel = pd.read_parquet(snap)
    panel = prepare_panel(raw_panel)
    val_offset = _val_split_offset(panel)
    log.info("Validation begins at time_idx %d", val_offset)

    train_ds, val_ds, predict_ds = build_datasets(
        panel,
        val_start_offset=val_offset,
        encoder_length=cfg["encoder_length"],
        prediction_length=cfg["prediction_length"],
        target=cfg["target"],
    )

    train_loader = train_ds.to_dataloader(train=True, batch_size=cfg["batch_size"], num_workers=0)
    val_loader = val_ds.to_dataloader(train=False, batch_size=cfg["batch_size"], num_workers=0)

    model = build_deepar(
        train_ds,
        hidden_size=cfg["hidden_size"],
        rnn_layers=cfg["rnn_layers"],
        dropout=cfg["dropout"],
        learning_rate=cfg["learning_rate"],
    )
    log.info("Using device: %s", pick_device())

    trainer = build_trainer(
        max_epochs=max_epochs,
        early_stopping_patience=cfg["early_stopping_patience"],
        gradient_clip_val=cfg["gradient_clip_val"],
        log_dir=str(config.REPO_ROOT / "lightning_logs" / f"deepar_seed{args.seed}"),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    preds = _emit_predictions(model, predict_ds, panel, seed=args.seed)

    for partition in ("val", "holdout"):
        slice_df = preds[preds["partition"] == partition]
        out = config.RESULTS_DIR / "predictions" / f"deepar_seed{args.seed}_{partition}.parquet"
        save_predictions(slice_df, out)
        log.info("Wrote %s (%d rows)", out, len(slice_df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

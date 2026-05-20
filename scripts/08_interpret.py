"""Extract VSN ranking + attention heatmaps from a saved TFT checkpoint.

Activities A4.1 + A4.2 + A4.3 from memoria §3.3.2.

Usage:
    uv run scripts/08_interpret.py [--seed 42]
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tfm_volatility import config
from tfm_volatility.interpret.attention import (
    aggregate_attention_heads,
    plot_attention_heatmap,
)
from tfm_volatility.interpret.vsn import (
    extract_vsn_weights,
    load_tft_checkpoint,
    rank_features_from_vsn_dict,
)
from tfm_volatility.models.common import build_datasets, prepare_panel
from tfm_volatility.models.deepar import pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("interpret")


def _latest_snapshot() -> Path:
    files = sorted(glob.glob(str(config.DATA_PROCESSED / "snapshot_*.parquet")))
    return Path(files[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    ckpt = config.REPO_ROOT / "checkpoints" / f"tft_seed{args.seed}.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No TFT checkpoint at {ckpt}; run scripts/05_train_tft.py first"
        )

    log.info("Loading checkpoint: %s", ckpt)
    model = load_tft_checkpoint(ckpt)

    raw_panel = pd.read_parquet(_latest_snapshot())
    panel = prepare_panel(raw_panel)
    _, _, predict_ds = build_datasets(
        panel,
        val_start_offset=int(0.8 * panel["time_idx"].max()),
        encoder_length=60,
        prediction_length=21,
        target="log_rv",
    )

    device = pick_device()
    log.info("Extracting VSN weights on device %s", device)
    vsn_dict = extract_vsn_weights(model, predict_ds, device=device)
    ranking = rank_features_from_vsn_dict(vsn_dict)

    interpret_dir = config.RESULTS_DIR / "interpret"
    figures_dir = config.RESULTS_DIR / "figures"
    interpret_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ranking.to_parquet(
        interpret_dir / f"vsn_ranking_seed{args.seed}.parquet", index=False
    )
    log.info("Wrote VSN ranking")

    loader = predict_ds.to_dataloader(train=False, batch_size=256, num_workers=0)
    interp = model.interpret_output(loader, reduction="none")
    att = interp.get("attention")
    if att is not None:
        att_np = att.detach().cpu().numpy()
        if att_np.ndim == 4:
            att_avg = aggregate_attention_heads(att_np).mean(axis=0)
        else:
            att_avg = att_np.mean(axis=0)
        np.save(interpret_dir / f"attention_pooled_seed{args.seed}.npy", att_avg)
        fig = plot_attention_heatmap(
            att_avg, title=f"TFT attention (seed {args.seed})"
        )
        fig.savefig(
            figures_dir / f"attention_pooled_seed{args.seed}.png", dpi=150
        )
        log.info("Wrote attention artifacts")

    print(ranking.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

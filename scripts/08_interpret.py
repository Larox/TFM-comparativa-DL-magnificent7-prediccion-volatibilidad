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
    extract_raw_prediction,
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
        raise FileNotFoundError(f"No TFT checkpoint at {ckpt}; run scripts/05_train_tft.py first")

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
    log.info("Extracting raw predictions on device %s (single pass shared by VSN + attention)", device)
    raw = extract_raw_prediction(model, predict_ds, device=device)

    log.info("Computing VSN weights (reduction=sum)")
    vsn_interp = model.interpret_output(raw.output, reduction="sum")
    vsn_weights = vsn_interp["encoder_variables"].detach().cpu().numpy()
    feature_names = (
        vsn_interp.get("encoder_variables_names")
        or getattr(predict_ds, "encoder_variables", None)
        or getattr(predict_ds, "reals", None)
        or []
    )
    vsn_dict = {
        "pooled": {
            (str(feature_names[i]) if i < len(feature_names) else f"feat_{i}"): float(
                vsn_weights[i]
            )
            for i in range(len(vsn_weights))
        }
    }
    ranking = rank_features_from_vsn_dict(vsn_dict)

    interpret_dir = config.RESULTS_DIR / "interpret"
    figures_dir = config.RESULTS_DIR / "figures"
    interpret_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ranking.to_parquet(interpret_dir / f"vsn_ranking_seed{args.seed}.parquet", index=False)
    log.info("Wrote VSN ranking")

    log.info("Computing attention (reduction=none)")
    att_interp = model.interpret_output(raw.output, reduction="none")
    att = att_interp.get("attention")
    if att is None:
        att = att_interp.get("decoder_attention")
    if att is not None:
        att_np = att.detach().cpu().numpy()
        # The shape pytorch-forecasting returns varies across versions:
        # - (n_samples, n_heads, enc_len, dec_len) -> mean heads then samples -> 2D
        # - (n_samples, n_heads, enc_len)          -> mean heads then samples -> 1D
        # - (n_samples, enc_len, dec_len)          -> mean samples -> 2D
        # - (n_samples, enc_len)                   -> mean samples -> 1D
        log.info("Raw attention tensor shape: %s", att_np.shape)
        if att_np.ndim == 4:
            att_avg = aggregate_attention_heads(att_np).mean(axis=0)
        elif att_np.ndim == 3:
            # Either (n, heads, enc) or (n, enc, dec). Heads are typically <= 8;
            # encoder/decoder lengths are larger. Collapse accordingly.
            att_avg = (
                att_np.mean(axis=(0, 1)) if att_np.shape[1] <= 8 else att_np.mean(axis=0)
            )
        elif att_np.ndim == 2:
            att_avg = att_np.mean(axis=0)
        else:
            att_avg = att_np

        np.save(interpret_dir / f"attention_pooled_seed{args.seed}.npy", att_avg)

        if att_avg.ndim == 2:
            fig = plot_attention_heatmap(
                att_avg, title=f"TFT attention (seed {args.seed})"
            )
        else:
            # 1D attention: bar plot over encoder positions (lag importance).
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.bar(range(len(att_avg)), att_avg, color="#1f77b4")
            ax.set_title(f"TFT encoder attention by lag (seed {args.seed})")
            ax.set_xlabel("Encoder step (0 = most recent)")
            ax.set_ylabel("Attention weight")
            fig.tight_layout()

        fig.savefig(figures_dir / f"attention_pooled_seed{args.seed}.png", dpi=150)
        log.info("Wrote attention artifacts (shape %s)", att_avg.shape)
    else:
        log.warning("interpret_output() returned no attention tensor; skipping heatmap")

    print(ranking.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

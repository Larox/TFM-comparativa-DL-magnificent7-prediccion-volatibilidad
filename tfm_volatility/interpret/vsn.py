"""Extract Variable Selection Network (VSN) importance from a trained TFT.

Limitations are clearly stated in the memoria (Cap. 3 A4, §2.4): VSN weights are
not causal attributions but are the standard interpretability hook for TFT.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tft_checkpoint(checkpoint_path: Path):
    """Load a saved TFT checkpoint into a `TemporalFusionTransformer` instance."""
    from pytorch_forecasting import TemporalFusionTransformer

    return TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))


def extract_raw_prediction(model, predict_dataset, *, device: str = "cpu"):
    """Wrap `model.predict(mode='raw', return_x=True)` to feed `interpret_output`.

    pytorch-forecasting 1.7 requires the raw prediction dict (not a DataLoader)
    as the input to `interpret_output`.
    """
    model = model.to(device).eval()
    loader = predict_dataset.to_dataloader(train=False, batch_size=256, num_workers=0)
    return model.predict(loader, mode="raw", return_x=True, return_index=True)


def extract_vsn_weights(
    model,
    predict_dataset,
    *,
    device: str = "cpu",
) -> dict[str, dict[str, float]]:
    """Run `interpret_output` and return a nested dict {'pooled': {feature: weight}}.

    Encoder variable importances are summed over the prediction window by the
    library; we then map them back to their feature names via the dataset.
    """
    raw = extract_raw_prediction(model, predict_dataset, device=device)
    interp = model.interpret_output(raw.output, reduction="sum")

    feature_names = (
        interp.get("encoder_variables_names")
        or getattr(predict_dataset, "encoder_variables", None)
        or getattr(predict_dataset, "reals", None)
        or []
    )
    weights = interp["encoder_variables"].detach().cpu().numpy()
    out: dict[str, dict[str, float]] = {"pooled": {}}
    for i in range(len(weights)):
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        out["pooled"][str(name)] = float(weights[i])
    return out


def rank_features_from_vsn_dict(
    vsn_dict: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Convert the nested VSN dict to a long DataFrame with per-ticker ranks."""
    rows: list[dict] = []
    for ticker, feat_weights in vsn_dict.items():
        total = sum(feat_weights.values()) or 1.0
        sorted_feats = sorted(feat_weights.items(), key=lambda kv: -kv[1])
        for rank, (feat, w) in enumerate(sorted_feats, start=1):
            rows.append(
                {
                    "ticker": ticker,
                    "feature": feat,
                    "weight": w,
                    "weight_normalized": w / total,
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)

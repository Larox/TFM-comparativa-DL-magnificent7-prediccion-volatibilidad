"""Temporal Fusion Transformer factory (Lim et al., 2021)."""

from __future__ import annotations

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss


def build_tft(
    train_ds: TimeSeriesDataSet,
    *,
    hidden_size: int,
    attention_head_size: int,
    hidden_continuous_size: int,
    dropout: float,
    learning_rate: float,
    quantiles: list[float],
) -> TemporalFusionTransformer:
    """Build TFT with quantile loss at the requested quantiles."""
    return TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(quantiles=quantiles),
        output_size=len(quantiles),
        log_interval=20,
        reduce_on_plateau_patience=4,
    )

"""DeepAR factory (Salinas et al., 2020) using pytorch-forecasting.

The factory is split from training so the model + trainer constructions are
unit-testable without performing optimization.
"""

from __future__ import annotations

import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import NormalDistributionLoss


def pick_device() -> str:
    """mps -> cuda -> cpu, with TFM_FORCE_CPU=1 escape hatch."""
    if os.environ.get("TFM_FORCE_CPU") == "1":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_deepar(
    train_ds: TimeSeriesDataSet,
    *,
    hidden_size: int,
    rnn_layers: int,
    dropout: float,
    learning_rate: float,
) -> DeepAR:
    """Construct DeepAR with Gaussian likelihood over the target (log_rv)."""
    return DeepAR.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
        loss=NormalDistributionLoss(),
    )


def build_trainer(
    *,
    max_epochs: int,
    early_stopping_patience: int,
    gradient_clip_val: float,
    log_dir: str,
) -> Trainer:
    """Construct a Lightning Trainer with sensible TFM defaults."""
    device = pick_device()
    accelerator = "gpu" if device in ("mps", "cuda") else "cpu"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        default_root_dir=log_dir,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

"""TFT multi-head attention extraction and visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def aggregate_attention_heads(att: np.ndarray) -> np.ndarray:
    """Average over the heads dimension.

    Input shape:  (n_samples, n_heads, encoder_length, decoder_length)
    Output shape: (n_samples, encoder_length, decoder_length)
    """
    return att.mean(axis=1)


def plot_attention_heatmap(att: np.ndarray, *, title: str) -> plt.Figure:
    """Plot a single (encoder x decoder) attention matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(att, cmap="viridis", ax=ax, cbar=True)
    ax.set_xlabel("Decoder step (forecast horizon)")
    ax.set_ylabel("Encoder step (past observation)")
    ax.set_title(title)
    fig.tight_layout()
    return fig

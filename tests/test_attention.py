import matplotlib
import numpy as np

matplotlib.use("Agg")

from tfm_volatility.interpret.attention import (
    aggregate_attention_heads,
    plot_attention_heatmap,
)


def test_aggregate_attention_heads_collapses_head_dim():
    rng = np.random.default_rng(0)
    att = rng.random((4, 4, 20, 10))
    agg = aggregate_attention_heads(att)
    assert agg.shape == (4, 20, 10)


def test_plot_attention_heatmap_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    att = rng.random((20, 10))
    out = tmp_path / "att.png"
    fig = plot_attention_heatmap(att, title="AAPL")
    fig.savefig(out)
    assert out.exists()

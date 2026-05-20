# tfm_code

Experimental codebase for the TFM "Comparative volatility forecasting on the Magnificent 7 (GARCH(1,1), DeepAR, Temporal Fusion Transformer)" — Master's in AI, UNIR.

Companion memoria repo: `../master_ai_thesis/`.
Design spec: `../master_ai_thesis/docs/superpowers/specs/2026-05-19-tfm-code-design.md`.

## Setup

```bash
uv sync
cp .env.example .env  # then paste your FRED API key into .env
```

## Pipeline — Plan 1 (OE1: dataset)

```bash
uv run scripts/01_download_data.py     # yfinance OHLCV + FRED macros
uv run scripts/02_build_dataset.py     # log returns + 21d RV + calendar align
uv run pytest -v -m "not network"      # unit + smoke tests
```

After step 2 the processed snapshot lives at `data/processed/snapshot_<date>.parquet` with a JSON manifest next to it.

## Pipeline — Plan 2 (OE2: models)

Single seed (default 42):

```bash
uv run scripts/03_fit_garch.py
uv run scripts/04_train_deepar.py
uv run scripts/05_train_tft.py
```

All three models × three seeds (42, 1337, 2024):

```bash
./scripts/run_all_models.sh
```

Outputs:

- `results/predictions/garch_seed{S}_{val|holdout}.parquet`
- `results/predictions/deepar_seed{S}_{val|holdout}.parquet`
- `results/predictions/tft_seed{S}_{val|holdout}.parquet`
- `checkpoints/tft_seed{S}.ckpt` (kept for OE4 interpretability in Plan 3)

All prediction files share the same schema; see `tfm_volatility/models/predictions.py::PREDICTION_COLUMNS`.

### Hardware

Trains on Apple Silicon MPS by default. To force CPU (debugging or MPS op fallbacks):

```bash
TFM_FORCE_CPU=1 uv run scripts/04_train_deepar.py
```

## Pipeline — Plan 3 (OE3 + OE4)

```bash
uv run scripts/06_evaluate.py        # QLIKE/MAE/RMSE/MAPE per cell + summary
uv run scripts/07_dm_tests.py        # Diebold-Mariano + HLN per pair x horizon x scope
uv run scripts/08_interpret.py       # VSN ranking + attention heatmap (seed 42 by default)
```

Outputs:

- `results/metrics/metrics_long.parquet` — per (model, ticker, horizon, seed) row
- `results/metrics/metrics_summary.parquet` + `.md` + `.tex`
- `results/dm_tests/dm_<partition>_<aggregation>.parquet` + `.csv`
- `results/interpret/vsn_ranking_seed{S}.parquet`
- `results/interpret/attention_pooled_seed{S}.npy`
- `results/figures/*.png`

Copy the figures and tables you want to cite from `results/` into `../master_ai_thesis/memoria/figures/` manually.

## Layout

- `tfm_volatility/` — importable package (all logic)
- `scripts/` — thin entry points, 1:1 with memoria Cap. 3 activities
- `configs/` — YAML hyperparameter configs (added in Plan 2)
- `data/` — gitignored, populated by scripts
- `results/` — gitignored, predictions + metrics + figures (Plan 3)
- `tests/` — pytest suite

## Reproducibility

- Seeds: `tfm_volatility.utils.seeds.set_global_seed(seed)` (default 42).
- 2025 hold-out is gated: `load_holdout()` requires `confirm=True` or `TFM_HOLDOUT_OK=1`.
- Library versions captured in `uv.lock` and in each `manifest_*.json`.

## Author

Sebastian Arias — sole author.

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

To be added once Plan 2 is written.

## Pipeline — Plan 3 (OE3 + OE4)

To be added once Plan 3 is written.

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

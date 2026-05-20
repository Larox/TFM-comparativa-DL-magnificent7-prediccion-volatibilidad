# tfm_code

Experimental codebase for the TFM "Comparative volatility forecasting on the Magnificent 7 (GARCH(1,1), DeepAR, Temporal Fusion Transformer)" — Master's in AI, UNIR.

Companion memoria repo: `../master_ai_thesis/`.

## Setup

```bash
uv sync
cp .env.example .env  # then paste your FRED API key
```

## Running

End-to-end pipeline (run in order):

```bash
uv run scripts/01_download_data.py
uv run scripts/02_build_dataset.py
# Plan 2: 03–05 (GARCH, DeepAR, TFT)
# Plan 3: 06–08 (eval, DM, interpret)
```

## Layout

- `tfm_volatility/` — importable package (all logic)
- `scripts/` — thin entry points, 1:1 with memoria Cap. 3 activities
- `configs/` — YAML hyperparameter configs
- `data/` — gitignored, populated by scripts
- `results/` — gitignored, predictions + metrics + figures
- `tests/` — pytest suite

## Author

Sebastian Arias — sole author.

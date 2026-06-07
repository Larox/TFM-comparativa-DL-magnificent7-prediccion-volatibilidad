# Comparativa de Modelos de Deep Learning para la Predicción de la Volatilidad Realizada del grupo Magnificent 7

Master: Máster Universitario en Inteligencia Artificial

Trabajo fin de estudio presentado por: Andrés Sebastián Arias Tabango

Tipo de trabajo: Tipo 3 – Comparativa de Soluciones

Director/a: Dagoberto Mayorca Torres

Fecha: 6 de junio de 2026

---

🇪🇸 [Español](#español) · 🇬🇧 [English](#english)

---

## Español

Código experimental para el TFM "Comparativa de Modelos de Deep Learning para la Predicción de la Volatilidad Realizada del grupo Magnificent 7"

### Instalación

```bash
uv sync
cp .env.example .env  # luego pega tu API key de FRED en .env
```

### Pipeline — Plan 1 (OE1: dataset)

```bash
uv run scripts/01_download_data.py     # OHLCV de yfinance + macros de FRED
uv run scripts/02_build_dataset.py     # log-retornos + RV de 21 días + alineación de calendario
uv run pytest -v -m "not network"      # tests unitarios + de humo
```

Tras el paso 2, el snapshot procesado queda en `data/processed/snapshot_<date>.parquet` con su manifiesto JSON al lado.

### Pipeline — Plan 2 (OE2: modelos)

Semilla única (42 por defecto):

```bash
uv run scripts/03_fit_garch.py
uv run scripts/04_train_deepar.py
uv run scripts/05_train_tft.py
```

Los tres modelos × tres semillas (42, 1337, 2024):

```bash
./scripts/run_all_models.sh
```

Salidas:

- `results/predictions/garch_seed{S}_{val|holdout}.parquet`
- `results/predictions/deepar_seed{S}_{val|holdout}.parquet`
- `results/predictions/tft_seed{S}_{val|holdout}.parquet`
- `checkpoints/tft_seed{S}.ckpt` (se conserva para la interpretabilidad de OE4 en el Plan 3)

Todos los ficheros de predicciones comparten el mismo esquema; ver `tfm_volatility/models/predictions.py::PREDICTION_COLUMNS`.

#### Hardware

Entrena en Apple Silicon (MPS) por defecto. Para forzar CPU (depuración o caídas a CPU por operaciones no soportadas en MPS):

```bash
TFM_FORCE_CPU=1 uv run scripts/04_train_deepar.py
```

### Pipeline — Plan 3 (OE3 + OE4)

```bash
uv run scripts/06_evaluate.py        # QLIKE/MAE/RMSE/MAPE por celda + resumen
uv run scripts/07_dm_tests.py        # Diebold-Mariano + HLN por par × horizonte × ámbito
uv run scripts/08_interpret.py       # ranking VSN + heatmap de atención (semilla 42 por defecto)
```

Salidas:

- `results/metrics/metrics_long.parquet` — una fila por (modelo, ticker, horizonte, semilla)
- `results/metrics/metrics_summary.parquet` + `.md` + `.tex`
- `results/dm_tests/dm_<partition>_<aggregation>.parquet` + `.csv`
- `results/interpret/vsn_ranking_seed{S}.parquet`
- `results/interpret/attention_pooled_seed{S}.npy`
- `results/figures/*.png`

Las figuras y tablas que vayas a citar se copian manualmente desde `results/` hacia `../master_ai_thesis/memoria/figures/`.

### Estructura del repositorio

- `tfm_volatility/` — paquete importable (toda la lógica)
- `scripts/` — entry points finos, 1:1 con las actividades del Cap. 3 de la memoria
- `configs/` — configs YAML de hiperparámetros (añadidos en el Plan 2)
- `data/` — gitignored, lo pueblan los scripts
- `results/` — gitignored, predicciones + métricas + figuras (Plan 3)
- `tests/` — suite de pytest

### Reproducibilidad

- Semillas: `tfm_volatility.utils.seeds.set_global_seed(seed)` (42 por defecto).
- El hold-out de 2025 está gateado: `load_holdout()` requiere `confirm=True` o `TFM_HOLDOUT_OK=1`.
- Versiones de librerías fijadas en `uv.lock` y en cada `manifest_*.json`.

### Autor

Sebastian Arias — autor único.

---

## English

Experimental codebase for the TFM "Comparativa de Modelos de Deep Learning para la Predicción de la Volatilidad Realizada del grupo Magnificent 7"

### Setup

```bash
uv sync
cp .env.example .env  # then paste your FRED API key into .env
```

### Pipeline — Plan 1 (OE1: dataset)

```bash
uv run scripts/01_download_data.py     # yfinance OHLCV + FRED macros
uv run scripts/02_build_dataset.py     # log returns + 21d RV + calendar align
uv run pytest -v -m "not network"      # unit + smoke tests
```

After step 2 the processed snapshot lives at `data/processed/snapshot_<date>.parquet` with a JSON manifest next to it.

### Pipeline — Plan 2 (OE2: models)

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

#### Hardware

Trains on Apple Silicon MPS by default. To force CPU (debugging or MPS op fallbacks):

```bash
TFM_FORCE_CPU=1 uv run scripts/04_train_deepar.py
```

### Pipeline — Plan 3 (OE3 + OE4)

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

### Layout

- `tfm_volatility/` — importable package (all logic)
- `scripts/` — thin entry points, 1:1 with memoria Cap. 3 activities
- `configs/` — YAML hyperparameter configs (added in Plan 2)
- `data/` — gitignored, populated by scripts
- `results/` — gitignored, predictions + metrics + figures (Plan 3)
- `tests/` — pytest suite

### Reproducibility

- Seeds: `tfm_volatility.utils.seeds.set_global_seed(seed)` (default 42).
- 2025 hold-out is gated: `load_holdout()` requires `confirm=True` or `TFM_HOLDOUT_OK=1`.
- Library versions captured in `uv.lock` and in each `manifest_*.json`.

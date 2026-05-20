#!/usr/bin/env bash
# Run GARCH + DeepAR + TFT across the three seeds.
# Expects scripts 01 + 02 already executed.

set -euo pipefail

SEEDS=(42 1337 2024)

for SEED in "${SEEDS[@]}"; do
  echo "=== seed=$SEED — GARCH ==="
  uv run scripts/03_fit_garch.py --seed "$SEED"

  echo "=== seed=$SEED — DeepAR ==="
  uv run scripts/04_train_deepar.py --seed "$SEED"

  echo "=== seed=$SEED — TFT ==="
  uv run scripts/05_train_tft.py --seed "$SEED"
done

echo "All runs complete. Predictions in results/predictions/"
ls -la results/predictions/

#!/usr/bin/env bash
# Run GARCH + DeepAR + TFT across the three seeds.
# Expects scripts 01 + 02 already executed.

set -euo pipefail

SEEDS=(42 1337 2024)
START_TIME=$(date +%s)
START_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")

echo "=== TFM multi-seed run started at $START_HUMAN ==="

for SEED in "${SEEDS[@]}"; do
  echo "=== seed=$SEED — GARCH ==="
  uv run scripts/03_fit_garch.py --seed "$SEED"

  echo "=== seed=$SEED — DeepAR ==="
  uv run scripts/04_train_deepar.py --seed "$SEED"

  echo "=== seed=$SEED — TFT ==="
  uv run scripts/05_train_tft.py --seed "$SEED"
done

END_TIME=$(date +%s)
END_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")
ELAPSED_SEC=$((END_TIME - START_TIME))
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
N_PRED=$(ls -1 results/predictions/*.parquet 2>/dev/null | wc -l | tr -d ' ')

# Flag file with a timestamp — easy to grep for "did it finish?"
FLAG_FILE="results/DONE_$(date +%Y%m%d_%H%M%S).flag"
mkdir -p results
{
  echo "Started:  $START_HUMAN"
  echo "Finished: $END_HUMAN"
  echo "Elapsed:  ${ELAPSED_H}h ${ELAPSED_M}m"
  echo "Prediction files: $N_PRED"
} > "$FLAG_FILE"

echo ""
echo "==============================================================="
echo "ALL RUNS COMPLETE — $END_HUMAN"
echo "Elapsed:          ${ELAPSED_H}h ${ELAPSED_M}m"
echo "Prediction files: $N_PRED (expected 18 = 3 models × 3 seeds × 2 partitions)"
echo "Flag file:        $FLAG_FILE"
echo "==============================================================="
ls -la results/predictions/

# macOS desktop notification + spoken alert. Harmless on non-macOS (the
# commands silently fail because `osascript` / `say` are missing).
if command -v osascript >/dev/null 2>&1; then
  osascript -e "display notification \"Multi-seed run done: $N_PRED files in ${ELAPSED_H}h ${ELAPSED_M}m\" with title \"tfm_code\" sound name \"Glass\""
fi
if command -v say >/dev/null 2>&1; then
  say "T F M models finished" &
fi

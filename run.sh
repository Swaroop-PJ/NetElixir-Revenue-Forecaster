#!/usr/bin/env bash
set -euo pipefail

# Accept arguments, fall back to defaults for local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "=================================================="
echo "🤖 ROBOT RUNTIME STARTED: Executing Track A Pipeline"
echo "=================================================="

# 1. Generate the features the model expects from the data
python src/generate_features.py \
  --data-dir "$DATA_DIR" \
  --out ./src/processed_features.parquet

# 2. Load the pickled model and produce predictions
python src/predict.py \
  --features ./src/processed_features.parquet \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
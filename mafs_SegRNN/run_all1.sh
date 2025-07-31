#!/bin/bash
set -x  
export CUDA_VISIBLE_DEVICES=7

SCRIPT_DIR="./scripts/SegRNN_agent"

scripts=(
  "AQShunyi.sh"
  "AQWan.sh"
  "ETTh1.sh"
  "ETTh2.sh"
  "ETTm1.sh"
  "ETTm2.sh"
)

  # "AQShunyi.sh"
  # "AQWan.sh"
for script in "${scripts[@]}"; do
  echo "Running $script..."
  (bash "$SCRIPT_DIR/$script")
  echo "$script finished."
  echo "------------------------"
done
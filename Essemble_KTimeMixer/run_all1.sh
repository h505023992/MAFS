#!/bin/bash
set -x  
export CUDA_VISIBLE_DEVICES=2
export model_name=Ensemble_TMixer

SCRIPT_DIR="./scripts/Ensemble_TMixer"

scripts=(
  "AQShunyi.sh"
  "AQWan.sh"
  "ETTh1.sh"
  "ETTh2.sh"
  "ETTm1.sh"
  "ETTm2.sh"
)

for script in "${scripts[@]}"; do
  echo "Running $script..."
  bash "$SCRIPT_DIR/$script"
  echo "$script finished."
  echo "------------------------"
done

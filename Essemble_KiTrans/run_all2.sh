#!/bin/bash
set -x  
export CUDA_VISIBLE_DEVICES=0
export model_name=Ensemble_iTransformer

SCRIPT_DIR="./scripts/Ensemble_iTransformer"

scripts=(
  'pm2_5.sh'
  'temp.sh'
  "ZafNoo.sh"
  "CzeLan.sh"
  "weather.sh"
)

for script in "${scripts[@]}"; do
  echo "Running $script..."
  bash "$SCRIPT_DIR/$script"
  echo "$script finished."
  echo "------------------------"
done

#!/bin/bash
set -x  
export CUDA_VISIBLE_DEVICES=6
export model_name=Ensemble_TMixer
SCRIPT_DIR="./scripts/Ensemble_TMixer"

scripts=(
  'pm2_5.sh'
)

for script in "${scripts[@]}"; do
  echo "Running $script..."
  bash "$SCRIPT_DIR/$script"
  echo "$script finished."
  echo "------------------------"
done

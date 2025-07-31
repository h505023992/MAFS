#!/bin/bash
set -x  
export CUDA_VISIBLE_DEVICES=7
export model_name=Ensemble_TMixer 
SCRIPT_DIR="./scripts/Ensemble_TMixer"

scripts=(
  'temp.sh'
  'CzeLan'
)

for script in "${scripts[@]}"; do
  echo "Running $script..."
  bash "$SCRIPT_DIR/$script"
  echo "$script finished."
  echo "------------------------"
done

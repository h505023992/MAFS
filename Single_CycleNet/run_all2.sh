#!/bin/bash
set -x  
export CUDA_VISIBLE_DEVICES=1
export model_name=CycleNet
SCRIPT_DIR="./scripts/CycleNet"
# scripts=(
#   "AQShunyi.sh"
# )
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

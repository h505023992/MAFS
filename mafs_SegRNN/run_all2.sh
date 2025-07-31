#!/bin/bash
set -x  
export CUDA_VISIBLE_DEVICES=3

SCRIPT_DIR="./scripts/SegRNN_agent"

scripts=(
  'pm2_5.sh'
  'temp.sh'
  "ZafNoo.sh"
  "CzeLan.sh"
  "weather.sh"
)

  # "AQShunyi.sh"
  # "AQWan.sh"
for script in "${scripts[@]}"; do
  echo "Running $script..."
  (bash "$SCRIPT_DIR/$script")
  echo "$script finished."
  echo "------------------------"
done
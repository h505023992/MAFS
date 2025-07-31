#!/bin/bash
set -x  
export CUDA_VISIBLE_DEVICES=2


SCRIPT_DIR="./scripts/PatchTST_agent"
  # "AQShunyi.sh"
  #  "AQWan.sh"
  #   "ETTh1.sh"
  # "ETTh2.sh"
  # "ETTm1.sh"
  # "ETTm2.sh"
  # 'pm2_5.sh'
scripts=(
  'temp.sh'
  "ZafNoo.sh"
  "CzeLan.sh"
  "weather.sh"
)


for script in "${scripts[@]}"; do
  echo "Running $script..."
  (bash "$SCRIPT_DIR/$script")
  echo "$script finished."
  echo "------------------------"
done


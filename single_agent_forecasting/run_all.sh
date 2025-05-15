#!/bin/bash
set -x  


SCRIPT_DIR="./scripts/iTransformer_single"

scripts=(
  "AQShunyi.sh"
  "AQWan.sh"
  "Electricity.sh"
  "ETTh1.sh"
  "ETTh2.sh"
  "ETTm1.sh"
  "ETTm2.sh"
  "Traffic.sh"
  'pm2_5.sh'
  'wind.sh'
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

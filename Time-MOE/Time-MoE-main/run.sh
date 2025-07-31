#!/bin/bash
export CUDA_VISIBLE_DEVICES=2


max_length=1024
num_train_epochs=1
#!/bin/bash

for data in   pm2_5 temp weather AQShunyi AQWan  ETTh2   CzeLan   ETTm1 ETTm2        ZafNoo; do
model_path=${data}/timemoe_run

python main.py \
  --data_path ./jsonl_outputs/${data}.jsonl \
  --model_path Maple728/TimeMoE-50M \
  --output_path ${model_path} \
  --max_length $max_length \
  --stride 1 \
  --learning_rate 1e-4 \
  --min_learning_rate 5e-5 \
  --num_train_epochs $num_train_epochs \
  --precision bf16 \
  --attn_implementation flash_attention_2 \
  --global_batch_size 256 \
  --micro_batch_size 64 \
  --normalization_method zero \
  --evaluation_strategy no \
  --save_strategy epoch \
  --save_steps 100 \
  --logging_steps 10 \
  --save_total_limit 3 \
  --save_only_model \
  --dataloader_num_workers 4






for PRED_LEN in 96 192 336 720; do

  DATA_PATH="./dataset/${data}.csv"              
                               
  BATCH_SIZE=256                    
  NUM_GPUS=1                         

  # ===== 自动设置 context_length =====
  if [[ $PRED_LEN -eq 96 ]]; then
    CONTEXT_LEN=96
  elif [[ $PRED_LEN -eq 192 ]]; then
    CONTEXT_LEN=192
  elif [[ $PRED_LEN -eq 336 ]]; then
    CONTEXT_LEN=336
  elif [[ $PRED_LEN -eq 720 ]]; then
    CONTEXT_LEN=720
  else
    CONTEXT_LEN=96
  fi

  # ===== 单卡评估 =====
  python run_eval.py \
  --model "$model_path" \
  --data "$DATA_PATH" \
  --prediction_length $PRED_LEN \
  --context_length $CONTEXT_LEN \
  --batch_size $BATCH_SIZE >./eval_log/${data}_2_${PRED_LEN}.log
done
done
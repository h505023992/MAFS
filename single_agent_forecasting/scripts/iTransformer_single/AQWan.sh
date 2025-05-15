if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=iTransformer

root_path_name=./dataset/
data_path_name=AQWan.csv
model_id_name=AQWan
data_name=custom

export CUDA_VISIBLE_DEVICES=7


for pred_len in 96 192 336 720
do
seq_len=$pred_len
  for lr in  1e-3 #5e-3  5e-4 1e-4
  do
    for agent_num in 1
    do
    for mode in  'single'
    do
    agent_root_name="./Single_agent/Forecasting_Agent_iTrans_${mode}"
    log_dir="${agent_root_name}/${model_id_name}/logs"
    mkdir -p $log_dir
      log_file="${log_dir}/${model_name}_A${agent_num}_LR${lr}_${seq_len}_${pred_len}.log"
      echo "Running pred_len=$pred_len, agent_num=$agent_num, lr=$lr"
      python -u run_single.py \
        --task_name long_term_forecast \
        --mode $mode \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${pred_len}_A${agent_num}_LR${lr} \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --patch_len 24 \
        --enc_in 11 \
        --agent_num $agent_num \
        --train_epochs 10 \
        --patience 5 \
        --itr 1 \
        --d_model 128 \
        --d_ff 128 \
        --factor 3 \
        --e_layers 2 \
        --d_layers 1 \
        --save_agent_root $agent_root_name\
        --batch_size 128 \
        --learning_rate $lr \
        > "$log_file" 2>&1
    done
  done
done
done
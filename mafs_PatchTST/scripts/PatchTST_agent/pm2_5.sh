if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=Agent_PatchTST_Cooperation
root_path_name=./dataset/
data_path_name=pm2_5.csv
model_id_name=pm2_5
data_name=custom
# seq_len=720
# trap 'kill 0' EXIT
# export CUDA_VISIBLE_DEVICES=0


for pred_len in 96 192 336 720
do
seq_len=$pred_len
  for lr in  1e-3 
  do
    for agent_num in  4
    do
    for mode in   'star'  
    do
    agent_root_name="./Homo_Agent_Log/Forecasting_${model_name}_${mode}"
    log_dir="${agent_root_name}/${model_id_name}/logs"
    mkdir -p $log_dir
      log_file="${log_dir}/${model_name}_A${agent_num}_LR${lr}_${seq_len}_${pred_len}.log"
      echo "Running pred_len=$pred_len, agent_num=$agent_num, lr=$lr"
      python -u run.py \
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
        --enc_in 184 \
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
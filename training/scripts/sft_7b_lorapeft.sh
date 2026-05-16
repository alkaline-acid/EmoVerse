#!/bin/bash

NPROC_PER_NODE=2
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

deepspeed=./scripts/zero3.json

llm=/home/xxx/LoRA_Qwen/models/Qwen2.5-VL-3B-Instruct

lr=1e-6
batch_size=4
grad_accum_steps=4

entry_file=qwenvl/train/train_qwen_lorapeft.py

datasets=emo
datasets=emo_200704

run_name="qwen2.5vl-baseline"
output_dir=./output_lorapeft_1028

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 200704 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb "

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

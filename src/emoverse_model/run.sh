#!/bin/bash

set -e

echo "================================================================="
echo "Starting Qwen2.5-VL LoRA Multi-GPU Training Script..."
echo "================================================================="

export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "INFO: Using GPUs: $CUDA_VISIBLE_DEVICES"


accelerate launch LoRA.py \
    --data_dir /home/xxx/dataset/EmoPro/json \
    --output_dir qwen2.5vl-emo-1024-lora-multigpu \
    --batch_size 1 \
    --epochs 2 \
    --lr 1e-4 \
    --use_lm_aux \
    --intensity_scale 0_1 \
    --lambda_vec 1.0 \
    --lambda_int 0.5 \
    --lambda_emo 0.5 \
    --lambda_lm 0.1

echo "================================================================="
echo "Training script finished."
echo "================================================================="

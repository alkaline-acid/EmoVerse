#!/bin/bash


GPUS_PER_NODE=8

while [[ $
    case $1 in
        --gpus_per_node)
            GPUS_PER_NODE=$2
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: ./classify_accelerate.sh --gpus_per_node <GPU数量>"
            exit 1
            ;;
    esac
done



training_params=""
training_params+=" --model_name_or_path "/mnt/bn/icvg-ec-dexiang-lf-hw/dexiang/Qwen3-VL/qwen-vl-finetune/model_output/3b_2_fuse_only_emo/checkpoint-6000""
training_params+=" --train_data "/mnt/bn/icvg-ec-dexiang-lf-hw/dexiang/data/emotion_data/train1027_2_bboxwith_attribute.jsonl""
training_params+=" --eval_data "/mnt/bn/icvg-ec-dexiang-lf-hw/dexiang/data/emotion_data/test1022_1with_attribute.jsonl""
training_params+=" --output_dir "./model_output/classify_attention_layers_with_train_with_attribute""
training_params+=" --per_device_train_batch_size 4"
training_params+=" --per_device_eval_batch_size 8"
training_params+=" --gradient_accumulation_steps 1"
training_params+=" --learning_rate 1e-5"
training_params+=" --num_train_epochs 3"
training_params+=" --weight_decay 0.01"
training_params+=" --warmup_ratio 0.05"
training_params+=" --logging_steps 10"
training_params+=" --save_steps 1000"
training_params+=" --save_total_limit 10"
training_params+=" --eval_strategy steps"
training_params+=" --eval_steps 1000"
training_params+=" --bf16 true"
training_params+=" --gradient_checkpointing false"
training_params+=" --dataloader_num_workers 4"
training_params+=" --unfreeze_last_layers true "
training_params+=" --feature_fusion attention"



cat << EOF
🚀 单机多卡训练配置信息 (使用accelerate) 🚀
每节点GPU数: $GPUS_PER_NODE
训练参数: $training_params
EOF

sleep 5

exec accelerate launch \
    --multi_gpu \
    --num_processes=${GPUS_PER_NODE} \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    qwenvl/train/train_qwen_classify.py \
    ${training_params}

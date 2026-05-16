
process_path='/home/xxx/LoRA_Qwen/models/Qwen2.5-VL-3B-Instruct'
model_path='/home/xxx/LoRA_Qwen/clone/Qwen3-VL/qwen-vl-finetune/output_1027'
data_path='/home/xxx/dataset/EmoPro/prompt/split_for_qwen/train1022_1.jsonl'
resume=False
SAVE_DIR='./features_1102_woimage_2_test_ttt'

python inference_for_second.py    --process_path=${process_path} \
                                --model_path=${model_path} \
                                --data_path=${data_path} \
                                --resume=${resume} \
                                --SAVE_DIR=${SAVE_DIR}

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
import requests
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
import json
import torch, contextlib, gc, traceback, sys
import math
import numpy as np

process_path = '/home/xxx/LoRA_Qwen/models/Qwen2.5-VL-3B-Instruct'
path = '/home/xxx/LoRA_Qwen/clone/Qwen3-VL/qwen-vl-finetune/output_1027'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path)
processor = AutoProcessor.from_pretrained(process_path)
model = model.eval()
model = model.to('cuda')
data = []
with open('/home/xxx/dataset/EmoPro/prompt/split_for_qwen/train1022_1.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"跳过无法解析的行: {e}")

with open('output_3B_alltrained_1028_2.jsonl', "w", encoding="utf-8") as f:
    for ind, i in enumerate(data):
        image = Image.open(i['image'])
        width, height = image.size
        if width * height > 3210 * 2120:
            continue
            model = model.to('cpu')
        else:
            model = model.to('cuda')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": i['conversations'][0]['value']},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            generate_ids = model.generate(**inputs.to(model.device), max_new_tokens=512)


            full_input_ids = torch.cat([inputs["input_ids"], generate_ids[:, inputs["input_ids"].shape[1]:]], dim=-1)


            with torch.no_grad():
                outputs = model(input_ids=full_input_ids.to(model.device),
                                pixel_values=inputs["pixel_values"].to(model.device),
                                image_grid_thw=inputs["image_grid_thw"].to(model.device),
                                output_hidden_states=True,
                                return_dict=True)


            last_hidden = outputs.hidden_states[-1]


            last_hidden = last_hidden[:, inputs["input_ids"].shape[1]:, :]

            save_dir = "./hidden_last"
            os.makedirs(save_dir, exist_ok=True)
            image_path = i['image']
            feat_path = os.path.join(save_dir, f"feat_{ind:05d}.pt")
            torch.save(last_hidden, feat_path)

            mid_hidden = outputs.hidden_states[18]


            mid_hidden = mid_hidden[:, inputs["input_ids"].shape[1]:, :]

            save_dir = "./hidden_mid"
            os.makedirs(save_dir, exist_ok=True)
            image_path = i['image']
            mid_feat_path = os.path.join(save_dir, f"feat_{ind:05d}.pt")
            torch.save(mid_hidden, mid_feat_path)

            shallow_hidden = outputs.hidden_states[1]


            shallow_hidden = shallow_hidden[:, inputs["input_ids"].shape[1]:, :]

            save_dir = "./hidden_shallow"
            os.makedirs(save_dir, exist_ok=True)
            image_path = i['image']
            shallow_feat_path = os.path.join(save_dir, f"feat_{ind:05d}.pt")
            torch.save(shallow_hidden, shallow_feat_path)


            record = {
                "index": ind,
                "image_path": image_path,
                "feature_path": feat_path,
                "mid_feature_path": mid_feat_path,
                "shallow_feature_path": shallow_feat_path,
                "seq_len": last_hidden.shape[1],
                "hidden_dim": last_hidden.shape[2],
            }


            with open(os.path.join(save_dir, "index.jsonl"), "a", encoding="utf-8") as f2:
                f2.write(json.dumps(record, ensure_ascii=False) + "\n")
        prompt_len = inputs["input_ids"].shape[1]
        generate_ids = generate_ids[:, prompt_len:]
        out1 = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        out1 = {"index": ind, "result": f"{out1}", "answer": f"{i['conversations'][1]['value']}"}

        json_line = json.dumps(out1, ensure_ascii=False)
        f.write(json_line + "\n")

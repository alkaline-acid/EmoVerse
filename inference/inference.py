import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
import requests
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
import json
import torch, contextlib, gc, traceback, sys
import math

def mem():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**2
        r = torch.cuda.memory_reserved() / 1024**2
        return f"alloc={a:.0f}MB, reserved={r:.0f}MB"
    return "cpu"

@contextlib.contextmanager
def step(name):
    try:
        print(f"\n==> {name} (before): {mem()}")
        yield
        print(f"==> {name} (after):  {mem()}")
    except RuntimeError as e:
        print(f"[OOM @ {name}] {e}")
        print(traceback.format_exc())
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



process_path = '/home/xxx/LoRA_Qwen/models/Qwen2.5-VL-3B-Instruct'
path = '/home/xxx/LoRA_Qwen/clone/Qwen3-VL/qwen-vl-finetune/output_200704_1029'
path = '/home/xxx/LoRA_Qwen/clone/Qwen3-VL/qwen-vl-finetune/output_200704_1102/checkpoint-1000'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path)
processor = AutoProcessor.from_pretrained(process_path)
model = model.eval()
model = model.to('cuda')
data = []
with open('/home/xxx/dataset/EmoPro/prompt/split_for_qwen/test1022_1.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"跳过无法解析的行: {e}")














with open('output_3B_1102.jsonl', "w", encoding="utf-8") as f:
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


        prompt_len = inputs["input_ids"].shape[1]
        generate_ids = generate_ids[:, prompt_len:]
        out1 = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        out1 = {"index": ind, "result": f"{out1}", "answer": f"{i['conversations'][1]['value']}"}
        json_line = json.dumps(out1, ensure_ascii=False)
        f.write(json_line + "\n")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import json
import torch
from pathlib import Path

SAVE_DIR = "./features"
os.makedirs(SAVE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(SAVE_DIR, "index.jsonl")
emo_class = ["Amusement", "Anger", "Awe", "Contentment", "Disgust", "Excitement", "Fear", "Sadness"]

SHARD_SIZE = 1024
shard_buf = {
    "last": [],
    "mid":  [],
    "shallow": [],
    "meta": []
}
shard_id = 0
D = 2048

def flush_shard():
    global shard_id, shard_buf
    if len(shard_buf["meta"]) == 0:
        return
    shard_name = f"shard_{shard_id:05d}.pt"
    shard_path = os.path.join(SAVE_DIR, shard_name)


    lens = [x.size(0) for x in shard_buf["last"]]
    L_max = max(lens)

    def pad_stack(tensors):
        if tensors[0].dim() == 1:
            return torch.stack([t for t in tensors], dim=0)

        D_ = tensors[0].size(-1)
        N  = len(tensors)
        out = torch.zeros(N, L_max, D_, dtype=torch.float16)
        mask = torch.zeros(N, L_max, dtype=torch.bool)
        for i, t in enumerate(tensors):
            L = t.size(0)
            out[i, :L] = t
            mask[i, :L] = True
        return out, mask


    last_list = [t.to(torch.float16).cpu() for t in shard_buf["last"]]
    mid_list  = [t.to(torch.float16).cpu()  for t in shard_buf["mid"]]
    shl_list  = [t.to(torch.float16).cpu()  for t in shard_buf["shallow"]]

    last_padded, last_mask = pad_stack(last_list)

    mid_packed   = pad_stack(mid_list)
    shl_packed   = pad_stack(shl_list)

    pack = {
        "last": last_padded,
        "last_mask": last_mask,
        "mid":  mid_packed[0] if isinstance(mid_packed, tuple) else mid_packed,
        "mid_mask": mid_packed[1] if isinstance(mid_packed, tuple) else None,
        "shallow": shl_packed[0] if isinstance(shl_packed, tuple) else shl_packed,
        "shallow_mask": shl_packed[1] if isinstance(shl_packed, tuple) else None,
        "meta": shard_buf["meta"],
    }
    torch.save(pack, shard_path)


    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        for row, m in enumerate(shard_buf["meta"]):
            f.write(json.dumps({
                "ind": m["ind"],
                "image_path": m["image_path"],
                "label": m["label"],
                "gen_len": m["gen_len"],
                "shard": shard_name,
                "row": row
            }, ensure_ascii=False) + "\n")


    shard_buf = {"last": [], "mid": [], "shallow": [], "meta": []}
    shard_id += 1






def add_sample_to_shard(ind, image_path, label, last_hidden, mid_hidden, shallow_hidden):

    last_ = last_hidden.squeeze(0).contiguous().cpu()
    mid_  = mid_hidden.squeeze(0).contiguous().cpu()
    shl_  = shallow_hidden.squeeze(0).contiguous().cpu()





    shard_buf["last"].append(last_)
    shard_buf["mid"].append(mid_)
    shard_buf["shallow"].append(shl_)
    shard_buf["meta"].append({
        "ind": int(ind),
        "image_path": image_path,
        "label": label,
        "gen_len": int(last_.size(0))
    })
    if len(shard_buf["meta"]) >= SHARD_SIZE:
        flush_shard()

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

with open('output_3B_alltrained_1029.jsonl', "w", encoding="utf-8") as f:
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
            image_path = i['image']

            mid_hidden = outputs.hidden_states[18]


            mid_hidden = mid_hidden[:, inputs["input_ids"].shape[1]:, :]

            shallow_hidden = outputs.hidden_states[1]


            shallow_hidden = shallow_hidden[:, inputs["input_ids"].shape[1]:, :]

            label = ''
            for j in emo_class:
                if j in image_path:
                    label = j

            add_sample_to_shard(ind, image_path, label, last_hidden, mid_hidden, shallow_hidden)


            record = {
                "index": ind,
                "image_path": image_path,
                "seq_len": last_hidden.shape[1],
                "hidden_dim": last_hidden.shape[2],
            }

        prompt_len = inputs["input_ids"].shape[1]
        generate_ids = generate_ids[:, prompt_len:]
        out1 = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        out1 = {"index": ind, "result": f"{out1}", "answer": f"{i['conversations'][1]['value']}"}

        json_line = json.dumps(out1, ensure_ascii=False)
        f.write(json_line + "\n")
        if(ind == 10): break
    flush_shard()

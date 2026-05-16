import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import json
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import gc

parser = argparse.ArgumentParser(description="Inference for push")


parser.add_argument('--process_path', type=str, required=True, help="Path to the process directory")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--data_path', type=str, required=True, help="Path to the data")
parser.add_argument('--resume', type=str, required=True, help="resume or not")
parser.add_argument('--SAVE_DIR', type=str, required=True, help="Path to the data output")


args = parser.parse_args()
process_path = args.process_path
model_path = args.model_path
data_path = args.data_path
resume = True if args.resume == 'True' else False

SAVE_DIR = args.SAVE_DIR
os.makedirs(SAVE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(SAVE_DIR, "index.jsonl")
emo_class = ["Amusement", "Anger", "Awe", "Contentment", "Disgust", "Excitement", "Fear", "Sadness"]

SHARD_SIZE = 1024
shard_buf = {
    "hidden": [],
    "meta": []
}
shard_id = 0
D = 2048
def clear_cuda_memory():
    """清理 CUDA 显存空间"""


    gc.collect()


    torch.cuda.empty_cache()





def flush_shard():
    global shard_id, shard_buf
    if len(shard_buf["meta"]) == 0:
        return
    shard_name = f"shard_{shard_id:05d}.pt"
    shard_path = os.path.join(SAVE_DIR, shard_name)


    lens = [x.size(0) for x in shard_buf["hidden"]]
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


    hidden_list = [t.to(torch.float16).cpu() for t in shard_buf["hidden"]]



    pack = {
        "hidden": hidden_list,

        "meta": shard_buf["meta"],
    }
    torch.save(pack, shard_path)


    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        for row, m in enumerate(shard_buf["meta"]):
            f.write(json.dumps({
                "ind": m["ind"],
                "image_path": m["image_path"],
                "label": m["label"],

                "shard": shard_name,
                "row": row
            }, ensure_ascii=False) + "\n")


    shard_buf = {
        "hidden": [],
        "meta": []
    }
    shard_id += 1







def add_sample_to_shard(ind, image_path, label, hidden_states):

    hidden_ = hidden_states.squeeze(0).squeeze(0).contiguous().cpu()





    shard_buf["hidden"].append(hidden_)
    shard_buf["meta"].append({
        "ind": int(ind),
        "image_path": image_path,
        "label": label,
    })
    if len(shard_buf["meta"]) >= SHARD_SIZE:
        flush_shard()


def main():

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(process_path)
    model = model.eval()



    data = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"跳过无法解析的行: {e}")

    if resume:
        with open(INDEX_PATH, 'r') as f:
            resume_line = ''
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    resume_line = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"跳过无法解析的行: {e}")

        resume_ind = resume_line['ind']
        shard_id = int((resume_ind + 1) / SHARD_SIZE)
        print('resume', resume_ind, shard_id)







    for ind, i in enumerate(tqdm(data, desc="Processing data", unit="item")):
        if ind == 5: break
        if resume:
            if ind <= resume_ind:
                continue

        image = Image.open(i['image'])





        messages = [
            {
                "role": "user",
                "content": [

                    {"type": "text", "text": i['conversations'][0]['value']},
                ],
            },
            {
                "role": "assistant",
                "content": [

                    {"type": "text", "text": i['conversations'][1]['value']},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text[:-1-9-12]
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[1] > 10000:
            continue



        pattern = torch.tensor([73353, 1210, 364])

        matches = (inputs["input_ids"][0].unfold(0, len(pattern), 1) == pattern).all(dim=1).nonzero(as_tuple=True)[0]



        indices = matches.item() + 3





























        with torch.inference_mode():
            output = model(input_ids=inputs["input_ids"].to(model.device),


                            output_hidden_states=True,
                            return_dict=True,
                            logits_to_keep=1
                            )





























        hidden_states = output['hidden_states'][-1]

        hidden_states = hidden_states[:, indices, :]




        image_path = i['image']


        label = ''
        for j in emo_class:
            if j in image_path:
                label = j
        add_sample_to_shard(ind, image_path, label, hidden_states)


    flush_shard()

if __name__ == "__main__":
    main()

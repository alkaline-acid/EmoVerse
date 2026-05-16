


import os
import json
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torch.distributed as dist

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer, TrainingArguments, PreTrainedModel,
)

from peft import LoraConfig, get_peft_model





MODEL_NAME = "/home/xxx/LoRA_Qwen/models/Qwen2.5-VL-7B-Instruct"
SPECIAL_TOKEN = "<EMO_VEC>"
EMO_DIM = 1024
EMO_LABELS = ["Amusement", "Anger", "Awe", "Contentment", "Disgust", "Excitement", "Fear", "Sadness"]


EMO_BASE_DIR = Path("/home/xxx/dataset/EmoPro/EmoPro")
EMO_FOLDERS = ["Amusement", "Anger", "Awe", "Contentment", "Disgust", "Excitement", "Fear", "Sadness"]





def sanitize_and_normalize_bboxes(bboxes, img_w, img_h, min_size=1.0):
    if not bboxes:
        return []
    out = []
    for b in bboxes:
        try:
            if isinstance(b, dict):
                x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            else:
                x1, y1, x2, y2 = map(float, b[:4])
        except Exception:
            continue
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        x1 = max(0.0, min(x1, img_w - 1))
        x2 = max(0.0, min(x2, img_w - 1))
        y1 = max(0.0, min(y1, img_h - 1))
        y2 = max(0.0, min(y2, img_h - 1))
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        out.append([x1 / max(img_w, 1), y1 / max(img_h, 1), x2 / max(img_w, 1), y2 / max(img_h, 1)])
    return out





def try_recover_image_path(item: Dict[str, Any]) -> Optional[str]:
    p = item.get("image_path")
    if isinstance(p, str) and p:
        return p
    id_base = (item.get("id") or "").strip()
    emotion = (item.get("emotion") or "").strip()
    if not id_base:
        return None
    if emotion and (EMO_BASE_DIR / emotion / f"{id_base}.jpg").exists():
        return str(EMO_BASE_DIR / emotion / f"{id_base}.jpg")
    for sub in EMO_FOLDERS:
        cand = EMO_BASE_DIR / sub / f"{id_base}.jpg"
        if cand.exists():
            return str(cand)
    return None





class EmoJsonDirDataset(Dataset):
    def __init__(self, data_dir: str, intensity_scale: str = "raw"):
        self.dir = Path(data_dir)
        if not self.dir.exists() or not self.dir.is_dir():
            raise FileNotFoundError(f"data_dir 不存在或不是目录: {data_dir}")
        files = sorted(glob.glob(str(self.dir / "*.json")))
        if not files:
            raise FileNotFoundError(f"{data_dir} 下未找到 *.json")
        self.items = []
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                self.items.append(json.load(f))
        self.intensity_scale = intensity_scale

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ex = self.items[i]
        img_path = try_recover_image_path(ex)
        if img_path is None:
            raise KeyError(f"[Dataset] 样本#{i} 缺 image_path 且无法恢复。keys={list(ex.keys())}")


        user_text = (
            f"请分析这张图像及其背景信息，并在 {SPECIAL_TOKEN} 处聚合出高维情感向量。\n"
            f"图像分析：{ex.get('analysis','')}\n"
            f"背景：{ex.get('background','')}\n"
            f"提示：情感形容词={ex.get('adjective','')}, 名词={ex.get('noun','')}, 类别={ex.get('emotion','')}, 强度={ex.get('intensity',0.0)}\n"
            f"{SPECIAL_TOKEN}"
        )


        lm_target = {
            "background": ex.get("background", ""),
            "adjective": ex.get("adjective", ""),
            "noun": ex.get("noun", ""),
            "emotion": ex.get("emotion", ""),
            "intensity": float(ex.get("intensity", 0.0) or 0.0),
            "bbox": ex.get("bbox", None),
        }
        lm_target_str = json.dumps(lm_target, ensure_ascii=False)

        intensity = float(ex.get("intensity", 0.0) or 0.0)
        if self.intensity_scale == "0_1":
            intensity = intensity / 5.0

        emo = ex.get("emotion", "")
        emo_idx = EMO_LABELS.index(emo) if emo in EMO_LABELS else -1

        return {
            "id": ex.get("id"),
            "emotion": emo,
            "image_path": img_path,
            "user_text": user_text,
            "lm_target_str": lm_target_str,
            "emotion_idx": emo_idx,
            "intensity": intensity,
            "bboxes": ex.get("bbox", []) or [],
        }





class EmoHead(nn.Module):
    def __init__(self, hidden_size: int, emo_dim: int, num_emotions: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, emo_dim),
        )
        self.intensity_head = nn.Sequential(
            nn.LayerNorm(emo_dim),
            nn.Linear(emo_dim, emo_dim // 2),

            nn.GELU(),
            nn.Linear(emo_dim // 2, 1)
        )
        self.emotion_head = nn.Sequential(
            nn.LayerNorm(emo_dim),
            nn.Linear(emo_dim, emo_dim // 2),
            nn.GELU(),
            nn.Linear(emo_dim // 2, num_emotions)
        )

    def forward(self, h_last: torch.Tensor):
        z = self.proj(h_last)
        intensity = self.intensity_head(z).squeeze(-1)
        emotion_logits = self.emotion_head(z)
        return z, intensity, emotion_logits





class QwenEmoModel(nn.Module):
    def __init__(self, model_name: str, special_token: str, emo_dim: int, use_lm_aux: bool, processor: AutoProcessor):
        super().__init__()
        self.processor = processor
        self.model: PreTrainedModel = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.config.output_hidden_states = True

        self.tokenizer = processor.tokenizer


        if special_token not in self.tokenizer.get_vocab():
            raise ValueError(
                f"特殊 token '{special_token}' 未在 tokenizer 中找到。"
                "请确保在模型初始化前已将其添加并保存。"
            )
        self.special_id = self.tokenizer.convert_tokens_to_ids(special_token)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "hidden_size_qwen2", None) or self.model.config.hidden_size
        self.emo_head = EmoHead(hidden_size, emo_dim, num_emotions=len(EMO_LABELS))

        self.use_lm_aux = use_lm_aux
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss()

    def forward(
        self,
        labels_lm=None,
        emotion_idx=None,
        intensity=None,
        bboxes: Optional[List[List[List[float]]]] = None,
        lambda_vec: float = 1.0,
        lambda_int: float = 0.5,
        lambda_emo: float = 0.5,
        lambda_lm: float = 0.1,
        **model_inputs,
    ):



        outputs = self.model(**model_inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]

        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("input_ids 缺失，无法定位 <EMO_VEC>。")
        emo_mask = (input_ids == self.special_id)
        if not emo_mask.any():
            raise RuntimeError(f"输入未包含 {SPECIAL_TOKEN} 标记。")
        idxs = emo_mask.float().argmax(dim=1)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        emo_hidden = last_hidden[batch_idx, idxs, :]

        z, pred_intensity, emo_logits = self.emo_head(emo_hidden)

        loss = 0.0

        if lambda_vec > 0:
            loss = loss + lambda_vec * (z.pow(2).mean())



        if intensity is not None and lambda_int > 0:
            loss = loss + lambda_int * self.mse(pred_intensity.float(), intensity.float())



        if emotion_idx is not None and (emotion_idx >= 0).any() and lambda_emo > 0:
            valid = (emotion_idx >= 0)
            if valid.any():
                ce_loss = self.ce(emo_logits[valid], emotion_idx[valid])
                loss = loss + lambda_emo * ce_loss



        if self.use_lm_aux and labels_lm is not None and lambda_lm > 0:
            lm_logits = outputs.logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels_lm[:, 1:].contiguous()
            lm_loss = self.ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss + lambda_lm * lm_loss


        return {"loss": loss, "z": z, "intensity": pred_intensity, "emotion_logits": emo_logits}


    @property
    def config(self):
        return self.model.config

    def gradient_checkpointing_enable(self, **kwargs):

        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.model.resize_token_embeddings(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.model, "prepare_inputs_for_generation"):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)
        return super().prepare_inputs_for_generation(*args, **kwargs)


    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)








@dataclass
class EmoCollator:
    processor: AutoProcessor
    use_lm_aux: bool = True

    def __call__(self, batch: List[Dict[str, Any]]):

        images_batch = []
        templated_texts = []


        prompts_only_texts = []

        for ex in batch:

            img_path = ex["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            images_batch.append(img)


            prompt = ex["user_text"]
            tgt_suffix = f"\n请输出以下JSON：{ex['lm_target_str']}" if self.use_lm_aux else ""
            full_text = prompt + tgt_suffix


            full_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_text}]}]



            final_text = self.processor.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            templated_texts.append(final_text)


            if self.use_lm_aux:
                prompt_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                prompt_text = self.processor.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=False
                )
                prompts_only_texts.append(prompt_text)


        self.processor.tokenizer.padding_side = "right"
        inputs = self.processor(
            text=templated_texts,
            images=images_batch,
            return_tensors="pt",
            padding=True
        )


        labels_lm = inputs["input_ids"].clone()
        if self.use_lm_aux:

            prompt_lens = [
                len(t) for t in self.processor.tokenizer(prompts_only_texts, padding=False)["input_ids"]
            ]

            for i in range(len(batch)):
                prompt_len = prompt_lens[i]

                labels_lm[i, :prompt_len] = -100
        else:
            labels_lm[:] = -100

        inputs["labels_lm"] = labels_lm


        inputs["emotion_idx"] = torch.tensor([ex["emotion_idx"] for ex in batch], dtype=torch.long)
        inputs["intensity"] = torch.tensor([ex["intensity"] for ex in batch], dtype=torch.float)

        return inputs





def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/xxx/dataset/EmoPro/json", help="目录，内含每图一个 json")
    parser.add_argument("--output_dir", type=str, default="qwen25vl-emo-lora")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--use_lm_aux", action="store_true", help="启用 LM 辅助 JSON CE")
    parser.add_argument("--intensity_scale", type=str, default="raw", choices=["raw", "0_1"])
    parser.add_argument("--lambda_vec", type=float, default=1.0)
    parser.add_argument("--lambda_int", type=float, default=0.5)
    parser.add_argument("--lambda_emo", type=float, default=0.5)
    parser.add_argument("--lambda_lm", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()





    is_distributed = "RANK" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend='nccl', device_id=local_rank)
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] Distributed setup complete. World size: {world_size}")
    else:
        rank = 0

    torch.manual_seed(args.seed)






    prepared_model_dir = Path(args.output_dir) / "prepared_model"

    if rank == 0:
        print("Rank 0: Preparing processor and model config...")
        prepared_model_dir.mkdir(parents=True, exist_ok=True)

        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
        tok = processor.tokenizer


        if SPECIAL_TOKEN not in tok.get_vocab():
            print(f"Adding special token: {SPECIAL_TOKEN}")
            tok.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN]})

        if tok.pad_token is None:
            print("Setting pad_token to eos_token.")
            tok.pad_token = tok.eos_token


        processor.save_pretrained(prepared_model_dir)
        print(f"Processor saved to {prepared_model_dir}")


    if is_distributed:
        print(f"111")

        print(f"222")
        print(f"[Rank {rank}] Barrier passed. All ranks will now load from prepared directory.")



    processor = AutoProcessor.from_pretrained(prepared_model_dir, trust_remote_code=True, use_fast=False)


    ds = EmoJsonDirDataset(args.data_dir, intensity_scale=args.intensity_scale)
    n = len(ds)
    n_val = max(1, int(0.05 * n))
    train_set, val_set = torch.utils.data.random_split(
        ds, [n - n_val, n_val], generator=torch.Generator().manual_seed(args.seed)
    )




    base = QwenEmoModel(MODEL_NAME, SPECIAL_TOKEN, EMO_DIM, use_lm_aux=args.use_lm_aux, processor=processor)



    base.model.resize_token_embeddings(len(processor.tokenizer))


    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )
    base.model = get_peft_model(base.model, peft_cfg)


    if rank == 0:
        base.model.print_trainable_parameters()

    collator = EmoCollator(processor=processor, use_lm_aux=args.use_lm_aux)

    class EmoTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            labels_lm = inputs.pop("labels_lm", None)
            emotion_idx = inputs.pop("emotion_idx", None)
            intensity = inputs.pop("intensity", None)
            bboxes = inputs.pop("bboxes", None)

            outputs = model(
                labels_lm=labels_lm,
                emotion_idx=emotion_idx,
                intensity=intensity,
                bboxes=bboxes,
                lambda_vec=args.lambda_vec,
                lambda_int=args.lambda_int,
                lambda_emo=args.lambda_emo,
                lambda_lm=args.lambda_lm,
                **inputs
            )
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=20,
        save_steps=1000,
        bf16=True,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = EmoTrainer(
        model=base,
        args=train_args,
        data_collator=collator,
        train_dataset=train_set,
        eval_dataset=val_set,
    )

    trainer.train()


    if rank == 0:
        print("Training finished. Saving model and artifacts...")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        base.model.save_pretrained(args.output_dir)
        torch.save(base.emo_head.state_dict(), os.path.join(args.output_dir, "emo_head.bin"))
        processor.save_pretrained(args.output_dir)
        cfg = {
            "model_name": MODEL_NAME,
            "special_token": SPECIAL_TOKEN,
            "emo_dim": EMO_DIM,
            "labels": EMO_LABELS,
            "intensity_scale": args.intensity_scale,
            "use_lm_aux": args.use_lm_aux,
        }
        with open(os.path.join(args.output_dir, "emo_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("Saved to:", args.output_dir)

    print("Training finished. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()

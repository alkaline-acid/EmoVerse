


import os, json, argparse, glob, sys
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from transformers.trainer_utils import get_last_checkpoint


from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training



class EmoDataset(Dataset):
    """
    懒加载 + 容错设计：
    1. 初始化时只记录文件路径。
    2. 获取数据时实时读取。
    3. 如果遇到坏文件（无法解析、内容为空、缺少关键字段），则返回 None。
    """

    def __init__(self, json_dir: str):
        print(f"Initializing Lazy EmoDataset from: {json_dir}")
        self.files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        if not self.files:
            raise FileNotFoundError(f"FATAL: No json files found under {json_dir}")
        print(f"--> Found {len(self.files)} json file paths.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        file_path = self.files[idx]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)


            if not data or "image_path" not in data or not data["image_path"]:
                print(
                    f"\n[Warning] Skipping invalid data: content is empty or 'image_path' is missing. File: {file_path}")
                return None

            return data
        except Exception as e:

            print(f"\n[Warning] Skipping corrupted file that failed to load. File: {file_path}, Error: {e}")
            return None


class EmotionHead(nn.Module):
    def __init__(self, hidden_size: int, emo_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, emo_dim),
        )
    def forward(self, h):
        z = self.net(h)
        z = F.normalize(z, dim=-1)
        return z

class Classifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)

class Regressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.fc(x)


class EmoModel(nn.Module):
    def __init__(self, base, tokenizer, emo_token: str, n_emotions: int,
                 n_bg: int = 0, n_adj: int = 0, n_noun: int = 0,
                 use_bbox: bool = True, emo_dim: int = 1024):
        super().__init__()
        self.base = base
        self.tokenizer = tokenizer
        self.emo_token = emo_token
        self.emo_id = tokenizer.convert_tokens_to_ids(emo_token)
        hidden_size = base.config.hidden_size

        self.emotion_head = EmotionHead(hidden_size, emo_dim)
        self.cls_emotion  = Classifier(emo_dim, n_emotions)
        self.reg_intensity = Regressor(emo_dim, 1)

        self.n_bg, self.n_adj, self.n_noun = n_bg, n_adj, n_noun
        if n_bg > 0:  self.cls_bg = Classifier(emo_dim, n_bg)
        if n_adj > 0: self.cls_adj = Classifier(emo_dim, n_adj)
        if n_noun > 0:self.cls_noun = Classifier(emo_dim, n_noun)

        self.use_bbox = use_bbox
        if use_bbox:  self.reg_bbox = Regressor(emo_dim, 4)





    def forward(self, **inputs):
        outputs = self.base(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states[-1]
        input_ids = inputs["input_ids"]

        emo_mask = (input_ids == self.emo_id)
        if emo_mask.sum() == 0:
            emo_pos = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        else:
            emo_pos = emo_mask.float().argmax(dim=1)

        idx = torch.arange(input_ids.size(0), device=input_ids.device)
        h_emo = hs[idx, emo_pos, :]

        z = self.emotion_head(h_emo)
        out = {
            "z": z,
            "logits_emotion": self.cls_emotion(z),
            "pred_intensity": self.reg_intensity(z),
        }
        if self.n_bg  > 0: out["logits_bg"]   = self.cls_bg(z)
        if self.n_adj > 0: out["logits_adj"]  = self.cls_adj(z)
        if self.n_noun> 0: out["logits_noun"] = self.cls_noun(z)
        if self.use_bbox:   out["pred_bbox"]  = self.reg_bbox(z)
        return out



class EmoCollator:
    def __init__(self, processor, emo_token: str,
                 emotion2id: Dict[str, int],
                 bg2id: Optional[Dict[str, int]] = None,
                 adj2id: Optional[Dict[str, int]] = None,
                 noun2id: Optional[Dict[str, int]] = None):
        self.processor = processor
        self.emo_token = emo_token
        self.emotion2id = emotion2id
        self.bg2id = bg2id or {}
        self.adj2id = adj2id or {}
        self.noun2id = noun2id or {}

    @staticmethod
    def _norm_intensity(x: float) -> float:
        return float(x) / 5.0

    @staticmethod
    def _norm_bbox_xyxy(b, w, h):
        return [b["x1"] / w, b["y1"] / h, b["x2"] / w, b["y2"] / h]

    def _build_messages(self, ex: Dict[str, Any]) -> List[Dict[str, Any]]:

        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image_path"]},
                {"type": "text", "text": f"{self.emo_token} Analysis: {ex['analysis']}"}
            ]
        }]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:



        if not batch:
            return {}


        images = []
        texts = []
        for ex in batch:

            try:
                img = Image.open(ex["image_path"]).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not open image {ex.get('image_path')}. Replacing with a grey image. Error: {e}")
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))


            messages = self._build_messages(ex)
            text_str = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text_str)


        self.processor.tokenizer.padding_side = "right"
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )


        emo_ids = torch.tensor([self.emotion2id[ex["emotion"]] for ex in batch], dtype=torch.long)
        intens = torch.tensor([self._norm_intensity(ex.get("intensity", 0.0)) for ex in batch],
                              dtype=torch.float32).unsqueeze(-1)

        out = {**inputs, "labels_emotion": emo_ids, "labels_intensity": intens}

        if self.bg2id:
            out["labels_bg"] = torch.tensor([self.bg2id.get(ex.get("background", ""), 0) for ex in batch],
                                            dtype=torch.long)
        if self.adj2id:
            out["labels_adj"] = torch.tensor([self.adj2id.get(ex.get("adjective", ""), 0) for ex in batch],
                                             dtype=torch.long)
        if self.noun2id:
            out["labels_noun"] = torch.tensor([self.noun2id.get(ex.get("noun", ""), 0) for ex in batch],
                                              dtype=torch.long)

        need_bbox = any(ex.get("bbox") for ex in batch)
        if need_bbox:
            norm_boxes = []

            for i, ex in enumerate(batch):
                if ex.get("bbox"):
                    b0 = ex["bbox"][0]
                    w, h = images[i].size
                    norm_boxes.append(self._norm_bbox_xyxy(b0, w, h))
                else:
                    norm_boxes.append([0., 0., 0., 0.])
            out["labels_bbox"] = torch.tensor(norm_boxes, dtype=torch.float32)

        return out


class EmoTrainer(Trainer):
    def __init__(self, *args, w_arc=1.0, w_int=0.5, w_aux=0.2, w_box=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_arc, self.w_int, self.w_aux, self.w_box = w_arc, w_int, w_aux, w_box
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.SmoothL1Loss()

    def compute_loss(self, model: EmoModel, inputs, return_outputs=False, **kwargs):
        labels_emotion = inputs.pop("labels_emotion")
        labels_intensity = inputs.pop("labels_intensity")
        labels_bg  = inputs.pop("labels_bg",  None)
        labels_adj = inputs.pop("labels_adj", None)
        labels_noun= inputs.pop("labels_noun",None)
        labels_bbox= inputs.pop("labels_bbox",None)

        outputs = model(**inputs)
        loss = 0.0

        loss_emotion   = self.ce(outputs["logits_emotion"], labels_emotion)
        loss_intensity = self.l1(outputs["pred_intensity"], labels_intensity)
        loss = loss + self.w_arc*loss_emotion + self.w_int*loss_intensity

        aux_terms = []
        if labels_bg  is not None and "logits_bg"  in outputs: aux_terms.append(self.ce(outputs["logits_bg"],  labels_bg))
        if labels_adj is not None and "logits_adj" in outputs: aux_terms.append(self.ce(outputs["logits_adj"], labels_adj))
        if labels_noun is not None and "logits_noun"in outputs: aux_terms.append(self.ce(outputs["logits_noun"],labels_noun))
        if aux_terms:
            loss_aux = sum(aux_terms)/len(aux_terms)
            loss = loss + self.w_aux*loss_aux
        else:
            loss_aux = torch.tensor(0.0, device=loss_emotion.device)

        if labels_bbox is not None and "pred_bbox" in outputs:
            loss_box = self.l1(outputs["pred_bbox"], labels_bbox)
            loss = loss + self.w_box*loss_box
        else:
            loss_box = torch.tensor(0.0, device=loss_emotion.device)

        self.log({
            "loss_emotion": loss_emotion.detach(),
            "loss_intensity": loss_intensity.detach(),
            "loss_aux": loss_aux.detach(),
            "loss_bbox": loss_box.detach(),
            "z_norm_mean": outputs["z"].norm(dim=-1).mean().detach()
        })
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str,
                    default="/home/xxx/LoRA_Qwen/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train_json_dir", type=str,
                    default="/home/xxx/dataset/EmoPro/json")
    ap.add_argument("--eval_json_dir", type=str,
                    default="/home/xxx/dataset/EmoPro/json")
    ap.add_argument("--output_dir", type=str,
                    default="/home/xxx/LoRA_Qwen/outputs/qwen25vl_emo1024")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--freeze_projector", action="store_true")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--w_arc", type=float, default=1.0)
    ap.add_argument("--w_int", type=float, default=0.5)
    ap.add_argument("--w_aux", type=float, default=0.2)
    ap.add_argument("--w_box", type=float, default=0.0)
    ap.add_argument("--emotion_list", type=str,
                    default="Amusement,Anger,Awe,Contentment,Disgust,Excitement,Fear,Sadness")
    ap.add_argument("--bg_list", type=str, default="")
    ap.add_argument("--adj_list", type=str, default="")
    ap.add_argument("--noun_list", type=str, default="")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--load_best_model_at_end", action="store_true")
    ap.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    ap.add_argument("--greater_is_better", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default="auto",
                    help="'auto' 自动找最后的 ckpt；也可填具体路径 /path/to/checkpoint-1234；"
                         "设为 'none' 表示不续训")

    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    EMO_TOKEN = "[EMO]"
    if EMO_TOKEN not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens":[EMO_TOKEN]})


    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path, dtype=torch.bfloat16
    )
    model.resize_token_embeddings(len(processor.tokenizer))


    if args.freeze_vision:
        vis = getattr(model, "vision_tower", None) or getattr(model, "visual", None)
        if vis is not None:
            for p in vis.parameters(): p.requires_grad = False
    if args.freeze_projector:
        proj = getattr(model, "visual_projector", None) or getattr(model, "multi_modal_projector", None)
        if proj is not None:
            for p in proj.parameters(): p.requires_grad = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)


    emotion_list = [s for s in args.emotion_list.split(",") if s]
    bg_list  = [s for s in args.bg_list.split(",") if s]
    adj_list = [s for s in args.adj_list.split(",") if s]
    noun_list= [s for s in args.noun_list.split(",") if s]
    emotion2id = {k:i for i,k in enumerate(emotion_list)}
    bg2id  = {k:i for i,k in enumerate(bg_list)} if bg_list else {}
    adj2id = {k:i for i,k in enumerate(adj_list)} if adj_list else {}
    noun2id= {k:i for i,k in enumerate(noun_list)} if noun_list else {}

    emo_model = EmoModel(
        base=model, tokenizer=processor.tokenizer, emo_token=EMO_TOKEN,
        n_emotions=len(emotion2id),
        n_bg=len(bg2id), n_adj=len(adj2id), n_noun=len(noun2id),
        use_bbox=True
    )


    train_ds = EmoDataset(args.train_json_dir)
    eval_ds  = EmoDataset(args.eval_json_dir)
    collator = EmoCollator(processor, EMO_TOKEN, emotion2id, bg2id, adj2id, noun2id)


    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,


        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,


        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,

        bf16=True,
        report_to="none",
        save_safetensors=True,

        remove_unused_columns=False,
    )

    trainer = EmoTrainer(
        model=emo_model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        w_arc=args.w_arc, w_int=args.w_int, w_aux=args.w_aux, w_box=args.w_box
    )


    resume_arg = None
    if args.resume_from_checkpoint.lower() == "auto":
        last_ckpt = get_last_checkpoint(args.output_dir)
        if last_ckpt is not None:
            print(f"[Resume] Found last checkpoint: {last_ckpt}")
            resume_arg = last_ckpt
    elif args.resume_from_checkpoint.lower() == "none":
        resume_arg = None
    else:

        resume_arg = args.resume_from_checkpoint
        print(f"[Resume] Using provided checkpoint: {resume_arg}")

    trainer.train(resume_from_checkpoint=resume_arg)


    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training done. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()

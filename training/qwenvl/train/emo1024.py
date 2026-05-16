
import os, json, glob
from typing import Dict, Any, List, Optional
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer


class EmoDataset(Dataset):
    REQUIRED = ["image_path", "analysis", "emotion", "intensity"]

    def __init__(self, json_dir: str, allow_alias=True, max_report=20):
        self.files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        if not self.files:
            raise FileNotFoundError(f"No json files under {json_dir}")
        self.data, self.bad = [], []
        aliases = [
            ("image_path", ["image", "img_path", "path", "imagePath"]),
            ("analysis",   ["caption", "text", "description"]),
        ] if allow_alias else []

        def norm_keys(ex: dict):
            for key, cand in aliases:
                if key not in ex:
                    for a in cand:
                        if a in ex:
                            ex[key] = ex[a]; break
            return ex

        for p in self.files:
            try:
                ex = json.load(open(p, "r", encoding="utf-8"))
                ex = norm_keys(ex)
                miss = [k for k in self.REQUIRED if k not in ex or ex[k] in (None, "")]
                if miss:
                    self.bad.append((p, f"missing {miss}")); continue
                if not os.path.isfile(ex["image_path"]):
                    self.bad.append((p, f"image not found: {ex['image_path']}")); continue
                self.data.append(ex)
            except Exception as e:
                self.bad.append((p, f"parse err: {e}"))
        if self.bad:
            print(f"[EmoDataset] skipped {len(self.bad)} broken items (showing up to {max_report}):")
            for path, msg in self.bad[:max_report]:
                print("  -", path, ":", msg)
        if not self.data:
            raise RuntimeError("All samples invalid. Please fix JSONs.")

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class EmoCollator:
    def __init__(self, processor, emo_token: str,
                 emotion2id: Dict[str,int],
                 bg2id: Optional[Dict[str,int]]=None,
                 adj2id: Optional[Dict[str,int]]=None,
                 noun2id: Optional[Dict[str,int]]=None):
        self.processor = processor
        self.emo_token = emo_token
        self.emotion2id = emotion2id
        self.bg2id  = bg2id  or {}
        self.adj2id = adj2id or {}
        self.noun2id= noun2id or {}

    @staticmethod
    def _norm_intensity(x: float) -> float: return float(x)/5.0

    def _build_messages(self, ex: Dict[str, Any]):
        img = ex.get("image_path") or ex.get("image") or ex.get("img_path") or ex.get("path") or ex.get("imagePath")
        ana = ex.get("analysis")   or ex.get("caption") or ex.get("text")    or ex.get("description") or ""
        return [{
            "role":"user",
            "content":[
                {"type":"image","image":img},
                {"type":"text","text":f"{self.emo_token} Analysis: {ana}"}
            ]
        }]

    def __call__(self, batch: List[Dict[str,Any]]) -> Dict[str, Any]:
        messages = [self._build_messages(ex) for ex in batch]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True,
            return_tensors="pt", return_dict=True
        )
        vis = self.processor(images=[ex["image_path"] for ex in batch], return_tensors="pt")
        inputs["pixel_values"] = vis["pixel_values"]

        out = {
            **inputs,
            "labels_emotion": torch.tensor([self.emotion2id[ex["emotion"]] for ex in batch], dtype=torch.long),
            "labels_intensity": torch.tensor([self._norm_intensity(ex["intensity"]) for ex in batch], dtype=torch.float32).unsqueeze(-1),
        }
        if self.bg2id:
            out["labels_bg"] = torch.tensor([self.bg2id.get(ex.get("background",""),0) for ex in batch], dtype=torch.long)
        if self.adj2id:
            out["labels_adj"]= torch.tensor([self.adj2id.get(ex.get("adjective",""),0) for ex in batch], dtype=torch.long)
        if self.noun2id:
            out["labels_noun"]=torch.tensor([self.noun2id.get(ex.get("noun",""),0) for ex in batch], dtype=torch.long)


        need_bbox = any(ex.get("bbox") for ex in batch)
        if need_bbox:
            boxes=[]
            for ex in batch:
                if ex.get("bbox"):
                    b0=ex["bbox"][0]
                    with Image.open(ex["image_path"]) as im:
                        w,h=im.size
                    boxes.append([b0["x1"]/w, b0["y1"]/h, b0["x2"]/w, b0["y2"]/h])
                else:
                    boxes.append([0.,0.,0.,0.])
            out["labels_bbox"]=torch.tensor(boxes, dtype=torch.float32)
        return out


class EmotionHead(nn.Module):
    def __init__(self, hidden_size:int, emo_dim:int=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, emo_dim),
        )
    def forward(self, h):
        z = self.net(h)
        return F.normalize(z, dim=-1)

class Classifier(nn.Module):
    def __init__(self, in_dim:int, n:int): super().__init__(); self.fc=nn.Linear(in_dim,n)
    def forward(self,x): return self.fc(x)

class Regressor(nn.Module):
    def __init__(self, in_dim:int, out:int): super().__init__(); self.fc=nn.Linear(in_dim,out)
    def forward(self,x): return self.fc(x)

class EmoWrapper(nn.Module):
    """
    包装官方Qwen多模态模型：forward返回我们需要的多任务输出
    依赖 tokenizer 的特殊 token [EMO] 定位隐藏态
    """
    def __init__(self, base, tokenizer, emo_token:str, n_emotions:int,
                 n_bg:int=0, n_adj:int=0, n_noun:int=0, use_bbox:bool=True, emo_dim:int=1024):
        super().__init__()
        self.base = base
        self.tokenizer = tokenizer
        self.emo_id = tokenizer.convert_tokens_to_ids(emo_token)
        hidden = base.config.hidden_size

        self.h_emo = EmotionHead(hidden, emo_dim)
        self.cls_emotion = Classifier(emo_dim, n_emotions)
        self.reg_intensity= Regressor(emo_dim, 1)
        self.n_bg, self.n_adj, self.n_noun = n_bg, n_adj, n_noun
        if n_bg  >0: self.cls_bg   = Classifier(emo_dim, n_bg)
        if n_adj >0: self.cls_adj  = Classifier(emo_dim, n_adj)
        if n_noun>0: self.cls_noun = Classifier(emo_dim, n_noun)
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

        z = self.h_emo(h_emo)
        out = {
            "z": z,
            "logits_emotion": self.cls_emotion(z),
            "pred_intensity": self.reg_intensity(z),
        }
        if self.n_bg >0:  out["logits_bg"]  = self.cls_bg(z)
        if self.n_adj>0:  out["logits_adj"] = self.cls_adj(z)
        if self.n_noun>0: out["logits_noun"]= self.cls_noun(z)
        if self.use_bbox: out["pred_bbox"]  = self.reg_bbox(z)
        return out


class EmoTrainer(Trainer):
    def __init__(self, *args, w_arc=1.0, w_int=0.5, w_aux=0.2, w_box=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_arc, self.w_int, self.w_aux, self.w_box = w_arc, w_int, w_aux, w_box
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.SmoothL1Loss()

    def compute_loss(self, model: EmoWrapper, inputs, return_outputs=False):
        labels_emotion = inputs.pop("labels_emotion")
        labels_intensity = inputs.pop("labels_intensity")
        labels_bg  = inputs.pop("labels_bg", None)
        labels_adj = inputs.pop("labels_adj", None)
        labels_noun= inputs.pop("labels_noun",None)
        labels_bbox= inputs.pop("labels_bbox",None)

        outputs = model(**inputs)
        loss = 0.0
        loss_emotion   = self.ce(outputs["logits_emotion"], labels_emotion)
        loss_intensity = self.l1(outputs["pred_intensity"], labels_intensity)
        loss = loss + self.w_arc*loss_emotion + self.w_int*loss_intensity

        aux_terms=[]
        if labels_bg  is not None and "logits_bg"  in outputs: aux_terms.append(self.ce(outputs["logits_bg"],  labels_bg))
        if labels_adj is not None and "logits_adj" in outputs: aux_terms.append(self.ce(outputs["logits_adj"], labels_adj))
        if labels_noun is not None and "logits_noun"in outputs: aux_terms.append(self.ce(outputs["logits_noun"],labels_noun))
        loss_aux = sum(aux_terms)/len(aux_terms) if aux_terms else torch.tensor(0.0, device=loss_emotion.device)
        loss += self.w_aux*loss_aux

        if labels_bbox is not None and "pred_bbox" in outputs:
            loss_box = self.l1(outputs["pred_bbox"], labels_bbox)
        else:
            loss_box = torch.tensor(0.0, device=loss_emotion.device)
        loss += self.w_box*loss_box


        self.log({
            "loss_emotion": loss_emotion.detach(),
            "loss_intensity": loss_intensity.detach(),
            "loss_aux": loss_aux.detach(),
            "loss_bbox": loss_box.detach(),
            "z_norm_mean": outputs["z"].norm(dim=-1).mean().detach()
        })
        return (loss, outputs) if return_outputs else loss


def build_emo_components(processor, tokenizer, base_model, data_args, training_args):

    EMO_TOKEN = "[EMO]"
    if EMO_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens":[EMO_TOKEN]})
        base_model.resize_token_embeddings(len(tokenizer))


    emotion_list = [s for s in data_args.emotion_list.split(",") if s]
    bg_list  = [s for s in data_args.bg_list.split(",") if s]
    adj_list = [s for s in data_args.adj_list.split(",") if s]
    noun_list= [s for s in data_args.noun_list.split(",") if s]
    emotion2id = {k:i for i,k in enumerate(emotion_list)}
    bg2id  = {k:i for i,k in enumerate(bg_list)} if bg_list else {}
    adj2id = {k:i for i,k in enumerate(adj_list)} if adj_list else {}
    noun2id= {k:i for i,k in enumerate(noun_list)} if noun_list else {}


    train_ds = EmoDataset(data_args.train_json_dir)
    eval_ds  = EmoDataset(data_args.eval_json_dir)
    collator = EmoCollator(processor, EMO_TOKEN, emotion2id, bg2id, adj2id, noun2id)


    wrapper = EmoWrapper(
        base=base_model,
        tokenizer=tokenizer,
        emo_token=EMO_TOKEN,
        n_emotions=len(emotion2id),
        n_bg=len(bg2id), n_adj=len(adj2id), n_noun=len(noun2id),
        use_bbox=True
    )


    trainer_kwargs = dict(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        w_arc=data_args.w_arc, w_int=data_args.w_int,
        w_aux=data_args.w_aux, w_box=data_args.w_box
    )
    return wrapper, trainer_kwargs

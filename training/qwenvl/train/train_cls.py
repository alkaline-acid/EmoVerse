


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from collections import Counter
from sklearn.metrics import classification_report




SAVE_DIR = "./hidden_features"
INDEX_FILE = os.path.join(SAVE_DIR, "index.jsonl")
OUT_DIR = "./cls_out"
SEED = 42
BATCH_SIZE = 64
LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_EPOCH = 50
PATIENCE = 8
VAL_SPLIT = 0.1
POOLING = "mean"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "Amusement", "Anger", "Awe", "Contentment",
    "Disgust", "Excitement", "Fear", "Sadness"
]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}

os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)




def infer_label_from_path(image_path: str) -> int:
    """
    假设路径类似：
    .../EmoPro/Amusement/Amusement000001.jpg
    取上一级目录名作为类别
    """
    p = Path(image_path)


    for part in p.parts[::-1]:
        if part in CLS2ID:
            return CLS2ID[part]

    for part in p.parents:
        name = Path(part).name
        if name in CLS2ID:
            return CLS2ID[name]
    raise ValueError(f"无法从路径推断类别: {image_path}")

class HiddenFeatDataset(Dataset):
    def __init__(self, index_file: str, pooling: str = "mean"):
        self.records: List[Dict] = []
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        if len(self.records) == 0:
            raise RuntimeError("index.jsonl 为空")
        self.pooling = pooling


        first = torch.load(self.records[0]["feature_path"], map_location="cpu")

        if first.dim() == 3:
            self.hidden_dim = first.size(-1)
        elif first.dim() == 2:
            self.hidden_dim = first.view(-1).shape[0]
        else:
            raise ValueError(f"未知特征形状: {list(first.shape)}")

    def __len__(self):
        return len(self.records)

    def _pool(self, feat: torch.Tensor) -> torch.Tensor:
        """
        输入 feat: [1, L, D] 或 [L, D] 或 [D]
        输出 pooled: [D]
        """
        if feat.dim() == 3:

            feat = feat.squeeze(0)
        if feat.dim() == 2:
            if self.pooling == "mean":
                return feat.mean(dim=0)
            else:
                raise NotImplementedError(f"未实现的 pooling: {self.pooling}")
        elif feat.dim() == 1:
            return feat
        else:
            raise ValueError(f"未知特征形状: {list(feat.shape)}")

    def __getitem__(self, idx):
        rec = self.records[idx]
        feat = torch.load(rec["feature_path"], map_location="cpu")
        pooled = self._pool(feat).float()
        label = infer_label_from_path(rec["image_path"])
        return pooled, label




class Classifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.net(x)




def compute_class_weight(dataset: Dataset) -> torch.Tensor:
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y)
    cnt = Counter(labels)

    total = len(dataset)
    weights = []
    for cls_id in range(len(CLASSES)):
        c = cnt.get(cls_id, 1)
        w = total / (len(CLASSES) * c)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)

def evaluate(model, loader, device) -> Tuple[float, List[int], List[int]]:
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
    acc = correct / max(1, total)
    return acc, y_true, y_pred

def main():
    ds = HiddenFeatDataset(INDEX_FILE, pooling=POOLING)


    N = len(ds)
    n_test = max(1, int(0.1 * N))
    n_val = max(1, int(VAL_SPLIT * (N - n_test)))
    n_train = N - n_val - n_test
    train_set, val_set, test_set = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )


    train_subset = torch.utils.data.Subset(ds, train_set.indices)
    class_weight = compute_class_weight(train_subset).to(DEVICE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = Classifier(in_dim=ds.hidden_dim, num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH)

    best_val, patience = 0.0, PATIENCE
    best_path = os.path.join(OUT_DIR, "best_cls.pt")

    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        scheduler.step()
        train_loss = total_loss / max(1, len(train_loader.dataset))
        val_acc, _, _ = evaluate(model, val_loader, DEVICE)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")


        if val_acc > best_val:
            best_val = val_acc
            patience = PATIENCE
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "in_dim": ds.hidden_dim,
                    "num_classes": len(CLASSES),
                    "classes": CLASSES,
                    "pooling": POOLING,
                }
            }, best_path)
            print(f"  ↳ New best! saved to {best_path}")
        else:
            patience -= 1
            if patience == 0:
                print("  ↳ Early stopping.")
                break


    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_acc, y_true, y_pred = evaluate(model, test_loader, DEVICE)
    print(f"[Test] acc={test_acc:.4f}")


    report = classification_report(
        y_true, y_pred, target_names=CLASSES, digits=4
    )
    print(report)
    with open(os.path.join(OUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()

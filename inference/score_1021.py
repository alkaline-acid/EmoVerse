

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import json, re, ast, sys, math, statistics, argparse, cv2
from pathlib import Path


W_BBOX = 1/3
W_TEXT = 1/3
W_INT  = 1/3


imagedata = []
with open('/home/xxx/dataset/EmoPro/prompt/split_for_qwen/test1022_1_bbox.jsonl', 'r') as f:
    for i in f:
        t = json.loads(i)
        imagedata.append(t['image'])
imageind = -1
def draw_two_groups(line: str, save_path: str = None, thickness: int = None):
    """
    解析形如：
    o/EmoPro/Sadness/Sadness012101.jpg 0.0054 [...] [...]
    的一整行字符串，读取图片并用绿色画第一组bbox、红色画第二组bbox。
    """



    m = re.match(r"(.+?)\s+(\[.*?\])\s+(\[.*\])\s*$", line.strip())
    if not m:
        raise ValueError("输入行格式不符合预期。")
    img_path, gt_str, pred_str = m.groups()
    gt_boxes = ast.literal_eval(gt_str)
    pred_boxes = ast.literal_eval(pred_str)


    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{img_path}")

    h, w = img.shape[:2]
    if thickness is None:

        thickness = max(2, min(h, w) // 400)


    def clamp(v, lo, hi):
        return int(max(lo, min(hi, float(v))))

    def draw_boxes(boxes, color):
        for b in boxes:
            x1 = clamp(b.get("x1", 0), 0, w-1)
            y1 = clamp(b.get("y1", 0), 0, h-1)
            x2 = clamp(b.get("x2", 0), 0, w-1)
            y2 = clamp(b.get("y2", 0), 0, h-1)

            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            if x2 > x1 and y2 > y1:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


    draw_boxes(gt_boxes,  (0, 255, 0))
    draw_boxes(pred_boxes,(0, 0, 255))


    if save_path is None:
        base, ext = os.path.splitext(img_path)
        save_path = f"{base}_compare{ext}"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, img)
    return save_path


class TextEncoder:
    def __init__(self):
        self.backend = None
        self.device = "cuda" if self._cuda_available() else "cpu"

        try:
            import open_clip
            self.open_clip = open_clip
            local_path = '/home/xxx/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.laion2b_e16/snapshots/825e019510c276bde39060d9bc8925d901acde74/open_clip_model.safetensors'
            pre = 'ViT-B-32'
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                pre, pretrained=local_path, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(pre)
            self.backend = "open_clip"
        except Exception:
            self.open_clip = None

            try:
                import clip as oclip
                import torch
                self.oclIP = oclip
                self.torch = torch
                self.model, _ = oclip.load("ViT-B/32", device=self.device)
                self.backend = "clip"
            except Exception:
                self.oclIP = None
                self.torch = None

                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                self.TfidfVectorizer = TfidfVectorizer
                self.cosine_similarity = cosine_similarity
                self.backend = "tfidf"

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def encode_pair_cosine(self, a: str, b: str) -> float:
        if (a is None) or (b is None):
            return 0.0
        a = a.strip()
        b = b.strip()
        if a == "" and b == "":
            return 1.0
        if self.backend == "open_clip":
            import torch
            with torch.no_grad():
                ta = self.tokenizer([a])
                tb = self.tokenizer([b])
                ea = self.model.encode_text(ta.to(self.device))
                eb = self.model.encode_text(tb.to(self.device))
                ea = ea / ea.norm(dim=-1, keepdim=True)
                eb = eb / eb.norm(dim=-1, keepdim=True)
                sim = (ea @ eb.T).item()
                return (sim + 1) / 2
        elif self.backend == "clip":
            import torch
            with torch.no_grad():
                ea = self.model.encode_text(self.oclIP.tokenize([a]).to(self.device))
                eb = self.model.encode_text(self.oclIP.tokenize([b]).to(self.device))
                ea = ea / ea.norm(dim=-1, keepdim=True)
                eb = eb / eb.norm(dim=-1, keepdim=True)
                sim = (ea @ eb.T).item()
                return (sim + 1) / 2
        else:
            vec = self.TfidfVectorizer().fit_transform([a, b])
            sim = self.cosine_similarity(vec[0:1], vec[1:2])[0, 0]
            return float(sim)


DICT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def extract_last_dict_str(s: str):
    if not isinstance(s, str):
        return None
    matches = list(DICT_PATTERN.finditer(s))
    if not matches:
        return None
    return matches[-1].group(0)

def parse_payload(s: str):
    dstr = extract_last_dict_str(s)
    if dstr is None:
        return {}
    try:
        return ast.literal_eval(dstr)
    except Exception:
        try:
            j = dstr.replace("'", '"')
            return json.loads(j)
        except Exception:
            return {}

def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def to_number(x, default=None):
    """尽量从任意类型中抽出数值（优先 float），抽不到返回 default。"""
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = NUM_RE.search(s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return default
    return default


def _area_xyxy(b):
    w = max(0.0, float(b["x2"]) - float(b["x1"]))
    h = max(0.0, float(b["y2"]) - float(b["y1"]))
    return w * h

def _inter_xyxy(a, b):
    xA = max(float(a["x1"]), float(b["x1"]))
    yA = max(float(a["y1"]), float(b["y1"]))
    xB = min(float(a["x2"]), float(b["x2"]))
    yB = min(float(a["y2"]), float(b["y2"]))
    w = max(0.0, xB - xA)
    h = max(0.0, yB - yA)
    return w * h

def iou_xyxy(a, b):
    try:
        inter = _inter_xyxy(a, b)
        areaA = _area_xyxy(a)
        areaB = _area_xyxy(b)
        denom = areaA + areaB - inter
        return 0.0 if denom <= 0 else inter / denom
    except Exception:
        return 0.0

def center_distance_score(a, b):
    """1 - min(1, dist / diag_union) ∈ [0,1]"""
    try:
        cx1 = (float(a["x1"]) + float(a["x2"])) / 2.0
        cy1 = (float(a["y1"]) + float(a["y2"])) / 2.0
        cx2 = (float(b["x1"]) + float(b["x2"])) / 2.0
        cy2 = (float(b["y1"]) + float(b["y2"])) / 2.0
        dist = math.hypot(cx1 - cx2, cy1 - cy2)

        x1 = min(float(a["x1"]), float(b["x1"]))
        y1 = min(float(a["y1"]), float(b["y1"]))
        x2 = max(float(a["x2"]), float(b["x2"]))
        y2 = max(float(a["y2"]), float(b["y2"]))
        diag = math.hypot(x2 - x1, y2 - y1)
        if diag <= 0:
            return 0.0
        return max(0.0, 1.0 - min(1.0, dist / diag))
    except Exception:
        return 0.0

def coverage_PRF(a, b):
    """
    召回 R = inter/area_gt
    精度 P = inter/area_pred
    F1 = 2PR/(P+R)
    """
    try:
        inter = _inter_xyxy(a, b)
        area_gt = _area_xyxy(a)
        area_pd = _area_xyxy(b)
        P = 0.0 if area_pd <= 0 else inter / area_pd
        R = 0.0 if area_gt <= 0 else inter / area_gt
        F1 = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)

        return float(max(0.0, min(1.0, P))), float(max(0.0, min(1.0, R))), float(max(0.0, min(1.0, F1)))
    except Exception:
        return 0.0, 0.0, 0.0

def greedy_match(b1, b2, score_fn):
    """
    通用贪心匹配：基于 score_fn(a,b) 最大化。
    返回 matched_scores(list)，未匹配不计入平均。
    """
    if not isinstance(b1, list): b1 = []
    if not isinstance(b2, list): b2 = []
    used1, used2 = set(), set()
    scores = []

    M = [[score_fn(a, b) for b in b2] for a in b1]
    while True:
        best = -1.0
        sel = (-1, -1)
        for i in range(len(b1)):
            if i in used1: continue
            for j in range(len(b2)):
                if j in used2: continue
                if M[i][j] > best:
                    best = M[i][j]
                    sel = (i, j)
        if best <= 0:
            break
        i, j = sel
        used1.add(i); used2.add(j)
        scores.append(M[i][j])
    return scores

def greedy_match_prf(b1, b2):
    """
    针对 PRF 的贪心：以 IoU 为匹配准则（稳定），随后对匹配对计算 P/R/F1。
    返回：p_list, r_list, f1_list
    """
    if not isinstance(b1, list): b1 = []
    if not isinstance(b2, list): b2 = []
    used1, used2 = set(), set()
    pairs = []
    M = [[iou_xyxy(a, b) for b in b2] for a in b1]
    while True:
        best = -1.0
        sel = (-1, -1)
        for i in range(len(b1)):
            if i in used1: continue
            for j in range(len(b2)):
                if j in used2: continue
                if M[i][j] > best:
                    best = M[i][j]
                    sel = (i, j)
        if best <= 0:
            break
        i, j = sel
        used1.add(i); used2.add(j)
        pairs.append((i, j))
    P_list, R_list, F1_list = [], [], []
    for i, j in pairs:
        P, R, F1 = coverage_PRF(b1[i], b2[j])
        P_list.append(P); R_list.append(R); F1_list.append(F1)
    return P_list, R_list, F1_list

def avg_or_zero(xs):
    return sum(xs)/len(xs) if xs else 0.0

def to_box_list(boxes):
    """
    将多种 bbox 表达统一为 dict 列表：{'x1','y1','x2','y2'}，并保证 x1<=x2, y1<=y2
    支持：
      - [{'x1':..,'y1':..,'x2':..,'y2':..}, ...]
      - [[x1,y1,x2,y2], (x1,y1,x2,y2), ...]
    过滤非法项。
    """
    out = []
    def _is_seq4(x):
        try:
            return (hasattr(x, "__len__") and len(x) == 4 and not isinstance(x[0], dict) and not isinstance(x[0], list))
        except Exception:
            return False
    if _is_seq4(boxes):
        boxes = [boxes]
    if not isinstance(boxes, list):
        return out
    for b in boxes:
        if isinstance(b, dict) and all(k in b for k in ("x1","y1","x2","y2")):
            try:
                x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            except:
                print(boxes)
                return []
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = map(float, b)
        else:
            continue
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def per_line_scores(enc: TextEncoder, gold: dict, pred: dict):
    global imageind
    global imagedata

    bbox_gold = safe_get(gold, "bbox", [])
    bbox_pred = safe_get(pred, "bbox", [])
    if bbox_pred == []:
        bbox_pred = safe_get(pred, "bbox_2d", [])




    bbox_gold = to_box_list(bbox_gold)
    bbox_pred = to_box_list(bbox_pred)





    iou_list   = greedy_match(bbox_gold, bbox_pred, iou_xyxy)
    ctr_list   = greedy_match(bbox_gold, bbox_pred, center_distance_score)
    P_list, R_list, F1_list = greedy_match_prf(bbox_gold, bbox_pred)

    iou_score = avg_or_zero(iou_list)















    ctr_score = avg_or_zero(ctr_list)
    covP = avg_or_zero(P_list)
    covR = avg_or_zero(R_list)
    covF1 = avg_or_zero(F1_list)
    bbox_agg = (iou_score + ctr_score + covF1) / 3.0


    desc_g = str(safe_get(gold, "description", "") or "")
    desc_p = str(safe_get(pred, "description", "") or "")
    clip_desc = enc.encode_pair_cosine(desc_g, desc_p)

    emo_g = str(safe_get(gold, "emotion", "") or "")
    emo_p = str(safe_get(pred, "emotion", "") or "")

    clip_emotion = 1 if emo_g == emo_p else 0

    bag_g = " ".join([str(safe_get(gold, k, "") or "") for k in ["background", "adjective", "noun"]]).strip()
    bag_p = " ".join([str(safe_get(pred, k, "") or "") for k in ["background", "adjective", "noun"]]).strip()
    clip_bag = enc.encode_pair_cosine(bag_g, bag_p)


    text_agg = (clip_desc + clip_bag) / 2.0


    inten_g = to_number(safe_get(gold, "intensity", None))
    inten_p = to_number(safe_get(pred, "intensity", None))



    if inten_g is None or inten_p is None:
        intensity_score = 0.0
    else:
        inten_g = max(0.0, min(10.0, float(inten_g)))
        inten_p = max(0.0, min(10.0, float(inten_p)))
        intensity_score = 1.0 - min(1.0, abs(inten_g - inten_p) / 10.0)


    overall = W_BBOX * bbox_agg + W_TEXT * text_agg + W_INT * intensity_score




























    return {
        "iou": iou_score,
        "center": ctr_score,
        "cov_precision": covP,
        "cov_recall": covR,
        "cov_f1": covF1,
        "bbox_agg": bbox_agg,
        "clip_desc": clip_desc,
        "clip_emotion": clip_emotion,
        "clip_bag": clip_bag,
        "text_agg": text_agg,
        "intensity": intensity_score,
        "overall": overall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=str, help="Path to out.jsonl")
    ap.add_argument("--save_csv", type=str, default="scores_v2_alltrained.csv", help="Where to save detailed scores")
    args = ap.parse_args()

    enc = TextEncoder()
    print(f"[Info] Text similarity backend = {enc.backend}")

    rows = []
    bad_lines = 0

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            idx = rec.get("index", line_no)

            global imageind
            imageind = idx
            result_raw = rec.get("result", "")
            answer_raw = rec.get("answer", "")


            gold = parse_payload(answer_raw)
            pred = parse_payload(result_raw)

            scores = per_line_scores(enc, gold, pred)
            row = {"index": idx, **scores}
            rows.append(row)

    if not rows:
        print("[Error] No valid rows parsed.")
        if bad_lines > 0:
            print(f"[Hint] Malformed JSON lines: {bad_lines}")
        sys.exit(1)


    def mean_var(vals):
        m = statistics.fmean(vals)
        v = statistics.pvariance(vals) if len(vals) > 1 else 0.0
        return m, v

    keys_for_stats = [
        "iou","center","cov_precision","cov_recall","cov_f1","bbox_agg",
        "clip_desc","clip_emotion","clip_bag","text_agg",
        "intensity","overall"
    ]
    stats = {k: mean_var([r[k] for r in rows]) for k in keys_for_stats}

    print("\n===== Summary =====")
    print(f"Total lines: {len(rows)}  |  backend: {enc.backend}  |  malformed lines skipped: {bad_lines}")
    for k in keys_for_stats:
        m, v = stats[k]
        print(f"{k:>13s} -> mean: {m:.4f}, var: {v:.6f}")


    worst = sorted(rows, key=lambda r: r["overall"])[:min(50, len(rows))]
    print("\n===== Worst 50 by overall score =====")
    for r in worst:
        print(
            "index={index}, overall={overall:.4f} | bbox(iou={iou:.3f},ctr={center:.3f},covF1={cov_f1:.3f}) "
            "| text(desc={clip_desc:.3f},emo={clip_emotion:.3f},bag={clip_bag:.3f}) "
            "| intensity={intensity:.3f}".format(**r)
        )


    try:
        import csv
        keys = ["index"] + keys_for_stats
        with open(args.save_csv, "w", newline="", encoding="utf-8") as wf:
            writer = csv.DictWriter(wf, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in keys})
        print(f"\n[Saved] Detailed scores -> {args.save_csv}")
    except Exception as e:
        print(f"[Warn] Failed to save CSV: {e}")

if __name__ == "__main__":
    main()



"""
bbox_diff_eval_top10.py
读取 bbox.jsonl，解析预测与标注框，
计算 IoU / 中心距离 / 面积差，并输出指标；
同时保存 IoU 最低的 10 个样本。
"""

import json, ast, re
from pathlib import Path
import numpy as np
from tqdm import tqdm



def parse_boxes(text):
    """
    从任意字符串中解析出框坐标。
    可能的格式：
      - "[{'x1':..,'y1':..,'x2':..,'y2':..}, {...}]"
      - "[102, 30, 245, 160]"
    返回：List[[x1,y1,x2,y2]]
    """
    if not text or not isinstance(text, str):
        return []
    text = text.strip()


    if text.startswith('[') and text.endswith(']'):
        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict):
                data = [data]

            if all(isinstance(x, (int, float)) for x in data) and len(data) == 4:
                return [data]

            elif isinstance(data, list):
                out = []
                for d in data:
                    if isinstance(d, dict):
                        out.append([d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2")])
                    elif isinstance(d, (list, tuple)) and len(d) == 4:
                        out.append(d)
                return out
        except Exception:
            pass


    import re
    matches = re.findall(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", text)
    if matches:
        return [[float(a), float(b), float(c), float(d)] for (a,b,c,d) in matches]

    return []

def box_iou(box1, box2):
    try:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
        area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0.0
    except Exception as e:
        print(f"[Warning] box_iou() 跳过异常框: {box1}, {box2}，错误: {e}")
        return 0.0

def box_center_distance(b1, b2):
    try:
        cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
        cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    except Exception as e:
        return 0.0

def box_area_diff(b1, b2):
    try:
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return abs(a1 - a2) / max(a1, a2) if max(a1, a2) > 0 else 0.0
    except Exception as e:
        return 0.0

def match_boxes(pred_boxes, gt_boxes):
    """按顺序一一匹配"""
    if not pred_boxes or not gt_boxes:
        return [], [], []
    ious, dists, diffs = [], [], []
    for p, g in zip(pred_boxes, gt_boxes):
        iou = box_iou(p, g)
        dist = box_center_distance(p, g)
        diff = box_area_diff(p, g)
        ious.append(iou)
        dists.append(dist)
        diffs.append(diff)
    return ious, dists, diffs



in_path = Path("output_3B_alltrained_bbox_180.jsonl")

out_path = Path("bbox_metrics_alltrained.jsonl")









































results = []

with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Processing"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        pred_text = obj.get("result", "")

        pred_text = pred_text.split('\n')[-1]


        gt_text = obj.get("answer", "")

        pred_boxes = parse_boxes(pred_text)
        gt_boxes = parse_boxes(gt_text)

        ious, dists, diffs = match_boxes(pred_boxes, gt_boxes)
        if ious:
            mean_iou = float(np.mean(ious))
            mean_dist = float(np.mean(dists))
            mean_diff = float(np.mean(diffs))
        else:
            mean_iou, mean_dist, mean_diff = 0.0, 0.0, 0.0

        record = {
            "index": obj.get("index"),
            "pred_boxes": pred_boxes,
            "gt_boxes": gt_boxes,
            "metrics": {
                "mean_iou": mean_iou,
                "mean_center_dist": mean_dist,
                "mean_area_diff": mean_diff
            }
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        results.append(record)


if results:
    all_ious = [r["metrics"]["mean_iou"] for r in results]
    all_dists = [r["metrics"]["mean_center_dist"] for r in results]
    all_diffs = [r["metrics"]["mean_area_diff"] for r in results]

    stats = {
        "total_samples": len(results),
        "iou": {
            "mean": float(np.mean(all_ious)),
            "std": float(np.std(all_ious)),
            "min": float(np.min(all_ious)),
            "max": float(np.max(all_ious)),
        },
        "center_dist": {
            "mean": float(np.mean(all_dists)),
            "std": float(np.std(all_dists)),
            "min": float(np.min(all_dists)),
            "max": float(np.max(all_dists)),
        },
        "area_diff": {
            "mean": float(np.mean(all_diffs)),
            "std": float(np.std(all_diffs)),
            "min": float(np.min(all_diffs)),
            "max": float(np.max(all_diffs)),
        },
    }


    fout_path = out_path
    with fout_path.open("a", encoding="utf-8") as fstat:
        fstat.write("# === Overall Statistics ===\n")
        fstat.write(json.dumps(stats, ensure_ascii=False, indent=2))
        fstat.write("\n")

    print("\n✅ 全部统计结果：")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
else:
    print("⚠️ 没有有效样本，无法计算总体统计。")

results_sorted = sorted(results, key=lambda x: x["metrics"]["mean_iou"])
lowest10 = results_sorted[:10]

with out_path.open("a", encoding="utf-8") as f_low:
    for r in lowest10:
        f_low.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ 已保存所有结果到 {out_path}")
print(f"⚠️ 已输出 IoU 最低的 10 条样本到 {out_path}")

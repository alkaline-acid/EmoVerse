


import json
import ast
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
from tqdm import tqdm


BASE_IMG_DIR = Path("/home/xxx/dataset/EmoPro/EmoPro")
CSV_PATH = Path("/home/xxx/dataset/EmoPro/final.csv")
XLSX_PATH = Path("/home/xxx/dataset/EmoPro/dino_out.xlsx")
OUTPUT_JSON_DIR = BASE_IMG_DIR.parent / "json"
OUTPUT_JSONL = OUTPUT_JSON_DIR / "index.jsonl"


EMO_FOLDERS = ["Amusement", "Anger", "Awe", "Contentment", "Disgust", "Excitement", "Fear", "Sadness"]


def try_parse_bbox(cell) -> Optional[List[float]]:
    """
    解析单元格为 [x1,y1,x2,y2]（字符串/list/tuple）
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, (list, tuple)) and len(cell) >= 4:
        return [float(cell[0]), float(cell[1]), float(cell[2]), float(cell[3])]
    s = str(cell).strip()
    if not s:
        return None
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
    except Exception:
        pass
    parts = [p.strip() for p in s.replace("[", "").replace("]", "").split(",")]
    if len(parts) >= 4:
        try:
            return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
        except Exception:
            return None
    return None


def read_final_csv(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    读取 final.csv（逗号分隔、首行表头）
    必需列：id, analysis, background, adjective, noun, emotion, intensity（大小写不敏感）
    """
    df = pd.read_csv(csv_path, header=0, sep=",", dtype=str, keep_default_na=False)
    col_map = {c.lower().strip(): c for c in df.columns}
    required = ["id", "analysis", "background", "adjective", "noun", "emotion", "intensity"]
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}；检测到的列：{list(df.columns)}")

    data: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        _id = str(row[col_map["id"]]).strip()
        id_base = Path(_id).stem
        try:
            intensity = float(str(row[col_map["intensity"]]).strip())
        except Exception:
            intensity = None

        data[id_base] = {
            "id": id_base,
            "analysis": str(row[col_map["analysis"]]).strip(),
            "background": str(row[col_map["background"]]).strip(),
            "adjective": str(row[col_map["adjective"]]).strip(),
            "noun": str(row[col_map["noun"]]).strip(),
            "emotion": str(row[col_map["emotion"]]).strip(),
            "intensity": intensity,
        }
    return data


def read_dino_xlsx_multi(xlsx_path: Path) -> Dict[str, List[List[float]]]:
    """
    读取 dino_out.xlsx（首行表头，可能对同一图片多行）
    约定：
      - 第1列：文件名（含 .jpg）
      - 第5列：bbox 字符串
    返回 {id_base: [[x1,y1,x2,y2], ...]}
    """
    df = pd.read_excel(xlsx_path, header=0)
    if df.shape[1] < 5:
        raise ValueError(f"xlsx 列数不足，期望至少 5 列，实际 {df.shape[1]}")

    boxes_map: Dict[str, List[List[float]]] = {}
    for _, row in df.iterrows():
        fname = str(row.iloc[0]).strip()
        if not fname or pd.isna(fname):
            continue
        id_base = Path(fname).stem

        bbox = try_parse_bbox(row.iloc[4])
        if bbox is None or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = map(float, bbox[:4])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        boxes_map.setdefault(id_base, []).append([x1, y1, x2, y2])


    for k, lst in boxes_map.items():
        uniq, seen = [], set()
        for b in lst:
            key = tuple(round(v, 2) for v in b)
            if key not in seen:
                seen.add(key)
                uniq.append(b)
        boxes_map[k] = uniq

    return boxes_map


def find_image_path(base_dir: Path, emotion: Optional[str], id_base: str) -> Optional[Path]:
    """根据 emotion（类别目录）与 id 构造路径；若不存在，退化到8个类别中搜索。"""
    if emotion:
        p = base_dir / emotion / f"{id_base}.jpg"
        if p.exists():
            return p
    for sub in EMO_FOLDERS:
        p = base_dir / sub / f"{id_base}.jpg"
        if p.exists():
            return p
    return None


def main():
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading CSV ...")
    csv_data = read_final_csv(CSV_PATH)
    print(f"  CSV items: {len(csv_data)}")

    print("Reading XLSX with multi boxes per image ...")
    xlsx_boxes = read_dino_xlsx_multi(XLSX_PATH)
    print(f"  XLSX image entries with boxes: {len(xlsx_boxes)}")

    missing_img, written = 0, 0


    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout_idx:
        for id_base, ann in tqdm(csv_data.items(), desc="Writing JSON"):
            emotion = ann.get("emotion")
            img_path = find_image_path(BASE_IMG_DIR, emotion, id_base)
            if img_path is None:
                missing_img += 1
                continue

            bboxes = xlsx_boxes.get(id_base, [])
            item = {
                "id": id_base,
                "image_path": str(img_path),
                "analysis": ann.get("analysis", ""),
                "background": ann.get("background", ""),
                "adjective": ann.get("adjective", ""),
                "noun": ann.get("noun", ""),
                "emotion": emotion,
                "intensity": ann.get("intensity", None),
                "bbox": [
                    {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}
                    for b in bboxes
                ],
                "mask_path": None
            }


            out_path = OUTPUT_JSON_DIR / f"{id_base}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)


            fout_idx.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Written {written} JSON files to {OUTPUT_JSON_DIR}")
    if missing_img:
        print(f"  Missing images: {missing_img}")

if __name__ == "__main__":
    main()

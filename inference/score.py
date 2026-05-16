

"""
从 JSONL 读取每行的:
  - result 中 assistant 段落(assistant\n 之后)
  - answer 参考
计算 BERTScore(F1)，输出所有行分数与总体统计，以及最低分若干条。

Dependencies:
  pip install bert-score transformers torch numpy
"""

import json
import argparse
import numpy as np
from bert_score import score
import torch
from pathlib import Path

def extract_assistant_text(result_str: str) -> str:
    """
    从 result 字段中抽取 assistant 的一句话:
    假定格式包含 'assistant\n'，取其后的文本。
    """
    if not isinstance(result_str, str):
        return ""
    key = "assistant\n"
    if key in result_str:
        return result_str.split(key, 1)[1].strip()

    return result_str.strip()

def load_pairs(jsonl_path: str):
    indices, cands, refs = [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "result" not in obj or "answer" not in obj:
                continue
            cand = extract_assistant_text(obj["result"])
            ref = obj["answer"].strip() if isinstance(obj["answer"], str) else ""
            indices.append(obj.get("index", len(indices)))
            cands.append(cand)
            refs.append(ref)
    return indices, cands, refs

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--output", "-o", default="bertscore_report.json", help="输出 JSON 文件路径")
    ap.add_argument("--model", default="roberta-large", help="BERTScore底模 (如 roberta-large / microsoft/deberta-xlarge-mnli)")
    ap.add_argument("--lang", default="en", help="语言 (默认 en)")
    ap.add_argument("--device", default=None, help="cuda / cpu；默认自动选择")
    ap.add_argument("--worst-k", type=int, default=None, help="保存最低分的前K条；若不设，则按比例 --worst-ratio")
    ap.add_argument("--worst-ratio", type=float, default=0.01, help="按比例保存最低分(默认 1%)；当 --worst-k 为空时生效")
    args = ap.parse_args()

    inp = 'output_3B_alltrained.jsonl'
    indices, cands, refs = load_pairs(inp)
    n = len(cands)
    if n == 0:
        raise ValueError("未在输入文件中读到有效 (result, answer) 对。")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Loaded {n} samples. Using device={device}, model={args.model}")



    P, R, F1 = score(cands, refs, lang=args.lang, model_type=args.model, device=device)

    f1 = F1.detach().cpu().numpy().astype(float)
    mean_score = float(np.mean(f1))
    var_score = float(np.var(f1))
    std_score = float(np.std(f1))
    min_score = float(np.min(f1))
    max_score = float(np.max(f1))


    scores = [{"index": int(idx), "bertscore_f1": float(s)} for idx, s in zip(indices, f1)]


    if args.worst_k is not None:
        k = max(1, min(args.worst_k, n))
    else:
        k = max(1, min(int(round(n * args.worst_ratio)), n))
    worst_idx_sorted = np.argsort(f1)[:k]
    worst = []
    for pos in worst_idx_sorted:
        worst.append({
            "index": int(indices[pos]),
            "bertscore_f1": float(f1[pos]),
            "candidate": cands[pos],
            "reference": refs[pos]
        })

    summary = {
        "num_samples": n,
        "model": args.model,
        "lang": args.lang,
        "device": device,
        "mean": mean_score,
        "std": std_score,
        "var": var_score,
        "min": min_score,
        "max": max_score,
        "worst_k": k
    }

    output = {
        "summary": summary,
        "scores": scores,
        "worst": worst
    }

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved report -> {out_path.resolve()}")
    print(f"Mean={mean_score:.4f}  Std={std_score:.4f}  Var={var_score:.6f}  Min={min_score:.4f}  Max={max_score:.4f}")
    print(f"Worst K = {k}")

if __name__ == "__main__":
    main()

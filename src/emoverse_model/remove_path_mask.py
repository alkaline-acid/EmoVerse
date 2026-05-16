import os
import json
from tqdm import tqdm


json_dir = "/home/xxx/dataset/EmoPro/json/"


for root, _, files in os.walk(json_dir):
    for fname in tqdm(files, desc=f"Processing {root}"):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(root, fname)
        try:

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)


            if "mask_path" in data:
                del data["mask_path"]


            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

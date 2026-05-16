import os
import json
import glob
from tqdm import tqdm


def enhanced_data_validator(json_dir):
    """
    对指定目录下的JSON文件进行深度验证。
    """
    problematic_files = []
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)

    if not json_files:
        print(f"错误：在目录 '{json_dir}' 及其子目录中没有找到任何 .json 文件。")
        return

    print(f"正在对 {len(json_files)} 个文件进行深度验证...")

    for file_path in tqdm(json_files, desc="验证进度"):

        if os.path.getsize(file_path) == 0:
            problematic_files.append((file_path, "文件为空"))
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)


            if not isinstance(data, dict) or not data:
                problematic_files.append((file_path, "JSON内容不是一个非空字典 (可能是空的 '{}' 或 '[]')"))
                continue


            image_path_value = data.get('image_path')
            if image_path_value is None:
                problematic_files.append((file_path, "缺少 'image_path' 键"))
            elif not isinstance(image_path_value, str) or not image_path_value.strip():
                problematic_files.append((file_path, "'image_path' 的值为空字符串或不是字符串类型"))

        except json.JSONDecodeError:
            problematic_files.append((file_path, "JSON 格式无效，无法解析"))
        except Exception as e:
            problematic_files.append((file_path, f"发生未知错误: {e}"))

    if problematic_files:
        print("\n===================================")
        print("🚨 验证完成！发现以下存在问题的文件：")
        print("===================================")
        for p_file, reason in problematic_files:
            print(f"文件: {p_file}\n原因: {reason}\n")
    else:
        print("\n✅ 验证完成！所有 JSON 文件均通过检查。")




train_json_directory = "/home/xxx/dataset/EmoPro/json"
enhanced_data_validator(train_json_directory)

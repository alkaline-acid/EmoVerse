import os
import json


def transform_and_aggregate_json(source_dir, base_dest_dir):
    """
    将源目录下的JSON文件转换为新的对话格式。
    - 单个的JSON文件保存到 'base_dest_dir/single/'
    - 汇总的JSONL文件保存到 'base_dest_dir/'

    Args:
        source_dir (str): 包含原始JSON文件的源文件夹路径。
        base_dest_dir (str): 用于存放输出文件的基础目标文件夹路径。
    """


    single_json_output_dir = os.path.join(base_dest_dir, 'single')

    jsonl_output_path = os.path.join(base_dest_dir, 'dataset.jsonl')


    os.makedirs(single_json_output_dir, exist_ok=True)
    print(f"单个JSON文件的输出目录: '{single_json_output_dir}'")
    print(f"汇总JSONL文件的输出路径: '{jsonl_output_path}'")


    try:
        file_list = [f for f in os.listdir(source_dir) if f.endswith('.json')]
        if not file_list:
            print(f"警告：在源文件夹 '{source_dir}' 中没有找到任何 .json 文件。")
            return
    except FileNotFoundError:
        print(f"错误：源文件夹 '{source_dir}' 不存在。")
        return

    print(f"\n找到 {len(file_list)} 个JSON文件，开始处理...")

    processed_count = 0

    with open(jsonl_output_path, 'w', encoding='utf-8') as jsonl_file:
        for filename in file_list:
            source_file_path = os.path.join(source_dir, filename)

            dest_file_path = os.path.join(single_json_output_dir, filename)

            try:

                with open(source_file_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)


                image_path = original_data.get("image_path")
                background_description = original_data.get("bbox")

                if not image_path or not background_description:
                    print(f"跳过文件 {filename}：缺少 'image_path' 或 'background' 字段。")
                    continue

                transformed_data = {
                    "image": image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Output the pixel position of the main subject of the image. The image may have more than one subject.\n<image>"
                        },
                        {
                            "from": "gpt",
                            "value": background_description
                        }
                    ]
                }


                with open(dest_file_path, 'w', encoding='utf-8') as f:
                    json.dump(transformed_data, f, indent=2, ensure_ascii=False)


                jsonl_line = json.dumps(transformed_data, ensure_ascii=False)
                jsonl_file.write(jsonl_line + '\n')

                processed_count += 1

            except json.JSONDecodeError:
                print(f"错误：文件 {filename} 不是一个有效的JSON文件，已跳过。")
            except Exception as e:
                print(f"处理文件 {filename} 时发生未知错误: {e}")

    print(f"\n处理完成！")
    print(f"成功转换了 {processed_count} 个文件。")
    print(f"独立的JSON文件已保存至: {single_json_output_dir}")
    print(f"汇总的JSONL文件已保存为: {jsonl_output_path}")


if __name__ == "__main__":

    source_directory = "/home/xxx/dataset/EmoPro/json/"
    base_destination_directory = "/home/xxx/dataset/EmoPro/prompt/bbox"


    transform_and_aggregate_json(source_directory, base_destination_directory)

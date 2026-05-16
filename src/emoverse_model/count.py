import os


def count_json_files(directory):
    """
    递归地统计一个文件夹及其所有子文件夹中 .json 文件的总数。

    参数:
    directory (str): 你要搜索的文件夹的路径。

    返回:
    int: .json 文件的总数。
    """
    json_count = 0


    if not os.path.isdir(directory):
        print(f"错误：路径 '{directory}' 不是一个有效的文件夹或不存在。")
        return 0


    for root, dirs, files in os.walk(directory):
        for file in files:

            if file.endswith('.json'):
                json_count += 1

    return json_count






target_folder = '/home/xxx/dataset/EmoPro/prompt/intensity/single/'



total_files = count_json_files(target_folder)

if os.path.isdir(target_folder):
    print(f"在文件夹 '{target_folder}' 及其所有子文件夹中，共找到 {total_files} 个 JSON 文件。")

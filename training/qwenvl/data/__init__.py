import re


CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

emo_200704 = {
    "annotation_path": "/home/xxx/dataset/EmoPro/prompt/split_for_qwen/train1027_2_bbox.jsonl",
    "data_path": "",
}

emo = {
    "annotation_path": "/home/xxx/dataset/EmoPro/prompt/split_for_qwen/train1027_1_bbox.jsonl",
    "data_path": "",
}

emo_onlyemo = {
    "annotation_path": "/home/xxx/dataset/EmoPro/prompt/split_for_qwen/train1022_1_ab.jsonl",
    "data_path": "",
}

test = {
    "annotation_path": "/home/xxx/LoRA_Qwen/clone/Qwen3-VL/qwen-vl-finetune/demo/single_images.json",
    "data_path": "/home/xxx/LoRA_Qwen/clone/Qwen3-VL/qwen-vl-finetune",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "emo": emo,
    "emo_200704": emo_200704,
    "test": test,
    "emo_onlyemo": emo_onlyemo,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)

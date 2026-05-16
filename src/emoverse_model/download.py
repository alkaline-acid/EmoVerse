import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel
import os



model_name = "Qwen/Qwen2.5-VL-7B-Instruct"







print(f"步骤 1/2: 正在下载并加载 '{model_name}' 的处理器...")
try:

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("处理器加载成功！")
except Exception as e:
    print(f"加载处理器时发生错误: {e}")
    exit()



print(f"\n步骤 2/2: 正在下载并加载 '{model_name}' 的模型权重...")
print("这个过程可能会花费很长时间，并且需要大量的磁盘空间。请耐心等待。")

try:


    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    print("\n模型权重加载成功！")
    print("-" * 30)
    print("模型已成功加载到以下设备：")
    print(model.hf_device_map)
    print("-" * 30)

except Exception as e:
    print(f"加载模型时发生错误: {e}")


print("\n所有必要的模型和处理器都已准备就绪。")

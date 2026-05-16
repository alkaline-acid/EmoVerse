from accelerate import Accelerator
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, get_scheduler
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import gc
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import re
import transformers
from qwenvl.model.qwen_classify import ClassifyModelImprove


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_args(model_args, data_args, training_args, output_dir):
    """保存所有参数到JSON文件"""

    os.makedirs(output_dir, exist_ok=True)


    args_dict = {
        "model_args": model_args.__dict__,
        "data_args": data_args.__dict__,
        "training_args": training_args.__dict__
    }


    with open(os.path.join(output_dir, "training_args.json"), "w") as f:

        json.dump(args_dict, f, indent=4)

    print(f"参数已保存到 {os.path.join(output_dir, 'training_args.json')}")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/bn/icvg-ec-dexiang-lf-hw/offline_weight/Qwen2.5-VL-3B-Instruct")
    lora_adapter_path: Optional[str] = field(default=None)
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    freeze_backbone: bool = field(default=True)
    unfreeze_last_layers: bool = field(default=False)
    feature_fusion: str = field(default="attention")


@dataclass
class DataArguments:
    train_data: str = field(default="/mnt/bn/icvg-ec-dexiang-lf-hw/dexiang/data/emotion_data/train1022_1_ab.jsonl")
    eval_data: str = field(default="/mnt/bn/icvg-ec-dexiang-lf-hw/dexiang/data/emotion_data/test1022_1_ab.jsonl")
    max_image_size: int = field(default=3210*2120)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)

@dataclass
class TrainingArguments:
    output_dir: str = field(default="./output")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    num_train_epochs: int = field(default=3)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.05)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=1000)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=False)
    dataloader_num_workers: int = field(default=4)
    local_rank: int = field(default=-1)

def extract_emotion_from_answer(answer):
    pattern = r"'emotion':\s*'([^']+)'"





    match = re.search(pattern, answer)
    if match:
        emotion_word = match.group(1)

        return emotion_word
    else:
        return None


class ImageTextDataset(Dataset):
    def __init__(self, data, max_image_size=3210*2120):
        self.data = data
        self.max_image_size = max_image_size
        self.label_map = {"amusement": 0, "anger": 1, "awe": 2, "contentment": 3,
                         "disgust": 4, "excitement": 5, "fear": 6, "sadness": 7}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:

            image_path = item['image']
            image = Image.open(image_path)
            width, height = image.size


            if width * height > self.max_image_size:
                return None


            answer = item['conversations'][-1]['value']
            emotion_label = extract_emotion_from_answer(answer)

            label = self.label_map.get(emotion_label.lower(), 7)

            return {
                'image': image,
                'query': item['conversations'][0]['value'],
                'label': label,
                'index': idx
            }
        except Exception as e:
            logger.warning(f"加载数据项 {idx} 时出错: {e}")
            return None


def collate_fn(batch):

    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images = [item['image'] for item in batch]
    queries = [item['query'] for item in batch]
    labels = [item['label'] for item in batch]
    indices = [item['index'] for item in batch]

    return {
        'images': images,
        'queries': queries,
        'labels': torch.tensor(labels, dtype=torch.long),
        'indices': indices
    }



def load_data_from_jsonl(jsonl_path):
    start_time = time.time()
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    with ThreadPoolExecutor(max_workers=8) as executor:
        parsed_data = list(executor.map(
            lambda line: json.loads(line.strip()) if line.strip() else None,
            lines
        ))

    data = [item for item in parsed_data if item is not None]
    logger.info(f"加载 {len(data)} 条数据耗时: {time.time() - start_time:.2f}秒")
    return data


def process_batch_messages(batch, processor):
    batch_messages = []
    for img, query in zip(batch['images'], batch['queries']):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": query},
            ],
        }]
        batch_messages.append(messages)


    texts = []
    image_inputs_batch = []

    for messages in batch_messages:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        texts.append(text)
        image_inputs_batch.append(messages[0]['content'][0]['image'])

    return texts, image_inputs_batch


def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    accelerator = Accelerator(
        mixed_precision="bf16" if training_args.bf16 else "no",
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=training_args.output_dir
    )


    os.makedirs(training_args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        save_args(model_args, data_args, training_args, training_args.output_dir)


    logger.info(f"从 {model_args.model_name_or_path} 加载模型...")
    processor = AutoProcessor.from_pretrained('/mnt/bn/icvg-ec-dexiang-lf-hw/offline_weight/Qwen2.5-VL-3B-Instruct', padding_side='left')
    model = ClassifyModelImprove(model_args.model_name_or_path, processor, model_args.lora_adapter_path, max_pixels=data_args.max_pixels, min_pixels=data_args.min_pixels, feature_fusion=model_args.feature_fusion)


    if model_args.freeze_backbone:
        model.freeze_backbone()

    if model_args.unfreeze_last_layers:
        model.unfreeze_last_layers()


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )


    logger.info("加载训练数据...")
    train_data = load_data_from_jsonl(data_args.train_data)
    train_dataset = ImageTextDataset(train_data, max_image_size=data_args.max_image_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True
    )

    logger.info("加载评估数据...")
    eval_data = load_data_from_jsonl(data_args.eval_data)
    eval_dataset = ImageTextDataset(eval_data, max_image_size=data_args.max_image_size)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True
    )


    num_update_steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * training_args.warmup_ratio),
        num_training_steps=max_train_steps,
    )


    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    accelerator.init_trackers("qwen_classify_training")
    logger.info("开始训练...")

    global_step = 0

    for epoch in range(training_args.num_train_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{training_args.num_train_epochs}")

        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue

            with accelerator.accumulate(model):

                texts, image_inputs_batch = process_batch_messages(batch, processor)


                logits = model(texts, image_inputs_batch, accelerator.device)


                loss = criterion(logits, batch['labels'].to(accelerator.device))


                accelerator.backward(loss)


                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)


                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())


            if accelerator.is_main_process and global_step % training_args.logging_steps == 0:
                accelerator.log({
                    "train_loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step
                }, step=global_step)


            if accelerator.is_main_process and global_step % training_args.save_steps == 0 and global_step > 0:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)


                torch.save({
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step
                }, os.path.join(checkpoint_dir, 'checkpoint.pt'))

                logger.info(f"已保存检查点到 {checkpoint_dir}")

            global_step += 1


            if global_step % training_args.eval_steps == 0 and accelerator.is_main_process:
                model.eval()
                eval_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch in tqdm(eval_dataloader, desc="评估", disable=not accelerator.is_local_main_process):
                        if batch is None:
                            continue

                        texts, image_inputs_batch = process_batch_messages(batch, processor)
                        logits = model(texts, image_inputs_batch, accelerator.device)

                        loss = criterion(logits, batch['labels'].to(accelerator.device))
                        eval_loss += loss.item()


                        predictions = torch.argmax(logits, dim=-1)
                        correct += (predictions == batch['labels'].to(accelerator.device)).sum().item()
                        total += batch['labels'].size(0)

                eval_accuracy = correct / total if total > 0 else 0
                logger.info(f"Step {global_step} 评估结果: 损失={eval_loss/len(eval_dataloader)}, 准确率={eval_accuracy:.4f}")

                with open(os.path.join(training_args.output_dir, f"eval_{global_step}results.json"), "w") as f:
                    json.dump({
                        "epoch": epoch,
                        "step": global_step,
                        "eval_loss": eval_loss/len(eval_dataloader),
                        "eval_accuracy": eval_accuracy
                    }, f, indent=4)

                accelerator.log({
                    "eval_loss": eval_loss/len(eval_dataloader),
                    "eval_accuracy": eval_accuracy,
                    "epoch": epoch
                }, step=global_step)


    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_checkpoint_dir = os.path.join(training_args.output_dir, "final")
        os.makedirs(final_checkpoint_dir, exist_ok=True)

        torch.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'epoch': training_args.num_train_epochs,
            'global_step': global_step
        }, os.path.join(final_checkpoint_dir, 'model.pt'))

        logger.info(f"已保存最终模型到 {final_checkpoint_dir}")

    accelerator.end_training()

if __name__ == '__main__':
    train()

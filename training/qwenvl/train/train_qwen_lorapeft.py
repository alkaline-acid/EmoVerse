# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename="my_output.log",
    filemode="a",
    encoding="utf-8",
)

import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer
from peft import LoraConfig, get_peft_model
import re

from transformers import TrainerCallback

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

LORA_PATTERNS = [
    r"\.q_proj$", r"\.k_proj$", r"\.v_proj$", r"\.o_proj$",
    r"\.up_proj$", r"\.down_proj$", r"\.gate_proj$",
    r"\.qkv$", r"\.proj$", r"\.fc1$", r"\.fc2$"
]

def wanted_linear(name, module):
    if not isinstance(module, torch.nn.Linear):
        return False


    if not name.startswith("model.language_model"):
        return False
    if name.startswith("model.visual.merger"):
        return False

    return any(re.search(pat, name) for pat in LORA_PATTERNS)

class WatchParamCallback(TrainerCallback):
    def __init__(self, pattern, every_n_steps=100, use_regex=True, log_grad=True):
        self.pattern = pattern
        self.every_n_steps = every_n_steps
        self.use_regex = use_regex
        self.log_grad = log_grad
        self._resolved = False
        self._param_name = None

    def _match_name(self, name):
        return (re.search(self.pattern, name) is not None) if self.use_regex else (self.pattern in name)

    def _unwrap_engine(self, model):
        """
        HF+DeepSpeed 时，trainer.model 是 DeepSpeedEngine；真正模型在 engine.module
        PEFT 时，LoRA 包裹在 PeftModel -> base_model 层级。
        """
        eng = model
        inner = getattr(model, "module", model)

        try:
            if hasattr(inner, "get_base_model"):
                inner = inner.get_base_model()
        except Exception:
            pass
        return eng, inner

    def _resolve_once(self, model):
        if self._resolved:
            return
        engine, inner = self._unwrap_engine(model)
        candidates = [n for n, _ in inner.named_parameters()]
        if self.pattern in candidates and not self.use_regex:
            self._param_name = self.pattern
        else:
            hits = [n for n in candidates if self._match_name(n)]
            if len(hits) == 1:
                self._param_name = hits[0]
            elif len(hits) > 1:
                logging.warning(f"[WatchParam] pattern 命中多个参数，取第一个：{hits[0]}\n全部命中={hits}")
                self._param_name = hits[0]
            else:
                logging.warning(f"[WatchParam] 未找到匹配的参数：{self.pattern}")
                self._param_name = None
        self._resolved = True

    def _is_rank0(self, args):
        try:
            return (args.local_rank in (-1, 0)) and (getattr(args, "process_index", 0) == 0)
        except Exception:
            return True

    def _gather_and_log(self, args, model, step):
        if not self._is_rank0(args) or self._param_name is None:
            return

        engine, inner = self._unwrap_engine(model)
        p = dict(inner.named_parameters()).get(self._param_name, None)
        if p is None:
            logging.info(f"[WatchParam] 参数在当前 rank 未找到：{self._param_name}")
            return


        ctx = torch.no_grad()
        try:
            import deepspeed
            from deepspeed import zero

            enabled = hasattr(engine, "zero_optimization") and getattr(engine, "zero_optimization", {}).get("stage", 0) > 0
            ctx = zero.GatheredParameters([p], modifier_rank=0) if enabled else torch.no_grad()
        except Exception:
            pass

        with ctx:
            if p.data is None or p.data.numel() == 0:
                logging.info(f"[WatchParam] step={step} | {self._param_name}: EMPTY (sharded?)")
                return
            data = p.data.detach()
            mean = data.float().mean().item()
            std  = data.float().std().item()
            minv = data.float().min().item()
            maxv = data.float().max().item()
            msg = (f"[WatchParam] step={step} | {self._param_name}: "
                   f"mean={mean:.6f}, std={std:.6f}, min={minv:.6f}, max={maxv:.6f}, "
                   f"shape={tuple(data.shape)}, dtype={data.dtype}, device={data.device}")
            if self.log_grad and (p.grad is not None) and (p.grad.numel() > 0):
                try:
                    gnorm = p.grad.data.float().norm().item()
                    msg += f", grad_norm={gnorm:.6f}"
                except Exception:
                    pass
            logging.info(msg)

    def on_train_begin(self, args, state, control, **kwargs):
        self._resolve_once(kwargs["model"])
        if self._param_name:
            logging.info(f"[WatchParam] 监控参数：{self._param_name}（every {self.every_n_steps} steps）")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            self._gather_and_log(args, kwargs["model"], state.global_step)

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(
            model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    target_modules = []
    for n, m in model.named_modules():
        if wanted_linear(n, m):
            print(n,m)

            leaf = n.split(".")[-1]
            target_modules.append(leaf)


    target_modules = sorted(set(target_modules))
    print("LoRA target leaf names =", target_modules)

    lora_cfg = LoraConfig(
        r=256,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",

        modules_to_save=["model.visual.merger"]
    )

    model = get_peft_model(model, lora_cfg)
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(processor, data_args=data_args)
    for n, p in model.language_model.named_parameters():
        if 'lora' in n:
            p.requires_grad = True
        print(n, p)
    watch = WatchParamCallback(
        pattern=r"language_model\.layers\.0\.self_attn\.q_proj\.lora_A$",
        every_n_steps=10,
        use_regex=True,
        log_grad=True
    )
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, callbacks=[watch], **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

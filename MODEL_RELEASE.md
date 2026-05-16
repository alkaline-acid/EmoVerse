# EmoVerse Model Release

This repository includes the code and metadata needed to reproduce the EmoVerse Qwen2.5-VL fine-tuning experiments. Large checkpoint weights are intentionally not stored in Git.

## Included Checkpoint

The lightweight PEFT LoRA checkpoint prepared for release is:

```text
qwen2.5vl-emo-lora-1023
```

The public release is available at:

```text
https://huggingface.co/alkalol/EmoVerse-LoRA
```

Expected contents when published through a model-hosting flow:

- `adapter_config.json`
- `adapter_model.safetensors`
- tokenizer and processor files required by Qwen2.5-VL
- `trainer_state.json` and `training_args.bin` for reproducibility

Publish these files from the root of the prepared PEFT adapter directory. Do not upload `checkpoint-*`, `global_step*`, DeepSpeed optimizer states, RNG states, or scheduler state files as part of the public model package.

The adapter was trained on top of:

```text
Qwen2.5-VL-3B-Instruct
```

The full-model export inspected during release preparation was the `output_200704_1102` experiment from the Qwen-VL fine-tuning workspace.

If a public full-model checkpoint is needed, publish only the exported model shards and model/tokenizer/processor metadata through a model-hosting release flow, then link it from this file. Do not publish training checkpoint folders or optimizer-state folders.

## Code Layout

- `src/emoverse_model/`: EmoVerse-specific LoRA and emotion-head training utilities.
- `training/qwenvl/`: Qwen-VL fine-tuning modules used by the experiments.
- `training/scripts/`: launch scripts and DeepSpeed configs.
- `inference/`: inference and scoring helpers.
- `examples/`: compact data-format examples.
- Model weights: [Hugging Face](https://huggingface.co/alkalol/EmoVerse-LoRA)

## Notes

The scripts still contain local path defaults from the training machine. Before running them on another machine, update model, dataset, and output paths in `training/qwenvl/data/__init__.py` and `training/scripts/*.sh`.

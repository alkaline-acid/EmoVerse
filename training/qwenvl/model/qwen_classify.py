from torch import nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassifyModelImprove(nn.Module):
    def __init__(self, qwen_model_path, processor, lora_adapter_path=None, feature_fusion="attention", max_pixels=None, min_pixels=None):
        super().__init__()
        self.feature_fusion = feature_fusion
        self.processor = processor
        if max_pixels is not None:
            self.processor.max_pixels = max_pixels
        if min_pixels is not None:
            self.processor.min_pixels = min_pixels
        self.build_qwen(qwen_model_path, lora_adapter_path)
        self.build_clssify_head()
        self.freeze_backbone()

    def build_qwen(self, model_path, lora_adapter_path=None):
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=None
        )

        if lora_adapter_path:
            self.qwen_model.load_adapter(lora_adapter_path)

    def build_clssify_head(self):
        hidden_size = self.qwen_model.config.text_config.hidden_size


        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(dtype=torch.bfloat16)


        if self.feature_fusion == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size // 2,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ).to(dtype=torch.bfloat16)
        elif self.feature_fusion == "learning_attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size // 2,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ).to(dtype=torch.bfloat16)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size // 2, dtype=torch.bfloat16))
            self.token_proj_layer = nn.Linear(hidden_size // 2, hidden_size // 2).to(dtype=torch.bfloat16)



        self.classify_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, 8)
        ).to(dtype=torch.bfloat16)

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def forward(self, texts, image_inputs_batch, device):
        inputs = self.processor(
                text=texts,
                images=image_inputs_batch,
                padding=True,
                return_tensors="pt",
            )

        pixel_values = inputs["pixel_values"].to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)


        kwargs = {}
        if "image_grid_thw" in inputs:
            kwargs["image_grid_thw"] = inputs["image_grid_thw"].to(device)


        encoder_hidden_states = self.qwen_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
            output_hidden_states=True,
        )


        hidden_states_list = encoder_hidden_states.hidden_states


        num_layers_to_fuse = 4
        fused_hidden = torch.stack(hidden_states_list[-num_layers_to_fuse:], dim=0).mean(dim=0)


        split_hidden_states = self._extract_masked_hidden(fused_hidden, attention_mask)





        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )


        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)


        projected_features = self.feature_projection(prompt_embeds)


        if self.feature_fusion == "attention":
            attended_features, _ = self.attention(projected_features, projected_features, projected_features)

            sequence_repr = torch.sum(attended_features, dim=1) / torch.sum(torch.ones_like(attended_features), dim=1)
        elif self.feature_fusion == "learning_attention":

            cls_tokens = self.cls_token.expand(projected_features.size(0), -1, -1)
            token_features = torch.cat([cls_tokens, projected_features], dim=1)
            attended_features, _ = self.attention(token_features, token_features, token_features)
            sequence_repr = attended_features[:, 0, :]
            sequence_repr = self.token_proj_layer(sequence_repr)
        else:

            sequence_repr = torch.mean(projected_features, dim=1)


        logits = self.classify_head(sequence_repr)

        return logits

    def freeze_backbone(self):
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        logger.info("Qwen模型主干已冻结，仅训练分类头")

    def unfreeze_last_layers(self, num_layers=2):
        """解冻最后几层以进行微调"""
        for param in self.qwen_model.model.language_model.layers[-num_layers:].parameters():
            param.requires_grad = True
        logger.info(f"已解冻最后{num_layers}层进行微调")

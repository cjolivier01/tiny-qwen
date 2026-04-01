import json
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import load_checkpoint_and_dispatch
from .vision import VisionEncoder, VisionConfig

# Set to True to use flash_attn.cute (FA4) instead of F.scaled_dot_product_attention
USE_FLASH_ATTN_CUTE = False

def enable_flash_attn_cute(enable=True):
    global USE_FLASH_ATTN_CUTE
    USE_FLASH_ATTN_CUTE = enable


@dataclass
class ModelConfig:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int  # dense MLP intermediate size

    n_vocab: int
    tie_word_embeddings: bool

    rope_theta: float
    rms_norm_eps: float

    # MoE parameters
    d_head: Optional[int] = None
    n_experts: Optional[int] = None
    n_experts_per_token: Optional[int] = None
    n_moe_mlp: Optional[int] = None


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_head
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.register_buffer("inv_freq", 1.0 / (t ** (r / d)).float(), persistent=False)

        self.mrope_section = [24, 20, 20]

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.to(dtype=torch.float32, device=x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """[TTT...HHH...WWW] -> [THWTHWTHW...TT]"""
        freqs_t = freqs[0]  # start with temporal dimension
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed

        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.n_embed, bias=False)

        self.q_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if USE_FLASH_ATTN_CUTE:
            # flash_attn.cute expects (B, T, H, D) and handles GQA natively
            q = q.transpose(1, 2)  # (B, T, n_heads, D)
            k = k.transpose(1, 2)  # (B, T, n_kv_heads, D)
            v = v.transpose(1, 2)  # (B, T, n_kv_heads, D)
            from flash_attn.cute import flash_attn_func as fa_cute
            y, _ = fa_cute(q, k, v, causal=True)
            y = y.contiguous().view(B, T, self.n_heads * self.d_head)
        else:
            if self.n_kv_heads < self.n_heads:
                num_repeat = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(num_repeat, dim=1)
                v = v.repeat_interleave(num_repeat, dim=1)

            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)

        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (SelfAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (SelfAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_experts
        self.hidden_size = config.n_embed
        self.expert_dim = config.n_moe_mlp

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        gate_up = torch.einsum("th,ehq->teq", x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_outputs = torch.einsum("teq,eqh->teh", F.silu(gate) * up, self.down_proj)
        weighted = expert_outputs * routing_weights.unsqueeze(-1)
        return weighted.sum(dim=1)


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embed
        self.expert_dim = config.n_moe_mlp
        self.num_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = MoEExperts(config)

    def forward(self, x):
        B, T, _ = x.shape
        hidden = x.reshape(-1, self.hidden_size)

        router_logits = self.gate(hidden)
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        routed = torch.zeros_like(router_logits)
        topk_weights = topk_weights.to(router_logits.dtype)
        routed.scatter_(1, topk_indices, topk_weights)
        routed = routed / (routed.sum(dim=-1, keepdim=True) + 1e-9)
        routed = routed.to(hidden.dtype)
        expert_out = self.experts(hidden, routed)
        combined = expert_out.view(B, T, self.hidden_size)

        return combined


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = MoEMLP(config) if config.n_experts else DenseMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embed,
        vision_embed=None,
        vision_residuals=None,
        vision_mask=None,
        position_ids=None,
    ):
        if vision_embed is not None and vision_mask is not None:
            input_embed[vision_mask] = vision_embed

        cos, sin = self.rotary_emb(input_embed, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            input_embed = layer(input_embed, cos, sin)
            if vision_residuals and vision_mask is not None:
                # deepstack process
                vision_residual = vision_residuals.get(layer_idx)
                if vision_residual is not None:
                    input_embed[vision_mask] = (
                        input_embed[vision_mask] + vision_residual
                    )

        input_embed = self.norm(input_embed)
        return input_embed


class Qwen3VL(nn.Module):
    def __init__(
        self, config: ModelConfig, vision_config: Optional[VisionConfig] = None
    ):
        super().__init__()
        self.config = config
        self.vision_config = vision_config

        self.model = nn.Module()
        self.model.language_model = Model(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

        if vision_config is not None:
            self.model.visual = VisionEncoder(vision_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.model.language_model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)

        if pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed, vision_residuals = self.model.visual(
                pixels=pixels, d_image=d_image
            )
            image_pad_token = getattr(self.config, "image_token_id", 151655)
            vision_mask = input_ids == image_pad_token
            output = self.model.language_model(
                input_embed=input_embeds,
                vision_embed=vision_embed,
                vision_residuals=vision_residuals,
                vision_mask=vision_mask,
                position_ids=position_ids,
            )
        else:
            output = self.model.language_model(
                input_embed=input_embeds, position_ids=position_ids
            )

        logits = (
            output @ self.model.language_model.embed_tokens.weight.T
            if self.lm_head is None
            else self.lm_head(output)
        )
        return logits

    def _get_position_ids(
        self, input_ids: torch.Tensor, d_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        image_pad_token = getattr(self.config, "image_token_id", 151655)

        # text-only case: sequential position IDs repeated 3 times
        if d_image is None:
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(3, B, -1)
            return position_ids

        # text + vision case: 3D position IDs
        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=input_ids.device)
        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            text_idx, image_idx, seq_idx = 0, 0, 0
            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if token_id == image_pad_token:
                    # start of an vision block (image(s))
                    text_idx, image_idx, seq_idx = self._emit_image_block(
                        position_ids=position_ids,
                        batch_idx=batch_idx,
                        seq_idx=seq_idx,
                        text_idx=text_idx,
                        image_idx=image_idx,
                        d_image=d_image,
                    )
                else:
                    # treat as regular text token
                    position_ids[:, batch_idx, seq_idx] = text_idx
                    text_idx, image_idx, seq_idx = text_idx + 1, image_idx, seq_idx + 1

        return position_ids

    def _emit_image_block(
        self,
        position_ids: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
        text_idx: int,
        image_idx: int,
        d_image: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int, int]:
        t_img, h_img, w_img = d_image[image_idx]
        t_img = int(t_img.item())
        h_img = int((h_img // spatial_merge_size).item())
        w_img = int((w_img // spatial_merge_size).item())

        image_token_count = h_img * w_img
        video_token_count = t_img * image_token_count
        for offset in range(video_token_count):
            target_idx = seq_idx + offset
            remaining = offset % image_token_count
            h_pos = remaining // w_img
            w_pos = remaining % w_img

            position_ids[:, batch_idx, target_idx] = text_idx
            position_ids[1, batch_idx, target_idx] = text_idx + h_pos
            position_ids[2, batch_idx, target_idx] = text_idx + w_pos

        return text_idx + 1, image_idx + 1, seq_idx + video_token_count

    @classmethod
    def from_pretrained(cls, weights_path: str, device_map: str = "auto"):
        model_path = Path(weights_path)

        with open(model_path / "config.json", "r") as f:
            hf_config = json.load(f)

        llm_config = hf_config["text_config"]
        config = ModelConfig(
            n_embed=llm_config["hidden_size"],
            n_heads=llm_config["num_attention_heads"],
            n_kv_heads=llm_config["num_key_value_heads"],
            n_layer=llm_config["num_hidden_layers"],
            n_mlp=llm_config["intermediate_size"],
            n_vocab=llm_config["vocab_size"],
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            rope_theta=llm_config["rope_theta"],
            rms_norm_eps=llm_config["rms_norm_eps"],
            d_head=llm_config.get("head_dim"),
            n_experts=llm_config.get("num_experts"),
            n_experts_per_token=llm_config.get("num_experts_per_tok"),
            n_moe_mlp=llm_config.get("moe_intermediate_size"),
        )

        vision_config = None
        vision_config_data = hf_config.get("vision_config")
        if vision_config_data is not None:
            vision_config = VisionConfig(
                n_embed=vision_config_data["hidden_size"],
                n_layer=vision_config_data["depth"],
                n_heads=vision_config_data["num_heads"],
                n_output_embed=vision_config_data["out_hidden_size"],
                n_mlp=vision_config_data["intermediate_size"],
                deepstack_visual_indexes=vision_config_data["deepstack_visual_indexes"],
                num_position_embeddings=vision_config_data["num_position_embeddings"],
                in_channels=vision_config_data["in_channels"],
                temporal_patch_size=vision_config_data["temporal_patch_size"],
                patch_size=vision_config_data["patch_size"],
                spatial_merge_size=vision_config_data["spatial_merge_size"],
            )

        model = cls(config, vision_config=vision_config)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path),
            device_map=device_map,
            no_split_module_classes=["Block", "VisionBlock"],
            dtype=torch.bfloat16,
        )

        return model

    def forward_sample(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> dict:
        """Autoregressive sampling with log-prob collection.

        Args:
            input_ids: (B, prompt_len) prompt token IDs
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = standard, <1 = sharper)

        Returns:
            dict with:
                - sampled_tokens: (B, prompt_len + gen_len) full sequence
                - sampled_logprobs: (B, prompt_len + gen_len) per-token log-probs
                  (prompt positions are 0.0)
                - prompt_length: int
        """
        B, prompt_len = input_ids.shape
        device = input_ids.device

        sampled_tokens = input_ids.clone()
        sampled_logprobs = torch.zeros(B, prompt_len, device=device, dtype=torch.float32)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids=sampled_tokens)
                last_logits = logits[:, -1, :].float()  # (B, vocab)

                log_probs = F.log_softmax(last_logits, dim=-1)

                if temperature <= 0:
                    next_token = last_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

                token_log_probs = log_probs.gather(-1, next_token).squeeze(-1)  # (B,)

                sampled_tokens = torch.cat([sampled_tokens, next_token], dim=1)
                sampled_logprobs = torch.cat(
                    [sampled_logprobs, token_log_probs.unsqueeze(1)], dim=1
                )

        if was_training:
            self.train()

        return {
            "sampled_tokens": sampled_tokens,
            "sampled_logprobs": sampled_logprobs,
            "prompt_length": prompt_len,
        }

    def _generate_core(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor],
        d_image: Optional[torch.Tensor],
        max_new_tokens: int,
        stop_tokens: Optional[list],
    ):
        if stop_tokens is None:
            # <|im_end|>, <|im_start|>, <|endoftext|>
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        generated_ids = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(
                    input_ids=generated_ids, pixels=pixels, d_image=d_image
                )
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                token_id = next_token[0].item()
                yield token_id, generated_ids

                if token_id in stop_tokens:
                    break

    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
    ):
        generated_ids = input_ids

        for _, generated_ids in self._generate_core(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            pass

        return generated_ids

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
    ):
        for token_id, _ in self._generate_core(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            yield token_id

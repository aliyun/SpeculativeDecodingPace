import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3MLP,
    Gemma3PreTrainedModel,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from model.cache import KVCache


class Gemma3KVCache(KVCache):

    def __init__(self, config, max_n_beams, max_len, device, dtype, batch_size):
        super().__init__(
            config=config,
            max_n_beams=max_n_beams,
            max_len=max_len,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        self.positions = torch.zeros(
            (batch_size, max_len), device=self.device, dtype=torch.long
        )
        self._active_batch = 0

    def record_positions(self, position_ids: torch.Tensor, cache_seq_indices: torch.Tensor):
        self.positions[self._active_batch, cache_seq_indices] = position_ids[0]

    def gather(self, batch_idx, src_indices, dest_indices):
        super().gather(batch_idx, src_indices, dest_indices)

        src = src_indices.to(self.device)
        dest = dest_indices.to(self.device)
        if src.dtype == torch.bool:
            src = torch.where(src)[0]

        self.positions[batch_idx, dest] = self.positions[batch_idx, src]
        self.positions[batch_idx, dest.max() + 1 :] = 0

    @contextmanager
    def prefill_context(self, n_beams: int, batch_idx: int):

        previous = self._active_batch
        self._active_batch = batch_idx
        with super().prefill_context(n_beams, batch_idx):
            yield
        self._active_batch = previous


class Gemma3Attention(nn.Module):

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )
        self.is_causal = True

        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=bias
        )
        self.q_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: KVCache,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_batch_indices,
            cache_seq_indices,
        )

        attention_mask = self._apply_sliding_window(
            attention_mask, position_ids, past_key_values
        )

        attn_output = self._sdpa_attention(
            query_states, key_states, value_states, attention_mask
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _apply_sliding_window(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: KVCache,
    ):
        """Drop keys further away than the window, keeping the engine's own mask intact."""
        if self.sliding_window is None:
            return attention_mask

        key_positions = past_key_values.positions[:, None, None, :]
        within_window = (
            position_ids[:, None, :, None] - key_positions
        ) < self.sliding_window

        if attention_mask.dtype == torch.bool:
            return attention_mask & within_window

        return attention_mask.masked_fill(
            ~within_window, torch.finfo(attention_mask.dtype).min
        )

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        dropout: float = 0.0,
    ):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=False,
        )

        return attn_output.transpose(1, 2).contiguous()


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: KVCache,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class Gemma3Model(Gemma3PreTrainedModel):

    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False


        self._tied_weights_keys = {}
        self.all_tied_weights_keys = {}
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: KVCache,
    ):
        hidden_states = self.embed_tokens(input_ids)
        past_key_values.record_positions(position_ids, cache_seq_indices)


        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings[
                    self.config.layer_types[layer_idx]
                ],
                past_key_values=past_key_values,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values


class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):

    kv_cache_cls = Gemma3KVCache
    config_class = Gemma3TextConfig

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


        self.all_tied_weights_keys = {}
        self.post_init()

        if getattr(config, "tie_word_embeddings", True):
            self.tie_weights()

    def tie_weights(self, *args, **kwargs):
        """Gemma-3 checkpoints carry no lm_head tensor: the head is the embedding matrix."""
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
        return None

    @classmethod
    def from_checkpoint(cls, name: str, dtype: torch.dtype, device: torch.device):
        from transformers.models.auto.configuration_auto import AutoConfig

        config = AutoConfig.from_pretrained(name)
        if getattr(config, "model_type", None) != "gemma3":
            model = cls.from_pretrained(name, config=config, torch_dtype=dtype)
            model.eval()
            return model.to(device)

        text_config = config.text_config
        try:
            model = cls._from_config(text_config, dtype=dtype)
        except TypeError:
            model = cls(text_config).to(dtype)

        from safetensors.torch import safe_open

        state = model.state_dict()
        loaded = set()
        for shard in sorted(Path(name).glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as handle:
                for key in handle.keys():
                    if not key.startswith("language_model."):
                        continue
                    local = key[len("language_model.") :]
                    target = state.get(local)
                    if target is None:
                        continue
                    target.copy_(handle.get_tensor(key))
                    loaded.add(local)

        missing = sorted(set(state) - loaded - {"lm_head.weight"})
        model.tie_weights()
        model.eval()
        model.to(device)
        if missing:
            raise RuntimeError(
                f"{len(missing)} text tensors were not found in {name}, e.g. {missing[:3]}"
            )
        return model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: KVCache,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        logits = self.lm_head(hidden_states)
        softcap = getattr(self.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap

        return logits, past_key_values

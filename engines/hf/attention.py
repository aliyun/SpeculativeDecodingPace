# HF 5.16 treats ANY config attention name containing "flash" as its
# own lazy-loader request, even after AttentionInterface.register().
NAME = 'triage_fixed_kv'
_FLASH_OP = None
_SDPA = None


def sdpa_kv(query, keys, values, lengths, scale):
    """Bottom-right causal fixed-bank attention with SDPA instead of Flash.

    Same semantics as ``flash_attn_with_kvcache(causal=True, cache_seqlens=…)``:
    query ``i`` of row ``b`` reads keys ``0..lengths[b]-Q+i``. Used only when the
    environment asks for it (``TRIAGE_HF_KV_SDPA=1``); otherwise the caller needs
    compiled ``flash_attn`` kernels. Every tensor op is CUDA-graph capturable, and the
    query/key head mismatch is resolved with ``enable_gqa`` instead of a copy.
    """
    import torch
    # flash_attn's kvcache layout is (batch, seqlen, heads, headdim); SDPA wants
    # (batch, heads, seqlen, headdim), and the caller keeps the flash layout.
    q = query.transpose(1, 2)
    k = keys.transpose(1, 2)
    v = values.transpose(1, 2)
    heads, width = q.shape[1], q.shape[2]
    keys_len = k.shape[2]
    positions = torch.arange(width, device=q.device)
    # visible key index per (row, query position); clamped so a row whose KV is
    # shorter than the query still sees its first key instead of emitting NaN.
    visible = (lengths.view(-1, 1) - width + positions.view(1, -1)).clamp(min=0)
    key_index = torch.arange(keys_len, device=q.device).view(1, 1, 1, keys_len)
    mask = key_index <= visible.view(-1, 1, width, 1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, is_causal=False,
        enable_gqa=k.shape[1] != heads)
    return out.transpose(1, 2).contiguous()


def _register_op():
    global _FLASH_OP
    if _FLASH_OP is not None:
        return
    import os
    import torch
    use_sdpa = os.environ.get('TRIAGE_HF_KV_SDPA', '0') == '1'
    if use_sdpa:
        kernel = sdpa_kv
    else:
        from flash_attn import flash_attn_with_kvcache

        def kernel(query, keys, values, lengths, scale):
            return flash_attn_with_kvcache(
                query, keys, values, cache_seqlens=lengths,
                softmax_scale=scale, causal=True)

    # Opaque to Inductor, but CUDA-graph capturable. Without a fake kernel,
    # tracing a vendor Python/C++ binding can create a graph break per layer.
    @torch.library.custom_op('triage_hf::flash_kv', mutates_args=())
    def flash_kv(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                 lengths: torch.Tensor, scale: float) -> torch.Tensor:
        return kernel(query, keys, values, lengths, scale).contiguous()

    @flash_kv.register_fake
    def fake(query, keys, values, lengths, scale):
        return query.new_empty(query.shape)

    _FLASH_OP = flash_kv


def attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, *, triage_fixed_kv_lengths=None, **kwargs):
    if triage_fixed_kv_lengths is None:
        # Required for initial AND chunked DynamicCache prefill. The custom
        # attention name also registers SDPA's mask provider, not "no mask".
        return _SDPA(module, query, key, value, attention_mask,
                     dropout=dropout, scaling=scaling, **kwargs)
    if (module.training or dropout != 0.0 or kwargs.get('sliding_window') is not None
            or kwargs.get('is_causal') is False or kwargs.get('output_attentions', False)):
        raise ValueError('fixed KV Flash requires full causal inference without attention weights')
    scale = query.shape[-1]**-0.5 if scaling is None else scaling
    output = _FLASH_OP(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                       triage_fixed_kv_lengths, scale)
    return output, None


def install(backend):
    """Register a NEW attention/mask name; fail before any bank allocation."""
    global _SDPA
    torch, model = backend.torch, backend.model
    if backend.device.type != 'cuda' or backend.dtype != torch.bfloat16:
        raise ValueError('fixed KV Flash currently requires CUDA BF16')
    if getattr(model.config, '_attn_implementation', None) not in ('sdpa', NAME):
        raise ValueError('fixed KV Flash requires an SDPA prefill model')
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    _register_op()  # missing/unsupported extension is a startup error, not fallback
    _SDPA = sdpa_attention_forward
    AttentionInterface.register(NAME, attention_forward)
    AttentionMaskInterface.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS['sdpa'])
    model.set_attn_implementation(NAME)
    if model.config._attn_implementation != NAME:
        raise RuntimeError('HF model refused fixed KV Flash registration')


def fixed_kwargs(torch, positions, attention):
    if attention == 'sdpa':
        return {}
    if attention != 'flash-kv':
        raise ValueError('unknown fixed attention')
    return {'triage_fixed_kv_lengths': (positions[:, -1]+1).to(torch.int32)}

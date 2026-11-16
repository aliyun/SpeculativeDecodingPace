from pathlib import Path


def context_length(config):
    """Prompt/KV budget declared by a model config, wrapper-aware.

    A multimodal wrapper (gemma-3) keeps ``max_position_embeddings`` inside
    ``text_config`` only; reading the top level alone yields 0 and every prefill
    is then rejected with ``prefill exceeds model context``. Prefer the top level
    and fall back to the text config, and refuse a config that declares neither
    instead of silently running with a zero-token budget.
    """
    text_config = getattr(config, 'text_config', config)
    value = (getattr(config, 'max_position_embeddings', None)
             or getattr(text_config, 'max_position_embeddings', 0))
    if not value:
        raise ValueError('cannot determine max_position_embeddings for this model')
    return int(value)


class HFBackend:
    """Greedy forwards for one model on one device, one request row at a time."""

    def __init__(self, path, *, device=None, dtype='bfloat16', attention='sdpa',
                 model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = getattr(torch, dtype)
        self.model = model if model is not None else AutoModelForCausalLM.from_pretrained(
            str(Path(path)), dtype=self.dtype, attn_implementation=attention,
            local_files_only=True)
        if model is None:
            self.model.to(self.device)
        self.model.eval()
        import os
        if model is None and os.environ.get('TRIAGE_HF_COMPILE') == '1':
            # Optional draft-side graph compilation; the cloud stays eager so a
            # compile failure cannot take the target down.
            self.model = torch.compile(self.model, mode='max-autotune-no-cudagraphs', dynamic=True)
        self.tokenizer = tokenizer
        config = self.model.config
        text_config = getattr(config, 'text_config', config)
        # gemma-3's config.json carries no vocab_size at all (neither top level nor in text_config);
        # transformers fills it from the text config's class default. Read it the same way, and fall
        # back to the tokenizer, whose length is the vocabulary in every case.
        _vocab = getattr(config, 'vocab_size', None) or getattr(text_config, 'vocab_size', None)
        if not _vocab and self.tokenizer is not None:
            _vocab = len(self.tokenizer)
        if not _vocab:
            raise ValueError('cannot determine vocab_size for this model')
        self.vocab_size = int(_vocab)
        self.context = context_length(config)
        self.layers = int(getattr(text_config, 'num_hidden_layers'))
        heads = int(getattr(text_config, 'num_attention_heads'))
        self.kv_heads = int(getattr(text_config, 'num_key_value_heads', heads))
        self.head_dim = int(getattr(text_config, 'head_dim', 0)
                             or int(getattr(text_config, 'hidden_size'))//heads)
        self.attention = attention
        # transformers 5.x dropped the explicit cache_position argument; the cache
        # grows from get_seq_length() and position_ids alone.
        import inspect
        self.cache_position_supported = 'cache_position' in inspect.signature(
            self.model.forward).parameters
        import transformers
        try:
            crop_version = tuple(int(part) for part in transformers.__version__.split('.')[:2])
        except ValueError:
            crop_version = (5, 12)
        # 5.16 deprecates absolute crop lengths in favour of "remove N" negatives.
        self.crop_negative = crop_version >= (5, 16)

    def kv_bytes_per_token(self):
        # BF16 K and V for every layer: no blocks, no fragmentation headroom.
        return 2*2*self.layers*self.kv_heads*self.head_dim

    def kv_budget_tokens(self, fraction=0.90):
        if self.device.type != 'cuda':
            return 1 << 20
        torch = self.torch
        try:
            torch.cuda.synchronize(self.device)
            free, _ = torch.cuda.mem_get_info(self.device)
        except Exception:
            return 1 << 20
        return max(0, int(free*float(fraction))//self.kv_bytes_per_token())

    def new_cache(self):
        from transformers import DynamicCache
        return DynamicCache()

    def new_slab(self, slots):
        import os
        if os.environ.get('TRIAGE_HF_FIXED_SLAB', '0') == '1':
            from .fixed_slab import FixedSlab, bank_cap, bank_rows, bucket_size
            context = int(os.environ.get('TRIAGE_HF_FIXED_CONTEXT', '1024'))
            graphs = os.environ.get('TRIAGE_HF_FIXED_GRAPHS', '0') == '1'
            # Graph banks must keep their captured addresses alive across resizes.
            cap = bank_cap(slots)
            physical_rows = (sum(size for size in bank_rows() if size <= cap)
                             if graphs else slots)
            if context*physical_rows > self.kv_budget_tokens(fraction=.75):
                raise ValueError('fixed KV bank exceeds available device budget')
            attention = os.environ.get('TRIAGE_HF_FIXED_ATTENTION', 'sdpa')
            # Compiled shapes retain Inductor scratch buffers alongside graph pools.
            # TRIAGE_HF_FIXED_COMPILE=0 disables torch.compile for the fixed bank.
            compile_model = os.environ.get('TRIAGE_HF_FIXED_COMPILE', '1') == '1'
            slab = FixedSlab(self, slots, context, compile_model=compile_model, use_graphs=graphs,
                             attention=attention,
                             prefill_flash=os.environ.get('TRIAGE_HF_PREFILL_BUCKETS') is not None)
            slab.warm()
            return slab
        return SlabBatch(self, slots)

    def new_prefill_graphs(self, buckets, context):
        from .prefill_graph import PrefillGraphs
        return PrefillGraphs(self, buckets, context)

    def new_fused_draft(self, capacity, context):
        from .fused_draft import FusedDraftPool
        return FusedDraftPool(self, capacity, context)

    def crop(self, cache, length):
        if not isinstance(length, int) or length < 0:
            raise ValueError('invalid cache length')
        current = self.length(cache)
        if length > current:
            raise ValueError('cache crop beyond current length')
        if length == current:
            return
        cache.crop(length-current if self.crop_negative else length)

    def length(self, cache):
        return int(cache.get_seq_length())

    def _prediction_values(self, logits, *, root_top2=False):
        torch = self.torch
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError('nonfinite HF forward logits')
        tokens = logits.argmax(-1).tolist()
        if not root_top2:
            return tokens
        if len(tokens) != 1:
            raise ValueError('root top2 requires one actual draft step')
        from engines.common.root_shadow import validate_top2
        primary = tokens[0]
        scores = logits[0].float().clone()
        scores[primary] = -float('inf')
        alternate = int(scores.argmax().item())
        gap = float((logits[0, primary].float()-logits[0, alternate].float()).item())
        return tokens, validate_top2(dict(primary=primary, alternate=alternate, gap=gap), primary)

    def predict(self, cache, tokens, *, keep=None, require_finite=False, root_top2=False):
        """Forward `tokens`, returning one greedy argmax per new position.

        keep=1 keeps only the final position's logits, which is what prompt
        prefill and draft rounds need; verify keeps every new position.
        """
        if not tokens:
            raise ValueError('nonempty forward tokens required')
        torch = self.torch
        past = self.length(cache)
        input_ids = torch.tensor([list(tokens)], dtype=torch.long, device=self.device)
        cache_position = torch.arange(past, past+len(tokens), dtype=torch.long, device=self.device)
        count = len(tokens) if keep is None else keep
        if not isinstance(count, int) or not 1 <= count <= len(tokens):
            raise ValueError('invalid kept logits')
        # One contiguous row: the model's own causal mask and cache positions are
        # exactly what an explicit mask/position_ids would repeat.
        call = dict(input_ids=input_ids, past_key_values=cache,
                    use_cache=True, logits_to_keep=count, return_dict=True)
        if self.cache_position_supported:
            call['cache_position'] = cache_position
        with torch.inference_mode():
            output = self.model(**call)
        logits = output.logits[0, -count:, :]
        if require_finite and not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError('nonfinite HF prefill logits')
        return self._prediction_values(logits, root_top2=root_top2)

    def prefill(self, cache, tokens, *, chunk=2048):
        """Append a prompt without materialising vocabulary logits for it."""
        if len(tokens) > self.context:
            raise ValueError('prefill exceeds model context')
        for start in range(0, len(tokens), chunk):
            self.predict(cache, tokens[start:start+chunk], keep=1)

    def prefill_first(self, cache, tokens, *, chunk=2048):
        """Full prompt KV plus its final greedy prediction; never a second decode."""
        if not tokens or len(tokens) > self.context:
            raise ValueError('invalid prefill-first prompt')
        prediction = None
        for start in range(0, len(tokens), chunk):
            prediction = self.predict(cache, tokens[start:start+chunk], keep=1,
                                      require_finite=True)[0]
        return prediction

    # ---- fixed-length StaticCache path (edge drafts, one compiled q=1 shape) ----

    def static_cache(self, max_len):
        from transformers import StaticCache
        if type(max_len) is not int or not 1 <= max_len <= self.context:
            raise ValueError('invalid static cache length')
        return StaticCache(config=self.model.config, max_batch_size=1,
                           max_cache_len=max_len, device=self.device, dtype=self.dtype)

    def compile_draft(self):
        """Compile the single-token draft step and trace it before serving."""
        import os
        if os.environ.get('TRIAGE_HF_DRAFT_GRAPH') == '1':
            # The draft step is always batch=1 with one token against a fixed-length
            # cache, so nothing about it needs dynamic shapes. Static shapes let
            # Inductor replay the step as a CUDA graph instead of paying the
            # dispatch cost on every token.
            self.draft_model = self.torch.compile(
                self.model, mode='reduce-overhead', dynamic=False)
        else:
            self.draft_model = self.torch.compile(
                self.model, mode='max-autotune-no-cudagraphs', dynamic=True)
        # Inductor traces at the first call, not at wrap time. Do it here so the
        # service never answers health with a multi-minute compile still pending.
        warm = self.static_cache(min(256, self.context))
        self.static_prefill(warm, [1]*8, 0)
        self.static_predict(warm, [1], 8, compiled=True)

    def compile_verify(self, q=4, *, cache_length=None):
        """Trace the serving KV shape before ready, not an unrelated 256-slot shape."""
        if type(q) is not int or not 1 <= q <= 16:
            raise ValueError('invalid verify width')
        capacity = min(256, self.context) if cache_length is None else cache_length
        if type(capacity) is not int or not 8+q <= capacity <= self.context:
            raise ValueError('invalid verify warmup cache length')
        import logging
        import time
        logger = logging.getLogger(__name__)
        started = time.monotonic()
        logger.info('HF_VERIFY_WARM_START q=%d cache_length=%d', q, capacity)
        self.draft_model = self.torch.compile(
            self.model, mode='max-autotune-no-cudagraphs', dynamic=False)
        warm = self.static_cache(capacity)
        self.static_prefill(warm, [1]*8, 0)
        self.static_predict(warm, [1]*q, 8, compiled=True)
        logger.info('HF_VERIFY_WARM_DONE q=%d cache_length=%d elapsed_s=%.3f',
                    q, capacity, time.monotonic()-started)

    def _static_call(self, cache, tokens, count, compiled, *, root_top2=False):
        torch = self.torch
        module = self.draft_model if compiled else self.model
        # StaticCache tracks its own cumulative length; explicit positions and
        # masks are what breaks the model's own causal bookkeeping.
        with torch.inference_mode():
            output = module(input_ids=torch.tensor([list(tokens)], dtype=torch.long,
                                                   device=self.device),
                            past_key_values=cache, use_cache=True,
                            logits_to_keep=count, return_dict=True)
        return self._prediction_values(output.logits[0, -count:, :], root_top2=root_top2)

    def static_length(self, cache):
        return int(cache.get_seq_length())

    def static_prefill(self, cache, tokens, position):
        if self.static_length(cache) != position:
            raise ValueError('static prefill position mismatch')
        if position+len(tokens) > int(cache.get_max_cache_shape()):
            raise ValueError('static prefill exceeds its cache')
        if tokens:
            # A one-token rewind gap shares the compiled q=1 draft shape.
            compiled = getattr(self, 'draft_model', None) is not None and len(tokens) == 1
            return self._static_call(cache, tokens, 1, compiled=compiled)
        return None

    def static_predict(self, cache, tokens, position, *, compiled=False, root_top2=False):
        if self.static_length(cache) != position:
            raise ValueError('static predict position mismatch')
        return self._static_call(cache, tokens, len(tokens), compiled=compiled, root_top2=root_top2)

    def static_rewind(self, cache, position):
        """Reset the seen length and drop the abandoned KV slots."""
        current = self.static_length(cache)
        if not 0 <= position <= current:
            raise ValueError('invalid static rewind')
        if position == current:
            return
        with self.torch.inference_mode():
            for layer in cache.layers:
                layer.keys[:, :, position:current] = 0
                layer.values[:, :, position:current] = 0
                # cumulative_length is a tensor; keep its type for the next add_.
                layer.cumulative_length = layer.cumulative_length-(current-position)


class SlabBatch:
    """One physical KV slab with per-slot valid lengths and a 4D attention mask.

    The cache batch contains exactly the admitted slots. A forward may verify a
    subset of them; the other active rows join with masked dummy queries so the
    batch dimension stays fixed without reallocating the slab. Accepted nodes are
    copied to the front of each selected row, then the slab is trimmed to the
    longest real prefix.
    """

    def __init__(self, backend, slots):
        if type(slots) is not int or not 1 <= slots <= 64:
            raise ValueError('invalid slab slot count')
        self.backend, self.capacity = backend, slots
        self.cache = backend.new_cache()
        self.valid = [0]*slots
        self.order = []
        self.phys = 0
        self.block_start = 0
        self.ready = False
        self.selected = []
        self.stats = dict(install=0, free=0, forwards=0, rows=0, query_rows=0,
                          commits=0, trims=0, batch_resizes=0)

    def _torch(self):
        return self.backend.torch

    def _empty_layers(self, batch):
        backend, torch = self.backend, self._torch()
        empty = torch.zeros(batch, backend.kv_heads, 0, backend.head_dim,
                            dtype=backend.dtype, device=backend.device)
        for layer in self.cache.layers:
            layer.keys = empty
            layer.values = empty

    def _ensure(self, batch):
        if not self.ready:
            torch, backend = self._torch(), self.backend
            ids = torch.zeros(batch, 1, dtype=torch.long, device=backend.device)
            positions = torch.zeros(batch, 1, dtype=torch.long, device=backend.device)
            mask = torch.ones(batch, 1, 1, 1, dtype=torch.bool, device=backend.device)
            with torch.inference_mode():
                backend.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                              past_key_values=self.cache, use_cache=True,
                              logits_to_keep=1, return_dict=True)
            self._empty_layers(batch)
            self.ready = True
            self.phys = 0
            return
        current = self.cache.layers[0].keys.shape[0] if self.cache.layers else 0
        if current == batch:
            return
        torch = self._torch()
        for layer in self.cache.layers:
            zeros = torch.zeros(batch, layer.keys.shape[1], self.phys, layer.keys.shape[3],
                                dtype=layer.keys.dtype, device=layer.keys.device)
            keep = min(current, batch)
            if keep:
                zeros[:keep, :, :self.phys] = layer.keys[:keep, :, :self.phys]
            layer.keys = zeros
            zeros = torch.zeros(batch, layer.values.shape[1], self.phys, layer.values.shape[3],
                                dtype=layer.values.dtype, device=layer.values.device)
            if keep:
                zeros[:keep, :, :self.phys] = layer.values[:keep, :, :self.phys]
            layer.values = zeros
        self.stats['batch_resizes'] += 1

    def _grow(self, length):
        if length <= self.phys:
            return
        torch = self._torch()
        pad = length-self.phys
        for layer in self.cache.layers:
            zeros = torch.zeros(layer.keys.shape[0], layer.keys.shape[1], pad, layer.keys.shape[3],
                                dtype=layer.keys.dtype, device=layer.keys.device)
            layer.keys = torch.cat([layer.keys, zeros], dim=2)
            layer.values = torch.cat([layer.values, zeros], dim=2)
        self.phys = length

    def install(self, slot, cache, prefix):
        """Copy one prefilled row into its slot; ``prefix`` must match the cache."""
        if type(slot) is not int or not 0 <= slot < self.capacity:
            raise ValueError('invalid slab slot')
        length = self.backend.length(cache)
        if len(prefix) != length:
            raise ValueError('slab install prefix / cache mismatch')
        if slot in self.order:
            index = self.order.index(slot)
        else:
            self.order.append(slot)
            index = len(self.order)-1
        self._ensure(len(self.order))
        self._grow(length)
        with self._torch().inference_mode():
            for cache_index, layer in enumerate(self.cache.layers):
                source = cache.layers[cache_index]
                if length:
                    layer.keys[index, :, :length] = source.keys[0, :, :length]
                    layer.values[index, :, :length] = source.values[0, :, :length]
        self.valid[slot] = length
        self.stats['install'] += 1
        self.trim()

    def free(self, slot):
        if type(slot) is not int or not 0 <= slot < self.capacity:
            raise ValueError('invalid slab slot')
        self.valid[slot] = 0
        if slot in self.order:
            index = self.order.index(slot)
            last = len(self.order)-1
            if index != last:
                with self._torch().inference_mode():
                    for layer in self.cache.layers:
                        layer.keys[index, :, :self.phys] = layer.keys[last, :, :self.phys]
                        layer.values[index, :, :self.phys] = layer.values[last, :, :self.phys]
                self.order[index] = self.order[last]
            self.order.pop()
            self._ensure(len(self.order))
        self.stats['free'] += 1
        self.trim()

    def _mask(self, q, parents):
        torch = self._torch()
        total = self.phys+q
        mask = torch.zeros(len(self.order), 1, q, total, dtype=torch.bool,
                           device=self.backend.device)
        physical = torch.arange(self.phys, device=self.backend.device)
        for index, slot in enumerate(self.order):
            mask[index, :, :, :self.phys] = physical < self.valid[slot]
        from engines.common.tree import ancestor_mask
        mask[:, :, :, self.phys:] = torch.tensor(ancestor_mask(parents), dtype=torch.bool,
                                                 device=self.backend.device)

        return mask

    def predict(self, rows, keep=None, *, parents=None):
        """Forward ``rows`` = [(slot, tokens)]; other active slots run masked dummy rows."""
        if not rows:
            raise ValueError('slab forward requires rows')
        q = len(rows[0][1])
        if any(len(tokens) != q for _, tokens in rows) or not q:
            raise ValueError('slab rows must share one query length')
        from engines.common.tree import depths, validate
        parents = tuple(range(-1, q-1)) if parents is None else tuple(parents)
        validate(parents)
        if len(parents) != q:
            raise ValueError('slab parent / query width mismatch')
        logical_depths = depths(parents)
        selected = {}
        for slot, tokens in rows:
            if not 0 <= slot < self.capacity or slot not in self.order:
                raise ValueError('invalid slab slot')
            selected[slot] = list(tokens)
        if len(selected) != len(rows):
            raise ValueError('duplicate slab rows')
        if not self.ready:
            raise ValueError('slab predict before install')
        torch, backend = self._torch(), self.backend
        total = self.phys+q
        batch = [selected.get(slot, [0]*q) for slot in self.order]
        input_ids = torch.tensor(batch, dtype=torch.long, device=backend.device)
        position_ids = torch.tensor([[self.valid[slot]+step for step in logical_depths]
                                     for slot in self.order], dtype=torch.long,
                                    device=backend.device)
        count = q if keep is None else keep
        aligned = (parents == tuple(range(-1, q-1)) and
                   all(self.valid[slot] == self.phys for slot in self.order))
        if aligned:
            mask = torch.ones(len(self.order), self.phys+q, dtype=torch.long,
                              device=backend.device)
        else:
            mask = self._mask(q, parents)
        call = dict(input_ids=input_ids, attention_mask=mask,
                    position_ids=position_ids, past_key_values=self.cache,
                    use_cache=True, logits_to_keep=count, return_dict=True)
        if backend.cache_position_supported:
            call['cache_position'] = torch.arange(self.phys, total, dtype=torch.long,
                                                  device=backend.device)
        self.block_start = self.phys
        with torch.inference_mode():
            output = backend.model(**call)
        self.phys = total
        self.selected = [slot for slot, _ in rows]
        self.stats['forwards'] += 1
        self.stats['rows'] += len(rows)
        self.stats['query_rows'] += len(self.order)
        logits = output.logits[:, -count:, :]
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError('nonfinite HF slab logits')
        predictions = logits.argmax(-1).tolist()
        return [predictions[self.order.index(slot)] for slot in self.selected]

    def commit(self, slot, offsets):
        """Copy accepted query rows [base+offset] to this row's committed prefix."""
        if slot not in self.selected:
            raise ValueError('slab commit outside the selected rows')
        q = self.phys-self.block_start
        if (type(offsets) is not list or not offsets or len(offsets) != len(set(offsets))
                or any(type(offset) is not int or not 0 <= offset < q for offset in offsets)):
            raise ValueError('invalid slab commit offsets')
        index = self.order.index(slot)
        destination = self.valid[slot]
        source = [self.block_start+offset for offset in offsets]
        with self._torch().inference_mode():
            for layer in self.cache.layers:
                keys = layer.keys[index][:, source, :].clone()
                layer.keys[index, :, destination:destination+len(source)] = keys
                values = layer.values[index][:, source, :].clone()
                layer.values[index, :, destination:destination+len(source)] = values
        self.valid[slot] = destination+len(source)
        self.stats['commits'] += 1

    def trim(self):
        target = max((self.valid[slot] for slot in self.order), default=0)
        if target >= self.phys:
            return
        with self._torch().inference_mode():
            for layer in self.cache.layers:
                layer.keys = layer.keys[:, :, :target]
                layer.values = layer.values[:, :, :target]
        self.phys = target
        self.stats['trims'] += 1

    def status(self):
        return dict(slots=self.capacity, active=list(self.order), valid=list(self.valid),
                    physical=self.phys, batching='PADDED_SLAB_BATCH', stats=dict(self.stats))

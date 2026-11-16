from types import SimpleNamespace
from .attention import fixed_kwargs
from engines.common.tree import ancestor_mask, depths, path_to, validate

# Families whose attention this bank can express. Everything else still refuses.
SLAB_FAMILIES = ('qwen3', 'qwen2', 'gemma3', 'gemma3_text')
MASK_KINDS = ('full_attention', 'sliding_attention')


def family_attention(config):
    """Return (model_type, hybrid, window) for a fixed-slab candidate config.

    gemma3 ships an *empty* ``layer_types`` and derives the sliding pattern from
    ``sliding_window`` when the model is built, yet the built model still asks the
    mask mapping for ``sliding_attention``. A config-only reader therefore has to
    treat gemma3 as hybrid, otherwise the first warm forward raises KeyError.
    The multimodal wrapper keeps the attention layout on its ``text_config``.
    """
    text = getattr(config, 'text_config', None) or config
    model_type = str(getattr(text, 'model_type', None) or getattr(config, 'model_type', ''))
    types = getattr(text, 'layer_types', None) or getattr(config, 'layer_types', None)
    window = int(getattr(text, 'sliding_window', 0) or getattr(config, 'sliding_window', 0) or 0)
    if types and any(kind not in MASK_KINDS for kind in types):
        raise ValueError('fixed slab has no sliding/hybrid attention support')
    hybrid = bool(types) and any(kind != 'full_attention' for kind in types)
    if not hybrid and model_type.startswith('gemma3') and window > 0:
        hybrid = True
    if hybrid and window <= 0:
        raise ValueError('fixed slab sliding attention needs a window')
    return model_type, hybrid, window


def tree_mask(torch, starts, keys, ancestors):
    """Physical append offsets are distinct from a tree node's logical depth."""
    q = ancestors.shape[0]
    relative = keys[None, :]-starts
    indices = relative.clamp(0, q-1)[:, None, :].expand(-1, q, -1)
    visible = ancestors[None, :, :].expand(starts.shape[0], -1, -1).gather(2, indices)
    query = (relative[:, None, :] >= 0) & (relative[:, None, :] < q) & visible
    return ((relative[:, None, :] < 0) | query)[:, None, :, :]


def bank_rows():
    """Row banks this slab captures. Default is the full 1/4/16 bank.

    ``TRIAGE_HF_FIXED_ROWS`` narrows the bank so a C1-only workload captures
    only one copy of each topology; every bank that is not declared becomes
    unserviceable, so a batch above the declared maximum is a startup/serving
    error rather than a silent fallback.
    """
    import os
    value = os.environ.get('TRIAGE_HF_FIXED_ROWS')
    if value is None:
        return (1, 4, 16)
    items = [item.strip() for item in str(value).split(',') if item.strip()]
    try:
        rows = tuple(sorted({int(item) for item in items}))
    except (TypeError, ValueError):
        raise ValueError('HF_FIXED_ROWS') from None
    if not rows or any(row not in (1, 4, 16) for row in rows):
        raise ValueError('HF_FIXED_ROWS')
    return rows


def bank_cap(capacity):
    """Largest declared bank that still fits the slab's physical capacity."""
    if type(capacity) is not int or not 1 <= capacity <= 16:
        raise ValueError('fixed slab needs 1..16 rows')
    declared = [size for size in bank_rows() if size <= capacity]
    if not declared:
        raise ValueError('HF_FIXED_ROWS declares no bank within %d rows' % (capacity,))
    return max(declared)


def bucket_size(rows):
    if type(rows) is not int or not 1 <= rows <= 16:
        raise ValueError('fixed slab needs 1..16 rows')
    declared = [size for size in bank_rows() if size >= rows]
    if not declared:
        raise ValueError('fixed slab row bank %r is not declared' % (rows,))
    return min(declared)


class FixedKV:
    """Duck-typed HF cache; the caller supplies masks and positions explicitly."""
    def __init__(self, backend, rows, context):
        self.layers = []
        self.positions = None
        self.context = context
        torch = backend.torch
        for _ in range(backend.layers):
            keys = torch.zeros(rows, backend.kv_heads, context, backend.head_dim,
                               dtype=backend.dtype, device=backend.device)
            self.layers.append(SimpleNamespace(keys=keys, values=torch.zeros_like(keys)))

    def update(self, keys, values, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        indices = self.positions[:, None, :, None].expand_as(keys)
        layer.keys.scatter_(2, indices, keys)
        layer.values.scatter_(2, indices, values)
        return layer.keys, layer.values

    def get_seq_length(self, layer_idx=0):
        # Explicit position IDs and masks carry the bank's logical positions.
        # Never infer per-row length from nonzero KV values.
        raise RuntimeError('fixed cache requires explicit per-row positions')


class FixedGraph:
    """One fixed KV bank and full topology; retain all replay storage."""
    def __init__(self, owner, cache, batch, parents):
        self.backend, self.cache, self.module = owner.backend, cache, owner.module
        self.parents = tuple(parents)
        q = len(self.parents)
        self.batch, self.q, self.context = batch, q, owner.context
        self.tree = self.parents != tuple(range(-1, q-1))
        # KVFlash's bottom-right causal mask cannot express an alternate root.
        # Real-side shapes deliberately use declared SDPA graphs, never an
        # operator-error fallback. Chain shapes retain their requested kernel.
        self.attention = 'sdpa' if self.tree else owner.attention
        # Hybrid masking is a property of the model, not of the graph shape.
        self.hybrid, self.window = owner.hybrid, owner.window
        torch = self.backend.torch
        self.depths = torch.tensor(depths(self.parents), dtype=torch.long, device=self.backend.device)
        self.ancestors = torch.tensor(ancestor_mask(self.parents), dtype=torch.bool, device=self.backend.device)
        self.inputs = torch.empty(batch, q+1, dtype=torch.long, device=self.backend.device)
        self.inputs[:, :q].fill_(1)
        self.inputs[:, q].fill_(8)
        self.steps = torch.arange(q, device=self.backend.device)
        self.keys = torch.arange(self.context, device=self.backend.device)
        self.sliding = None
        self.retained = ()
        stream = torch.cuda.Stream(device=self.backend.device)
        stream.wait_stream(torch.cuda.current_stream(self.backend.device))
        # Trace/autotune outside capture, with the exact packed input strides.
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self._forward()
        stream.synchronize()
        self.retained = ()
        self.cache.positions = None
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
            self.outputs = self._forward()
        torch.cuda.current_stream(self.backend.device).wait_stream(stream)

    def masks(self, torch, mask, positions):
        """Attention mask mapping handed to the model.

        Full-attention families get exactly the single key they had before. Hybrid
        families (gemma3) additionally get a sliding-window view of the same bank:
        a key is visible to a sliding layer only inside ``window`` tokens below the
        query. Both tensors stay alive for the replay through ``retained``.
        """
        if not self.hybrid:
            return {'full_attention': mask}
        self.sliding = mask & (self.keys[None, None, None, :] >
                               (positions[:, None, :, None] - self.window))
        return {'full_attention': mask, 'sliding_attention': self.sliding}

    def _forward(self):
        torch = self.backend.torch
        starts = self.inputs[:, self.q:self.q+1]
        writes = starts+self.steps[None, :]
        positions = starts+self.depths[None, :]
        mask = (tree_mask(torch, starts, self.keys, self.ancestors) if self.tree else
                self.keys[None, None, None, :] <= positions[:, None, :, None])
        self.cache.positions = writes
        metadata = fixed_kwargs(torch, positions, self.attention)
        masks = self.masks(torch, mask, positions)
        logits = self.module(input_ids=self.inputs[:, :self.q], past_key_values=self.cache,
                             position_ids=positions, attention_mask=masks,
                             use_cache=True, logits_to_keep=self.q, return_dict=True,
                             **metadata).logits
        predictions = logits.argmax(-1)
        finite = torch.isfinite(logits).all().to(torch.long)
        packed = torch.cat((predictions.reshape(-1), finite.reshape(1)))
        self.retained = (writes, positions, mask, self.sliding, logits, predictions, finite, metadata)
        return packed

    def replay(self, ids, lengths):
        torch = self.backend.torch
        host_values = [list(tokens)+[length] for tokens, length in zip(ids, lengths)]
        with torch.inference_mode():
            self.inputs.copy_(torch.tensor(host_values, dtype=torch.long))
            self.graph.replay()
        values = self.outputs.tolist()
        if values[-1] != 1:
            raise RuntimeError('nonfinite fixed graph logits')
        return [values[i*self.q:(i+1)*self.q] for i in range(self.batch)]


class FixedSlab:
    """Runtime Slab API, with fixed context and bucketed physical batch size."""
    def __init__(self, backend, slots, context, *, compile_model=False, use_graphs=False,
                 attention='sdpa', prefill_flash=False):
        config = backend.model.config
        model_type, hybrid, window = family_attention(config)
        if model_type not in SLAB_FAMILIES:
            raise ValueError('fixed slab currently requires Qwen3, Qwen2 or Gemma3')
        self.hybrid, self.window = hybrid, window
        if type(slots) is not int or not 1 <= slots <= 16:
            raise ValueError('fixed slab slot bound')
        if type(context) is not int or not 8 <= context <= backend.context:
            raise ValueError('fixed slab context bound')
        if use_graphs and backend.device.type != 'cuda':
            raise ValueError('CUDA graphs require a CUDA device')
        if attention not in ('sdpa', 'flash-kv'):
            raise ValueError('unknown fixed attention')
        if attention == 'flash-kv' or prefill_flash:
            # Finalize routing BEFORE any decode tracing. Prefill's separate
            # metadata still selects Flash, while metadata-free sdpa decode
            # remains SDPA. A later config flip would invalidate Dynamo guards.
            from .attention import install
            install(backend)
        self.attention = attention
        self.backend, self.capacity, self.context = backend, slots, context
        self.use_graphs = use_graphs
        from engines.common.growth import KS, enabled, parents as growth_parents
        self.adaptive = enabled()
        layouts = {(-1,), (-1, 0, 1, 2)}
        if self.adaptive:
            for k in KS:
                for side in (False, True):
                    layouts.add(growth_parents(k, side))
        # Equal query widths can have different ancestry and attention kernels.
        self.layouts = tuple(sorted(layouts, key=lambda ps: (len(ps), ps)))
        if context < 8+max(map(len, self.layouts)):
            raise ValueError('fixed slab context cannot fit the declared warm shapes')
        self.banks, self.graphs = {}, {}
        self._warming, self._graphs_ready, self._ready_event = False, False, None
        self.valid = [0]*slots
        self.order, self.cache = [], None
        self.batch = 0
        self.selected, self.query_width, self.query_parents = [], 0, ()
        self.stats = dict(forwards=0, rows=0, physical_rows=0, installs=0,
                          frees=0, commits=0, resizes=0, graph_replays=0,
                          tree_sdpa_forwards=0, nonlinear_commits=0)
        self.module = (backend.torch.compile(backend.model,
                       mode='max-autotune-no-cudagraphs', dynamic=False)
                       if compile_model else backend.model)
        self.compiled = compile_model
        self.warm_shapes = []

    def warm(self):
        """Run every declared shape without Dynamo's default eight-variant cutoff."""
        # Growth declares all negotiated topologies across the row banks. Exceeding
        # Dynamo's default limit must not silently warm eager instead. Scope the
        # raised, bounded allowance to startup; serving only replays these graphs.
        torch = self.backend.torch
        with torch._dynamo.config.patch(recompile_limit=64, accumulated_recompile_limit=256):
            self._warm()

    def _warm(self):
        import logging
        import time
        logger = logging.getLogger(__name__)
        torch = self.backend.torch
        self._warming = True
        cap = bank_cap(self.capacity)
        for rows in bank_rows():
            if rows > cap:
                continue
            live = min(rows, self.capacity)
            self.order = list(range(live))
            self.valid = [8 if slot < live else 0 for slot in range(self.capacity)]
            self._resize(live)
            for parents in self.layouts:
                q = len(parents)
                started = time.monotonic()
                logger.info('HF_FIXED_WARM_START batch=%d q=%d parents=%s context=%d',
                            rows, q, parents, self.context)
                self.predict([(slot, [1]*q) for slot in self.order], parents=parents)
                elapsed = time.monotonic()-started
                self.warm_shapes.append(dict(batch=rows, q=q, parents=list(parents),
                                             context=self.context, elapsed_s=elapsed))
                logger.info('HF_FIXED_WARM_DONE batch=%d q=%d parents=%s elapsed_s=%.3f',
                            rows, q, parents, elapsed)
        # These synthetic KV entries never become owned request state.
        if self.cache is not None:
            with torch.inference_mode():
                for layer in self.cache.layers:
                    layer.keys.zero_()
                    layer.values.zero_()
        self.order, self.selected = [], []
        self.valid = [0]*self.capacity
        for key in self.stats:
            self.stats[key] = 0
        self._record_mutation()
        self._warming, self._graphs_ready = False, True

    def _wait_mutation(self):
        if self.use_graphs and self._ready_event is not None:
            self.backend.torch.cuda.current_stream(self.backend.device).wait_event(self._ready_event)

    def _record_mutation(self):
        if self.use_graphs:
            self._ready_event = self.backend.torch.cuda.Event()
            self._ready_event.record(self.backend.torch.cuda.current_stream(self.backend.device))

    @property
    def phys(self):
        return max((self.valid[slot] for slot in self.order), default=0)

    def _resize(self, rows):
        batch = bucket_size(rows)
        if batch == self.batch:
            return
        self._wait_mutation()
        if self.use_graphs:
            if batch not in self.banks:
                self.banks[batch] = FixedKV(self.backend, batch, self.context)
            new = self.banks[batch]
        else:
            new = FixedKV(self.backend, batch, self.context)
        if self.cache is not None:
            count = min(self.batch, batch)
            with self.backend.torch.inference_mode():
                for old, dest in zip(self.cache.layers, new.layers):
                    dest.keys[:count].copy_(old.keys[:count])
                    dest.values[:count].copy_(old.values[:count])
        self.cache, self.batch = new, batch
        self.stats['resizes'] += 1
        self._record_mutation()

    def install(self, slot, cache, prefix):
        if type(slot) is not int or not 0 <= slot < self.capacity:
            raise ValueError('fixed slab slot bound')
        if self.use_graphs and not self._graphs_ready:
            raise RuntimeError('fixed graphs must be prewarmed before admitting requests')
        self._wait_mutation()
        length = self.backend.length(cache)
        if length != len(prefix) or length > self.context:
            raise ValueError('fixed slab install length mismatch')
        if slot not in self.order:
            self.order.append(slot)
        index = self.order.index(slot)
        self._resize(len(self.order))
        with self.backend.torch.inference_mode():
            for layer_index, layer in enumerate(self.cache.layers):
                layer.keys[index].zero_()
                layer.values[index].zero_()
                if length:
                    layer.keys[index, :, :length].copy_(cache.layers[layer_index].keys[0])
                    layer.values[index, :, :length].copy_(cache.layers[layer_index].values[0])
        self.valid[slot] = length
        self.stats['installs'] += 1
        self._record_mutation()

    def free(self, slot):
        if type(slot) is not int or not 0 <= slot < self.capacity:
            raise ValueError('fixed slab free slot bound')
        if slot not in self.order:
            return
        self._wait_mutation()
        index, last = self.order.index(slot), len(self.order)-1
        with self.backend.torch.inference_mode():
            for layer in self.cache.layers:
                if index != last:
                    layer.keys[index].copy_(layer.keys[last])
                    layer.values[index].copy_(layer.values[last])
                layer.keys[last].zero_()
                layer.values[last].zero_()
        self.order[index] = self.order[last]
        self.order.pop()
        self.valid[slot] = 0
        self.stats['frees'] += 1
        self._record_mutation()

    def predict(self, rows, keep=None, *, parents=None):
        if not rows:
            raise ValueError('fixed slab forward requires rows')
        selected = {}
        q = len(rows[0][1])
        if not q or (keep is not None and not 1 <= keep <= q):
            raise ValueError('fixed slab invalid query width')
        parents = tuple(range(-1, q-1)) if parents is None else tuple(parents)
        validate(parents)
        if len(parents) != q:
            raise ValueError('fixed slab parent / query width mismatch')
        is_tree = parents != tuple(range(-1, q-1))
        if self.use_graphs and (parents not in self.layouts or keep not in (None, q)):
            raise ValueError('fixed graph only declares full q1/q4 logits' if not self.adaptive
                             else 'fixed graph undeclared growth topology')
        if self.use_graphs and not (self._graphs_ready or self._warming):
            raise RuntimeError('fixed graphs must be prewarmed before predict')
        for slot, tokens in rows:
            if slot not in self.order or slot in selected or len(tokens) != q:
                raise ValueError('fixed slab row binding/width')
            selected[slot] = list(tokens)
        if any(self.valid[slot]+q > self.context for slot in self.order):
            raise ValueError('fixed slab query exceeds context')
        self._resize(len(self.order))
        self._wait_mutation()
        torch, backend = self.backend.torch, self.backend
        ids = [selected.get(slot, [0]*q) for slot in self.order]
        lengths = [self.valid[slot] for slot in self.order]
        ids += [[0]*q for _ in range(self.batch-len(ids))]
        lengths += [0]*(self.batch-len(lengths))
        if self.use_graphs:
            key = (self.batch, parents)
            if key not in self.graphs:
                if not self._warming:
                    raise RuntimeError('undeclared fixed graph during serving')
                self.graphs[key] = FixedGraph(self, self.cache, self.batch, parents)
            predictions = self.graphs[key].replay(ids, lengths)
            self.stats['graph_replays'] += 1
        else:
            input_ids = torch.tensor(ids, dtype=torch.long, device=backend.device)
            starts = torch.tensor(lengths, dtype=torch.long, device=backend.device)
            writes = starts[:, None]+torch.arange(q, device=backend.device)[None, :]
            positions = starts[:, None]+torch.tensor(depths(parents), device=backend.device)[None, :]
            physical_keys = torch.arange(self.context, device=backend.device)
            mask = (tree_mask(torch, starts[:, None], physical_keys,
                              torch.tensor(ancestor_mask(parents), dtype=torch.bool, device=backend.device))
                    if is_tree else physical_keys[None, None, None, :] <= positions[:, None, :, None])
            self.cache.positions = writes
            count = q if keep is None else keep
            with torch.inference_mode():
                output = self.module(input_ids=input_ids, past_key_values=self.cache,
                                     position_ids=positions,
                                     attention_mask={'full_attention': mask},
                                     use_cache=True, logits_to_keep=count, return_dict=True,
                                     **fixed_kwargs(torch, positions, 'sdpa' if is_tree else self.attention))
            logits = output.logits[:, -count:, :]
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError('nonfinite fixed slab logits')
            predictions = logits.argmax(-1).tolist()
        self._record_mutation()
        self.selected, self.query_width, self.query_parents = list(selected), q, parents
        self.stats['tree_sdpa_forwards'] += int(is_tree)
        self.stats['forwards'] += 1
        self.stats['rows'] += len(rows)
        self.stats['physical_rows'] += self.batch
        return [predictions[self.order.index(slot)] for slot, _ in rows]

    def commit(self, slot, offsets):
        if slot not in self.selected:
            raise ValueError('fixed slab commit outside selected rows')
        if (type(offsets) is not list or not offsets or
                any(type(i) is not int or not 0 <= i < self.query_width for i in offsets) or
                offsets != path_to(self.query_parents, offsets[-1])):
            raise ValueError('fixed slab requires one canonical accepted path')
        if offsets != list(range(len(offsets))):
            self._wait_mutation()
            index, start = self.order.index(slot), self.valid[slot]
            torch = self.backend.torch
            source = torch.tensor([start+i for i in offsets], dtype=torch.long, device=self.backend.device)
            with torch.inference_mode():
                for layer in self.cache.layers:
                    keys = layer.keys[index].index_select(1, source)
                    values = layer.values[index].index_select(1, source)
                    layer.keys[index, :, start:start+len(offsets)].copy_(keys)
                    layer.values[index, :, start:start+len(offsets)].copy_(values)
            self._record_mutation()  # ownership extends THROUGH destination copies
            self.stats['nonlinear_commits'] += 1
        self.valid[slot] += len(offsets)
        self.selected.remove(slot)
        self.stats['commits'] += 1

    def trim(self):
        # Never slice fixed-capacity storage: future/rejected KV is masked out.
        pass

    def status(self):
        return dict(slots=self.capacity, active=list(self.order), valid=list(self.valid),
                    physical=self.phys, batch=self.batch, context=self.context,
                    batching='FIXED_SCATTER_BUCKETS_1_4_16', compiled=self.compiled,
                    attention=self.attention, tree_attention='DECLARED_ANCESTOR_SDPA',
                    adaptive_growth=self.adaptive,
                    declared_query_widths=sorted({len(ps) for ps in self.layouts}),
                    declared_topologies=[dict(q=len(ps), parents=list(ps)) for ps in self.layouts],
                    cuda_graphs=self.use_graphs,
                    graph_shapes=[[batch, len(ps)] for batch, ps in self.graphs],
                    graph_topologies=[dict(batch=batch, q=len(ps), parents=list(ps))
                                      for batch, ps in self.graphs],
                    warm_shapes=list(self.warm_shapes), stats=dict(self.stats))

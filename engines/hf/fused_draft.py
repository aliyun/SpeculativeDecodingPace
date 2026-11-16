from types import SimpleNamespace
from .attention import fixed_kwargs, install
from .fixed_slab import FixedKV


class DraftGraph:
    def __init__(self, owner, handle, fill, k=3, root_top2=False):
        self.owner, self.handle, self.fill = owner, handle, fill
        self.k, self.root_top2 = k, root_top2
        torch = owner.backend.torch
        self.inputs = torch.tensor([[1, 8, 1]], dtype=torch.long, device=owner.backend.device)
        self.keys = torch.arange(owner.context, device=owner.backend.device)
        self.retained = []
        stream = torch.cuda.Stream(device=owner.backend.device)
        stream.wait_stream(torch.cuda.current_stream(owner.backend.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3): self._forward()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
            self.outputs = self._forward()
        torch.cuda.current_stream(owner.backend.device).wait_stream(stream)

    def _call(self, ids, positions, *, fill=False):
        self.handle.cache.positions = positions
        mask = self.keys[None, None, None, :] <= positions[:, None, :, None]
        kwargs = dict(input_ids=ids, past_key_values=self.handle.cache,
                      position_ids=positions, attention_mask={'full_attention': mask},
                      use_cache=True, return_dict=True,
                      **fixed_kwargs(self.owner.backend.torch, positions, 'flash-kv'))
        if fill:
            result = self.owner.backbone(**kwargs).last_hidden_state
        else:
            result = self.owner.module(**kwargs, logits_to_keep=1).logits
        self.retained.append((positions, mask, kwargs, result))
        return result

    def _forward(self):
        torch = self.owner.backend.torch
        self.retained = []
        start = self.inputs[:, 1:2]
        current = self.inputs[:, :1]
        if self.fill:
            hidden = self._call(self.inputs[:, 2:3], start, fill=True)
            finite = torch.isfinite(hidden).all()
            start = start+1
        else:
            finite = torch.ones((), dtype=torch.bool, device=self.owner.backend.device)
        tokens, root_metadata = [], []
        for i in range(self.k):
            logits = self._call(current, start+i)
            current = logits.argmax(-1)
            tokens.append(current)
            finite = finite & torch.isfinite(logits).all()
            if i == 0 and self.root_top2:
                # Tie order must agree with greedy argmax, not topk's tie order.
                masked = logits.scatter(-1, current.unsqueeze(-1), -float('inf'))
                alternate = masked.argmax(-1)
                gap = (logits.gather(-1, current.unsqueeze(-1)).float()-
                       logits.gather(-1, alternate.unsqueeze(-1)).float()).reshape(1)
                bits = gap.view(torch.int32).to(torch.long) & 0xffffffff
                root_metadata = [alternate.reshape(1), bits]
                finite = finite & torch.isfinite(gap).all() & (gap >= 0).all()
                self.retained.append((masked, alternate, gap, bits))
        self.predictions = tokens  # keep every intermediate, not reused aliases
        return torch.cat([t.reshape(-1) for t in tokens]+root_metadata+[finite.long().reshape(1)])

    def replay(self, root, valid, tail):
        torch = self.owner.backend.torch
        with torch.inference_mode():
            self.inputs.copy_(torch.tensor([[root, valid, tail or 0]], dtype=torch.long))
            self.graph.replay()
        out = self.outputs.tolist()
        if out[-1] != 1:
            raise RuntimeError('nonfinite fused draft graph result')
        if self.root_top2:
            from engines.common.root_shadow import decode_top2
            return out[:self.k], decode_top2(out[0], out[self.k], out[self.k+1])
        return out[:self.k]


class FusedDraftPool:
    """Caller serializes dispatch; events also protect cross-thread CUDA streams."""
    def __init__(self, backend, capacity, context):
        cfg = backend.model.config
        # Full-attention Qwen2 and Qwen3 share this KV view. Hybrid families refuse.
        if getattr(cfg, 'model_type', None) not in ('qwen3', 'qwen2') or any(
                t != 'full_attention' for t in (getattr(cfg, 'layer_types', None) or [])):
            raise ValueError('fused draft requires Qwen3 or Qwen2 full attention')
        if type(capacity) is not int or not 1 <= capacity <= 16:
            raise ValueError('fused draft pool bound')
        if type(context) is not int or not 256 <= context <= backend.context:
            raise ValueError('fused draft context bound')
        install(backend)  # explicit CUDA/BF16, missing extension is not fallback
        if capacity*context > backend.kv_budget_tokens(fraction=.75):
            raise ValueError('fused draft bank exceeds device budget')
        self.backend, self.capacity, self.context = backend, capacity, context
        from engines.common.growth import KS, enabled
        self.adaptive = enabled()
        self.actions = ([(k, root) for k in KS for root in (False, True)]
                        if self.adaptive else [(3, False)])
        self.module = backend.torch.compile(backend.model, mode='max-autotune-no-cudagraphs', dynamic=False)
        self.backbone = backend.torch.compile(backend.model.model, mode='max-autotune-no-cudagraphs', dynamic=False)
        self.handles, self.available = [], []
        self.stats = dict(rounds=0, graph_replays=0, fill_rounds=0, rebuilds=0, rewinds=0,
                          allocations=0, frees=0)
        import logging
        logger = logging.getLogger(__name__)
        for slot in range(capacity):
            handle = SimpleNamespace(cache=FixedKV(backend, 1, context), valid=0,
                                     tail=None, active=False, ready=None, graphs={})
            self.handles.append(handle)
            for k, root_top2 in self.actions:
                for fill in (False, True):
                    logger.info('HF_DRAFT_GRAPH_WARM_START slot=%d fill=%s k=%d roots=%d',
                                slot, fill, k, 2 if root_top2 else 1)
                    key = fill if (k, root_top2) == (3, False) else (fill, k, root_top2)
                    handle.graphs[key] = DraftGraph(self, handle, fill, k, root_top2)
                    logger.info('HF_DRAFT_GRAPH_WARM_DONE slot=%d fill=%s k=%d roots=%d',
                                slot, fill, k, 2 if root_top2 else 1)
            self._clear(handle)
            self.available.append(handle)

    def _wait(self, handle):
        if handle.ready is not None:
            self.backend.torch.cuda.current_stream(self.backend.device).wait_event(handle.ready)

    def _record(self, handle):
        handle.ready = self.backend.torch.cuda.Event()
        handle.ready.record(self.backend.torch.cuda.current_stream(self.backend.device))

    def _clear(self, handle):
        self._wait(handle)
        with self.backend.torch.inference_mode():
            for layer in handle.cache.layers:
                layer.keys.zero_(); layer.values.zero_()
        handle.valid, handle.tail = 0, None
        self._record(handle)

    def _check(self, handle):
        if not any(h is handle for h in self.handles) or not handle.active:
            raise ValueError('inactive or foreign fused draft handle')

    def allocate(self):
        if not self.available:
            raise ValueError('fused draft pool exhausted')
        handle = self.available.pop()
        self._wait(handle)
        handle.active = True
        self.stats['allocations'] += 1
        return handle

    def install(self, handle, prefix):
        self._check(handle)
        if len(prefix)+3 > self.context:
            raise ValueError('fused draft prefix exceeds cache bound')
        self._wait(handle)
        cache = self.backend.new_cache()
        self.backend.prefill(cache, prefix)
        with self.backend.torch.inference_mode():
            for i, dest in enumerate(handle.cache.layers):
                dest.keys.zero_(); dest.values.zero_()
                if prefix:
                    dest.keys[0, :, :len(prefix)].copy_(cache.layers[i].keys[0])
                    dest.values[0, :, :len(prefix)].copy_(cache.layers[i].values[0])
        handle.valid, handle.tail = len(prefix), None
        self.stats['rebuilds'] += 1
        self._record(handle)

    def align(self, handle, prefix, common):
        self._check(handle)
        desired = len(prefix)-1
        if len(prefix)+3 > self.context:
            raise ValueError('fused draft prefix exceeds cache bound')
        if not 0 <= common <= min(handle.valid, desired):
            raise ValueError('fused draft invalid common prefix')
        gap = desired-common
        if gap > 1:
            self.install(handle, prefix[:-1])
        else:
            if common < handle.valid: self.stats['rewinds'] += 1
            handle.valid = common
            handle.tail = prefix[common] if gap else None

    def draft(self, handle, root, *, k=3, root_top2=False):
        self._check(handle)
        if type(root) is not int or not 0 <= root < self.backend.vocab_size:
            raise ValueError('fused draft root outside vocabulary')
        if type(k) is not int or type(root_top2) is not bool or (k, root_top2) not in self.actions:
            raise ValueError('undeclared fused draft action')
        if handle.valid+int(handle.tail is not None)+k > self.context:
            raise ValueError('fused draft query exceeds cache bound')
        self._wait(handle)
        fill = handle.tail is not None
        key = fill if (k, root_top2) == (3, False) else (fill, k, root_top2)
        result = handle.graphs[key].replay(root, handle.valid, handle.tail)
        handle.valid += k+int(fill)
        handle.tail = None
        self._record(handle)
        self.stats['rounds'] += 1; self.stats['graph_replays'] += 1
        self.stats['fill_rounds'] += int(fill)
        action = 'k%d_roots%d_rounds' % (k, 2 if root_top2 else 1)
        self.stats[action] = self.stats.get(action, 0)+1
        return result

    def free(self, handle):
        self._check(handle)
        self._clear(handle)
        handle.active = False
        self.available.append(handle)
        self.stats['frees'] += 1

    def status(self):
        return dict(capacity=self.capacity, free=len(self.available), context=self.context,
                    cuda_graph=True, fused_steps=3, attention='flash-kv',
                    graph_variants=[('FILL1_' if fill else '')+'K%d'%k+('_ROOT2' if root else '')
                                    for k, root in self.actions for fill in (False, True)],
                    adaptive_growth=self.adaptive, stats=dict(self.stats))

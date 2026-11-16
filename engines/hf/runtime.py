import os

from engines.common.policy import (CONTEXT, DRAFT_TOKENS, MAX_OUTPUT, MAX_PROMPT,
                                   MAX_REQUESTS, PARENTS, WIDTH, common_prefix,
                                   fingerprint, token_ids)
from engines.common.tree import accept
from engines.common.contracts import EdgeAdmissionError


class ChainRuntime:
    backend = None

    def __init__(self, backend, *, role, context=CONTEXT, max_rows=MAX_REQUESTS,
                 kv_budget_tokens=None):
        if DRAFT_TOKENS != 3:
            raise ValueError('HF_CHAIN_REQUIRES_FIXED_K3')
        if role not in ('cloud', 'edge'):
            raise ValueError('invalid HF runtime role')
        if type(context) is not int or context < WIDTH+1:
            raise ValueError('invalid HF runtime context')
        if type(max_rows) is not int or not 1 <= max_rows <= MAX_REQUESTS:
            raise ValueError('invalid HF runtime row bound')
        from engines.common.growth import enabled
        self.growth = enabled()
        self.verify_width = 9 if self.growth else WIDTH
        self.growth_calls = {}
        self.backend, self.role = backend, role
        self.context, self.max_rows = context, max_rows
        self.requests = {}   # cloud: committed prefix, slab slot and reservation
        self.sessions = {}   # edge: cached context and last committed prefix
        self.fixed_slab = role == 'cloud' and os.environ.get('TRIAGE_HF_FIXED_SLAB') == '1'
        # The single-row dual-cache experiment and fixed batched bank are distinct.
        self.cloud_static = role == 'cloud' and os.environ.get('TRIAGE_HF_CLOUD_STATIC') == '1'
        if self.growth and self.cloud_static:
            raise ValueError('adaptive HF requires a shared slab, not legacy single-row static cloud')
        if self.fixed_slab and self.cloud_static:
            raise ValueError('fixed batched slab and single-row static mode are exclusive')
        self.slab = backend.new_slab(max_rows) if role == 'cloud' else None
        self.request_context = self.slab.context if self.fixed_slab else context
        self.prefill_first = role == 'cloud' and os.environ.get('TRIAGE_HF_PREFILL_FIRST') == '1'
        if self.prefill_first and not self.fixed_slab:
            raise ValueError('prefill-first requires the fixed HF cloud bank')
        prefill_buckets = os.environ.get('TRIAGE_HF_PREFILL_BUCKETS') if role == 'cloud' else None
        self.prefill_graphs = None
        if prefill_buckets is not None:
            if not self.prefill_first:
                raise ValueError('prefill graphs require prefill-first')
            self.prefill_graphs = backend.new_prefill_graphs(prefill_buckets, self.request_context)
        self.free_slots = list(range(max_rows))
        # Cloud may verify a single row through its own fixed-length StaticCache
        # (compiled q=WIDTH) and fall back to the shared slab for batches > 1.
        try:
            self.cloud_cache_len = int(os.environ.get('TRIAGE_HF_CLOUD_CACHE', '512'))
        except ValueError as exc:
            raise ValueError('invalid TRIAGE_HF_CLOUD_CACHE') from exc
        if self.cloud_static:
            if not 256 <= self.cloud_cache_len <= context:
                raise ValueError('cloud static cache length out of range')
            self.request_context = self.cloud_cache_len
            if hasattr(backend, 'compile_verify'):
                backend.compile_verify(WIDTH, cache_length=self.cloud_cache_len)
        # Optional edge path: fixed-length StaticCache plus one compiled q=1 step.
        self.static = role == 'edge' and os.environ.get('TRIAGE_HF_STATIC') == '1'
        self.fused = role == 'edge' and os.environ.get('TRIAGE_HF_FUSED_DRAFT') == '1'
        if os.environ.get('TRIAGE_HF_TRACE_ENV') == '1':
            # Opt-in trace of the role's resolved startup flags.
            print('HF_ENV_TRACE role=%r fused_env=%r static_env=%r fused=%r static=%r'
                  % (role, os.environ.get('TRIAGE_HF_FUSED_DRAFT'),
                     os.environ.get('TRIAGE_HF_STATIC'), self.fused, self.static), flush=True)
        if self.fused and self.static:
            raise ValueError('fused draft and legacy StaticCache modes are exclusive')
        self.draft_pool = None
        if self.fused:
            try:
                capacity = int(os.environ.get('TRIAGE_HF_DRAFT_POOL', str(min(4, max_rows))))
                bound = int(os.environ.get('TRIAGE_HF_DRAFT_CONTEXT', '1024'))
            except ValueError as exc:
                raise ValueError('invalid fused draft cache bounds') from exc
            if not 1 <= capacity <= max_rows or not 256 <= bound <= context:
                raise ValueError('fused draft cache bounds')
            self.draft_pool = backend.new_fused_draft(capacity, bound)
            self.request_context = self.draft_pool.context
        try:
            self.edge_cache_len = int(os.environ.get('TRIAGE_HF_EDGE_CACHE', '1024'))
        except ValueError as exc:
            raise ValueError('invalid TRIAGE_HF_EDGE_CACHE') from exc
        self.edge_capacity = self.draft_pool.capacity if self.fused else max_rows
        self.static_pool = []
        if self.static:
            if not 256 <= self.edge_cache_len <= context:
                raise ValueError('edge static cache length out of range')
            if hasattr(backend, 'compile_draft') and os.environ.get('TRIAGE_HF_COMPILE') == '1':
                backend.compile_draft()
            self.compiled_static = getattr(backend, 'draft_model', None) is not None
            try:
                pool = int(os.environ.get('TRIAGE_HF_EDGE_POOL', str(max_rows)))
            except ValueError as exc:
                raise ValueError('invalid TRIAGE_HF_EDGE_POOL') from exc
            if not 1 <= pool <= max_rows:
                raise ValueError('edge static cache pool bound')
            self.edge_capacity = pool
            self.request_context = self.edge_cache_len
            # Allocate and warm every pooled cache before the service is ready:
            # allocation and first-trace costs must not land on request paths.
            for _ in range(pool):
                cache = backend.static_cache(self.edge_cache_len)
                backend.static_prefill(cache, [1, 2, 3, 4], 0)
                backend.static_predict(cache, [1], 4, compiled=self.compiled_static)
                backend.static_rewind(cache, 0)
                self.static_pool.append(cache)
        else:
            self.compiled_static = False
        self.reserved_tokens = 0
        self.calls = dict(admit=0, first_token=0, cached_first_tokens=0, verify=0, ar_step=0, sync_draft=0,
                          warm=0, free=0, draft_tokens=0, verified_positions=0,
                          accepted_drafts=0, rejected_drafts=0, prefills=0,
                          batch_forwards=0, batch_rows=0)
        self.kv_budget_tokens = (kv_budget_tokens if kv_budget_tokens is not None
                                 else (backend.kv_budget_tokens()
                                       if hasattr(backend, 'kv_budget_tokens') else 1 << 20))
        if type(self.kv_budget_tokens) is not int or self.kv_budget_tokens < 0:
            raise ValueError('invalid HF KV token budget')

    # ---------------- shared ----------------

    def status(self):
        live = len(self.requests) if self.role == 'cloud' else len(self.sessions)
        capacity = self.edge_capacity if self.role == 'edge' else self.max_rows
        result = dict(role=self.role, engine='transformers', requests=len(self.requests),
                      sessions=len(self.sessions),
                      free_blocks=capacity-live,
                      capacity=capacity,
                      block_size=1, context=self.context,
                      draft_tokens=DRAFT_TOKENS, width=WIDTH, parents=list(PARENTS),
                      verify='GREEDY_CHAIN_CAUSAL_FORWARD',
                      kv=('FUSED_K3_FIXED_BANK' if self.fused else
                          'FIXED_SCATTER_BANK' if self.fixed_slab else
                          ('STATIC_CACHE_%d' % self.edge_cache_len) if self.static else
                          ('STATIC_CACHE_%d' % self.cloud_cache_len) if self.cloud_static else
                          ('PER_REQUEST_DYNAMIC_CACHE' if self.role == 'edge' else 'PADDED_SHARED_SLAB')),
                      batching=('PER_ROW_K3_CUDA_GRAPH' if self.fused else
                                'FIXED_SCATTER_BUCKETS_1_4_16' if self.fixed_slab else
                                'PER_ROW_FORWARD' if self.role == 'edge' else 'PADDED_SLAB_BATCH'),
                      effective_request_context=self.request_context,
                      tree=False, cuda_graph=(self.fused or self.fixed_slab and getattr(self.slab, 'use_graphs', False)),
                      fused_steps=self.fused, prefill_first_token=self.prefill_first,
                      decode_attention=(getattr(self.slab, 'attention', 'sdpa') if self.role == 'cloud' else
                                        'flash-kv' if self.fused else getattr(self.backend, 'attention', 'sdpa')),
                      prefill_attention=('flash-kv' if self.prefill_graphs is not None else
                                         getattr(self.backend, 'attention', 'sdpa')),
                      kv_reserved_tokens=self.reserved_tokens, kv_budget_tokens=self.kv_budget_tokens,
                      calls=dict(self.calls))
        if self.slab is not None:
            result['slab'] = self.slab.status()
        if self.draft_pool is not None:
            result['draft_pool'] = self.draft_pool.status()
        if self.prefill_graphs is not None:
            result['prefill_graphs'] = self.prefill_graphs.status()
        if self.growth:
            from engines.common.growth import capability
            result.update(growth=capability(), tree=True, verify='GREEDY_GROWTH_TREE',
                          growth_calls=dict(self.growth_calls), verify_batch_max=8,
                          max_query_width=9)
        return result

    def dispatch(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('invalid HF runtime payload')
        op = payload.get('op')
        if op == 'status':
            return self.status()
        if op in ('validate', 'configure_compaction'):
            return {op: 'NO_OP_FOR_TRANSFORMERS'}
        if op == 'configure_tree':
            mode = payload.get('mode')
            if mode != ('fork5' if self.growth else 'chain4'):
                raise ValueError('HF_TREE_MODES_UNSUPPORTED')
            return {'tree_mode': mode}
        if op == 'prepare_graphs':
            return {'enabled': bool(self.slab is not None and getattr(self.slab, 'use_graphs', False)),
                    'scope': 'STARTUP_DECLARED_HF_SHAPES_ONLY'}
        if op == 'export_verify_profile':
            return []
        if self.role == 'cloud':
            if op == 'admit':
                return self.admit(payload['requests'])
            if op == 'first_token':
                return self.single_step(payload['requests'], 'first_token')
            if op == 'ar_step':
                return self.single_step(payload['requests'], 'ar_step')
            if op == 'verify':
                return self.verify(payload['requests'])
            if op == 'free':
                return self.free(payload.get('ids'))
            raise ValueError('invalid HF cloud operation: '+str(op))
        if op == 'sync_draft':
            return self.sync_draft(payload['requests'])
        if op == 'warm':
            return self.warm(payload['requests'])
        if op == 'free':
            return self.free(payload.get('ids'))
        raise ValueError('invalid HF edge operation: '+str(op))

    def _ids(self, payload_ids, bound=None):
        if (not isinstance(payload_ids, list) or len(payload_ids) != len(set(payload_ids))
                or (bound is not None and len(payload_ids) > bound)
                or any(not isinstance(rid, str) or not rid for rid in payload_ids)):
            raise ValueError('invalid HF request id batch')
        return payload_ids

    # ---------------- cloud ----------------

    def _bind(self, item, vocabulary, *, allow_prefill=False):
        rid = item.get('id')
        if not isinstance(rid, str) or rid not in self.requests:
            raise ValueError('unknown HF request')
        row = self.requests[rid]
        if fingerprint(row['tokens']) != item.get('prefix_hash'):
            raise ValueError('STALE_REQUEST_PREFIX')
        prefilled = row.get('prefill_token') is not None
        if prefilled and not allow_prefill:
            raise ValueError('consume prefilled first token before decoding')
        expected = len(row['tokens'])-int(not prefilled)
        if row.get('slab_synced', True) and self.slab.valid[row['slot']] != expected:
            raise ValueError('HF slab / prefix drift')
        return row

    def _sync_slab(self, row):
        if row.get('slab_synced', True):
            return
        cache = self.backend.new_cache()
        if len(row['tokens']) > 1:
            self.backend.prefill(cache, row['tokens'][:-1])
        self.slab.install(row['slot'], cache, row['tokens'][:-1])
        row['slab_synced'] = True

    def admit(self, requests):
        if not isinstance(requests, list) or not requests:
            raise ValueError('invalid HF admission batch')
        accepted = []
        for item in requests:
            if not isinstance(item, dict):
                raise ValueError('invalid HF admission row')
            rid, tokens, maximum = item.get('id'), item.get('tokens'), item.get('max_tokens')
            if not isinstance(rid, str) or not rid:
                raise ValueError('invalid HF request id')
            if rid in self.requests:
                raise ValueError('DUPLICATE_REQUEST')
            tokens = token_ids(tokens)
            if (self.cloud_static and type(maximum) is int and
                    len(tokens)+maximum+self.verify_width > self.cloud_cache_len):
                raise ValueError('cloud static cache bound')
            if any(t >= self.backend.vocab_size for t in tokens):
                raise ValueError('HF admission prefix outside vocabulary')
            if (type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT
                    or len(tokens) > MAX_PROMPT
                    or len(tokens)+maximum+self.verify_width > self.request_context):
                raise ValueError('invalid HF request budget')
            reserved = len(tokens)+maximum+self.verify_width
            if (not self.free_slots or len(self.requests) >= self.max_rows
                    or self.reserved_tokens+reserved > self.kv_budget_tokens):
                break
            slot = self.free_slots.pop(0)
            cache = self.backend.new_cache()
            cached_first = None
            try:
                if self.prefill_first:
                    if self.prefill_graphs is not None:
                        cached_first = self.prefill_graphs.fill_slot(self.slab, slot, tokens)
                    else:
                        cached_first = self.backend.prefill_first(cache, tokens)
                    if type(cached_first) is not int or not 0 <= cached_first < self.backend.vocab_size:
                        raise ValueError('prefilled first token outside vocabulary')
                    self.calls['prefills'] += 1
                    # All prompt KV remains VALID until the token is consumed.
                    # Lowering valid by one here would let another row's dummy
                    # forward overwrite the final real prompt KV.
                    if self.prefill_graphs is None:
                        self.slab.install(slot, cache, tokens)
                else:
                    if len(tokens) > 1:
                        self.backend.prefill(cache, tokens[:-1])
                        self.calls['prefills'] += 1
                    self.slab.install(slot, cache, tokens[:-1])
            except BaseException:
                self.free_slots.insert(0, slot)
                self.slab.free(slot)
                raise
            row = dict(tokens=tokens, maximum=maximum, slot=slot,
                       reserved=reserved, pending=None,
                       static_synced=False, slab_synced=True, prefill_token=cached_first)
            if self.cloud_static:
                if len(tokens)+maximum+WIDTH > self.cloud_cache_len:
                    raise ValueError('cloud static cache bound')
                cache = self.backend.static_cache(self.cloud_cache_len)
                self.backend.static_prefill(cache, tokens[:-1], 0)
                row.update(cache=cache, s_position=len(tokens)-1, static_synced=True)
            self.requests[rid] = row
            self.reserved_tokens += reserved
            accepted.append(rid)
        self.calls['admit'] += len(accepted)
        return {'admitted': accepted}

    def _commit(self, row, path, emitted, nodes):
        if len(path) != len(emitted):
            raise ValueError('HF verify path / emitted mismatch')
        if emitted[:-1] != [nodes[i] for i in path[1:]]:
            raise ValueError('HF verify accepted nodes mismatch')
        new_tokens = row['tokens']+list(emitted)
        if len(new_tokens)-1 > self.context:
            raise ValueError('HF committed prefix exceeds context')
        self.slab.commit(row['slot'], list(path))
        if self.slab.valid[row['slot']] != len(new_tokens)-1:
            raise ValueError('HF slab commit length')
        row['tokens'] = new_tokens
        row['pending'] = None
        return new_tokens

    def _commit_static(self, row, path, emitted, nodes):
        if len(path) != len(emitted):
            raise ValueError('HF verify path / emitted mismatch')
        if emitted[:-1] != [nodes[i] for i in path[1:]]:
            raise ValueError('HF verify accepted nodes mismatch')
        new_tokens = row['tokens']+list(emitted)
        if len(new_tokens)-1 > self.cloud_cache_len:
            raise ValueError('HF committed prefix exceeds the static cache')
        # The forward wrote all nodes; keep only the accepted prefix so rejected
        # drafts cannot be attended by later steps.
        self.backend.static_rewind(row['cache'], len(new_tokens)-1)
        row['s_position'] = len(new_tokens)-1
        if self.backend.static_length(row['cache']) != row['s_position']:
            raise ValueError('HF static position mismatch')
        row['tokens'] = new_tokens
        row['pending'] = None
        row['slab_synced'] = False
        return new_tokens

    def _native_shape(self):
        """Actual bank, including a fixed bank retained after a free; not active B."""
        slab = self.slab
        if (not self.growth or os.environ.get('TRIAGE_STREAM_SELECTOR') != '1'
                or slab is None or not hasattr(slab, 'order')):
            return {}
        shape = dict(engine='hf', layout='fixed_bank' if self.fixed_slab else 'resident_rows',
                     physical_rows=slab.batch if self.fixed_slab else len(slab.order),
                     execution=('graph' if getattr(slab, 'use_graphs', False) else
                                'compiled' if getattr(slab, 'compiled', False) else 'eager'),
                     context=self.request_context)
        return dict(native_shape=shape)

    def single_step(self, requests, phase):
        if not isinstance(requests, list) or not requests:
            raise ValueError('invalid HF single-token batch')
        rows, bound = [], []
        for item in requests:
            row = self._bind(item, self.backend.vocab_size, allow_prefill=phase == 'first_token')
            if row['pending'] is not None:
                raise ValueError('single-token step with uncommitted speculation')
            self._sync_slab(row)
            if row.get('prefill_token') is None:
                rows.append((row['slot'], [row['tokens'][-1]]))
            bound.append((item, row))
        predictions = {}
        if rows:
            predictions = dict(zip((slot for slot, _ in rows), self.slab.predict(rows)))
            self.calls['batch_forwards'] += 1
            self.calls['batch_rows'] += len(rows)
        output = []
        for item, row in bound:
            cached = row.get('prefill_token')
            token = cached if cached is not None else predictions[row['slot']][0]
            nodes = [row['tokens'][-1]]
            path, emitted = [0], [token]
            if cached is not None:
                new_tokens = row['tokens']+emitted
                if self.slab.valid[row['slot']] != len(new_tokens)-1:
                    raise ValueError('prefilled first token / KV drift')
                row['tokens'], row['prefill_token'] = new_tokens, None
                self.calls['cached_first_tokens'] += 1
            else:
                new_tokens = self._commit(row, path, emitted, nodes)
            if self.cloud_static:
                row['static_synced'] = False
            output.append(dict(id=item['id'], path=path, emitted=emitted,
                               prefix_hash=item['prefix_hash'],
                               next_prefix_hash=fingerprint(new_tokens), **self._native_shape()))
            self.calls[phase] += 1
        self.slab.trim()
        return output

    def verify(self, requests):
        if self.growth:
            return self._verify_growth(requests)
        if not isinstance(requests, list) or not requests:
            raise ValueError('invalid HF verify batch')
        if self.cloud_static and len(requests) == 1:
            item = requests[0]
            row = self._bind(item, self.backend.vocab_size)
            nodes = item.get('nodes')
            if (not isinstance(nodes, (list, tuple)) or len(nodes) != WIDTH
                    or any(type(token) is not int or not 0 <= token < self.backend.vocab_size
                           for token in nodes)
                    or nodes[0] != row['tokens'][-1]):
                raise ValueError('invalid HF verify nodes')
            if row['pending'] is not None:
                raise ValueError('HF verify on uncommitted speculation')
            nodes = list(nodes)
            row['pending'] = nodes
            if not row.get('static_synced', False):
                cache = self.backend.static_cache(self.cloud_cache_len)
                self.backend.static_prefill(cache, row['tokens'][:-1], 0)
                row.update(cache=cache, s_position=len(row['tokens'])-1, static_synced=True)
            prediction = self.backend.static_predict(
                row['cache'], nodes, row['s_position'], compiled=True)
            if len(prediction) != WIDTH:
                raise ValueError('HF verify prediction binding')
            path, emitted = accept(PARENTS, nodes, prediction)
            new_tokens = self._commit_static(row, path, emitted, nodes)
            self.calls['batch_forwards'] += 1
            self.calls['batch_rows'] += 1
            self.calls['verify'] += 1
            self.calls['verified_positions'] += WIDTH
            self.calls['accepted_drafts'] += len(path)-1
            self.calls['rejected_drafts'] += WIDTH-len(path)
            return [dict(id=item['id'], path=path, emitted=emitted,
                         prefix_hash=item['prefix_hash'],
                         next_prefix_hash=fingerprint(new_tokens))]
        rows, bound = [], []
        for item in requests:
            row = self._bind(item, self.backend.vocab_size)
            nodes = item.get('nodes')
            if (not isinstance(nodes, (list, tuple)) or len(nodes) != WIDTH
                    or any(type(token) is not int or not 0 <= token < self.backend.vocab_size
                           for token in nodes)
                    or nodes[0] != row['tokens'][-1]):
                raise ValueError('invalid HF verify nodes')
            if row['pending'] is not None:
                raise ValueError('HF verify on uncommitted speculation')
            nodes = list(nodes)
            row['pending'] = nodes
            self._sync_slab(row)
            rows.append((row['slot'], nodes))
            bound.append((item, row, nodes))
        predictions = self.slab.predict(rows)
        self.calls['batch_forwards'] += 1
        self.calls['batch_rows'] += len(rows)
        output = []
        for (item, row, nodes), prediction in zip(bound, predictions):
            if len(prediction) != WIDTH:
                raise ValueError('HF verify prediction binding')
            path, emitted = accept(PARENTS, nodes, prediction)
            new_tokens = self._commit(row, path, emitted, nodes)
            if self.cloud_static:
                row['static_synced'] = False
            self.calls['verify'] += 1
            self.calls['verified_positions'] += WIDTH
            self.calls['accepted_drafts'] += len(path)-1
            self.calls['rejected_drafts'] += WIDTH-len(path)
            output.append(dict(id=item['id'], path=path, emitted=emitted,
                               prefix_hash=item['prefix_hash'],
                               next_prefix_hash=fingerprint(new_tokens)))
        self.slab.trim()
        return output

    def _verify_growth(self, requests):
        from engines.common.growth import Plan, parents as growth_parents
        if (type(requests) is not list or not 1 <= len(requests) <= 8
                or any(type(item) is not dict for item in requests)
                or len({item.get('id') for item in requests}) != len(requests)):
            raise ValueError('invalid HF growth verify batch')
        prepared = []
        # No _sync_slab, pending marker or model work until ALL rows bind.
        for item in requests:
            row = self._bind(item, self.backend.vocab_size)
            plan = Plan.bind(item.get('growth_plan'), item['id'], row['tokens'])
            nodes = item.get('nodes')
            if (not isinstance(nodes, (list, tuple)) or len(nodes) not in (plan.k+1, plan.k+2)
                    or len(nodes) == plan.k+2 and plan.branches != 2
                    or any(type(t) is not int or not 0 <= t < self.backend.vocab_size for t in nodes)
                    or nodes[0] != row['tokens'][-1]):
                raise ValueError('invalid HF growth verify nodes')
            if row['pending'] is not None:
                raise ValueError('HF verify on uncommitted speculation')
            previous = row.get('growth_plan')
            if previous is not None and (plan.generation != previous.generation or
                                         plan.round_id <= previous.round_id):
                raise ValueError('stale HF target growth round or generation')
            parents = growth_parents(plan.k, len(nodes) == plan.k+2)
            if len(row['tokens'])-1+len(nodes) > self.request_context:
                raise ValueError('HF growth query exceeds physical context')
            prepared.append((item, row, plan, list(nodes), parents))
        if len({(p.k, p.branches) for _, _, p, _, _ in prepared}) != 1:
            raise ValueError('HF growth verify requires one immutable action')
        groups = {}
        for entry in prepared:
            groups.setdefault(entry[-1], []).append(entry)
        results = {}
        for parents, entries in groups.items():
            rows = []
            for item, row, plan, nodes, _ in entries:
                self._sync_slab(row)
                row['pending'] = nodes
                rows.append((row['slot'], nodes))
            predictions = self.slab.predict(rows, parents=parents)
            if len(predictions) != len(entries):
                raise ValueError('HF growth verify prediction batch')
            self.calls['batch_forwards'] += 1
            self.calls['batch_rows'] += len(rows)
            for (item, row, plan, nodes, _), prediction in zip(entries, predictions):
                if len(prediction) != len(nodes):
                    raise ValueError('HF growth verify prediction width')
                path, emitted = accept(parents, nodes, prediction)
                new_tokens = self._commit(row, path, emitted, nodes)
                row['growth_plan'] = plan
                self.calls['verify'] += 1
                self.calls['verified_positions'] += len(nodes)
                self.calls['accepted_drafts'] += len(path)-1
                self.calls['rejected_drafts'] += len(nodes)-len(path)
                action = 'k%d_roots%d' % (plan.k, plan.branches)
                self.growth_calls[action] = self.growth_calls.get(action, 0)+1
                results[item['id']] = dict(id=item['id'], path=path, emitted=emitted,
                    prefix_hash=item['prefix_hash'], next_prefix_hash=fingerprint(new_tokens),
                    growth_plan=plan.wire(), **self._native_shape(), growth_meta=dict(
                        # `side` means the accepted path took the alternate root, not that
                        # the tree offered one: the pricing and survival models subtract it
                        # from the delivered count, and an offered-but-rejected alternate
                        # root would make that arithmetic claim the primary chain lost.
                        side=path == [0, plan.k+1], queries=len(nodes), free_blocks=len(self.free_slots),
                        capacity=self.max_rows))
            self.slab.trim()
        return [results[item['id']] for item in requests]

    def free(self, ids):
        ids = self._ids(ids)
        freed = []
        for rid in ids:
            row = self.requests.pop(rid, None) if self.role == 'cloud' else self.sessions.pop(rid, None)
            if row is None:
                continue
            if self.role == 'cloud':
                self.reserved_tokens -= row['reserved']
                self.slab.free(row['slot'])
                self.free_slots.append(row['slot'])
                self.free_slots.sort()
            elif self.fused:
                self.draft_pool.free(row['cache'])
            elif self.static:
                self.backend.static_rewind(row['cache'], 0)
                self.static_pool.append(row['cache'])
            freed.append(rid)
        if self.role == 'cloud' and freed:
            self.slab.trim()
        self.calls['free'] += len(freed)
        return dict(freed=freed, **(self._native_shape() if self.role == 'cloud' else {}))

    # ---------------- edge ----------------

    def _session(self, rid):
        session = self.sessions.get(rid)
        if session is None:
            if len(self.sessions) >= self.edge_capacity:
                raise EdgeAdmissionError('edge live session limit')
            if self.fused:
                session = dict(cache=self.draft_pool.allocate(), position=0,
                               context=[], committed=[], rounds=0)
            elif self.static:
                if not self.static_pool:
                    raise ValueError('edge static cache pool exhausted')
                session = dict(cache=self.static_pool.pop(),
                               position=0, context=[], committed=[], rounds=0)
            else:
                session = dict(cache=self.backend.new_cache(), context=[], committed=[], rounds=0)
            self.sessions[rid] = session
        return session

    def _align(self, session, prefix):
        """Cache keeps ``context[:-1]``; rewind by the real common prefix."""
        if self.fused:
            common = common_prefix(session['context'][:-1], prefix[:-1])
            self.draft_pool.align(session['cache'], prefix, common)
            session['position'] = len(prefix)-1
            session['context'] = list(prefix)
            session['rounds'] += 1
            return
        if self.static:
            desired = len(prefix)-1
            if desired > self.edge_cache_len:
                raise ValueError('edge static cache bound')
            common = common_prefix(session['context'][:-1], prefix[:-1])
            if common < self.backend.static_length(session['cache']):
                self.backend.static_rewind(session['cache'], common)
            if desired > common:
                self.backend.static_prefill(session['cache'], prefix[common:desired], common)
            session['position'] = desired
            session['context'] = list(prefix)
            session['rounds'] += 1
            return
        cached = session['context'][:-1]
        common = common_prefix(cached, prefix[:-1])
        self.backend.crop(session['cache'], common)
        if len(prefix)-1 > common:
            self.backend.prefill(session['cache'], prefix[common:len(prefix)-1])
        if self.backend.length(session['cache']) != len(prefix)-1:
            raise ValueError('HF edge cache length after rewind')
        session['context'] = list(prefix)
        session['rounds'] += 1

    def sync_draft(self, requests):
        if (not isinstance(requests, list) or not 1 <= len(requests) <= MAX_REQUESTS
                or any(type(row) is not dict for row in requests)
                or len({row.get('id') for row in requests}) != len(requests)):
            raise ValueError('invalid edge batch')
        from engines.common.growth import Plan
        prepared, creating = [], 0
        # Validate every row before allocating a session or advancing any KV.
        for item in requests:
            rid, prefix = item.get('id'), token_ids(item.get('tokens'))
            if not isinstance(rid, str) or not rid.startswith('ec-'):
                raise ValueError('invalid edge request id')
            if any(token >= self.backend.vocab_size for token in prefix):
                raise ValueError('edge prefix outside vocabulary')
            if 'rewind' in item and type(item['rewind']) is not bool:
                raise ValueError('edge rewind must be boolean')
            plan = Plan.bind(item['growth_plan'], rid, prefix) if 'growth_plan' in item else None
            if 'root_top2' in item and (type(item['root_top2']) is not bool or plan is not None):
                raise ValueError('edge stream root metadata requires an unplanned draft')
            if plan is not None and not self.growth:
                raise ValueError('HF_GROWTH_DISABLED')
            k = plan.k if plan is not None else DRAFT_TOKENS
            if len(prefix)+k > self.request_context:
                raise EdgeAdmissionError('edge context bound')
            existing = self.sessions.get(rid)
            if existing is None:
                creating += 1
            else:
                previous = existing.get('growth_plan')
                if plan is not None and previous is not None and (
                        plan.generation != previous.generation or plan.round_id <= previous.round_id):
                    raise ValueError('stale edge growth round or generation')
                if item.get('rewind') is not True:
                    committed = existing['committed']
                    if committed and (prefix[:len(committed)] != committed or len(prefix) <= len(committed)):
                        raise ValueError('stale or rewritten target prefix')
            prepared.append((item, prefix, plan, k))
        if any(p is not None for _, _, p, _ in prepared) and (
                any(p is None for _, _, p, _ in prepared) or len({k for _, _, _, k in prepared}) != 1):
            raise ValueError('HF_EDGE_MIXED_ACTION_BATCH')
        if len(self.sessions)+creating > self.edge_capacity:
            raise EdgeAdmissionError('edge live session limit')
        output = []
        for item, prefix, plan, k in prepared:
            session = self._session(item['id'])
            self._align(session, prefix)
            top2 = None
            root_top2 = (plan.branches == 2 if plan is not None
                         else self.growth and item.get('root_top2', False))
            if self.fused:
                if (k, root_top2) == (3, False):
                    generated = self.draft_pool.draft(session['cache'], prefix[-1])
                else:
                    result = self.draft_pool.draft(session['cache'], prefix[-1], k=k, root_top2=root_top2)
                    if root_top2:
                        generated, top2 = result
                    else:
                        generated = result
                session['context'].extend(generated)
                session['position'] = session['cache'].valid
            else:
                generated, current = [], prefix[-1]
                for step in range(k):
                    root = root_top2 and step == 0
                    options = dict(root_top2=True) if root else {}
                    if self.static:
                        result = self.backend.static_predict(
                            session['cache'], [current], self.backend.static_length(session['cache']),
                            compiled=self.compiled_static, **options)
                        session['position'] = self.backend.static_length(session['cache'])
                    else:
                        result = self.backend.predict(session['cache'], [current], **options)
                    if root:
                        predictions, top2 = result
                    else:
                        predictions = result
                    token = predictions[0]
                    if type(token) is not int or not 0 <= token < self.backend.vocab_size:
                        raise ValueError('edge draft outside vocabulary')
                    generated.append(token)
                    session['context'].append(token)
                    current = token
            if len(generated) != k or any(
                    type(t) is not int or not 0 <= t < self.backend.vocab_size for t in generated):
                raise ValueError('edge fused draft outside vocabulary')
            if root_top2:
                from engines.common.root_shadow import validate_top2
                top2 = validate_top2(top2, generated[0])
                if top2['alternate'] >= self.backend.vocab_size:
                    raise ValueError('edge alternate outside vocabulary')
            if plan is not None or not item.get('rewind'):
                session['committed'] = list(prefix)
            row = dict(id=item['id'], prefix_hash=fingerprint(prefix), tokens=generated)
            if root_top2 or plan is not None:
                row['root_top2'] = top2
            if plan is not None:
                session['growth_plan'] = plan
                row['growth_plan'] = plan.wire()
                action = 'k%d_roots%d' % (k, plan.branches)
                self.growth_calls[action] = self.growth_calls.get(action, 0)+1
            self.calls['draft_tokens'] += k
            output.append(row)
        self.calls['sync_draft'] += 1
        return output

    def warm(self, requests):
        if (not isinstance(requests, list) or not 1 <= len(requests) <= MAX_REQUESTS
                or any(not isinstance(row, dict) for row in requests)):
            raise ValueError('invalid edge warm batch')
        ids = [row.get('id') for row in requests]
        if (any(not isinstance(rid, str) or not rid.startswith('ec-') for rid in ids)
                or len(ids) != len(set(ids))):
            raise ValueError('invalid edge request id')
        prepared = []
        # Every input-domain refusal precedes ALL session allocation/prefill,
        # including generic/static paths and a bad row later in the batch.
        for item, rid in zip(requests, ids):
            prompt = token_ids(item.get('tokens'))
            if any(token >= self.backend.vocab_size for token in prompt):
                raise ValueError('edge warm prefix outside vocabulary')
            if len(prompt)+DRAFT_TOKENS > self.request_context:
                raise EdgeAdmissionError('edge fused warm context bound' if self.fused else 'edge warm context bound')
            if rid in self.sessions:
                raise ValueError('edge warm requires a new session')
            prepared.append((rid, prompt))
        if len(self.sessions)+len(prepared) > self.edge_capacity:
            raise EdgeAdmissionError('edge fused warm pool exhausted' if self.fused else 'edge warm pool exhausted')
        warmed = []
        for rid, prompt in prepared:
            session = self._session(rid)
            if len(prompt) > 1:
                if self.fused:
                    self.draft_pool.install(session['cache'], prompt[:-1])
                    session['position'] = len(prompt)-1
                elif self.static:
                    if len(prompt)-1 > self.edge_cache_len:
                        raise ValueError('edge static cache bound')
                    self.backend.static_prefill(session['cache'], prompt[:-1], 0)
                    session['position'] = len(prompt)-1
                else:
                    self.backend.prefill(session['cache'], prompt[:-1])
                self.calls['prefills'] += 1
            session['context'] = list(prompt)
            session['committed'] = list(prompt[:-1])
            warmed.append(rid)
        self.calls['warm'] += len(warmed)
        return {'warmed': warmed}

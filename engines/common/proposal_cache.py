import asyncio
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import uuid

from .policy import CONTEXT, DRAFT_TOKENS, MAX_OUTPUT, MAX_REQUESTS, WIDTH, fingerprint, token_ids


class ExactProposalCache:
    def __init__(self, identity, vocabulary, max_entries=4096, max_tokens=2097152):
        if (not isinstance(identity, tuple) or not identity or type(vocabulary) is not int
                or vocabulary < 1 or max_entries < 1 or max_tokens < 1):
            raise ValueError('invalid exact cache bounds / identity')
        self.identity, self.vocabulary = identity, vocabulary
        self.max_entries, self.max_tokens = max_entries, max_tokens
        self.entries = OrderedDict()
        self.tokens = self.hits = self.misses = self.evictions = self.invalidations = 0

    def key(self, identity, prefix):
        if identity != self.identity:
            self.clear()
            raise ValueError('proposal cache identity changed')
        return tuple(token_ids(prefix))

    def get(self, identity, prefix):
        key = self.key(identity, prefix)
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return list(value)

    def validate(self, proposal):
        value = tuple(token_ids(proposal))
        if len(value) != DRAFT_TOKENS or any(t >= self.vocabulary for t in value):
            raise ValueError('invalid exact cached proposal')
        return value

    def put(self, identity, prefix, proposal):
        key, value = self.key(identity, prefix), self.validate(proposal)
        if len(key)+len(value) > self.max_tokens:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.tokens -= len(key)+len(previous)
        self.entries[key] = value
        self.tokens += len(key)+len(value)
        while len(self.entries) > self.max_entries or self.tokens > self.max_tokens:
            old, value = self.entries.popitem(last=False)
            self.tokens -= len(old)+len(value)
            self.evictions += 1

    def clear(self):
        self.entries.clear()
        self.tokens = 0
        self.invalidations += 1

    def status(self):
        return dict(entries=len(self.entries), stored_tokens=self.tokens, hits=self.hits,
                    misses=self.misses, evictions=self.evictions, invalidations=self.invalidations,
                    max_entries=self.max_entries, max_tokens=self.max_tokens,
                    full_prefix_equality=True, stored_KV=False, stored_target_outputs=False)


class CachedDraft:
    """Serialize lookup/native misses/free with one logical committed-prefix guard.

    Hits create no native state. A later miss supplies the entire committed
    prefix to the existing adapter, which repairs only that request's own KV.
    Free still reaches the adapter (including partially cached trajectories).
    """
    def __init__(self, native, identity, vocabulary):
        self.native, self.identity = native, identity
        self.cache = ExactProposalCache(identity, vocabulary)
        self.lock, self.committed = asyncio.Lock(), {}
        self.native_batches = self.native_requests = 0

    async def dispatch(self, payload):
        async with self.lock:
            try:
                op = payload.get('op')
                if op == 'status':
                    result = await self.native(payload)
                    return dict(result, exact_proposals=self.status())
                if op == 'free':
                    ids = payload['ids']
                    if (not isinstance(ids, list) or len(ids) > MAX_REQUESTS
                            or any(not isinstance(rid, str) for rid in ids) or len(ids) != len(set(ids))):
                        raise ValueError('invalid cached edge free batch')
                    result = await self.native(payload)
                    for rid in ids:
                        self.committed.pop(rid, None)
                    return result
                if op != 'sync_draft':
                    raise ValueError('invalid cached edge operation')
                return await self.propose(payload['requests'])
            except BaseException:
                # Failed RPCs / cancellation never leave reusable token entries.
                self.cache.clear()
                raise

    async def propose(self, items):
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_REQUESTS:
            raise ValueError('invalid cached edge batch')
        prefixes, ids, budgets = [], [], []
        # Check every row before cache lookup, native allocation or KV write.
        for item in items:
            rid, prefix, maximum = item['id'], tuple(token_ids(item['tokens'])), item['max_tokens']
            if (not isinstance(rid, str) or not rid.startswith('ec-') or rid in ids
                    or type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT
                    or len(prefix)+maximum+WIDTH > CONTEXT
                    or any(t >= self.cache.vocabulary for t in prefix)):
                raise ValueError('invalid cached edge request')
            old = self.committed.get(rid)
            if old and (len(prefix) <= len(old[0]) or prefix[:len(old[0])] != old[0]
                        or len(prefix)+maximum != old[1]):
                raise ValueError('stale / rewritten cached target prefix or budget')
            ids.append(rid)
            prefixes.append(prefix)
            budgets.append(len(prefix)+maximum)
        if len(set(self.committed) | set(ids)) > MAX_REQUESTS:
            raise ValueError('cached edge live request limit')
        values, misses = {}, []
        for item, prefix in zip(items, prefixes):
            value = self.cache.get(self.identity, prefix)
            if value is None:
                misses.append(item)
            else:
                values[item['id']] = value
        if misses:
            rows = await self.native(dict(op='sync_draft', requests=misses))
            if not isinstance(rows, list) or len(rows) != len(misses):
                raise ValueError('cached edge native response batch binding')
            pending = []
            for item, row in zip(misses, rows):
                if row.get('id') != item['id'] or row.get('prefix_hash') != fingerprint(item['tokens']):
                    raise ValueError('cached edge native response prefix binding')
                pending.append((item, self.cache.validate(row.get('tokens'))))
            # Publish only after the entire native result batch is validated.
            for item, value in pending:
                self.cache.put(self.identity, item['tokens'], value)
                values[item['id']] = list(value)
            self.native_batches += 1
            self.native_requests += len(misses)
        for rid, prefix, budget in zip(ids, prefixes, budgets):
            self.committed[rid] = (prefix, budget)
        return [dict(id=rid, prefix_hash=fingerprint(prefix), tokens=values[rid])
                for rid, prefix in zip(ids, prefixes)]

    def status(self):
        return dict(self.cache.status(), live_requests=len(self.committed),
                    native_batches=self.native_batches, native_requests=self.native_requests,
                    session=self.identity[-1])


def install_cache(native, engine, model):
    mode = os.environ.get('TRIAGE_EXACT_PROPOSALS', 'off')
    if mode == 'off':
        return None
    if mode != 'exact':
        raise ValueError('invalid exact proposal cache mode')
    root = Path(model)
    config = json.loads((root/'config.json').read_text())
    files = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json')
    binding = tuple((name, hashlib.sha256((root/name).read_bytes()).hexdigest())
                    for name in files if (root/name).is_file())
    identity = (engine, importlib.metadata.version(engine), str(root.resolve()), binding,
                'BF16/native-RoPE/TP1/greedy-seed0/ignore-eos', DRAFT_TOKENS,
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), uuid.uuid4().hex)
    return CachedDraft(native, identity, config['vocab_size'])

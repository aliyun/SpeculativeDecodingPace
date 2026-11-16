from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import json
import os
import threading
import time

from .rate_gate import CACHE_SOURCES, rate_gate_from_env, seed_check_from_env
from .root_shadow import validate_top2
from .policy import DRAFT_TOKENS, WIDTH, MAX_REQUESTS, execution_mode, fingerprint, token_ids
from .contracts import Request


@dataclass(frozen=True)
class Entry:
    prefix: tuple
    stride: int
    future: Future
    index: int
    source: str


class Prefetch:
    ec_prefix = 'ec-'

    def __init__(self, client, *, root_top2_shadow=False, similar_context_drafts=False):
        if type(root_top2_shadow) is not bool:
            raise ValueError('ROOT_TOP2_SHADOW_MODE')
        self.root_top2_shadow, self.root_top2 = root_top2_shadow, {}
        if type(similar_context_drafts) is not bool:
            raise ValueError('similar-context drafts require a boolean')
        if similar_context_drafts and (root_top2_shadow or (DRAFT_TOKENS, WIDTH) not in ((3, 4), (5, 6), (7, 8))):
            raise ValueError('similar-context drafts require fixed K3 without root-top2')
        self.client = client
        self.similar_cache = None
        # Compatibility handles: both auxiliary caches are retired and the
        # unified cache owns their duties; callers may still read these.
        self.exact_cache = None
        self.cache = None
        if similar_context_drafts:
            from .similar_cache import (UnifiedContextCache, fuzzy_mode_from_env,
                                        fuzzy_threshold_from_env, prompt_width_from_env,
                                        min_width_from_env, repeat_gate_from_env,
                                        repeat_min_from_env, unique_off_from_env,
                                        auto_gate_from_env, gate_ratio_from_env,
                                        gate_window_from_env, per_request_gate_from_env,
                                        multi_suffix_from_env, progress_from_env)
            self.similar_cache = UnifiedContextCache(
                cross_request_only=False, fuzzy_mode=fuzzy_mode_from_env(),
                fuzzy_threshold=fuzzy_threshold_from_env(),
                prompt_width=prompt_width_from_env(),
                min_width=min_width_from_env(),
                repeat_gate=repeat_gate_from_env(),
                repeat_min=repeat_min_from_env(),
            unique_off=unique_off_from_env(),
            auto_gate=auto_gate_from_env(), gate_ratio=gate_ratio_from_env(),
                gate_window=gate_window_from_env(),
                per_request_gate=per_request_gate_from_env(),
                multi_suffix=multi_suffix_from_env(),
                progress=progress_from_env())
        # A cache proposal serves a round only while its
        # measured committed-token rate beats the edge-or-AR alternative.
        # Disabled by default; the switch is TRIAGE_UNIFIED_RATE_GATE.
        self.rate_gate = rate_gate_from_env()
        self.seed_check_drafts = seed_check_from_env()
        self.round_mark = OrderedDict()
        # What supplied the in-flight round for this request. Recorded where the
        # proposal is chosen, because the entry that served a round may be gone by
        # the time the target commits (a streamed plan is re-planned first).
        self.round_source = OrderedDict()
        # What the plan was priced against ('edge' means: do not build a proposal).
        self.cache_forecast = OrderedDict()
        # Proposal built for a round the gate did not let serve: scored against
        # the real commit instead of being thrown away.
        self.shadow_round = OrderedDict()
        self.seed_dropped = False
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='single-edge')
        self.lock = threading.RLock()
        self.entries, self.inflight, self.active_rids, self.live = {}, set(), set(), set()
        self.releasing, self.warming = {}, set()
        self._on_edge_free_ack = None
        self.stats = dict(edge_submit=0, suffix_hit=0, consumed=0, cleanup=0, ar_bypass=0, ar_consumed=0)
        if self.similar_cache is not None:
            self.stats['similar_context_hit'] = 0
        self.closed = False

    def event(self, event, **fields):
        value=dict(event=event, t_ns=time.monotonic_ns(), **fields)
        # Compact event encoding preserves every field without repeated nested
        # sorting. API and fingerprint serialization are unchanged.
        raw=(json.dumps(value,separators=(',',':')) if os.environ.get('TRIAGE_SHAPE_BUDGET_GROWTH')=='1'
             else json.dumps(value,sort_keys=True))
        logging.getLogger(__name__).info('SINGLE_EDGE %s', raw)

    def observe_committed(self, request):
        if self.similar_cache is not None:
            with self.lock:
                self._note_round(request)
                self.similar_cache.observe_committed(request)

    def _gate_scope(self, request, source, detail=None):
        return self._scope_for_tier(tuple(request.prompt), source, detail)

    def _scope_for_tier(self, prompt, source, detail=None):
        """Price the tier that actually proposed, and the suffix width that did.

        A single 'other' scope averaged the short suffix widths (noise) with the
        long ones (usually right) and then vetoed all of them together.
        """
        layer = (detail or {}).get('layer')
        width = (detail or {}).get('width')
        if source == 'exact_prefix' or layer == 'prompt':
            return ('prompt', self.similar_cache.prompt_origin(tuple(prompt)))
        if layer == 'window':
            return ('suffix', 'w%s' % width if width else 'any')
        if layer in ('fuzzy', 'content', 'suffix1'):
            return ('band', layer)
        if layer == 'lookup':
            return ('inline', 'any')
        return ('other', 'any')

    def forecast_allows(self, prompt, supply, detail=None):
        """Price a *forecast* source before the plan is chosen.

        A veto after the fact still leaves the round priced as a cache action, so
        the selector keeps learning the wrong thing about it. Asking here means a
        losing scope is planned as an edge round from the start.
        """
        if not self.rate_gate.enabled or self.similar_cache is None or supply == 'edge':
            return True
        scope = self._scope_for_tier(tuple(prompt), supply, detail)
        if self.rate_gate.allow(scope):
            return True
        self.stats['rate_gate_preveto'] = self.stats.get('rate_gate_preveto', 0) + 1
        return False

    def _check_seed_health(self):
        """A seed that never produced an accepted token is not evidence: drop it.

        Dropping means the process behaves exactly as if it had no seed, which is
        the only way a wrong seed can cost nothing. It cannot fire on a seed whose
        replays were never proposed; that case is inert already.
        """
        cache = self.similar_cache
        if self.seed_dropped or self.seed_check_drafts <= 0:
            return
        if cache is None or cache.seeded_records == 0:
            return
        drafts, accepted = cache.seed_health()
        if drafts < self.seed_check_drafts:
            return
        if accepted:
            return
        cache.clear()
        self.seed_dropped = True
        self.stats['cache_seed_dropped'] = self.stats.get('cache_seed_dropped', 0) + 1
        self.event('cache_seed_dropped', drafts=drafts, accepted=accepted,
                   threshold=self.seed_check_drafts)

    def _mark_source(self, rid, side, scope=None):
        if not self.rate_gate.enabled:
            return
        self.round_source[rid] = (side, scope)
        self.round_source.move_to_end(rid)
        while len(self.round_source) > 64:
            self.round_source.popitem(last=False)

    def _note_round(self, request):
        """Charge one committed round to the side that supplied it.

        Both sides are timed from the same boundary (the proposal that served
        this round) to the same event (the target's commit), so the comparison
        needs no absolute cost constant and no double counting of the wait.
        """
        if not self.rate_gate.enabled:
            return
        rid = str(request.rid)
        now = time.monotonic_ns()
        committed = len(request.generated)
        mark = self.round_mark.get(rid)
        shadow = self.shadow_round.pop(rid, None)
        if shadow is not None and mark is not None:
            # Score it against what the target actually committed after the prefix
            # the proposal was built on. The cost is the measured round, which is
            # the alternative's clock: a local round has no exposed wait, so using
            # this denominator makes a scope win on progress alone.
            previous, started = mark
            committed_tokens = request.generated[shadow['since']:]
            matched = 0
            for left, right in zip(shadow['tokens'], committed_tokens):
                if left != right:
                    break
                matched += 1
            # The correction token commits either way; progress is what matched.
            self.rate_gate.note_shadow(shadow['scope'], matched + 1, now - started)
        tag = self.round_source.get(rid)
        if tag is not None:
            side, scope = tag
        else:
            # Fallback for a round whose supply was not recorded: an entry's
            # source, or the edge stream that owns a request without one.
            entry = self.entries.get(rid)
            supply = (entry.source if entry is not None
                      else ('edge' if rid in getattr(self, 'stream_plans', {}) else None))
            side = None if supply is None else ('cache' if supply in CACHE_SOURCES else 'alt')
            scope = (self._gate_scope(request, supply) if side == 'cache' else None)
        self.rate_gate.count('rounds')
        if mark is None:
            self.rate_gate.count('first_observation')
        if side is None:
            self.rate_gate.count('no_source')
        if mark is not None and side is not None:
            previous, started = mark
            tokens = committed - previous
            if tokens <= 0:
                self.rate_gate.count('no_tokens')
            if tokens > 0 and now > started:
                self.rate_gate.count(side + '_round')
                self.rate_gate.note(side, scope if side == 'cache' else None, tokens, now - started)
        self.round_mark[rid] = (committed, now)
        self.round_mark.move_to_end(rid)
        while len(self.round_mark) > 64:
            self.round_mark.popitem(last=False)

    def _shadow(self, rid, tokens, scope, since=0):
        """Remember what a refused proposal would have delivered."""
        if not self.rate_gate.enabled or type(tokens) is not list or not tokens:
            return
        self.shadow_round[rid] = dict(tokens=list(tokens), scope=scope,
                                      since=since, t0=time.monotonic_ns())
        self.shadow_round.move_to_end(rid)
        while len(self.shadow_round) > 64:
            self.shadow_round.popitem(last=False)

    def _gate_allows(self, request, source, detail=None):
        """Check whether the cache proposal passes the rate gate."""
        if not self.rate_gate.enabled:
            return True
        if self.rate_gate.allow(self._gate_scope(request, source, detail)):
            return True
        self.stats['rate_gate_veto'] = self.stats.get('rate_gate_veto', 0) + 1
        return False

    def _call(self, items):
        ids = {r['id'] for r in items}
        with self.lock:
            self.active_rids.update(ids)
        try:
            self.event('edge_start', ids=[row['id'] for row in items],
                       prefix_lengths=[len(row['tokens']) for row in items])
            rows = self.client.propose(items)
            if not isinstance(rows, list) or len(rows) != len(items):
                raise ValueError('draft response batch binding')
            for item, row in zip(items, rows):
                if (row.get('id') != item['id'] or row.get('prefix_hash') != fingerprint(item['tokens'])
                        or len(token_ids(row.get('tokens'))) != DRAFT_TOKENS):
                    raise ValueError('draft response prefix / token binding')
            if self.root_top2_shadow:
                metadata = [(item['id'], row['prefix_hash'], validate_top2(row.get('root_top2'), row['tokens'][0]))
                            for item, row in zip(items, rows)]
                with self.lock:
                    if len(set(self.root_top2) | ids) > MAX_REQUESTS or ids.intersection(self.root_top2):
                        raise ValueError('ROOT_TOP2_PENDING_BINDING')
                    self.root_top2.update({rid: (digest, meta) for rid, digest, meta in metadata})
            self.event('edge_result', ids=[r['id'] for r in items], rows=rows)
            return [row['tokens'] for row in rows]
        finally:
            with self.lock:
                self.active_rids.difference_update(ids)
            self.event('edge_end', ids=[row['id'] for row in items])

    def _warm(self, items):
        self.event('edge_warm_start', ids=[row['id'] for row in items],
                   prefix_lengths=[len(row['tokens']) for row in items])
        warmed = self.client.warm(items)
        self.event('edge_warm_end', ids=warmed)
        return warmed

    def warm(self, prompts):
        """Order a prompt prefill on the edge before the first committed-prefix draft."""
        if self.closed:
            raise ValueError('closed prefetch')
        items = []
        for entry in prompts:
            rid, prompt = entry[0], entry[1]
            if not str(rid).startswith(self.ec_prefix):
                continue
            item = dict(id=str(rid), tokens=token_ids(list(prompt)))
            if len(entry) > 2:
                item['max_tokens'] = entry[2]
            items.append(item)
        if not items:
            return
        with self.lock:
            self._check_warm()
            for row in items:
                if row['id'] in self.live or row['id'] in self.entries:
                    raise ValueError('prompt warm requires a new request')
            self.live.update(row['id'] for row in items)
            if len(self.live-set(self.releasing)) > MAX_REQUESTS:
                raise ValueError('single-edge live request limit')
            self.warming.add(self.pool.submit(self._warm, items))
            self.stats['warm_submit'] = self.stats.get('warm_submit', 0) + len(items)

    def _check_warm(self):
        for future in [f for f in self.warming if f.done()]:
            self.warming.discard(future)
            future.result()

    def submit_committed(self, reqs, stride=WIDTH, phase='commit', mode=None):
        from .offload_experiment import reset_for_epoch
        reset_for_epoch(self)
        if stride != WIDTH or self.closed:
            raise ValueError('invalid policy width or closed prefetch')
        finished, eligible, misses = [], [], []
        for req in reqs:
            if not str(req.rid).startswith(self.ec_prefix):
                continue
            if req.finished() or getattr(req, 'is_retracted', False):
                finished.append(str(req.rid))
            elif req.output_ids:
                eligible.append(req)
        if mode is None:
            mode = execution_mode(len(req.origin_input_ids)+len(req.output_ids) for req in eligible).mode
        if mode not in ('ar', 'spec'):
            raise ValueError('unknown execution mode')
        self.release(finished)
        with self.lock:
            self._check_warm()
            self.inflight = {future for future in self.inflight if not future.done()}
            for req in eligible:
                rid = str(req.rid)
                self._note_round(req)
                if getattr(req, 'inflight_middle_chunks', 0):
                    raise ValueError('uncommitted native results unsupported')
                trusted = (self.similar_cache is not None and type(req) is Request
                           and type(req.prompt) is tuple and type(req.generated) is tuple)
                generated = req.generated if trusted else list(req.output_ids)
                if not generated:
                    continue
                prefix = req.prefix if trusted else tuple(token_ids(list(req.origin_input_ids) + generated))
                old = self.entries.get(rid)
                if old is not None:
                    if old.prefix != prefix:
                        raise ValueError('unconsumed proposal on a different prefix')
                    if old.source == 'cancelled' or (old.source == 'ar') == (mode == 'ar'):
                        continue
                    # A mode change invalidates only the proposal. In-flight
                    # native edge work remains ordered before eventual cleanup.
                    self.entries.pop(rid)
                progress = getattr(self.similar_cache, 'observe_progress', None)
                if progress is not None:
                    # Index incrementally per chunk so no request-end burst
                    # blocks the edge loop.
                    progress(prefix, generated, rid=rid)
                self.live.add(rid)
                if len(self.live-set(self.releasing)) > MAX_REQUESTS:
                    raise ValueError('single-edge live request limit')
                if mode == 'ar':
                    future = Future()
                    future.set_result([[0]*DRAFT_TOKENS])
                    self.entries[rid] = Entry(prefix, stride, future, 0, 'ar')
                    self._mark_source(rid, 'alt')
                    self.stats['ar_bypass'] += 1
                    continue
                if self.similar_cache is not None:
                    unified, source = self.similar_cache.propose_committed_with_source(req)
                    if unified and not self._gate_allows(req, source):
                        unified = []
                    if unified:
                        future = Future()
                        future.set_result([unified])
                        self.entries[rid] = Entry(prefix, stride, future, 0, 'similar_context')
                        self._mark_source(rid, 'cache', self._gate_scope(req, source))
                        self.stats['similar_context_hit'] += 1
                        self.stats['unified_context_hit'] = self.stats.get('unified_context_hit', 0) + 1
                        self.event('similar_context_proposal', rid=rid,
                                   prefix_hash=fingerprint(prefix), tokens=unified)
                        continue
                similar, similar_source = ((self.similar_cache.propose_committed_with_source(req))
                                           if self.similar_cache is not None else ([], None))
                if similar and not self._gate_allows(req, similar_source):
                    similar = []
                if similar:
                    future = Future()
                    future.set_result([similar])
                    self.entries[rid] = Entry(prefix, stride, future, 0, 'similar_context')
                    self._mark_source(rid, 'cache', self._gate_scope(req, similar_source))
                    self.stats['similar_context_hit'] += 1
                    self.event('similar_context_proposal', rid=rid,
                               prefix_hash=fingerprint(prefix), tokens=similar)
                    continue
                maximum = req.sampling_params.max_new_tokens
                misses.append(dict(id=rid, tokens=list(prefix), max_tokens=max(1, maximum-len(generated))))
            if misses:
                future = self.pool.submit(self._call, misses)
                self.inflight.add(future)
                for i, row in enumerate(misses):
                    self.entries[row['id']] = Entry(tuple(row['tokens']), stride, future, i, 'edge')
                    self._mark_source(row['id'], 'alt')
                self.stats['edge_submit'] += len(misses)

    def ready(self, rid, prefix):
        with self.lock:
            entry = self.entries.get(str(rid))
            if entry is None or entry.prefix != tuple(prefix):
                raise ValueError('missing or stale committed-prefix proposal')
            if not entry.future.done():
                return False
            tokens = entry.future.result()[entry.index]
            if len(token_ids(tokens)) != DRAFT_TOKENS:
                raise ValueError('incomplete draft')
            return True

    def shadow_snapshot(self, rid, prefix):
        with self.lock:
            if not self.root_top2_shadow or not self.ready(rid, prefix):
                raise ValueError('ROOT_TOP2_NOT_READY')
            entry = self.entries[str(rid)]
            # Growth still supplies its exact-prefix proposals on this shared
            # path; cached tokens never acquire an edge alternate root.
            if entry.source in ('suffix', 'exact_prefix'):
                return dict(source=entry.source, top2=None)
            if entry.source != 'edge':
                raise ValueError('ROOT_TOP2_WRONG_SOURCE')
            digest, meta = self.root_top2[str(rid)]
            if digest != fingerprint(prefix):
                raise ValueError('ROOT_TOP2_STALE_PREFIX')
            return dict(source='edge', top2=dict(meta))

    def take(self, rid, prefix, stride=WIDTH):
        if stride != WIDTH:
            raise ValueError('policy width mismatch')
        with self.lock:
            if not self.ready(rid, prefix):
                raise ValueError('draft not ready')
            if self.entries[str(rid)].source == 'ar':
                raise ValueError('AR marker is not a speculative proposal')
            entry = self.entries.pop(str(rid))
            self.root_top2.pop(str(rid), None)
            self.stats['consumed'] += 1
            return list(entry.future.result()[entry.index])

    def take_ar(self, rid, prefix):
        with self.lock:
            if not self.ready(rid, prefix) or self.entries[str(rid)].source not in ('ar', 'cancelled'):
                raise ValueError('AR mode / prefix binding mismatch')
            self.entries.pop(str(rid))
            self.stats['ar_consumed'] += 1

    def cancel(self, rid, prefix):
        """Make an aborted row runnable without waiting on its proposal.

        The placeholder is never a candidate. Finishing the row calls forget,
        which orders session cleanup after the original in-flight draft.
        """
        rid = str(rid)
        with self.lock:
            if rid not in self.live:
                return
            future = Future()
            future.set_result([[0]*DRAFT_TOKENS])
            self.entries[rid] = Entry(tuple(token_ids(prefix)), WIDTH, future, 0, 'cancelled')
            self.event('cancel', rid=rid)

    def _free(self, ids):
        self.client.free(ids)
        if self._on_edge_free_ack is not None:
            self._on_edge_free_ack(ids)
        with self.lock:
            self.live.difference_update(ids)
            for rid in ids:
                self.releasing.pop(rid, None)
                self.root_top2.pop(rid, None)
            self.stats['cleanup'] += len(ids)
        self.event('cleanup', ids=ids)

    def release(self, ids, retry_failed=False):
        """Enqueue ordered cleanup without blocking a native scheduler callback."""
        selected, previous = [], []
        with self.lock:
            for rid in dict.fromkeys(str(rid) for rid in ids):
                self.entries.pop(rid, None)
                self.round_mark.pop(rid, None)
                self.round_source.pop(rid, None)
                self.shadow_round.pop(rid, None)
                if self.similar_cache is not None:
                    self.similar_cache.release(rid)
                future = self.releasing.get(rid)
                if future is not None:
                    if retry_failed and future.done() and future.exception() is not None:
                        self.releasing.pop(rid)
                    else:
                        previous.append(future)
                        continue
                if rid in self.live:
                    selected.append(rid)
            if selected:
                future = self.pool.submit(self._free, selected)
                for rid in selected:
                    self.releasing[rid] = future
                previous.append(future)
        return tuple(dict.fromkeys(previous))

    def forget(self, rid):
        # Explicit blocking cleanup remains available for teardown and callers
        # outside the native hot path. No lock is held while waiting.
        for future in self.release([rid], retry_failed=True):
            future.result()

    def poll_cleanup(self):
        with self.lock:
            futures = set(self.releasing.values())
        for future in futures:
            if future.done():
                future.result()
        return any(not future.done() for future in futures)

    def snapshot(self):
        with self.lock:
            self.inflight = {f for f in self.inflight if not f.done()}
            result = dict(self.stats, live_entries=len(self.entries), inflight=len(self.inflight),
                          live_requests=len(self.live),
                          cleanup_pending=sum(not f.done() for f in set(self.releasing.values())))
            if self.root_top2_shadow:
                result['root_top2_pending'] = len(self.root_top2)
            result['rate_gate'] = self.rate_gate.snapshot()
            if self.similar_cache is not None:
                result.update(self.similar_cache.snapshot())
            return result

    def close(self):
        if self.closed:
            return
        errors = []
        for rid in list(self.live):
            try:
                self.forget(rid)
            except Exception as exc:
                errors.append(exc)
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.client.close()
        if self.similar_cache is not None:
            self.similar_cache.clear()
        self.closed = True
        if errors:
            raise RuntimeError('edge cleanup failed') from errors[0]

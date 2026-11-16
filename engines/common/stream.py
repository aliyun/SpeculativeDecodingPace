from concurrent.futures import Future
from dataclasses import dataclass, field
import os
import time

from .policy import CONTEXT, DRAFT_TOKENS, STREAM_AHEAD, WIDTH, MAX_REQUESTS, JOIN_MS, execution_mode, fingerprint, token_ids
from .prefetch import Entry, Prefetch
from .contracts import Request


def _resolved(value):
    future = Future()
    future.set_result(value)
    return future


@dataclass
class Path:
    epoch: int
    tokens: list
    opened: int
    handle: object = None
    done: bool = False
    error: object = None
    streamed: int = field(default=0)
    paused: bool = False
    requested: int = 0
    reason: str = ''
    created_ns: int = 0
    last_frame_ns: int = 0
    checked: int = 0
    alignment_failed: bool = False


from .draft_policy import CohortPolicy, FutureCohortPolicy


class StreamPrefetch(CohortPolicy, Prefetch):
    never_wait = True

    def __init__(self, client, *, ahead=12, kmin=1, max_rows=None,
                 policy='cohort', clock=time.monotonic_ns, similar_context_drafts=False,

                 similar_context_cross_request_only=False, **similar_extra):
        if type(ahead) is not int or not DRAFT_TOKENS <= ahead <= STREAM_AHEAD:
            raise ValueError('invalid stream horizon')
        if type(kmin) is not int or not 1 <= kmin <= DRAFT_TOKENS:
            raise ValueError('invalid minimum streamed draft')
        if max_rows is not None and (type(max_rows) is not int or not 1 <= max_rows <= MAX_REQUESTS):
            raise ValueError('invalid stream row limit')
        if policy != 'cohort':
            raise ValueError('cohort-pipeline is the only proposal policy')
        if similar_context_drafts and (getattr(client, 'weak_edge_multiedge', False)
                or os.environ.get('TRIAGE_EDGE_BACKEND', 'native') not in ('native', 'gpu')
                or (DRAFT_TOKENS, WIDTH) not in ((3, 4), (5, 6), (7, 8))):
            raise ValueError('similar-context drafts require a supported K and a single native edge')
        super().__init__(client,
                         similar_context_drafts=similar_context_drafts)
        self.prepare_policy(clock)
        self.stream_demand = 0
        self.open_batch = None
        self.local_only = False
        self.recovery = set()
        self.waits, self.budgets, self.target_step_ns = {}, {}, None
        self.ahead, self.kmin = ahead, kmin
        self.max_rows = max_rows if max_rows is not None else getattr(client, 'max_rows', None)
        self.context = getattr(client, 'context', None)
        if self.context is None:
            self.context = (int(os.environ.get('TRIAGE_CPU_CONTEXT', '12288'))
                            if os.environ.get('TRIAGE_EDGE_BACKEND') == 'cpu' else CONTEXT)
        self.paths, self.epochs, self.warm_futures = {}, {}, {}
        # Root top-2 the edge reports for this request's first streamed position.
        # On by default and paid once per request (one frame field, one
        # validation, one event); TRIAGE_ROOT_GAP=0 removes it entirely. Later
        # rounds of the same path neither carry nor validate the field.
        self.root_gap = os.environ.get('TRIAGE_ROOT_GAP', '1') != '0'
        self.gap_seen = {}
        if self.root_gap:
            self.stats['stream_gap'] = 0
        self.scores, self.pending_draft = {}, {}
        self.stats.update(stream_open=0, stream_resync=0, stream_continue=0, stream_tokens=0,
                          stream_stale=0, stream_draft=0, stream_draft_real=0, never_wait_ar=0,
                          stream_row_cap=0, stream_row_skip=0, stream_local_pause=0, stream_ready_notifications=0)
        if self.adaptive:
            self.stats.update(adaptive_waits=0, adaptive_step_ns=0)
        if self.cohort:
            self.stats.update(stream_recovery_requests=0)

    def observe_committed(self, request):
        with self.lock:
            before = self.exact_cache.conflicts if self.exact_cache is not None else 0
            super().observe_committed(request)
            after = self.exact_cache.conflicts if self.exact_cache is not None else 0
            if self.cohort and after > before and request.rid not in self.recovery:
                self.recovery.add(request.rid)
                self.stats['stream_recovery_requests'] += 1
                self.event('stream_cache_recovery', rid=request.rid, conflicts=after-before,
                           prefix_length=len(request.prefix), horizon=DRAFT_TOKENS)

    def warm(self, prompts):
        before = set(self.warming)
        super().warm(prompts)
        with self.lock:
            for future in set(self.warming)-before:
                for entry in prompts:
                    rid = entry[0]
                    if str(rid).startswith(self.ec_prefix):
                        self.warm_futures[str(rid)] = future

    def _extra(self, rid, prefix):
        """Streamed tokens past a committed prefix, or None when the path diverged from it."""
        path = self.paths.get(rid)
        if path is None:
            return []
        n = min(len(path.tokens), len(prefix))
        if path.tokens[:n] != list(prefix[:n]):
            return None
        return path.tokens[len(prefix):]

    def _horizon(self, rid=None):
        if self.cohort and rid in self.recovery:
            return DRAFT_TOKENS
        # Multiple live misses share one edge device. Do not let speculative
        # more than the next-step overlap queue ahead of siblings' first K3.
        # Full acceptance advances WIDTH=4; the next K3 needs at least7 tokens,
        # rounded to three native three-token rounds (9), not two rounds (6).
        overlap = ((WIDTH+DRAFT_TOKENS+DRAFT_TOKENS-1)//DRAFT_TOKENS)*DRAFT_TOKENS
        return min(self.ahead, overlap) if self.cohort and self.stream_demand > 1 else self.ahead

    def _notify_ready(self):
        # A hint only: the controller must still select/check/consume the native
        # proposal. In particular, do not inspect results or change any Plan here.
        wake = getattr(self, 'ready_wake', None)
        if wake is not None:
            self.stats['stream_ready_notifications'] += 1
            wake.notify(None)

    def _open(self, rid, tokens, reason, budget, confirmed=None):
        remaining = budget-len(tokens)
        ahead = min(self._horizon(rid), self.context-len(tokens), remaining)
        if ahead < 1:
            return
        epoch = self.epochs.get(rid, 0)+1
        self.epochs[rid] = epoch
        path = Path(epoch, list(tokens), len(tokens),requested=ahead,reason=reason,
                    created_ns=self.clock(),checked=len(tokens) if confirmed is None else confirmed)
        self.paths[rid] = path

        def received_frame(count):
            now=self.clock()
            previous=path.last_frame_ns
            path.last_frame_ns=now
            observer=getattr(self,'arrival_observer',None)
            if observer is not None:
                observer(rid,reason,now-(previous or path.created_ns),first=not previous,count=count)

        def on_token(i, token):
            with self.lock:
                if self.paths.get(rid) is path and not path.paused and len(path.tokens) == path.opened+i:
                    path.tokens.append(token)
                    path.streamed += 1
                    self.stats['stream_tokens'] += 1
                    if path.streamed % DRAFT_TOKENS == 0:
                        received_frame(DRAFT_TOKENS)
                        self._notify_ready()

        def on_chunk(i, tokens):
            with self.lock:
                if self.paths.get(rid) is path and not path.paused and len(path.tokens) == path.opened+i:
                    path.tokens.extend(tokens)
                    path.streamed += len(tokens)
                    self.stats['stream_tokens'] += len(tokens)
                    received_frame(len(tokens))
                    self._notify_ready()
        on_token.chunk = on_chunk

        if self.root_gap:
            def on_gap(rid, index, meta):
                with self.lock:
                    if (reason != 'first' or index != 0 or self.paths.get(rid) is not path
                            or rid in self.gap_seen):
                        return
                    self.gap_seen[rid] = dict(meta)
                    self.stats['stream_gap'] += 1
                    self.event('stream_gap', rid=rid, epoch=epoch, index=index, open_length=len(tokens),
                               gap=meta['gap'], alternate=meta['alternate'])
            on_token.gap = on_gap

        def on_end(sent, error):
            with self.lock:
                path.done, path.error = True, error
                if self.paths.get(rid) is path:
                    timing = getattr(path.handle, 'timings', None)
                    self.event('stream_end', rid=rid, epoch=epoch, sent=sent, streamed=path.streamed,
                               error=None if error is None else repr(error),
                               transport_timing=dict(timing) if timing is not None else None)
                    self._notify_ready()

        self.stats['stream_'+('resync' if reason == 'diverged' else 'continue' if reason == 'continue'
                              else 'open')] += 1
        self.event('stream_open', rid=rid, epoch=epoch, reason=reason, open_length=len(tokens), ahead=ahead)
        payload = dict(id=rid, tokens=list(tokens), ahead=ahead, max_tokens=remaining,
                       **({'gap': True} if self.root_gap else {}))
        if self.open_batch is None:
            path.handle = self.client.stream(payload, on_token, on_end)
        else:
            self.open_batch.append((path, payload, on_token, on_end))

    def _start_open_batch(self, pending):
        if not pending:
            return
        handles = []
        try:
            items = [(payload, token, end) for _, payload, token, end in pending]
            handles = (self.client.stream_batch(items) if len(items) > 1
                       else [self.client.stream(*items[0])])
            if not isinstance(handles, list) or len(handles) != len(pending):
                raise ValueError('stream batch handle binding')
            for (path, _, _, _), handle in zip(pending, handles):
                path.handle = handle
        except BaseException as exc:
            for path, _, _, _ in pending:
                path.error = exc
            for handle in handles:
                handle.close()
            raise

    def _pause(self, rid):
        """Stop new work once, retaining any useful buffered prefix and native KV."""
        path = self.paths.get(rid)
        if path is not None and path.error is not None:
            raise RuntimeError('edge stream failed') from path.error
        if path is not None and not path.paused:
            path.paused = True
            if path.handle is not None:
                path.handle.close()
            self.stats['stream_local_pause'] += 1
            self.event('stream_local_pause', rid=rid, epoch=path.epoch)

    def _maintain(self, rid, prefix, budget):
        """Keep the edge drafting ahead of the committed prefix; called with the lock held."""
        warm = self.warm_futures.get(rid)
        if warm is not None:
            if not warm.done():
                return
            warm.result()
            self.warm_futures.pop(rid)
        path = self.paths.get(rid)
        if path is not None and path.error is not None:
            raise RuntimeError('edge stream failed') from path.error
        extra = self._extra(rid, prefix)
        if path is None:
            self._open(rid, prefix, 'first', budget,confirmed=len(prefix))
        elif extra is None:
            if path.handle is not None:
                path.handle.close()
            self.stats['stream_stale'] += 1
            self._open(rid, prefix, 'diverged', budget,confirmed=len(prefix))
        elif path.paused:
            if len(extra) < min(DRAFT_TOKENS, budget-len(prefix)):
                self._open(rid, list(prefix)+extra, 'continue', budget,confirmed=len(prefix))
        elif path.done and len(extra) < self._horizon(rid):
            self._open(rid, list(prefix)+extra, 'continue', budget,confirmed=len(prefix))

    def _observe_alignment(self,rid,prefix):
        observer=getattr(self,'alignment_observer',None)
        path=self.paths.get(rid)
        if observer is None or path is None or path.alignment_failed:return
        matched=0
        while path.checked<min(len(prefix),len(path.tokens)):
            if path.tokens[path.checked]!=prefix[path.checked]:
                path.alignment_failed=True
                break
            matched+=1
            path.checked+=1
        if matched or path.alignment_failed:
            observer(rid,matched,path.alignment_failed)

    def _observe_accept(self, rid, prefix):
        """Update the recent stream accept EMA after a commit against the last streamed draft."""
        pending = self.pending_draft.pop(rid, None)
        if pending is None:
            return
        extra = self._extra(rid, prefix)
        if extra is None:
            rate = 0.0
        else:
            accepted = max(0, min(pending['real'], len(prefix) - pending['prefix_len']))
            rate = accepted / max(1, pending['real'])
        self.scores[rid] = 0.5 * self.scores.get(rid, 0.5) + 0.5 * rate

    def _select_stream_rows(self, prefixes):
        """At most max_rows live streams: sticky consistent paths first, then recent accept EMA."""
        rids = list(prefixes)
        if self.max_rows is None or len(rids) <= self.max_rows:
            return set(rids)

        def key(rid):
            extra = self._extra(rid, prefixes[rid])
            sticky = rid in self.paths and extra is not None
            buffered = 0 if extra is None else len(extra)
            return (1 if sticky else 0, buffered, self.scores.get(rid, 0.5), -rids.index(rid))

        chosen = set(sorted(rids, key=key, reverse=True)[:self.max_rows])
        for rid in (set(self.paths) & set(prefixes)) - chosen:
            path = self.paths.pop(rid, None)
            if path is not None and path.handle is not None:
                path.handle.close()
            self.pending_draft.pop(rid, None)
            self.stats['stream_row_cap'] += 1
            self.event('stream_row_cap', rid=rid, max_rows=self.max_rows)
        return chosen

    def submit_committed(self, reqs, stride=WIDTH, phase='commit', mode=None, *, defer_producer=()):
        deferred=set(defer_producer)
        external_cache=getattr(self,'external_cache',False)
        if deferred and (not external_cache or any(type(rid) is not str for rid in deferred)):
            raise ValueError('deferred producers require externally bound cache offers')
        if stride != WIDTH or self.closed:
            raise ValueError('invalid policy width or closed prefetch')
        finished, eligible = [], []
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
            prefixes, budgets = {}, {}
            for req in eligible:
                rid = str(req.rid)
                if getattr(req, 'inflight_middle_chunks', 0):
                    raise ValueError('uncommitted native results unsupported')
                trusted = (self.similar_cache is not None and type(req) is Request
                           and type(req.prompt) is tuple and type(req.generated) is tuple)
                generated = req.generated if trusted else list(req.output_ids)
                prefix = req.prefix if trusted else tuple(token_ids(list(req.origin_input_ids)+generated))
                prefixes[rid] = prefix
                budgets[rid] = len(req.origin_input_ids)+req.sampling_params.max_new_tokens
                self._observe_accept(rid, prefix)
                self._observe_alignment(rid,prefix)
            for req in eligible:
                rid = str(req.rid)
                prefix = prefixes[rid]
                generated = list(req.output_ids)
                old = self.entries.get(rid)
                self.budgets[rid] = budgets[rid]
                if old is None or old.prefix != prefix or (old.source == 'ar') != (mode == 'ar'):
                    self.waits.pop(rid, None)
                    self.live.add(rid)
                    if len(self.live-set(self.releasing)) > MAX_REQUESTS:
                        raise ValueError('single-edge live request limit')
                    exact = (self.exact_cache.propose(prefix, want=DRAFT_TOKENS)
                             if mode == 'spec' and not external_cache and self.exact_cache is not None else [])
                    # Growth's hybrid delegate explicitly shares the committed
                    # legacy suffix cache. Unified-only callers keep cache=None.
                    # Do not bypass this source merely because transport is stream.
                    tokens = exact or (self.cache.propose(generated, want=DRAFT_TOKENS)
                                       if mode == 'spec' and not external_cache and self.cache is not None else [])
                    if tokens:
                        source = 'exact_prefix' if exact else 'suffix'
                        self.entries[rid] = Entry(prefix, stride, _resolved([tokens]), 0, source)
                        self.stats['exact_prefix_hit' if exact else 'suffix_hit'] += 1
                        self.event('exact_prefix_proposal' if exact else 'suffix_proposal', rid=rid,
                                   prefix_hash=fingerprint(prefix), tokens=tokens)
                    else:
                        similar, similar_source = (
                            self.similar_cache.propose_committed_with_source(req)
                            if mode == 'spec' and not external_cache and self.similar_cache is not None else ([], None))
                        if similar:
                            # Keep the native source taxonomy: a complete prompt
                            # replay is exact_prefix; suffix/band matches are suffix.
                            self.entries[rid] = Entry(prefix, stride, _resolved([similar]), 0, similar_source)
                            self.stats[similar_source + '_hit'] = self.stats.get(similar_source + '_hit', 0) + 1
                            self.stats['similar_context_hit'] += 1
                            self.event(similar_source + '_proposal', rid=rid,
                                       prefix_hash=fingerprint(prefix), tokens=similar)
                        else:
                            self.entries[rid] = Entry(prefix, stride, _resolved(None), 0, mode == 'ar' and 'ar' or 'stream')
                            if mode == 'ar':
                                self.stats['ar_bypass'] += 1
                if self.entries[rid].source in ('exact_prefix', 'suffix', 'similar_context', 'ar', 'cancelled'):
                    self._pause(rid)
            # An external selector holds a real local offer. Do not start or
            # repair a speculative producer until it actually chooses that lane.
            # Existing in-flight supply is stopped only when a cache Plan wins.
            misses = {rid: prefix for rid, prefix in prefixes.items()
                      if self.entries[rid].source == 'stream' and rid not in deferred}
            if self.cohort:
                self.local_only = not misses
                self.never_wait = not self.local_only
            selected = self._select_stream_rows(misses) if mode == 'spec' else set()
            self.stream_demand = len(selected)
            batching = self.cohort and len(selected) > 1 and callable(getattr(self.client, 'stream_batch', None))
            self.open_batch = [] if batching else None
            try:
                for rid, prefix in misses.items():
                    if rid in selected:
                        self._maintain(rid, prefix, budgets[rid])
                    elif mode == 'spec':
                        self._pause(rid)
                        self.stats['stream_row_skip'] += 1
                if batching:
                    self._start_open_batch(self.open_batch)
            finally:
                self.open_batch = None

    def _entry(self, rid, prefix):
        entry = self.entries.get(str(rid))
        if entry is None or entry.prefix != tuple(prefix):
            raise ValueError('missing or stale committed-prefix proposal')
        if entry.source in ('exact_prefix', 'suffix', 'similar_context'):
            # A paused reader may report failure after the local hit. Do not
            # let cache readiness/consumption hide that or a completed warm error.
            self._check_warm()
            self._pause(str(rid))
        return entry

    def ready(self, rid, prefix):
        with self.lock:
            self._entry(rid, prefix)
            return True

    def has_draft(self, rid, prefix, count=None):
        with self.lock:
            if count is not None and (type(count) is not int or not 1<=count<=STREAM_AHEAD):
                raise ValueError('invalid requested stream depth')
            entry = self._entry(rid, prefix)
            if entry.source in ('exact_prefix', 'suffix', 'similar_context'):
                return count is None or len(entry.future.result()[entry.index])>=count
            if entry.source == 'cancelled':return True
            extra = self._extra(str(rid), tuple(prefix))
            required = (count if count is not None else
                        min(DRAFT_TOKENS, self.budgets[str(rid)]-len(prefix))
                        if self.adaptive else self.kmin)
            return entry.source == 'stream' and extra is not None and len(extra) >= max(1, required)

    def take(self, rid, prefix, stride=WIDTH, count=None):
        if stride != WIDTH:
            raise ValueError('policy width mismatch')
        rid = str(rid)
        with self.lock:
            entry = self._entry(rid, prefix)
            if entry.source == 'ar':
                raise ValueError('AR marker is not a speculative proposal')
            if count is not None and not self.has_draft(rid,prefix,count=count):
                raise ValueError('explicit verification requires real supplied drafts')
            if self.adaptive and entry.source == 'stream' and not self.has_draft(rid, prefix):
                raise ValueError('adaptive verification requires real streamed drafts')
            self.entries.pop(rid)
            self.waits.pop(rid, None)
            self.stats['consumed'] += 1
            if entry.source != 'stream':
                value=list(entry.future.result()[entry.index])
                return value if count is None else value[:count]
            depth=DRAFT_TOKENS if count is None else count
            extra = (self._extra(rid, tuple(prefix)) or [])[:depth]
            # The legacy fixed-K caller retains its old short-budget padding.
            # Explicit ready-time actions consume exactly their real depth.
            tokens = ((extra+[extra[-1] if extra else prefix[-1]]*depth)[:depth]
                      if count is None else extra)
            self.stats['stream_draft'] += 1
            self.stats['stream_draft_real'] += len(extra)
            if extra:
                self.pending_draft[rid] = dict(prefix_len=len(prefix), real=len(extra))
            self.event('stream_draft', rid=rid, prefix_length=len(prefix), tokens=tokens, real=len(extra))
            return tokens

    def take_ar(self, rid, prefix):
        with self.lock:
            entry = self._entry(rid, prefix)
            self.entries.pop(str(rid))
            self.waits.pop(str(rid), None)
            if entry.source in ('ar', 'cancelled'):
                self.stats['ar_consumed'] += 1
            else:
                self.stats['never_wait_ar'] += 1

    def release(self, ids, retry_failed=False):
        ids = tuple(dict.fromkeys(str(rid) for rid in ids))
        with self.lock:
            for rid in ids:
                path = self.paths.pop(rid, None)
                if path is not None and path.handle is not None:
                    path.handle.close()
                self.warm_futures.pop(rid, None)
                self.scores.pop(rid, None)
                self.pending_draft.pop(rid, None)
                self.gap_seen.pop(rid, None)
                self.waits.pop(rid, None)
                self.budgets.pop(rid, None)
                self.recovery.discard(rid)
        return super().release(ids, retry_failed)


class FuturePrefetch(FutureCohortPolicy, Prefetch):
    """Native root capabilities use RPC futures under the same target policy."""
    def __init__(self, client, *, exact_prefix_drafts=False, root_top2_shadow=True):
        super().__init__(client, root_top2_shadow=root_top2_shadow)
        self.prepare_future_policy()

    submit_native = Prefetch.submit_committed
    draft_ready = Prefetch.ready


def make_prefetch(client, *, max_rows=None, exact_prefix_drafts=False, root_top2_shadow=False,
                  similar_context_drafts=False):
    """One policy, with transport-specific bounded proposal ownership."""
    if type(similar_context_drafts) is not bool:
        raise ValueError('similar-context modes require booleans')
    if (similar_context_drafts
            and (getattr(client, 'weak_edge_multiedge', False) or root_top2_shadow)):
        raise ValueError('similar-context drafts do not support peer or root-top2 transports')
    if getattr(client, 'weak_edge_multiedge', False):
        raise ValueError('only single-edge execution is included')
    if root_top2_shadow:
        return FuturePrefetch(client)
    return StreamPrefetch(client, max_rows=max_rows, exact_prefix_drafts=exact_prefix_drafts,
                          similar_context_drafts=similar_context_drafts)

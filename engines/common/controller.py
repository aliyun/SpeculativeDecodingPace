import asyncio
import time
import json
import os
from collections import OrderedDict

from .policy import MAX_REQUESTS, join_ms_from_env
from .cache_seed import load_configured
from .fingerprint import fingerprint
from .admission import AdmissionQueue
from engines.common.clients import EdgeClient, make_edge_client
from engines.common.contracts import Request
from engines.common.executor import Executor
from engines.common.prefetch import Prefetch
from engines.common.growth_prefetch import GrowthPrefetch
from engines.common.stream import StreamPrefetch, make_prefetch
from engines.common.detokenization import stable_text


def growth_prefetch_class(policy, transport):
    """Pick the supply transport for adaptive growth.

    `stream` and `auto` keep the live producer and bind stream plans at ready
    time. `auto` can also schedule synchronous rounds. `sync` uses
    the base prefetch, which issues one `sync_draft` batch per round and waits
    for the whole K instead of consuming stream frames.
    """
    if getattr(policy, 'streaming_selector', False) and transport in ('stream', 'auto'):
        from .stream_selector import StreamingGrowthPrefetch
        return StreamingGrowthPrefetch
    return GrowthPrefetch


class _IngressTokenizer:
    """Cardinality facade for one service's fixed, never-mutated vocabulary.

    Cache len(tokenizer), not vocab_size: added-token bounds must stay identical.
    Encoding and all ingress validation still use the original tokenizer/code.
    """
    __slots__ = ('_tokenizer', '_size')

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._size = len(tokenizer)

    def __len__(self):
        return self._size

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


class _DecodeMemo:
    """Bounded pure detokenization memo, private to one fixed tokenizer.

    This never supplies generated IDs or bypasses native target verification.
    Keys contain the complete, already target-committed generated sequence.
    """
    def __init__(self, tokenizer, max_entries=4096, max_units=4_194_304):
        if any(type(n) is not int or n < 1 for n in (max_entries, max_units)):
            raise ValueError('positive decode memo bounds required')
        self.tokenizer, self.max_entries, self.max_units = tokenizer, max_entries, max_units
        self.entries = OrderedDict()
        self.units = self.hits = self.misses = self.evictions = 0

    def decode(self, tokens):
        key = tuple(tokens)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        text = self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        cost = len(key) + len(text)
        if cost <= self.max_units:
            self.entries[key] = text
            self.units += cost
            while len(self.entries) > self.max_entries or self.units > self.max_units:
                old, value = self.entries.popitem(last=False)
                self.units -= len(old) + len(value)
                self.evictions += 1
        return text

    def snapshot(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    entries=len(self.entries), token_character_units=self.units)


def request_view(request):
    # Only native-owned, append-only serving state opts in at submit(). Its
    # immutable snapshot is invalidated BEFORE every mutation in emit().
    # Changed snapshots and all unowned callers retain the original validation.
    stats = request.get('_view_stats')
    if stats is not None and request.get('_request_snapshot') is not None:
        stats['hits'] += 1
        return request['_request_snapshot']
    view = Request(request['id'], tuple(request['tokens'][:request['input_length']]),
                   tuple(request['generated']), request['max_tokens'], request['done'])
    if stats is not None:
        stats['misses'] += 1
        request['_request_snapshot'] = view
    return view


class NativeController:
    def __init__(self,llm,tokenizer,edge_url,out,phase_profile=False,verify_profile=False):
        self.llm,self.tokenizer,self.edge_url,self.out=llm,tokenizer,edge_url,out
        self.ingress_tokenizer=tokenizer
        from .round_timing import RoundTiming
        self.lock=asyncio.Lock();self.waiting=AdmissionQueue(maxsize=512)
        # In-memory round timing, off unless TRIAGE_ROUND_TIMING=1 asks for it.
        self.round_timing=RoundTiming(emit=lambda name,**fields:self.prefetch.event(name,**fields))
        self.active=OrderedDict();self.cancelled=set();self.failure=None
        self.token_limit=None
        self.submitted=self.completed=0;self.peak_active=0
        self.admit_calls=0;self.max_admit_batch=0
        self.next_enqueue_seq=0;self.last_admitted_seq=-1;self.admission_deferrals=0
        self.verify_profile=verify_profile
        self.phase_profile=phase_profile;self.profile_round=None;self.profile_sequence=0
        self.profile_records=0;self.profile_write_ns=0;self.profile_bytes=0
        self.view_stats=None;self.decode_memo=None
        self.native_progress=0;self.progress_yields=0


    async def timed(self,name,operation):
        start=time.monotonic_ns()
        try:return await operation
        finally:
            end=time.monotonic_ns()
            if self.profile_round is not None:
                self.profile_round['spans'].append(dict(stage=name,start_ns=start,end_ns=end))


    def begin_round(self):
        if not self.phase_profile:return
        self.profile_sequence+=1
        self.profile_round=dict(round_id=self.profile_sequence,start_ns=time.monotonic_ns(),spans=[],requests=[])


    def finish_round(self):
        record=self.profile_round;self.profile_round=None
        if record is None or not record['spans']:return
        record['end_ns']=time.monotonic_ns()
        start=time.monotonic_ns()
        line=json.dumps(record,separators=(',',':'))+'\n'
        with (self.out/'PHASE_TRACE.jsonl').open('a') as file:file.write(line)
        self.profile_write_ns+=time.monotonic_ns()-start
        self.profile_records+=1;self.profile_bytes+=len(line.encode())


    def trace(self,event,r,**extra):
        if self.out is None:return
        record=dict(event=event,id=r['id'],enqueue_seq=r.get('enqueue_seq'),
                    ns=time.monotonic_ns(),active=len(self.active),waiting=self.waiting.qsize(),**extra)
        with (self.out/'REQUEST_TRACE.jsonl').open('a') as file:
            file.write(json.dumps(record,separators=(',',':'))+'\n')


    def submit(self,r):
        if self.view_stats is not None:
            r['_view_stats']=self.view_stats
            r.pop('_request_snapshot',None)
        self.waiting.put_nowait(r)
        r['enqueue_seq']=self.next_enqueue_seq;self.next_enqueue_seq+=1
        r['enqueue_ns']=time.monotonic_ns();self.submitted+=1
        self.trace('enqueue',r,input_tokens=len(r['tokens']),max_tokens=r['max_tokens'])


    def resource_trace(self,resource,payload,start):
        if not self.phase_profile or payload['op'] not in ('admit','draft','verify','commit','free'):return
        row=dict(resource=resource,op=payload['op'],start_ns=start,end_ns=time.monotonic_ns(),
                 round_id=self.profile_round['round_id'] if self.profile_round else None,
                 ids=payload.get('ids',[r['id'] for r in payload.get('requests',[])]))
        with (self.out/'RESOURCE_TRACE.jsonl').open('a') as file:
            file.write(json.dumps(row,separators=(',',':'))+'\n')


    async def emit(self,verified):
        emit_start=time.monotonic_ns()
        finished=[]
        for v in verified:
            r=self.active[v['id']]
            if self.view_stats is not None:r.pop('_request_snapshot',None)
            limit=getattr(self,'token_limit',None)
            if limit is not None and any(type(t) is not int for t in v['emitted']):
                raise ValueError('target token outside negotiated text vocabulary')
            immutable=getattr(self,'immutable_views_enabled',False)
            remain=r['max_tokens']-len(r['generated'])
            emitted=v['emitted'][:remain]
            stop=None
            for i,t in enumerate(emitted):
                if t in r['eos_ids']:emitted=emitted[:i+1];stop='stop';break
            # The verifier may commit a tail beyond EOS/the request budget.
            # Only delivered IDs belong to the negotiated text protocol.
            if limit is not None and any(not 0<=t<limit for t in emitted):
                raise ValueError('target token outside negotiated text vocabulary')
            assert v['prefix_hash']==fingerprint(r['tokens'])
            if immutable:r['tokens']=tuple(r['tokens'])+tuple(v['emitted'])
            else:r['tokens'].extend(v['emitted'])
            assert v['next_prefix_hash']==fingerprint(r['tokens'])
            if self.profile_round is not None:
                self.profile_round['requests'].append(dict(id=r['id'],accepted_draft_tokens=len(v['path'])-1,
                    raw_emitted_tokens=len(v['emitted']),delivered_tokens=len(emitted)))
            if immutable:r['generated']=tuple(r['generated'])+tuple(emitted)
            else:r['generated'].extend(emitted)
            if stop is None and len(r['generated'])>=r['max_tokens']:stop='length'
            text_plan=getattr(self,'text_plan',None)
            decoded_ids=list(r['generated']) if immutable else r['generated']
            if self.decode_memo is not None and text_plan is None:
                text=self.decode_memo.decode(decoded_ids)
                if not stop:text=text.rstrip('\ufffd')
            else:
                text=stable_text(self.tokenizer,decoded_ids,text_plan,final=stop is not None)
            assert text.startswith(r['text']),'NONMONOTONIC_DETOKENIZATION'
            delta=text[len(r['text']):];r['text']=text
            if delta and not r.get('first_content_traced'):
                r['first_content_traced']=True;self.trace('first_content',r)
            await r['queue'].put({'text':delta,'token_ids':emitted,'finish_reason':stop})
            if stop:
                r['done']=True;finished.append(r['id']);self.completed+=1
                self.trace('complete',r,finish_reason=stop)
        if self.profile_round is not None:
            self.profile_round['spans'].append(dict(stage='emit_enqueue',start_ns=emit_start,end_ns=time.monotonic_ns()))
        return finished


    async def rpc(self, payload):
        # The original owner, not the HTTP waiter, holds the native lock. Even
        # repeated cancellation cannot cancel a to_thread job or release its KV.
        async def owned():
            from engines.common.growth import enabled
            async with self.lock:
                if enabled() and getattr(self,'native_rpc_failure',None) and payload.get('op')!='status':
                    raise RuntimeError('NATIVE_RPC_UNKNOWN_REQUIRES_BOTH_ENDPOINTS_RESTART')
                start = time.monotonic_ns()
                try:
                    result = (await asyncio.to_thread(self.llm.collective_rpc,
                        'tree_service_call', args=(payload,), timeout=120))[0]
                    observe = getattr(getattr(self, 'growth', None), 'observe_geometry', None)
                    if observe is not None:
                        for row in result if isinstance(result, list) else [result]:
                            if isinstance(row, dict):observe(row.get('native_shape'))
                    return result
                except Exception as exc:
                    # A timed-out collective does not prove native completion.
                    # Keep the growth worker quarantined: no replay or fake free.
                    if enabled():self.native_rpc_failure=repr(exc)
                    raise
                finally:
                    self.resource_trace('cloud', payload, start)
        work=asyncio.create_task(owned())
        jobs=getattr(self,'rpc_jobs',None)
        if jobs is None:self.rpc_jobs=jobs=set()
        jobs.add(work)
        def finished(original):
            jobs.discard(original)
            if not original.cancelled():original.exception()
        work.add_done_callback(finished)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            # Preserve the original join-on-cancel contract as well. Every
            # repeated cancel is absorbed by a fresh shield, never by the owner.
            while not work.done():
                try:await asyncio.shield(work)
                except asyncio.CancelledError:continue
                except Exception:break
            raise


class Controller(NativeController):
    def __init__(self, *args, exact_prefix_drafts=False, latency_aware_prefill=False,
                 cpu_prompt_warm=False, native_prompt_warm=False, cpu_stream_drafts=False, cpu_stream_rows=None,
                 weak_edge_multiedge=False, stream_policy='cohort', bounded_edge_admission=False,
                 edge_client=None, **kwargs):
        from .policy import MAX_REQUESTS
        if type(exact_prefix_drafts) is not bool:
            raise ValueError('invalid exact-prefix draft switch')
        if type(latency_aware_prefill) is not bool or latency_aware_prefill and not exact_prefix_drafts:
            raise ValueError('latency-aware prefill requires exact-prefix drafts')
        if type(cpu_prompt_warm) is not bool or type(native_prompt_warm) is not bool:
            raise ValueError('invalid prompt warm switch')
        from .growth import enabled
        if native_prompt_warm and (not enabled() or os.environ.get('TRIAGE_EDGE_BACKEND','native')!='native'):
            raise ValueError('native prompt warm requires adaptive native edge')
        if type(cpu_stream_drafts) is not bool:
            raise ValueError('invalid stream draft switch')
        if stream_policy != 'cohort':
            raise ValueError('cohort-pipeline is the only proposal policy')
        if cpu_stream_rows is not None and (type(cpu_stream_rows) is not int
                                            or not 1 <= cpu_stream_rows <= MAX_REQUESTS):
            raise ValueError('stream row limit requires a bound of 1..%d' % MAX_REQUESTS)
        if type(weak_edge_multiedge) is not bool or weak_edge_multiedge:
            raise ValueError('only single-edge execution is included')
        self.weak_edge_multiedge = weak_edge_multiedge
        self.cpu_prompt_warm = cpu_prompt_warm
        self.native_prompt_warm = native_prompt_warm
        root_shadow = kwargs.pop('root_top2_shadow', False)
        self.root_fork5 = kwargs.pop('root_fork5', False)
        if type(self.root_fork5) is not bool or self.root_fork5 and root_shadow:
            raise ValueError('ROOT_FORK5_CONTROLLER_MODE')
        if type(root_shadow) is not bool:
            raise ValueError('ROOT_TOP2_SHADOW_MODE')
        from engines.common.growth import enabled
        if (root_shadow or self.root_fork5) and not enabled() and (cpu_prompt_warm or os.environ.get('TRIAGE_EDGE_BACKEND','native') != 'native'):
            raise ValueError('ROOT_FORK5_REQUIRES_NATIVE_EDGE')
        super().__init__(*args, **kwargs)
        self.root_shadow = None
        self.growth = None
        from engines.common.growth import enabled
        from engines.common.growth_batch_profile import enabled as batch_enabled
        ready_mode=os.environ.get('TRIAGE_GROWTH_READY_WAKE','0')
        if (ready_mode not in ('0','1') or ready_mode=='1' and
                (not enabled() or not batch_enabled() or not self.root_fork5
                 or os.environ.get('TRIAGE_SHAPE_BUDGET_GROWTH')!='1')):
            raise ValueError('READY_WAKE_REQUIRES_COMPLETE_BATCH_PROFILE')
        self.ready_wake_enabled=ready_mode=='1'
        view_mode=os.environ.get('TRIAGE_GROWTH_IMMUTABLE_VIEWS','0')
        if (view_mode not in ('0','1') or view_mode=='1' and
                (not enabled() or not batch_enabled() or not self.root_fork5
                 or os.environ.get('TRIAGE_SHAPE_BUDGET_GROWTH')!='1')):
            raise ValueError('IMMUTABLE_VIEWS_REQUIRES_COMPLETE_BATCH_PROFILE')
        self.immutable_views_enabled=view_mode=='1'
        self._immutable_views=None
        self.progress_yield = True
        self.ingress_tokenizer=(_IngressTokenizer(self.tokenizer) if self.tokenizer is not None else None)
        self.view_stats=dict(hits=0,misses=0)
        self.decode_memo=_DecodeMemo(self.tokenizer)

        client = edge_client if edge_client is not None else (
            make_edge_client(self.edge_url, bounded_capacity=bounded_edge_admission)
            if os.environ.get('TRIAGE_MODEL_PATH')
            else EdgeClient(self.edge_url))
        if enabled():
            if not self.root_fork5 or root_shadow:raise ValueError('ADAPTIVE_GROWTH_CONTROLLER_SCOPE')
            from engines.common.growth import Policy,create_policy,draft_transport
            factory=Policy if os.environ.get('TRIAGE_GROWTH_EXPERIMENT')=='1' else create_policy
            self.growth=factory(lambda event,**fields:Prefetch.event(None,event,**fields))
            transport=draft_transport()
            self.draft_transport=transport
            prefetch_type=growth_prefetch_class(self.growth,transport)
            prefetch=prefetch_type(client,self.growth,exact_prefix_drafts=exact_prefix_drafts,
                max_rows=cpu_stream_rows,
                enable_stream=transport in ('stream','auto') and os.environ.get('TRIAGE_EDGE_BACKEND','native') in ('native','hf'))
        else:prefetch=make_prefetch(client, max_rows=cpu_stream_rows,
            root_top2_shadow=root_shadow or self.root_fork5)
        # One seed file, loaded before the service can answer, so the first real
        # request already sees the three lookup tiers. No-op when unset.
        load_configured(prefetch)
        self.executor = Executor(prefetch=prefetch, join_ms=join_ms_from_env())
        self.admission_capacity = getattr(client, 'session_capacity', MAX_REQUESTS) if bounded_edge_admission else MAX_REQUESTS
        self.growth_context = {}
        self.growth_rpc_ms = 0.
        self.growth_microbatch_deferrals=0
        self.growth_deferred=()
        if root_shadow:
            from engines.common.root_shadow import RootTop2Shadow
            self.root_shadow = RootTop2Shadow()

        self.latency_aware_prefill = latency_aware_prefill
        self.prefill_slices = self.ready_decode_turns = 0
        self.admitting = []

    @property
    def prefetch(self):
        return self.executor.prefetch

    @prefetch.setter
    def prefetch(self, value):
        self.executor.prefetch = value

    @property
    def ar_rounds(self):
        return self.executor.ar_rounds

    @property
    def spec_rounds(self):
        return self.executor.spec_rounds

    def _request_view(self,request):
        if not getattr(self,'immutable_views_enabled',False):return request_view(request)
        if getattr(self,'_immutable_views',None) is None:
            from engines.common.request_views import RequestViews
            self._immutable_views=RequestViews()
        return self._immutable_views.get(request,request_view)

    async def free(self, ids):
        self.executor.finish(ids)
        if ids:
            await self.rpc(dict(op='free', ids=ids))
            for rid in ids:
                self.active.pop(rid, None)
                self.cancelled.discard(rid)
                if getattr(self,'growth',None):self.growth_context.pop(rid,None)
            if getattr(self,'_immutable_views',None) is not None:self._immutable_views.forget(ids)

    async def advance(self, verified):
        growth=getattr(self,'growth',None)
        resident_length=max((len(r['tokens']) for r in self.active.values()),default=0)
        before={r['id']:(len(self.active[r['id']]['generated']),self.active[r['id']]['max_tokens']-len(self.active[r['id']]['generated']),len(self.active[r['id']]['tokens'])) for r in verified} if growth else {}
        if growth:
            from engines.common.growth import Plan
            # Reject a mismatched/ill-typed native echo before any client-visible
            # delivery. The post-emit check below remains an independent gate.
            for row in verified:
                plan=self.executor.growth_plans.get(row['id'])
                if plan is not None:
                    try:echo=Plan.bind(row.get('growth_plan'),row['id'],self.active[row['id']]['tokens'])
                    except (ValueError,TypeError) as exc:raise ValueError('GROWTH_CONTROLLER_RESULT_PLAN') from exc
                    if echo!=plan:raise ValueError('GROWTH_CONTROLLER_RESULT_PLAN')
        finished = await self.emit(verified)
        if growth:
            batch_feedback=[]
            for row in verified:
                rid=row['id'];n,remaining,prefix_length=before[rid];plan=self.executor.growth_plans.get(rid)
                if plan is not None and row.get('growth_plan')!=plan.wire():raise ValueError('GROWTH_CONTROLLER_RESULT_PLAN')
                meta=row.get('growth_meta',{})
                growth.resources(active=len(self.active)-len(finished),waiting=self.waiting.qsize(),free_blocks=meta.get('free_blocks'),capacity=meta.get('capacity'))
                feedback=dict(rid=rid,delivered=len(self.active[rid]['generated'])-n,remaining=remaining,raw_emitted=len(row['emitted']),prefix_length=prefix_length,plan=plan,source=self.growth_context.pop(rid,None),side=meta.get('side',False))
                if getattr(growth,'streaming_selector',False):
                    feedback['native_mode']=self.executor.executing_mode.get(rid)
                    feedback['native_shape']=row.get('native_shape')
                    feedback['resident_prefix_length']=resident_length
                    feedback['terminal_censored']=(rid in finished or remaining<=len(row['emitted']))
                if getattr(growth,'cost_mode',False) and plan is not None:
                    feedback['query_nodes']=meta.get('queries')
                if hasattr(growth,'commit_batch'):batch_feedback.append(feedback)
                else:growth.commit(**feedback,batch=len(verified),rpc_ms=self.growth_rpc_ms)
            if hasattr(growth,'commit_batch'):growth.commit_batch(batch_feedback,rpc_ms=self.growth_rpc_ms)
        self.executor.committed([self._request_view(self.active[row['id']]) for row in verified])
        await self.free(finished)
        if verified:
            self.native_progress += 1

    def ready_decode(self):
        # Only inspect completed proposal futures; never block admission on an RPC.
        return (self.latency_aware_prefill and bool(self.active)
                and all(rid.startswith('ec-') for rid in self.active)
                and any((self.prefetch.has_draft if getattr(self.prefetch, 'cohort', False) else
                         self.prefetch.ready)(r['id'], self._request_view(r).prefix)

                        for r in self.active.values()))

    def latency_snapshot(self):
        result=dict(prefill_slices=self.prefill_slices, ready_decode_turns=self.ready_decode_turns)
        if self.view_stats is not None:
            result.update(native_view_cache=dict(self.view_stats), decode_memo=self.decode_memo.snapshot(),
                          native_progress=self.native_progress, progress_yields=self.progress_yields,
                          growth_microbatch_deferrals=getattr(self,'growth_microbatch_deferrals',0),
                          growth_verify_batch_max=8 if getattr(self,'growth',None) else 16,
                          resident_limit=16)
        return result

    def _growth_decision(self,decision,views):
        if not getattr(self,'growth',None) or not decision.selected:return decision
        if not all(self.prefetch.has_draft(rid,views[rid].prefix) for rid in decision.selected):return decision
        from dataclasses import replace
        deferred=getattr(self,'growth_deferred',())
        ordered=tuple(rid for rid in deferred if rid in decision.selected)+tuple(rid for rid in decision.selected if rid not in deferred)
        plans={rid:self.prefetch.plan_for(rid,views[rid].prefix) for rid in ordered}
        if any(p is None for p in plans.values()):raise ValueError('GROWTH_SELECTION_MISSING_PLAN')
        first=plans[ordered[0]];action=(first.k,first.branches)
        chosen=tuple(rid for rid in ordered if (plans[rid].k,plans[rid].branches)==action)[:8]
        self.growth_deferred=tuple(rid for rid in ordered if rid not in chosen)
        if self.growth_deferred:
            self.growth_microbatch_deferrals+=len(self.growth_deferred)
            self.prefetch.event('growth_verify_microbatch',selected=list(ordered),executing=list(chosen),
                                resident=len(self.active),native_batch_max=8)
        self.executor.native_selection(chosen,'decode')
        return replace(decision,selected=chosen)

    async def tick(self):
        if getattr(self,'growth',None):
            self.growth.refresh();self.growth.resources(active=len(self.active),waiting=self.waiting.qsize())
        self.prefetch.poll_cleanup()
        await self.free([rid for rid in self.active if rid in self.cancelled])
        for rid in self.waiting.remove_cancelled(self.cancelled):
            self.cancelled.discard(rid)
        ready_decode = self.ready_decode()
        budget = self.executor.admission_budget(len(self.active))
        capacity = getattr(self, 'admission_capacity', MAX_REQUESTS)
        if capacity <= MAX_REQUESTS:
            # A full-size edge pool also owns slots until asynchronous free is
            # acknowledged; target completion alone cannot admit replacements.
            with self.prefetch.lock:
                resident = len(set(self.active) | set(self.prefetch.releasing))
            budget = min(budget, max(0, capacity-resident))
        # tick already offers decode after first_token; shorten only the intervening prefill.
        candidates = self.waiting.peek(min(budget, 1) if ready_decode else budget)
        if candidates:
            self.admitting = [r['id'] for r in candidates]
            self.admit_calls += 1
            self.max_admit_batch = max(self.max_admit_batch, len(candidates))
            result = await self.rpc(dict(op='admit', requests=[
                {key: r[key] for key in ('id', 'tokens', 'max_tokens')} for r in candidates]))
            admitted = result['admitted']
            if admitted != [r['id'] for r in candidates[:len(admitted)]]:
                raise ValueError('non-FIFO target admission')
            self.admission_deferrals += len(candidates)-len(admitted)
            for request in candidates[:len(admitted)]:
                if self.waiting.get_nowait() is not request:
                    raise ValueError('admission queue changed')
                self.active[request['id']] = request
            self.peak_active = max(self.peak_active, len(self.active))
            self.admitting.clear()
            if admitted:
                if self.cpu_prompt_warm or self.native_prompt_warm:
                    items=[(rid, self.active[rid]['tokens']) for rid in admitted]
                    if self.native_prompt_warm:
                        items=[(rid,tokens,self.active[rid]['max_tokens']) for rid,tokens in items]
                    self.prefetch.warm(items)
                verified = await self.rpc(dict(op='first_token', requests=[
                    dict(id=rid, prefix_hash=fingerprint(self.active[rid]['tokens'])) for rid in admitted]))
                if [r['id'] for r in verified] != admitted:
                    raise ValueError('target first-token batch binding')
                await self.advance(verified)
                if ready_decode:
                    self.prefill_slices += 1
        # Normal growth reuses this synchronous tick's snapshots. The explicit
        # immutable mode may also reuse one across waits, but only while owned
        # tuple identities and every Request control field are unchanged.
        views = ({rid: self._request_view(r) for rid, r in self.active.items()}
                 if getattr(self, 'growth', None) else None)
        decision = self.executor.plan(list(views.values()) if views is not None
                                      else [self._request_view(r) for r in self.active.values()])
        # B1..8 physical catalogue: preserve resident=16 and pending plans.
        # Oldest deferred plans run first; this is explicit batching, no replay.
        decision=self._growth_decision(decision,views)
        if decision.selected:
            # Both payloads refer to these exact same immutable tick snapshots.
            digests = ({rid: fingerprint(views[rid].prefix) for rid in decision.selected}
                       if views is not None else None)
            shadow_context = None
            if self.root_shadow is not None or getattr(self, 'root_fork5', False):
                from engines.common.policy import execution_mode
                if (execution_mode(len(self.active[rid]['tokens']) for rid in decision.selected).mode == 'spec'
                        and all(self.prefetch.has_draft(rid, self._request_view(self.active[rid]).prefix)
                                for rid in decision.selected)):
                    shadow_context = []
                    for rid in decision.selected:
                        request = self.active[rid]
                        shadow_context.append(dict(id=rid, prefix_hash=digests[rid] if digests is not None else fingerprint(request['tokens']),
                            remaining=request['max_tokens']-len(request['generated']),
                            prefix_length=len(request['tokens']),
                            ignore_eos=not request['eos_ids'], before=len(request['generated']), request=request,
                            **self.prefetch.shadow_snapshot(rid, request['tokens'])))
            dispatch = self.executor.begin([views[rid] if views is not None
                                            else self._request_view(self.active[rid]) for rid in decision.selected])
            proposals = [dict(id=p.rid, prefix_hash=digests[p.rid] if digests is not None else fingerprint(p.prefix),
                             **({} if dispatch.mode == 'ar' else dict(nodes=list(p.nodes))),
                             **({'growth_plan':p.growth_plan.wire()} if p.growth_plan is not None else {}))
                         for p in dispatch.proposals]
            start = time.monotonic() if self.weak_edge_multiedge else None
            if getattr(self, 'root_fork5', False) and dispatch.mode == 'spec':
                from engines.common.fork5 import pack
                proposals = pack(proposals, shadow_context)
            if getattr(self,'growth',None) and shadow_context is not None:
                self.growth_context.update({c['id']:c['source'] for c in shadow_context})
            growth_rpc_start=time.monotonic_ns()
            self.round_timing.issue([p['id'] for p in proposals])
            verified = await self.rpc(dict(op='ar_step' if dispatch.mode == 'ar' else 'verify', requests=proposals))
            self.round_timing.verified()
            if getattr(self,'growth',None):self.growth_rpc_ms=(time.monotonic_ns()-growth_rpc_start)/1e6
            if [r['id'] for r in verified] != list(decision.selected):
                raise ValueError('target verification batch binding')
            if self.weak_edge_multiedge and dispatch.mode == 'spec':
                self.prefetch.observe_target(len(proposals), time.monotonic()-start)
            await self.advance(verified)
            self.round_timing.done([r['id'] for r in verified])
            if ready_decode:
                self.ready_decode_turns += 1
            if shadow_context is not None and self.root_shadow is not None:
                for row in shadow_context:
                    row['delivered'] = len(row.pop('request')['generated'])-row.pop('before')
                self.root_shadow.observe(proposals, verified, shadow_context)

    async def loop(self):
        wake=None
        wake_owners=[self.prefetch]
        stream=getattr(self.prefetch,'stream',None)
        if stream is not None:wake_owners.append(stream)
        try:
            # Default native hybrid drafts arrive on reader threads. Wake the
            # owner promptly, without altering its readiness/selection checks.
            if getattr(self,'ready_wake_enabled',False) or stream is not None:
                from engines.common.ready_wake import CompletionWake
                if any(getattr(owner,'ready_wake',None) is not None for owner in wake_owners):
                    raise RuntimeError('GROWTH_READY_WAKE_ALREADY_ATTACHED')
                wake=CompletionWake()
                for owner in wake_owners:owner.ready_wake=wake
            while True:
                if wake is not None:wake.clear()
                before = self.native_progress
                await self.tick()
                progressed = self.progress_yield and self.native_progress != before
                if progressed:
                    self.progress_yields += 1
                if progressed or wake is None:
                    await asyncio.sleep(0 if progressed else .001)
                else:
                    await wake.wait(.001)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = repr(exc)
            try:
                import traceback
                self.prefetch.event('controller_failure',error=repr(exc),
                                    trace=traceback.format_exc()[-6000:])
            except Exception:
                pass
            requests = list(self.active.values())
            while not self.waiting.empty():
                requests.append(self.waiting.get_nowait())
            for request in requests:
                await request['queue'].put({'error': self.failure})
            await self.free(list(dict.fromkeys([*self.active, *self.admitting])))
        finally:
            # One last aggregate so the final window is not truncated mid-period.
            # getattr: loop() is also driven by test doubles without a timer.
            timing = getattr(self, 'round_timing', None)
            if timing is not None:
                timing.report(final=True)
            if wake is not None:
                wake.close()
                for owner in wake_owners:
                    if getattr(owner,'ready_wake',None) is wake:owner.ready_wake=None

from concurrent.futures import Future
from dataclasses import dataclass
from .prefetch import Prefetch,Entry
from .draft_policy import FutureCohortPolicy
import os
import threading
import time
from .policy import SuffixCache,WIDTH,MAX_REQUESTS,CONTEXT,JOIN_MS,execution_mode,fingerprint,token_ids
from .similar_cache import (UnifiedContextCache, fuzzy_mode_from_env,
                            fuzzy_threshold_from_env, prompt_width_from_env,
                            min_width_from_env, repeat_gate_from_env,
                            repeat_min_from_env, unique_off_from_env,
                            auto_gate_from_env, gate_ratio_from_env,
                            gate_window_from_env, per_request_gate_from_env, multi_suffix_from_env,
                            progress_from_env)
from .root_shadow import validate_top2
from .growth import Plan
from .contracts import Request

class GrowthSuffixCache(SuffixCache):
    def __init__(self):super().__init__(horizon=7)

@dataclass(frozen=True)
class GrowthEntry(Entry):
    plan:Plan|None=None

class GrowthPrefetch(FutureCohortPolicy, Prefetch):
    def __init__(self,client,growth,*,exact_prefix_drafts=False,max_rows=None,enable_stream=False):
        super().__init__(client,root_top2_shadow=True)
        self.prepare_future_policy()
        self.growth = growth
        # Short generated suffixes can supply cold-cache proposals; every
        # proposed token still requires target verification.
        self.cache = self.similar_cache = UnifiedContextCache(
            horizon=7, min_suffix=2, suffix_widths=(2, 3, 4, 8, 12),
            fuzzy_mode=fuzzy_mode_from_env(),
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
        # Opt-in per-round timing: where does a plan round's wall time go?
        # submit -> reply (client RTT + edge drafting), reply -> take (executor
        # scheduling), take -> commit (verification and delivery). Default off.
        self.plan_timing = os.environ.get('TRIAGE_PLAN_TIMING') == '1'
        self.top2_inbox = []
        self.plan_t0 = {}
        self.plan_t1 = {}
        self.plan_take = {}
        self.sync_priced_join = os.environ.get('TRIAGE_SYNC_PRICED_JOIN','1')=='1'
        self.plan_join_cap_ms = float(os.environ.get('TRIAGE_SYNC_JOIN_CAP_MS') or 300)
        self.plan_join_margin_ms = float(os.environ.get('TRIAGE_SYNC_JOIN_MARGIN_MS') or 15)
        self.plan_submitted = {}
        self.plan_rpc_ms = float(os.environ.get('TRIAGE_SYNC_RPC_MS') or 121)

        self.stats['similar_context_hit'] = 0
        self.stats['exact_prefix_hit'] = 0
        client.negotiate_growth()
        self.context=getattr(client,'context',None) or CONTEXT
        limit=getattr(client,'max_rows',MAX_REQUESTS) or MAX_REQUESTS
        # HF edge negotiates its real session slots, not the native row default.
        limit=min(limit,getattr(client,'session_capacity',MAX_REQUESTS))
        if max_rows is not None:
            if type(max_rows) is not int or not 1<=max_rows<=MAX_REQUESTS:raise ValueError('GROWTH_ROW_LIMIT')
            limit=min(limit,max_rows)
        self.max_rows=limit;self.edge_owned=set()
        self._on_edge_free_ack = self._discard_edge_owned
        self.backend=os.environ.get('TRIAGE_EDGE_BACKEND','native')
        # Legacy launch arguments remain accepted, but no second cache is built.
        self.exact_cache = None
        # Hybrid transport: the fixed baseline action (K3 single root) is supplied by
        # the edge stream so drafts overlap verification, exactly as the non-adaptive
        # path does. Only wider actions (K5/K7/double root) pay a plan RPC. The two
        # sides never own the same request at once, so each request is observed once.
        self.stream=None;self.stream_plans={};self.streamed_rids=set()
        self.fallback_native_plans={}
        if enable_stream:
            from .stream import StreamPrefetch
            if self.backend not in ('native','hf'):raise ValueError('GROWTH_STREAM_REQUIRES_NATIVE_EDGE')
            stream = StreamPrefetch(client, max_rows=max_rows)
            stream.similar_cache = self.similar_cache
            stream.stats['similar_context_hit'] = 0
            self.stream = stream
        # FutureCohortPolicy is RPC-only, but the hybrid delegate also needs the
        # service-owned readers prepared before requests, not Thread.start per open.
        self.uses_stream_readers=self.stream is not None
    def _call(self,items):
        ids={r['id'] for r in items}
        with self.lock:self.active_rids.update(ids)
        try:
            t_start=time.perf_counter() if getattr(self,'plan_timing',False) else None
            self.event('edge_start',ids=[r['id'] for r in items],prefix_lengths=[len(r['tokens']) for r in items])
            plans=[Plan.bind(r['growth_plan'],r['id'],r['tokens']) for r in items]
            from .growth_batch_profile import enabled as batch_growth_enabled,BATCH_MAX
            batched=batch_growth_enabled()
            if (len({(p.k,p.branches) for p in plans})!=1 or plans[0].k!=3
                    and not (1<=len(items)<=BATCH_MAX if batched else len(items)==1)):raise ValueError('GROWTH_EDGE_BATCH')
            if self.stream is not None:
                streamers=[item for item in items if item['id'] in self.streamed_rids]
                if streamers:
                    with self.stream.lock:
                        for item in streamers:
                            self.stream._pause(item['id'])
            try:
        # Respect the client's batching capability and never replay a mutating
        # request after a transport failure.
                rows=[]
                width=1 if self.backend in ('cpu','webgpu') else BATCH_MAX
                for start in range(0,len(items),width):rows.extend(self.client.propose(items[start:start+width]))
            except Exception as exc:
                from .clients import EdgeHTTPError
                if self.backend!='webgpu' or not isinstance(exc,EdgeHTTPError) or exc.status!=503:raise
                self.event('growth_edge_unavailable',ids=sorted(ids),reason='browser_503_no_replay')
                return [None]*len(items)
            if not isinstance(rows,list) or len(rows)!=len(items):raise ValueError('GROWTH_EDGE_RESPONSE')
            t_meta0=time.perf_counter() if t_start is not None else None
            metadata=[]
            for item,row,p in zip(items,rows,plans):
                if (row.get('id')!=p.rid or row.get('prefix_hash')!=p.prefix_hash
                        or Plan.bind(row.get('growth_plan'),p.rid,item['tokens'])!=p
                        or len(token_ids(row.get('tokens')))!=p.k+(p.fork_len if p.fork_at else 0)):raise ValueError('GROWTH_EDGE_BINDING')
                if p.branches==1:
                    if 'root_top2' not in row or row['root_top2'] is not None:raise ValueError('GROWTH_ROOT1_CAPABILITY')
                else:metadata.append((p.rid,p.prefix_hash,validate_top2(row.get('root_top2'),row['tokens'][0])))
            t_lock0=time.perf_counter() if t_start is not None else None
            if metadata:self.top2_inbox.append((ids,metadata))
            if t_start is not None:
                # Split the plan round: work before the HTTP call, the call itself,
                # then the metadata validation and the growth-lock merge.
                pre=getattr(self.client,'_propose_pre',None)
                post=getattr(self.client,'_propose_post',None)
                now=time.perf_counter()
                if pre is not None and post is not None:
                    self.event('propose_timing',rid=items[0]['id'],
                               pre_ms=round((pre-t_start)*1000.0,2),
                               http_ms=round((post-pre)*1000.0,2),
                               meta_ms=round(((t_lock0 or now)-post)*1000.0,2),
                               lock_ms=round((now-(t_lock0 or now))*1000.0,2),
                               total_ms=round((now-t_start)*1000.0,2))
            self.event('edge_result',ids=[r['id'] for r in items],rows=rows)
            if getattr(self,'plan_timing',False):
                t1=time.perf_counter()
                for row in rows:
                    try: gp=row['growth_plan']; key=(gp['rid'],int(gp['round_id']))
                    except Exception: continue
                    t0=self.plan_t0.pop(key,None)
                    if t0 is None: continue
                    self.plan_t1[gp['rid']]=t1
                    self.event('plan_timing',rid=gp['rid'],round_id=key[1],
                               rpc_ms=round((t1-t0)*1000.0,2))
            return [r['tokens'] for r in rows]
        finally:
            with self.lock:self.active_rids.difference_update(ids)
            self.event('edge_end',ids=[r['id'] for r in items])
    def _fold_top2(self):
        """Merge the plan workers' root top-2 entries under the ledger's lock.

        A newer round may replace metadata abandoned by AR or a lane switch.
        Repeated replies for the same committed prefix are counted separately
        and keep the existing metadata.
        """
        while self.top2_inbox:
            ids,metadata=self.top2_inbox.pop()
            if len(set(self.root_top2)|ids)>MAX_REQUESTS:
                raise ValueError('GROWTH_TOP2_PENDING')
            for rid,digest,meta in metadata:
                old=self.root_top2.get(rid)
                if old is not None and old[0]==digest:
                    self.stats['top2_duplicate']=self.stats.get('top2_duplicate',0)+1
                    continue
                if old is not None:
                    self.stats['top2_replaced']=self.stats.get('top2_replaced',0)+1
                self.root_top2[rid]=(digest,meta)

    def _plans(self,snapshots,prefixes,mode):
        """Price only bounded real native cohorts, preserving the resident 16."""
        from .growth_batch_profile import BATCH_MAX
        plans={}
        width=min(BATCH_MAX,self.max_rows) if getattr(self.growth,'live_runtime',False) else BATCH_MAX
        for start in range(0,len(snapshots),width):
            group=snapshots[start:start+width]
            all_prefixes=[(rid,prefixes[rid]) for r,rid,generated in group]
            fresh=[(rid,prefix) for rid,prefix in all_prefixes
                   if rid not in self.stream_plans
                   and (rid not in self.entries or (self.entries[rid].source=='ar')!=(mode=='ar'))]
            if hasattr(self.growth,'prepare_cohort'):
                max_total=max((len(prefixes[rid])+r.sampling_params.max_new_tokens-len(generated) for r,rid,generated in group),default=0)
                args=({'remaining_by_rid':{rid:r.sampling_params.max_new_tokens-len(generated) for r,rid,generated in group}} if getattr(self.growth,'budget_mode',False) else {})
                if getattr(self.growth,'source_mode',False):
                    from .growth_source_cost import source_hint
                    args['source_by_rid']={rid: 'edge' if getattr(self,'auto_transport',False) and self.second_answer
                        else self._forecast_source(rid,prefixes[rid],generated) for r,rid,generated in group}
                self.growth.prepare_cohort(fresh,all_prefixes,mode=mode,max_total_tokens=max_total,**args)
            if mode=='spec':
                for rid,prefix in fresh:plans[rid]=self.growth.choose(rid,prefix,len(group))
        return plans

    def _forecast_source(self,rid,prefix,generated):
        """Source the plan is priced against; recorded so the loop can honour it."""
        held=self.entries.get(rid)
        if held is not None and held.source in ('edge','suffix','exact_prefix'):
            hint=held.source
        else:
            from .growth_source_cost import source_hint
            hint=source_hint(self,prefix,generated)
        if self.rate_gate.enabled:
            self.cache_forecast[rid]=(hint!='edge')
            self.cache_forecast.move_to_end(rid)
            while len(self.cache_forecast)>64:self.cache_forecast.popitem(last=False)
        return hint

    def _discard_edge_owned(self, ids):
        with self.lock:
            self.edge_owned.difference_update(ids)

    def submit_native(self,reqs,stride=WIDTH,phase='commit',mode=None):
        from .offload_experiment import reset_for_epoch
        if not getattr(self.growth,'live_runtime',False):reset_for_epoch(self)
        if stride!=WIDTH or self.closed:raise ValueError('GROWTH_PREFETCH_STATE')
        eligible=[];finished=[]
        for req in reqs:
            if not str(req.rid).startswith(self.ec_prefix):continue
            if req.finished() or getattr(req,'is_retracted',False):finished.append(str(req.rid))
            elif req.output_ids:eligible.append(req)
        if mode is None:mode=execution_mode(len(r.origin_input_ids)+len(r.output_ids) for r in eligible).mode
        if mode not in ('ar','spec'):raise ValueError('GROWTH_EXECUTION_MODE')
        self.release(finished);misses=[];stream_requests=[]
        with self.lock:
            if self.top2_inbox:self._fold_top2()
            self._check_warm()
            for rid,plan in list(self.fallback_native_plans.items()):
                if rid not in self.pending_native:
                    self.growth.finish_fallback_attempt(rid,plan)
                    self.fallback_native_plans.pop(rid)
            self.inflight={f for f in self.inflight if not f.done()}
            snapshots=[(r,str(r.rid),r.generated if type(r) is Request and type(r.generated) is tuple
                        else list(r.output_ids)) for r in eligible]
            # Request already validated these immutable tuple fields at the
            # commit boundary. Other adapters still take the full validation.
            prefixes={rid:(r.prefix if type(r) is Request and type(r.prompt) is tuple and type(r.generated) is tuple
                           else tuple(token_ids(list(r.origin_input_ids)+list(generated))))
                      for r,rid,generated in snapshots}
            # A streamed row whose committed prefix moved on has a stale plan: retire it
            # here so this round plans it again for the new prefix.
            for r,rid,generated in snapshots:
                held=self.stream_plans.get(rid)
                if held is not None and held.prefix_hash!=fingerprint(prefixes[rid]):
                    self.stream_plans.pop(rid,None);self.growth.forget(rid)
                    self.event('growth_stream_replan',rid=rid,prefix_length=len(prefixes[rid]))
            plans=self._plans(snapshots,prefixes,mode)
            for req,rid,generated in snapshots:
                prefix=prefixes[rid]
                # Charge the committed round to whichever side supplied it. This
                # is the one place every native commit passes through, so the gate
                # sees the same rounds the cache does.
                self._note_round(req)
                self._check_seed_health()
                if getattr(req,'inflight_middle_chunks',0):raise ValueError('GROWTH_UNCOMMITTED_NATIVE')
                pending=self.stream_plans.get(rid)
                if pending is not None:
                    # Keep feeding the delegate: it needs each committed prefix to
                    # maintain or rewind its own stream. The plan is unchanged because
                    # the prefix hash still matches.
                    if pending.prefix_hash!=fingerprint(prefix):raise ValueError('GROWTH_STREAM_PREFIX')
                    stream_requests.append(req)
                    continue
                old=self.entries.get(rid)
                if old is not None:
                    if old.prefix!=prefix:raise ValueError('GROWTH_STALE_PENDING')
                    if (old.source=='ar')==(mode=='ar'):continue
                    self.entries.pop(rid)
                    if old.plan is not None:self.growth.issued.pop((rid,old.plan.prefix_hash),None)
                if type(req) is Request:
                    self.similar_cache.observe_committed(req)
                else:
                    self.similar_cache.observe(rid, req.origin_input_ids, generated)
                self.live.add(rid)
                if len(self.live-set(self.releasing))>MAX_REQUESTS:raise ValueError('GROWTH_LIVE_LIMIT')
                if mode=='ar':
                    f=Future();f.set_result([[0]*3]);self.entries[rid]=GrowthEntry(prefix,WIDTH,f,0,'ar');self._mark_source(rid,'alt');self.stats['ar_bypass']+=1;continue
                p=plans[rid]
                if (self.stream is not None and p.k==3 and p.branches==1
                        and rid not in getattr(self,'sync_requests',())):
                    # Streamed baseline action: the delegate owns this request.
                    self.stream_plans[rid]=p;self.streamed_rids.add(rid)
                    self._mark_source(rid,'alt')
                    stream_requests.append(req)
                    self.stats['stream_delegate']=self.stats.get('stream_delegate',0)+1
                    continue
                tokens, source, detail = [], None, {}
                # Build the proposal even when the forecast priced this round as an
                # edge round: it is the input to the shadow score, and a shadow
                # round costs a lookup rather than a wasted verification.
                if p.reason != 'auto_sync_dual_root' and (self.cache_forecast.get(rid, True) or self.rate_gate.enabled):
                    tokens, source = self.similar_cache.propose_committed_with_source(req, want=p.k, detail=detail)
                if tokens and not self.cache_forecast.get(rid, True):
                    self._shadow(rid, tokens, self._gate_scope(req, source, detail), len(generated))
                    tokens = []
                if tokens and not self._gate_allows(req, source, detail):
                    # Rate-gate veto: this scope's measured rate loses to
                    # the edge alternative, so the round asks the edge instead.
                    # The plan was priced as a cache action, so record that it was
                    # not served locally rather than letting the ledger assume so.
                    self.event('rate_gate_veto', rid=rid, prefix_hash=p.prefix_hash,
                               source=source, scope=list(self._gate_scope(req, source, detail)),
                               k=p.k, tokens=len(tokens))
                    self._shadow(rid, tokens, self._gate_scope(req, source, detail), len(generated))
                    tokens = []
                if tokens:
                    f=Future();f.set_result([tokens]);self.entries[rid]=GrowthEntry(prefix,p.k+1,f,0,source,p)
                    self._mark_source(rid,'cache',self._gate_scope(req,source,detail))
                    self.stats['exact_prefix_hit' if source=='exact_prefix' else 'suffix_hit'] += 1
                    self.stats['similar_context_hit'] += 1
                    self.event(source+'_proposal',rid=rid,prefix_hash=p.prefix_hash,tokens=tokens,growth_plan=p.wire())
                else:
                    if len(prefix)+p.k>self.context or rid not in self.edge_owned and len(self.edge_owned)>=self.max_rows:
                        self.growth.issued.pop((rid,p.prefix_hash),None)
                        f=Future();f.set_result([None])
                        self.entries[rid]=GrowthEntry(prefix,WIDTH,f,0,'bounded_ar');self._mark_source(rid,'alt')
                        self.event('growth_edge_bound',rid=rid,context=self.context,max_rows=self.max_rows,k=p.k)
                        continue
                    self.edge_owned.add(rid)
                    row=dict(id=rid,tokens=list(prefix),max_tokens=max(1,req.sampling_params.max_new_tokens-len(generated)),growth_plan=p.wire())
                    rewind = rid in self.streamed_rids
                    mode = os.environ.get('TRIAGE_HEDGE_REWIND') or '1'
                    if rewind and rid in getattr(self, 'sync_requests', ()):
                        if mode == '0':
                            # Experiment knob: prove the edge refuses a switched
                            # round that does not allow its session to be trimmed.
                            rewind = False
                        elif mode == 'first':
                            done = getattr(self, 'sync_rewound', None)
                            if done is None:
                                self.sync_rewound = done = set()
                            if rid in done:
                                rewind = False
                            else:
                                done.add(rid)
                    if rewind:
                        # The edge stream may already be ahead of the committed prefix;
                        # this plan must be allowed to rewind the same session.
                        row['rewind']=True
                    misses.append(row);self._mark_source(rid,'alt')
            from .growth_batch_profile import BATCH_MAX, enabled as batch_growth_enabled
            groups=[]
            for row in misses:
                action=(row['growth_plan']['k'],row['growth_plan']['branches'])
                limit=BATCH_MAX if action[0]==3 or batch_growth_enabled() else 1
                if (not groups or len(groups[-1])>=limit or
                        action!=(groups[-1][0]['growth_plan']['k'],groups[-1][0]['growth_plan']['branches'])):
                    groups.append([])
                groups[-1].append(row)
            for group in groups:
                if self.plan_timing:
                    t0=time.perf_counter()
                    for row in group:
                        try: rid0=row['growth_plan']['rid']; rnd0=int(row['growth_plan']['round_id'])
                        except Exception: continue
                        self.plan_t0[(rid0,rnd0)]=t0
                submitted_at=time.monotonic_ns()
                f=self.pool.submit(self._call,group);self.inflight.add(f)
                for i,row in enumerate(group):
                    p=Plan.bind(row['growth_plan'],row['id'],row['tokens']);self.entries[p.rid]=GrowthEntry(tuple(row['tokens']),p.k+1,f,i,'edge',p)
                    self.plan_submitted[p.rid]=submitted_at
                self.stats['edge_submit']+=len(group)
                wake=getattr(self,'ready_wake',None)
                if wake is not None:f.add_done_callback(wake.notify)
        # Outside the locked section: the stream delegate takes its own lock.
        if stream_requests:self.stream.submit_committed(stream_requests,stride,phase,mode='spec')
    def ready(self,rid,prefix):
        # The executor asks every live row each tick; a streamed row is validated by
        # the delegate, which owns its entry.
        if str(rid) in self.stream_plans:return self.stream.ready(rid,prefix)
        return super().ready(rid,prefix)
    def has_draft(self,rid,prefix):
        if str(rid) in self.stream_plans:return self.stream.has_draft(rid,prefix)
        return super().has_draft(rid,prefix)
    def plan_eta_ms(self,rid):
        """Predicted remaining round trip for this row's plan, in ms.

        Zero means "already in hand or nothing to wait for". The scheduler
        uses the estimate to set a deadline without blocking the tick.
        """
        rid=str(rid)
        with self.lock:
            e=self.entries.get(rid)
            if e is None or e.future.done():return 0.0
            t0=self.plan_submitted.get(rid)
        if t0 is None:return 0.0
        residual=self.plan_rpc_ms-(time.monotonic_ns()-t0)/1e6
        return min(self.plan_join_cap_ms,max(0.0,residual)+self.plan_join_margin_ms)

    def join_plan(self,rid,prefix):
        """Pay for a plan-lane draft exactly once, priced by measured arrival.

        The budget is the predicted residual round trip plus a small margin,
        capped so a hopeless row cannot stall the tick. A draft that is already
        past its predicted arrival is not worth waiting for at all.
        """
        rid=str(rid)
        with self.lock:
            e=self.entries.get(rid)
            if e is None or e.prefix!=tuple(prefix) or e.future.done():return 0.0
        budget=self.plan_eta_ms(rid)
        if budget<=0:return 0.0
        start=time.monotonic_ns()
        try:
            e.future.result(timeout=budget/1000.0)
        except TimeoutError:
            self.stats['plan_join_late']=self.stats.get('plan_join_late',0)+1
            self.event('hedge_plan_late',rid=rid,budget_ms=round(budget,1),
                       ewma_ms=round(self.plan_rpc_ms,1))
            return 0.0
        t0=self.plan_submitted.get(rid)
        if t0 is not None:
            measured=(time.monotonic_ns()-t0)/1e6
            self.plan_rpc_ms=0.8*self.plan_rpc_ms+0.2*measured
        return (time.monotonic_ns()-start)/1e6

    def plan_lane_ready(self,rid,prefix,wait_ms=None):
        """Readiness of a plan-RPC round without the `draft_ready` alias.

        Inspect the entry directly because the streaming delegate aliases
        draft_ready to has_draft, which can route back to this method.
        The optional wait joins an in-flight plan before considering AR;
        priced-join mode defers the round instead of blocking decision time.
        """
        rid=str(rid)
        if wait_ms is None and self.sync_priced_join:
            # Decision time never blocks and never lies: a draft that is not in
            # hand is not runnable. The round is deferred to the predicted
            # arrival instead (plan_eta_ms), so the tick wakes up when the draft
            # lands rather than sleeping in front of it.
            with self.lock:
                e=self.entries.get(rid)
                if e is None or e.prefix!=tuple(prefix):return False
                if e.source in ('ar','cohort_pending','cancelled'):return False
                if not e.future.done():return False
                value=e.future.result()[e.index]
                if value is None:return False
                if len(token_ids(value))!=(e.plan.k if e.plan else 3):raise ValueError('GROWTH_INCOMPLETE_DRAFT')
                return True
        with self.lock:
            e=self.entries.get(rid)
        if e is not None and not e.future.done() and wait_ms is None:
            wait_ms=JOIN_MS
        if e is not None and not e.future.done() and wait_ms:
            try:
                e.future.result(timeout=max(0.0,wait_ms)/1000.0)
            except TimeoutError:
                return False
        with self.lock:
            e=self.entries.get(rid)
            if e is None or e.prefix!=tuple(prefix):return False
            if e.source in ('ar','cohort_pending','cancelled') or not e.future.done():return False
            value=e.future.result()[e.index]
            if value is None:return False
            if len(token_ids(value))!=(e.plan.k if e.plan else 3):raise ValueError('GROWTH_INCOMPLETE_DRAFT')
            return True
    def draft_ready(self,rid,prefix):
        if str(rid) in self.stream_plans:return self.stream.has_draft(rid,prefix)
        with self.lock:
            e=self.entries.get(str(rid))
            if e is None or e.prefix!=tuple(prefix):raise ValueError('GROWTH_MISSING_PREFIX')
            if e.source=='cancelled' or not e.future.done():return False
            value=e.future.result()[e.index]
            if value is None:return False
            if len(token_ids(value))!=(e.plan.k if e.plan else 3):raise ValueError('GROWTH_INCOMPLETE_DRAFT')
            return True
    def shadow_snapshot(self,rid,prefix):
        from .growth_batch_profile import enabled as batch_growth_enabled
        if str(rid) in self.stream_plans:
            plan=self.stream_plans[str(rid)]
            if (plan.prefix_hash!=fingerprint(prefix)
                    or not self.stream.has_draft(rid,tuple(prefix))):
                raise ValueError('GROWTH_STREAM_PREFIX')
            # The native packer only knows edge/suffix/exact_prefix. A streamed token
            # produced by the edge is 'edge'; a cache hit keeps its cache source. These
            # rows are always single-root, so there is no root alternate.
            entry=self.stream.entries.get(str(rid))
            source=getattr(entry,'source',None)
            if source=='stream':source='edge'
            elif source=='similar_context':source='suffix'
            if source not in ('edge','suffix','exact_prefix'):
                raise ValueError('GROWTH_STREAM_SOURCE')
            return dict(source=source,top2=None)
        with self.lock:
            if self.top2_inbox:self._fold_top2()
            e=self.entries.get(str(rid))
            if (e is not None and e.source=='edge' and e.plan is not None
                    and e.plan.branches==2
                    and (not e.future.done() or str(rid) not in self.root_top2)):
                # Reply readiness can precede top-2 metadata folding. Report no
                # alternate root until the metadata is available, and count it.
                self.stats['shadow_no_top2']=self.stats.get('shadow_no_top2',0)+1
                self.event('hedge_shadow_no_top2',rid=str(rid),k=e.plan.k)
                return dict(source='edge',top2=None)
            if e is not None and e.plan is not None and e.plan.branches==1:
                if (not self.root_top2_shadow or not self.draft_ready(rid,prefix)
                        or e.plan.rid!=str(rid) or e.plan.prefix_hash!=fingerprint(prefix)):
                    raise ValueError('GROWTH_ROOT1_PREFIX')
                if str(rid) in self.root_top2:
                    # Root count is a per-round action. A single-root round must
                    # not reuse alternate-root metadata from a previous round.
                    self.stats['shadow_no_top2']=self.stats.get('shadow_no_top2',0)+1
                    self.event('hedge_shadow_no_top2',rid=str(rid),k=e.plan.k)
                    return dict(source=e.source,top2=None)
                if e.source not in ('edge','suffix','exact_prefix'):raise ValueError('GROWTH_ROOT1_SOURCE')
                return dict(source=e.source,top2=None)
            snap=super().shadow_snapshot(rid,prefix)
            if e is not None and e.plan is not None and e.plan.branches==2:
                getter=getattr(self.growth,'fork_gap_for',None)
                gap=getter(e.plan.k) if getter is not None else None
                if gap is not None:snap=dict(snap,fork_gap=gap)
            return snap
    def plan_for(self,rid,prefix):
        rid=str(rid)
        if rid in self.stream_plans:
            plan=self.stream_plans[rid]
            if plan.prefix_hash!=fingerprint(prefix):raise ValueError('GROWTH_PLAN_LOOKUP')
            return plan
        with self.lock:
            e=self.entries.get(str(rid))
            if e is None or e.prefix!=tuple(prefix):raise ValueError('GROWTH_PLAN_LOOKUP')
            return e.plan
    def _entry_ready(self,rid,prefix):
        """Strict readiness: the tokens are in hand and bind this prefix.

        The streaming delegate's has_draft can describe an in-flight supplier.
        Consumption requires a completed future: waiting while holding this
        lock would prevent the plan worker from completing that future.
        """
        with self.lock:
            e=self.entries.get(str(rid))
            if e is None or e.prefix!=tuple(prefix):return False
            if e.source=='cancelled' or not e.future.done():return False
            value=e.future.result()[e.index]
            return len(token_ids(value))==(e.plan.k if e.plan else 3)

    def take(self,rid,prefix,stride=WIDTH):
        rid=str(rid)
        if rid in self.stream_plans:
            if stride!=WIDTH:raise ValueError('GROWTH_TAKE_STATE')
            plan=self.stream_plans[rid]
            if plan.prefix_hash!=fingerprint(prefix):raise ValueError('GROWTH_STREAM_PREFIX')
            # The delegate may not have a usable draft yet; keep ownership and the
            # pending plan so a later attempt (or AR fallback) still sees them.
            tokens=self.stream.take(rid,prefix,stride)
            if len(token_ids(tokens))!=plan.k:raise ValueError('GROWTH_STREAM_LENGTH')
            self.stream_plans.pop(rid,None)
            self.growth.consumed(plan);self.stats['consumed']+=1
            return list(tokens)
        # The single wait of a plan round: priced, bounded, at consumption time.
        if self.sync_priced_join:self.join_plan(rid,prefix)
        with self.lock:
            self.plan_submitted.pop(rid,None)
            if stride!=WIDTH or not self._entry_ready(rid,prefix):raise ValueError('GROWTH_TAKE_STATE')
            e=self.entries.pop(str(rid))
            self.waits.pop(str(rid),None)
            if e.plan is None or e.source=='ar':raise ValueError('GROWTH_TAKE_AR')
            self.growth.consumed(e.plan);self.root_top2.pop(str(rid),None);self.stats['consumed']+=1
            return list(e.future.result()[e.index])
    def take_ar(self,rid,prefix):
        rid=str(rid);noted=False;plan=None
        if getattr(self.growth,'attempt_mode',False):
            with self.lock:
                entry=self.entries.get(rid)
                plan=self.stream_plans.get(rid)or getattr(entry,'plan',None)
                continuation=plan is None and rid in self.fallback_native_plans
                if continuation:plan=self.fallback_native_plans[rid]
                if plan is not None:
                    source=('edge'if rid in self.stream_plans or continuation else entry.source)
                    noted=self.growth.note_fallback_attempt(plan,tuple(prefix),source=source,
                                                           continuation=continuation)
        if rid in self.stream_plans:
            # Retire native ownership normally, retaining only separate billing
            # intent until AR feedback. The native AR payload still has no Plan.
            self.stream_plans.pop(rid,None)
            if noted:self.growth.retire_fallback_plan(rid)
            else:self.growth.forget(rid)
            self.stream.take_ar(rid,prefix);self.stats['ar_bypass']+=1
            if noted:self.growth.finish_fallback_attempt(rid,plan)
            return
        result=super().take_ar(rid,prefix)
        if noted:
            if rid in self.pending_native:self.fallback_native_plans[rid]=plan
            else:self.growth.finish_fallback_attempt(rid,plan)
        return result
    def cancel(self,rid,prefix):
        rid=str(rid)
        self.fallback_native_plans.pop(rid,None)
        if rid in self.stream_plans:
            self.stream_plans.pop(rid);self.growth.forget(rid)
            self.stream.cancel(rid,prefix);self.event('cancel',rid=rid,side='stream')
            return
        with self.lock:
            if rid not in self.live:return
            self.growth.forget(rid);f=Future();f.set_result([[0]*3]);self.entries[rid]=GrowthEntry(tuple(token_ids(prefix)),WIDTH,f,0,'cancelled')
            self.event('cancel',rid=rid)
    def release(self,ids,retry_failed=False):
        ids=list(ids)
        for rid in ids:
            self.fallback_native_plans.pop(str(rid),None)
            self.growth.forget(str(rid));self.stream_plans.pop(str(rid),None)
            self.streamed_rids.discard(str(rid))
        if self.stream is not None:self.stream.release(ids,retry_failed)
        return super().release(ids,retry_failed)
    def warm(self,prompts):
        # The hybrid delegate owns the edge session; submitting the same rid
        # again through the parent would issue a duplicate prompt warm.
        if self.stream is not None:
            return self.stream.warm(prompts)
        return super().warm(prompts)
    def snapshot(self):
        return dict(super().snapshot(),growth=self.growth.snapshot(),max_rows=self.max_rows,
                    edge_owned=len(self.edge_owned),edge_context=self.context,planning_batch_max=min(8,self.max_rows),
                    hybrid_stream=self.stream is not None,stream_rows=len(self.stream_plans),
                    stream_ready_wake=self.stream is not None and getattr(self.stream,'ready_wake',None) is not None,
                    stream_ready_notifications=self.stream.stats['stream_ready_notifications'] if self.stream is not None else 0,
                    hybrid_stream_stats=self.stream.snapshot() if self.stream is not None else None)

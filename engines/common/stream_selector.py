from collections import defaultdict, deque
from dataclasses import replace
import json
import math
import os
import time

from .growth import KS, CHOICES, FORCED, Plan, draft_transport
from .growth_prefetch import GrowthPrefetch
from .lane_learn import LaneLearner
from .growth_source_cost import SourceCostPolicy
from .growth_survival import PrefixSurvival
from .policy import WIDTH, MAX_REQUESTS, CONTEXT, fingerprint
from .draft_policy import ar_candidate_enabled
from .stream_value import Costs, Supply, StreamValue, WAIT, AR, ModelBudgetExceeded
from .stream_cache import CacheAgreement
from .stream_batch import BatchRate, Batch, Offer
from .sync_action import second_answer_requested

SELECTOR='stream_event_rollout_v2'

def rate_fastpath_enabled():
    """Deep-buffer ready-rate fast path that skips the H2 rollout.

    Default on for the streaming selector: with a full width buffered and a
    live producer the H2 contest uses a one-step ready-rate comparison.
    `TRIAGE_RATE_FASTPATH=0` disables the fast path.
    """
    return os.environ.get('TRIAGE_RATE_FASTPATH','1')!='0'


def enabled():
    flag=os.environ.get('TRIAGE_STREAM_SELECTOR','0')
    if flag not in ('0','1'):raise ValueError('STREAM_SELECTOR_FLAG')
    return flag=='1'


class StreamingPolicy(SourceCostPolicy):
    live_runtime=True
    streaming_selector=True
    selector_name=SELECTOR
    ACTIONS=tuple((k,1) for k in CHOICES)
    PROBE_EVERY=8
    PROBE_FRACTION=.10

    def __init__(self,event,clock=time.monotonic_ns):
        super().__init__(event,clock)
        def named(name,**fields):
            fields['selector']=SELECTOR
            event(name,**fields)
        self.event=named
        self.observations=deque(maxlen=self.WINDOW)
        self.cost_observations=deque(maxlen=self.COST_WINDOW)
        self.native_ms=defaultdict(lambda:deque(maxlen=32))
        self.request_rows=defaultdict(lambda:deque(maxlen=64))
        self.profile_cache={}
        self.ready_choices=0
        self.selector_elapsed_ms=0.
        self.selector_probe_ms=0.
        self.ar_ms=deque(maxlen=32)
        self.ar_seen=0
        self.joint_ar=ar_candidate_enabled()
        self.first_frame_ms=defaultdict(lambda:deque(maxlen=32))
        self.frame_period_ms=deque(maxlen=64)
        self.frame_seen=defaultdict(int)
        self.edge_agreement=defaultdict(lambda:[0,0])
        self.model_cache={}
        self.last_model_signature={}
        self.model_fallbacks=defaultdict(int)
        self.rate_fastpath_on=rate_fastpath_enabled()
        self.fastpath_choices=0
        self.cache_agreement=CacheAgreement()
        self.batch_rate=BatchRate()
        self.event('stream_selector_bound',selector=SELECTOR,plan_binding='READY_BEFORE_NATIVE_DISPATCH',
                   ks=list(KS),single_root_stream=True,native_math_changed=False,
                   probe_fraction=self.PROBE_FRACTION,probe_every=self.PROBE_EVERY)

    def refresh(self):
        if self.control is None:
            self.epoch=0
            self.control=dict(epoch=0,mode='adaptive',learn=True)

    @staticmethod
    def median(values):
        values=sorted(values)
        return values[len(values)//2]

    def profile(self,rid,source):
        rows=self.request_rows.get((rid,source),())
        last=rows[-1] if rows else {}
        version=(len(rows),last.get('generation'),last.get('cycle_id'))
        held=self.profile_cache.get((rid,source))
        if held is None or held[0]!=version:
            held=(version,PrefixSurvival(rows,minimum=2,batched_rows=True))
            self.profile_cache[rid,source]=held
        return held[1]

    def observe_arrival(self,rid,reason,delta_ns,*,first,count):
        if reason not in ('first','continue','diverged') or count<3 or delta_ns<=0:return
        key=(reason,first)
        self.frame_seen[key]+=1
        if self.frame_seen[key]==1:return  # cold first observation is not steady service
        value=delta_ns/1e6
        if first:self.first_frame_ms[reason].append(value)
        else:self.frame_period_ms.append(value*3/count)

    def observe_alignment(self,rid,matched,failed):
        if type(matched) is not int or matched<0 or type(failed) is not bool:
            raise ValueError('STREAM_ALIGNMENT_OBSERVATION')
        if matched or failed:
            values=self.edge_agreement[str(rid)]
            values[0]+=matched;values[1]+=int(failed)

    @staticmethod
    def residual(samples,age_ms):
        # Conditional remaining time, not median(T)-age clamped to zero.
        survivors=[value-age_ms for value in samples if value>age_ms]
        return sum(survivors)/len(survivors) if len(survivors)>=2 else None

    def rollout(self,rid,prefix,*,path,remaining,wait_ms,deadline_ms,now_ns,ahead=12):
        if not self.joint_ar:return None
        if self.ar_seen<16:return AR,'stream_ar_bootstrap',0.
        matched,failed=self.edge_agreement.get(rid,(0,0))
        measured=[(k,self.median(self.native_ms[1,k])) for k in CHOICES
                  if len(self.native_ms[1,k])>=2 and len(prefix)+k<=CONTEXT]
        if (len(self.ar_ms)<8 or matched+failed<8 or not measured
                or len(self.frame_period_ms)<4 or path is None):return None
        initial=self.first_frame_ms.get('first',())
        repair=self.first_frame_ms.get('diverged',())
        continuation=self.first_frame_ms.get('continue',())
        repair=repair if len(repair)>=4 else initial
        continuation=continuation if len(continuation)>=4 else initial
        if len(repair)<4 or len(continuation)<4:return None
        alpha=(matched+1.)/(matched+failed+2.)
        ar=self.median(self.ar_ms)
        live=0 if path.done or path.paused else max(0,path.requested-path.streamed)
        # Deep-buffer fast path: with a full width already buffered and a live
        # producer, WAIT only buys tokens beyond the buffer cap and AR is
        # rate-dominated by the ready wide K, so the H2 contest collapses to a
        # one-step ready-rate comparison. Skip the rollout (and its arrival
        # model) entirely; this is the common low-RTT regime.
        if self.rate_fastpath_on and live:
            lead=len(path.tokens)-len(prefix)
            if lead>=max(CHOICES):
                ready=[((1.+sum(alpha**j for j in range(1,min(k,remaining-1)+1)))/cost,-k,k)
                       for k,cost in measured if k<=lead]
                if ready and max(ready)[0]>1./ar:
                    self.fastpath_choices+=1
                    return max(ready)[2],'stream_deep_ready_fast',0.
        arrival=0.;recheck=0.
        if live:
            if path.last_frame_ns:
                samples=self.frame_period_ms;age=(now_ns-path.last_frame_ns)/1e6
            else:
                samples=self.first_frame_ms.get(path.reason,())
                if len(samples)<4:samples=initial
                age=(now_ns-path.created_ns)/1e6
            age=max(0.,age)
            arrival=self.residual(samples,age)
            recheck=min((value-age for value in samples if value>age),default=0.)
            if arrival is None:
                # Unknown future supply cannot erase a profitable real ready
                # candidate. With no continuation model, compare current rates.
                lead=len(path.tokens)-len(prefix)
                ready=[((1.+sum(alpha**j for j in range(1,min(k,remaining-1)+1)))/cost,-k,k)
                       for k,cost in measured if k<=lead]
                if ready and max(ready)[0]>1./ar:
                    self.model_fallbacks['unidentified_arrival_tail_ready']+=1
                    return max(ready)[2],'stream_ready_tail_unidentified',0.
                if not ready and any(k<=lead for k in CHOICES):
                    self.model_fallbacks['unidentified_arrival_tail_bootstrap']+=1
                    return None
                self.model_fallbacks['unidentified_arrival_tail']+=1
                return AR,'stream_arrival_tail_unidentified',0.
        params=(alpha,ar,tuple(measured),
                self.median(self.frame_period_ms),self.median(repair),
                self.median(continuation),ahead,deadline_ms)
        held=self.model_cache.get(rid)
        if held is None or held[0]!=params:
            held=(params,StreamValue(Costs(*params)))
            self.model_cache[rid]=held
        state=Supply(len(path.tokens)-len(prefix),live,arrival)
        try:
            result=held[1].choose_rate(state,remaining,wait_ms=max(0.,wait_ms),room=CONTEXT-len(prefix))
        except ModelBudgetExceeded:
            self.model_fallbacks['bounded_rollout_budget']+=1
            return None
        signature=(path.epoch,state.lead,state.remaining,result.action)
        if self.last_model_signature.get(rid)!=signature:
            self.last_model_signature[rid]=signature
            self.event('stream_value_decision',rid=rid,action=result.action,lead=state.lead,
                producer_remaining=live,arrival_ms=arrival,alpha=alpha,
                wait_budget_ms=wait_ms,expected_progress=result.progress,
                expected_ms=result.elapsed_ms,evaluations=result.evaluations,
                objective='BOUNDED_H2_RATE_ROOT_NOT_GLOBAL_OPTIMUM',
                repair_source='diverged' if len(self.first_frame_ms.get('diverged',()))>=4 else 'first_prior')
        return result.action,'stream_model_rollout',recheck if result.action==WAIT else 0.

    def choose_ready(self,rid,prefix,*,offered,remaining,source,batch=1):
        feasible=[k for k in CHOICES if k<=offered and len(prefix)+k<=CONTEXT]
        if FORCED is not None:
            return (FORCED[0],'stream_ready_forced') if FORCED[0] in feasible else None
        if not feasible:return None
        profile=self.profile(rid,source)
        measured={k:self.median(self.native_ms[batch,k]) for k in feasible if self.native_ms[batch,k]}
        baseline=measured.get(3,measured.get(min(feasible)))
        if baseline is None:return min(feasible),'stream_ready_cost_bootstrap'
        mature=[];pending=[]
        for k in feasible:
            value=profile.mean(k,remaining)
            if k in measured and value is not None and profile.supported(k,remaining):
                mature.append((value/measured[k],-k,k))
            elif k!=3:
                pending.append(k)
        credit=2*baseline+self.PROBE_FRACTION*self.selector_elapsed_ms-self.selector_probe_ms
        if pending and (self.ready_choices+1)%self.PROBE_EVERY==0:
            # An unmeasured shape is a bounded real experiment, not a fabricated
            # cheap winner. Its actual complete cycle debits the same time budget.
            k=max(pending)
            estimate=measured.get(k,max(measured.values())*1.1)
            if credit>=estimate:
                return k,'stream_ready_probe'
        if mature:return max(mature)[2],'stream_ready_rate'
        return min(feasible),'stream_ready_reward_bootstrap'

    @staticmethod
    def expected_progress(alpha,k,remaining):
        return 1.+sum(alpha**j for j in range(1,min(k,remaining-1)+1))

    def ready_rate(self,rid,k,remaining,batch=1):
        samples=self.native_ms.get((batch,k),())
        if len(samples)<2:return None
        value=self.profile(rid,'edge').mean(k,remaining)
        if value is None or not self.profile(rid,'edge').supported(k,remaining):
            good,bad=self.edge_agreement.get(rid,(0,0))
            value=self.expected_progress((good+1.)/(good+bad+2.),k,remaining)
        return value/self.median(samples)

    def cache_candidate(self,rid,prefix,offers,*,remaining,batch=1):
        """One-step rate comparison between retrieved cache and edge proposals."""
        feasible=[o for o in offers if len(o.tokens) in CHOICES
                  and len(prefix)+len(o.tokens)<=CONTEXT]
        if FORCED is not None:feasible=[o for o in feasible if len(o.tokens)==FORCED[0]]
        if not feasible:return None
        smallest=min(len(o.tokens) for o in feasible)
        known={len(o.tokens):self.median(self.native_ms[batch,len(o.tokens)]) for o in feasible
               if len(self.native_ms[batch,len(o.tokens)])>=2}
        if smallest not in known:
            offer=next(o for o in feasible if len(o.tokens)==smallest)
            return offer,'stream_cache_cost_bootstrap',None
        pending=[o for o in feasible if len(o.tokens) not in known]
        credit=2*known[smallest]+self.PROBE_FRACTION*self.selector_elapsed_ms-self.selector_probe_ms
        if pending and (self.ready_choices+1)%self.PROBE_EVERY==0 and credit>=1.1*max(known.values()):
            offer=max(pending,key=lambda o:len(o.tokens))
            return offer,'stream_cache_probe',None
        scored=[]
        for offer in feasible:
            k=len(offer.tokens);alpha,n=self.cache_agreement.quality(offer)
            if k not in known or n<8 and k!=smallest:continue
            rate=self.expected_progress(alpha,k,remaining)/known[k]
            scored.append((rate,-k,offer))
        if not scored:return None
        rate,_,offer=max(scored,key=lambda row:row[:2])
        return offer,'stream_cache_ready_rate',rate

    def bind_ready(self,rid,prefix,*,k,source,batch):
        digest=fingerprint(prefix)
        existing=self.issued.get((rid,digest))
        if existing is not None:
            self.bind_reused=getattr(self,'bind_reused',0)+1
            return existing
        self.round_id+=1
        self.ready_choices+=1
        k,reason=k
        plan=Plan(rid,digest,k,1,self.round_id,self.generation,reason)
        self.current=(plan.k,1)
        self.issued[rid,digest]=plan
        self.attempt_batches[plan]=batch
        self.event('growth_plan',plan=plan.wire(),selector=SELECTOR,produce_batch=batch,
                   source=source,ready_bound=True)
        return plan

    def prepare_cohort(self, fresh, all_prefixes, **kwargs):
        # Only the synchronous lane issues cohort plans. Stream and cache
        # drafts bind a single-root plan after their tokens are ready.
        super().prepare_cohort(fresh, all_prefixes, **kwargs)
        if (self.frame is not None and draft_transport() == 'auto'
                and second_answer_requested()):
            k, _ = self.frame['action']
            self.frame.update(action=(k, 2), reason='auto_sync_dual_root')

    def observe_geometry(self,shape):
        self.batch_rate.observe_geometry(shape)

    def commit_batch(self,rows,*,rpc_ms=0.):
        super().commit_batch(rows,rpc_ms=rpc_ms)
        if rows:
            plans=[r['plan'] for r in rows]
            speculative=all(p is not None and p.branches==1 for p in plans)
            ar=all(p is None and r.get('native_mode')=='ar' for p,r in zip(plans,rows))
            shapes=[r.get('native_shape') for r in rows]
            if (speculative or ar) and all(s is not None for s in shapes):
                if any(s!=shapes[0] for s in shapes):raise ValueError('STREAM_BATCH_SHAPE_BINDING')
                self.batch_rate.observe_geometry(shapes[0])
                length=max(r.get('resident_prefix_length',r['prefix_length']) for r in rows)
                self.batch_rate.observe(len(rows),plans[0].k if speculative else 0,length,rpc_ms,shapes[0])
            if speculative and len(rows)>1:
                for row in rows:
                    if row['delivered']<=0:continue
                    p=row['plan']
                    sample=dict(k=p.k,branches=1,delivered=row['delivered'],
                        query_nodes=row.get('query_nodes',p.k+1),side_accepts=int(row.get('side',False)),
                        actual_batch=len(rows),learning=self.control.get('learn',True),
                        terminal_censored=row.get('terminal_censored',row['remaining']<=row['raw_emitted']),
                        generation=self.generation,cycle_id=self.cycle_id)
                    self.request_rows[row['rid'],row['source']].append(sample)
        cycle=self.observations[-1]
        elapsed=cycle.get('cycle_ms')
        if elapsed is not None:self.selector_elapsed_ms+=elapsed
        if any(p and p['reason'] in ('stream_ready_probe','stream_cache_probe','stream_batch_probe') for p in cycle['plans']):
            self.selector_probe_ms+=elapsed or rpc_ms
        if not all(r['plan'] is not None for r in rows):
            # first_token may carry a previous growth_rpc_ms; without an actual
            # dispatch mode it must never become an AR timing sample.
            if (all(r.get('native_mode')=='ar' for r in rows) and len(rows)==1
                    and rows[0]['delivered']>0 and math.isfinite(rpc_ms) and rpc_ms>0):
                self.ar_seen+=1
                if self.ar_seen>1:self.ar_ms.append(rpc_ms)
            return
        cost=self.cost_observations[-1]
        if cost['branches']==1 and any(r['raw_emitted'] for r in rows):
            self.native_ms[len(rows),cost['k']].append(rpc_ms)
        if len(rows)==1 and cost.get('learning') and rows[0]['delivered']>0:
            self.request_rows[rows[0]['rid'],rows[0]['source']].append(cost)
        if self.cycle_id%self.WINDOW==0:
            retained={(r['generation'],r['cycle_id']) for values in self.cost_rows.values() for r in values}
            self.sample_tick={key:tick for key,tick in self.sample_tick.items() if key in retained}

    def forget(self,rid):
        super().forget(rid)
        self.cache_agreement.forget(rid)
        self.edge_agreement.pop(str(rid),None)
        self.model_cache.pop(str(rid),None)
        self.last_model_signature.pop(str(rid),None)
        for key in list(self.request_rows):
            if key[0]==str(rid):self.request_rows.pop(key)
        for key in list(self.profile_cache):
            if key[0]==str(rid):self.profile_cache.pop(key)
        return super().forget(rid)

    def snapshot(self):
        return dict(super().snapshot(),selector=SELECTOR,streaming_selector=True,
                    ready_choices=self.ready_choices,probe_elapsed_ms=self.selector_probe_ms,
                    ar_decode_samples=len(self.ar_ms),ar_decode_steps=self.ar_seen,
                    ar_native_median_ms=self.median(self.ar_ms) if self.ar_ms else None,
                    frame3_samples=len(self.frame_period_ms),
                    frame3_median_ms=self.median(self.frame_period_ms) if self.frame_period_ms else None,
                    rollout_fallbacks=dict(self.model_fallbacks),
                    rate_fastpath=dict(enabled=self.rate_fastpath_on,choices=self.fastpath_choices),
                    cache_agreement_scopes=len(self.cache_agreement.evidence),
                    concurrent=self.batch_rate.snapshot(),
                    cache_projection='CURRENT_READY_RATE_GATE_NOT_IDENTIFIED_JOINT_H2',
                    native_costs=[dict(batch=b,k=k,n=len(v),median_ms=self.median(v))
                                  for (b,k),v in self.native_ms.items() if v])


class StreamingGrowthPrefetch(GrowthPrefetch):
    """One producer owns the session; no Plan RPC for a change of single-root K."""
    def __init__(self,client,growth,**kwargs):
        if not getattr(growth,'streaming_selector',False):raise ValueError('STREAM_SELECTOR_POLICY')
        if any(os.environ.get(name)=='1' for name in ('TRIAGE_UNIFIED_AUTO_GATE','TRIAGE_UNIFIED_RATE_GATE')):
            raise ValueError('STREAM_CACHE_LEGACY_GATE_UNSUPPORTED')
        super().__init__(client,growth,**kwargs)
        if self.stream is None:raise ValueError('STREAM_SELECTOR_REQUIRES_STREAM')
        self.requests_by_rid={}
        self.suggested_actions={}
        self.ar_selected={}
        self.wait_rollout_cache={}
        self.cache_offers={}
        self.selected_offers={}
        self.batch_order={}
        self.stream.external_cache=True
        self.stream.max_rows=self.max_rows
        self.stream.arrival_observer=growth.observe_arrival
        self.stream.alignment_observer=growth.observe_alignment
        # One ownership ledger: release()/poll_cleanup() can see the delegate's
        # ordered native free, rather than submitting a second free for that rid.
        self.entries=self.stream.entries
        self.live=self.stream.live
        self.releasing=self.stream.releasing
        self.stream._on_edge_free_ack = self._discard_edge_owned
        self.selector_cpu_ns=0
        self.selector_wall_ns=0
        self.cache_cpu_ns=0
        self.cache_wall_ns=0
        # The explicit second-answer capability can force a sync action.
        # The lane learner separately compares observed delivery rates.
        self.auto_transport=draft_transport()=='auto'
        self.second_answer=second_answer_requested()
        if self.auto_transport and not self.second_answer:
            self.stream.root_gap=False
        # Per-round transport, not a request-level commitment: the decision is
        # taken at every round boundary and the switch cost B is charged when the
        # choice differs from the previous round's transport.
        self.sync_requests=set()
        self.round_transport={}
        self.sync_decisions={}
        self.sync_action_log=[]
        self.sync_calls={}
        # K this request actually ran last round (the plan is popped on commit,
        # so `growth.last_used` cannot answer this at the next round boundary).
        self.round_k={}
        self.sync_rtt_ms=float(os.environ.get('TRIAGE_SYNC_ACTION_RTT_MS','0') or 0)
        # The legacy per-round action needs an explicit cost or capability.
        # The independently enabled lane learner owns its own accounting.
        self.sync_enabled=self.sync_rtt_ms>0 or self.second_answer
        self.sync_c_ms=float(os.environ.get('TRIAGE_SYNC_ACTION_C_MS','85') or 85)
        self.sync_rescue_gain=float(os.environ.get('TRIAGE_SYNC_ACTION_RESCUE','0.03') or 0.03)
        from .sync_action import GapCurve
        self.gap_curve=GapCurve()
        # Root top-2 the edge forwarded on this request's first stream round.
        # Owned by the stream delegate (frames), read here per round: no RPC.
        self.gap_seen=self.stream.gap_seen
        self.hedge_forced=self.second_answer and not self.auto_transport
        self.sync_round_delta_ms=float(os.environ.get('TRIAGE_SYNC_ACTION_ROUND_DELTA_MS','0') or 0)
        # Estimate the per-request sync-minus-stream round-time difference.
        # Each bucket uses RTT as its initial cost until delta_min samples arrive.
        self.hedge_delta_online=os.environ.get('TRIAGE_HEDGE_DELTA_ONLINE','1')!='0'
        self.delta_min=max(1,int(os.environ.get('TRIAGE_HEDGE_DELTA_MIN','3') or 3))
        # Rounds of measured sync-lane delivery needed to trust the direct rate.
        self.sync_rate_min=max(1,int(os.environ.get('TRIAGE_HEDGE_RATE_MIN','3') or 3))
        self.delta_samples={}    # bucket -> [delta_ms]
        self.sync_rate={}        # bucket -> dict(rates=[tokens/ms per round])
        # Stream-side prior per class, same units as the lane cells, so the two can
        # be compared before a request has produced anything.
        self.stream_rates={}     # class -> [tokens/ms per round]
        # Admission-time choice pays no switch cost because no stream has yet
        # opened and there is nothing to drain, trim or reopen.
        self.admit_sync_on=os.environ.get('TRIAGE_ADMIT_SYNC','0')=='1'
        self.admit_sync_force=os.environ.get('TRIAGE_ADMIT_SYNC_FORCE','0')=='1'
        self.admit_margin=float(os.environ.get('TRIAGE_ADMIT_MARGIN','1.02') or 1.02)
        self.admit_min_tokens=max(1,int(os.environ.get('TRIAGE_ADMIT_MIN_TOKENS','64') or 64))
        self.admit_decided={}    # rid -> 'stream'|'sync'
        self.admit_sync_rows=set()   # rows the stream delegate must never own
        self.sync_learn=(LaneLearner(request_stride=max(0,int(os.environ.get('TRIAGE_SYNC_LEARN_STRIDE','0') or 0)),
                                     request_margin=float(os.environ.get('TRIAGE_SYNC_LEARN_MARGIN','1.05') or 1.05),
                                     min_request_samples=max(1,int(os.environ.get('TRIAGE_SYNC_LEARN_MIN_REQ','2') or 2)),
                                     price_mode=os.environ.get('TRIAGE_SYNC_PRICE','1')!='0',
                                     price_a=float(os.environ.get('TRIAGE_PRICE_A','-14.47') or -14.47),
                                     price_b=float(os.environ.get('TRIAGE_PRICE_B','7.64') or 7.64),
                                     price_p=(None if not os.environ.get('TRIAGE_PRICE_P')
                                              else float(os.environ['TRIAGE_PRICE_P'])),
                                     price_margin=float(os.environ.get('TRIAGE_PRICE_MARGIN','1.0') or 1.0),
                                     price_cap_ms=float(os.environ.get('TRIAGE_PRICE_CAP_MS','200') or 200),
                                     price_prior_var_a=float(os.environ.get('TRIAGE_PRICE_VAR_A','400') or 400),
                                     price_prior_var_b=float(os.environ.get('TRIAGE_PRICE_VAR_B','25') or 25),
                                     price_noise=float(os.environ.get('TRIAGE_PRICE_NOISE','25') or 25),
                                     price_every=int(os.environ.get('TRIAGE_PRICE_EVERY','3') or 3),
                                     price_floor=int(os.environ.get('TRIAGE_PRICE_FLOOR','8') or 8),
                                     price_min_labels=int(os.environ.get('TRIAGE_PRICE_MIN_LABELS','6') or 6),
                                     price_max_sd=float(os.environ.get('TRIAGE_PRICE_MAX_SD','6') or 6),
                                     lane_rule=os.environ.get('TRIAGE_LANE_RULE','1')=='1',
                                     lane_g=float(os.environ.get('TRIAGE_LANE_G','2.6') or 2.6),
                                     lane_p=float(os.environ.get('TRIAGE_LANE_P','0.13') or 0.13))
                         # Auto starts with root-gap and progress scoring. The learner is opt-in.
                         if os.environ.get('TRIAGE_SYNC_LEARN','0' if self.auto_transport else '1')!='0' else None)
        # The default-enabled rule owns both its decisions and accounting.
        # Explicitly disabling it removes both from the per-round path.
        self.lane_on=bool(self.sync_learn is not None
                          and getattr(self.sync_learn,'lane_rule',False))
        if self.sync_learn is not None and os.environ.get('TRIAGE_SYNC_LEARN_NOOP')=='1':
            self.sync_learn.noop=True
        self.learn_lock={}       # rid -> the lane the learner committed to
        self.learn_pin={}        # rid -> lane this row must finish on (the sample)
        self.learn_streak={}     # rid -> rounds decided since the last refresh
        self.learn_events={}     # rid -> ticks the learner answered
        self.learn_reason={}     # rid -> last reason logged
        self.learn_verdict={}    # rid -> (prefix key, choice, why) for this round
        self.lane_warm=max(1,int(os.environ.get('TRIAGE_LANE_WARM','8') or 8))
        self.lane_min_rounds=max(1,int(os.environ.get('TRIAGE_LANE_MIN_ROUNDS','48') or 48))
        self.lane_window=max(self.lane_min_rounds,int(os.environ.get('TRIAGE_LANE_WINDOW','4096') or 4096))
        self.lane_pool_done=False  # the estimate is frozen once lane_min_rounds arrive
        self.lane_explore_every=max(0,int(os.environ.get('TRIAGE_LANE_EXPLORE_EVERY','8') or 0))
        self.lane_flip_margin=float(os.environ.get('TRIAGE_LANE_FLIP_MARGIN','0.05') or 0.05)
        self.lane_flip_votes_needed=max(1,int(os.environ.get('TRIAGE_LANE_FLIP_VOTES','2') or 2))
        self.lane_min_rows=max(1,int(os.environ.get('TRIAGE_LANE_MIN_ROWS','6') or 6))
        # Disable the hybrid explicitly when a caller requires a fixed lane.
        self.lane_hybrid=os.environ.get('TRIAGE_LANE_HYBRID','1')=='1'
        self.lane_belief=None      # 'sync'|'stream'; starts at the threshold's answer, may flip
        self.lane_judged=[]        # rids given a lane, in order (the exploration counter).
        # Keep state names distinct from the lane_rows() occupancy helper.
        self.lane_samples={'stream':[], 'sync':[]}   # per-row tokens/second
        self.lane_flip_votes=0
        self.lane_flips=[]         # (old, new, stream_median, sync_median) for the report
        self.own_stream={}       # rid -> [rounds, tokens] on the stream lane, all rounds
        self.win_stream=deque(maxlen=self.lane_window)  # steady stream rounds, pre-freeze
        self.learn_cell={}       # rid -> the cell the learner used for it
        self.learn_cell_cache={} # rid -> (stamp, cell)
        self.learn_start={}      # rid -> perf_counter when the learner first saw it
        self.learn_refresh=int(os.environ.get('TRIAGE_SYNC_LEARN_REFRESH','8') or 8)
        self.cf_labels=os.environ.get('TRIAGE_SYNC_CF_LABEL','1')!='0'
        self.cf_done={}          # rid -> prefix length already labelled
        # Pre-flight probe (TRIAGE_PROBE=1). The window's first rows alternate lanes
        # and are pinned, so both lanes run over the SAME interval; the ratio of their
        # throughputs is the one number the price rule cannot derive but could measure.
        # Observation only: the decision path is untouched until the probe is trusted.
        # Observe-only: the learner answers 'stream' for every row and never locks.
        # A window in this mode is a pure-stream measurement that still records the two
        # quantities the criterion needs (g from the rounds, P from the draft labels),
        # i.e. the "stream first, then observe" half of the policy.
        self.observe_only=os.environ.get('TRIAGE_OBSERVE_ONLY','0')=='1'
        self.probe_on=os.environ.get('TRIAGE_PROBE','0')=='1'
        self.probe_per_lane=max(1,int(os.environ.get('TRIAGE_PROBE_PER_LANE','2') or 2))
        self.probe_rows=[]        # (lane, tokens, seconds, rid)
        self.probe_tag={}         # rid -> lane assigned by the probe
        self.probe_n=0
        self.probe_done=False
        # Optional persistence lets later requests reuse the traffic class's
        # observations across service restarts.
        self.learn_state_path=os.environ.get('TRIAGE_SYNC_LEARN_STATE','') or None
        if self.sync_learn is not None and self.learn_state_path:
            try:
                with open(self.learn_state_path) as fh:
                    loaded=self.sync_learn.load_state(json.load(fh))
                self.event('sync_learn_loaded',path=self.learn_state_path,cells=loaded)
            except FileNotFoundError:
                pass
            except Exception as exc:
                self.event('sync_learn_load_failed',path=self.learn_state_path,error=repr(exc))
        self.round_ts={}         # rid -> perf_counter of the last delivered round
        self.lane_round_ms={}    # rid -> {'stream': [ms], 'sync': [ms]}
        # Acceptance criterion (default 0 = off, the root gap is used instead).
        self.hedge_accept_min=float(os.environ.get('TRIAGE_HEDGE_ACCEPT_MIN','0') or 0)
        # Require several rounds so initial AR steps do not dominate the estimate.
        self.hedge_accept_rounds=max(1,int(os.environ.get('TRIAGE_HEDGE_ACCEPT_ROUNDS','4') or 4))
        self.hedge_once=os.environ.get('TRIAGE_HEDGE_ONCE','0' if self.auto_transport else '1')!='0'
        arm_default='0' if self.auto_transport else '3.5'
        self.hedge_arm_rate=float(os.environ.get('TRIAGE_HEDGE_ARM_RATE',arm_default) or arm_default)
        self.hedge_gap_cap=float(os.environ.get('TRIAGE_HEDGE_GAP_CAP','6') or 6)
        self.arm_rate=None
        # ---- bucketed Beta posterior over the rescue label (new criterion) ----
        # Keyed by (batch, prefix_length//128, k). Trials are hedged rounds whose
        # root top-2 was verified; a rescue is the target choosing the alternate.
        # Cold start never bets: with fewer than beta_min trials the request stays
        # on the stream, and a small exploration fraction hedges only to collect
        # labels (never counted as a bet).
        self.beta={}
        # Traffic class for the posterior keys (see _batch_key): constant per
        # window, never the cohort size.
        self.hedge_class=os.environ.get('TRIAGE_HEDGE_CLASS','0') or '0'
        self.beta_min=max(0,int(os.environ.get('TRIAGE_HEDGE_BETA_MIN','6') or 6))
        self.beta_explore=float(os.environ.get('TRIAGE_HEDGE_EXPLORE','0.1') or 0.1)
        # Deliberate cold-start hedges per bucket (see _decide_hedge): one request
        # is enough to fill beta_min, because labels and delta samples arrive per
        # round, not per request.
        self.hedge_burnin=max(0,int(os.environ.get('TRIAGE_HEDGE_BURNIN','1') or 0))
        self.hedge_min_obs=max(2,int(os.environ.get('TRIAGE_HEDGE_MIN_OBS','6') or 6))
        # Rows on the sync lane at which one round trip is shared widely enough to
        # pay for itself; below it the lane is priced at a penalty, above it the
        # lane is the regime this gate wants to reach (see lane_bucket()).
        self.hedge_lane_floor=max(1,int(os.environ.get('TRIAGE_HEDGE_LANE_FLOOR','4') or 4))
        self.burnin_used={}
        self.hedge_gap_samples=max(1,int(os.environ.get('TRIAGE_HEDGE_GAP_SAMPLES','3') or 3))
        self.hedge_round_budget=max(1,int(os.environ.get('TRIAGE_HEDGE_ROUND_BUDGET','8') or 8))
        self.gap_samples={}      # rid -> [(gap, k, prefix_len)] up to hedge_gap_samples
        self.round_count={}      # rid -> rounds observed since admission
        self.hedge_labels={}     # rid -> last label dict
        self.hedge_pending={}    # rid -> this hedged round's root candidates
        self.explore_draws=0
        self.req_tokens={}
        self._load_prior()
        self.round_delivered={}
        # An AR round is a fallback, not acceptance evidence: its single token must
        # not enter the per-round delivery estimate.
        self.ar_round={}
        self.sync_edge_ms=float(os.environ.get('TRIAGE_SYNC_ACTION_EDGE_MS','1.3') or 1.3)
        self.sync_switch_ms=float(os.environ.get('TRIAGE_SYNC_ACTION_SWITCH_MS','0') or 0)

    def observe_hedge_label(self,rid,*,gap,rescued,primary_matched,alternate_matched,k,prefix_length):
        """One hedged round's verified label; updates the bucket posterior."""
        if type(rescued) is not bool or type(primary_matched) is not bool:
            raise ValueError('HEDGE_LABEL_TYPES')
        bucket=(self._batch_key(), int(prefix_length)//128, int(k))
        cell=self.beta.setdefault(bucket,[1,1])   # Beta(1,1) prior
        cell[0]+=1 if rescued else 0
        cell[1]+=1 if not rescued else 0
        self.hedge_labels[str(rid)]=dict(gap=gap,rescued=rescued,primary_matched=primary_matched,
                                        alternate_matched=alternate_matched,k=k,bucket=list(bucket))
        self.event('hedge_label',rid=str(rid),rescued=rescued,primary_matched=primary_matched,
                   alternate_matched=alternate_matched,gap=gap,bucket=list(bucket),
                   trials=cell[0]+cell[1]-2,rescues=cell[0]-1)

    def _decide_hedge(self,rid,k,prefix,base_g,base_c,previous='stream'):
        """One bet, priced as an effective decode rate: LCB(Theta_hedge) > margin*Theta_stream.

        The observation window (3 gaps / 8 rounds) only reduces variance -- it is
        not itself the criterion. Both rates are E[N]/E[T] in tokens/ms:

            Theta_stream = mean(delivered) / mean(round time)
            Theta_hedge  = (g_stream + eta) / (C + W_sync + c_node + c_mask + c_shape + split + B)

        with B charged once for the single stream->sync transition, and eta from the
        bucket posterior's LCB. A hedge must also live long enough to amortise B:
        remaining rounds > B / (Theta_hedge - Theta_stream). Cold buckets never bet.
        """
        import math as _m
        cell=self._beta_cell(k,len(prefix))
        trials=0 if cell is None else cell[0]+cell[1]-2
        p_lcb=self.beta_lower(cell)
        observed=self.round_delivered.get(rid) or []
        if len(observed)<self.hedge_min_obs:
            return None,'awaiting_rate_evidence',dict(trials=trials,observed=len(observed),
                theta_stream=round(float(base_g)/max(1.0,float(base_c)),5))
        ordered=sorted(observed)
        g_s=ordered[len(ordered)//2]        # median: the first rounds are the thinnest
        own=[x for x in (self.lane_round_ms.get(rid) or {}).get('stream',[]) if x>0]
        round_ms=(sum(own)/len(own)) if own else float(self._measured_round_ms(base_c))
        theta_s=g_s/round_ms
        eta=p_lcb*self.gap_curve.rescue_tokens
        delta_ms=float(self.sync_round_delta_ms)
        delta_source='env' if delta_ms else 'prior'
        delta_cell=self._delta_cell(k,prefix)
        if self.hedge_delta_online and not self.sync_round_delta_ms:
            # Bucket median once it has evidence, else the conservative prior.
            samples=self.delta_samples.get(delta_cell) or []
            if len(samples)>=self.delta_min:
                ordered=sorted(samples)
                delta_ms=ordered[len(ordered)//2]
                delta_source='online'
            else:
                delta_ms=float(self.sync_rtt_ms) if self.sync_rtt_ms>0 else 0.0
                delta_source='prior'
        hedged_ms=max(1.0, round_ms+delta_ms)
        theta_h=(g_s+eta)/hedged_ms
        theta_h_ss=theta_h
        rate_source='model'
        # Use the sync lane's committed tokens per millisecond when observed;
        # its accepted-token count already includes alternate-root rescues.
        # Price the occupancy this request would create, preferring the floor
        # cell when that cell has enough evidence.
        lane_cell=self._lane_cell(k,self.lane_bucket())
        floor_cell=self._lane_cell(k,self.hedge_lane_floor)
        floor_rate=self.sync_rate.get(floor_cell)
        floor_ready=bool(floor_rate and len(floor_rate['rates'])>=self.sync_rate_min)
        rate=floor_rate if floor_ready else self.sync_rate.get(lane_cell)
        if floor_ready:
            lane_cell=floor_cell
        measured_rounds=0 if not rate else len(rate['rates'])
        if rate and len(rate['rates'])>=self.sync_rate_min:
            ordered_rate=sorted(rate['rates'])
            median_rate=ordered_rate[len(ordered_rate)//2]
            # One-sided 90% deflation for a small sample of per-round rates.
            theta_h=median_rate*(1.0-1.28/_m.sqrt(len(ordered_rate)))
            theta_h_ss=theta_h
            rate_source='measured'
        b_once=float(self.sync_rtt_ms) if previous!='sync' else 0.0
        margin=1.02
        budgets=getattr(self.stream,'budgets',{})
        remaining_tokens=max(0,budgets.get(rid,0)-len(prefix))
        remaining_rounds=int(_m.ceil(remaining_tokens/max(1e-9,g_s))) if remaining_tokens else 0
        # Tokens the switch costs = B_ms * Theta_stream; the hedge repays them at
        # the extra rate (Theta_hedge - Theta_stream), so the break-even is
        # B*Theta_s/(dTheta) milliseconds, i.e. that over one round time.
        need_rounds=(((b_once*theta_s)/(theta_h_ss-theta_s))/round_ms
                     if theta_h_ss>theta_s and b_once>0 else 0.0)
        detail=dict(trials=trials,p_lcb=round(p_lcb,4),g_stream=round(g_s,3),
                    theta_stream=round(theta_s,5),theta_hedge=round(theta_h,5),
                    theta_hedge_steady=round(theta_h_ss,5),eta=round(eta,3),
                    hedged_round_ms=round(hedged_ms,2),delta_ms=round(delta_ms,2),
                    delta_source=delta_source,delta_cell=list(delta_cell),
                    rate_source=rate_source,measured_rounds=measured_rounds,
                    lane_bucket=self.lane_bucket(),
                    b_ms=round(b_once,2),
                    remaining_rounds=remaining_rounds,
                    need_rounds=None if need_rounds is None else round(need_rounds,1))
        # Calibrate the configured cohort floor when its cell lacks evidence.
        # Sparse observations from lower occupancy do not establish the rate
        # at that floor, where several rows share the round-trip cost.
        if (self.hedge_lane_floor>1 and not floor_ready
                and self.lane_rows()<self.hedge_lane_floor):
            return True,'lane_floor_calibration',dict(detail,lane_floor=self.hedge_lane_floor,
                                                      lane_rows=self.lane_rows())
        if rate_source=='measured':
            # A measured sync rate needs no rescue labels: trials only guard the
            # modelled path. Still refuse when the request is too short or the
            # measured rate does not clear the stream's own rate by the margin.
            if remaining_rounds<2:
                return False,'request_too_short',detail
            if theta_h<=margin*theta_s:
                return False,'sync_rate_below_margin',detail
            if need_rounds is not None and remaining_rounds<=need_rounds:
                return False,'b_not_amortised',detail
            return True,'sync_rate_beats_margin',detail
        if trials<self.beta_min:
            cell_key=self._delta_cell(k,prefix)
            used=self.burnin_used.get(cell_key,0)
            if used<self.hedge_burnin:
                self.burnin_used[cell_key]=used+1
                return True,'explore_burn_in',dict(detail,burnin=used+1)
            draw=self.explore_draws
            self.explore_draws+=1
            exploring=(self.beta_explore>0 and (hash((rid,draw))%1000)/1000.0<self.beta_explore)
            if exploring:
                return True,'explore_label_collection',dict(detail,explore=True)
            return False,'cold_start_no_label',dict(detail,explore=False)
        if remaining_rounds<2:
            return False,'request_too_short',detail
        if theta_h<=margin*theta_s:
            return False,'theta_hedge_below_margin',detail
        if need_rounds is not None and remaining_rounds<=need_rounds:
            return False,'b_not_amortised',detail
        return True,'theta_lcb_beats_margin',detail

    def _measured_round_ms(self, base_c):
        """Round time used by the rate comparison (same-window measurement)."""
        return base_c if base_c and base_c>0 else 1.0

    def lane_rows(self):
        """Rows currently assigned to the sync lane.

        Use persistent round_transport assignments. The transient
        sync_requests set is rebuilt during each decision pass and cannot
        represent occupancy while that pass is in progress.
        """
        return sum(1 for lane in getattr(self,'round_transport',{}).values() if lane=='sync')

    def lane_bucket(self, extra=0):
        """Occupancy bucket of the sync lane (saturating at the floor)."""
        return min(self.hedge_lane_floor,max(1,self.lane_rows()+int(extra)))

    def stream_rewound_rows(self):
        """Rows for which a plan must be allowed to trim the edge session.

        That is exactly `streamed_rids`, so an admission-sync row can never appear
        here -- which is the structural reason it pays none of the switch cost.
        """
        return self.streamed_rids

    def _median_rate(self,rates):
        if not rates:
            return None,0
        ordered=sorted(rates)
        return ordered[len(ordered)//2],len(ordered)

    @staticmethod
    def _median_of(values):
        if not values:
            return None
        ordered=sorted(values)
        return ordered[len(ordered)//2]

    def _lane_observe(self,lane,tps,rid=None):
        """One finished row: what did its lane actually deliver, end to end?

        Only rows assigned by the rule are comparison evidence. Requests that
        streamed solely to fill the calibration pool were not chosen by the
        rule and may come from a different workload phase.
        """
        if not self.lane_on or not self.lane_hybrid:
            return
        if rid is not None and str(rid) not in self.lane_judged:
            return
        if lane not in self.lane_samples or not (tps>0):
            return
        self.lane_samples[lane].append(float(tps))
        self._lane_correct()

    def _lane_correct(self):
        """Let the window's own outcomes overrule the threshold, but only persistently.

        The threshold is a configurable prior. Consecutive votes and the
        configured margin are required before changing the belief; requests
        already committed keep their lane.
        """
        if self.lane_belief is None:
            return
        st=self._median_of(self.lane_samples['stream'])
        sy=self._median_of(self.lane_samples['sync'])
        if st is None or sy is None:
            return
        if (len(self.lane_samples['stream'])<self.lane_min_rows
                or len(self.lane_samples['sync'])<self.lane_min_rows):
            return
        margin=1.0+self.lane_flip_margin
        better=('sync' if sy>st*margin else 'stream' if st>sy*margin else None)
        if better is None or better==self.lane_belief:
            self.lane_flip_votes=0
            return
        self.lane_flip_votes+=1
        if self.lane_flip_votes<self.lane_flip_votes_needed:
            return
        self.lane_flips.append((self.lane_belief,better,round(st,4),round(sy,4)))
        self.event('lane_flip',old=self.lane_belief,new=better,
                   stream_median=round(st,4),sync_median=round(sy,4),
                   stream_rows=len(self.lane_samples['stream']),
                   sync_rows=len(self.lane_samples['sync']),margin=self.lane_flip_margin)
        self.lane_belief=better
        self.lane_flip_votes=0

    def _probe_lane(self,rid):
        """Alternate the first 2*per_lane admissions across the two lanes.

        This observes both lanes over the same interval before policy selection.
        """
        if not self.probe_on or self.probe_done:
            return None
        if rid in self.probe_tag:
            return self.probe_tag[rid]
        if self.probe_n >= 2*self.probe_per_lane:
            return None
        lane='sync' if self.probe_n % 2 else 'stream'
        self.probe_n+=1
        self.probe_tag[rid]=lane
        self.event('probe_assign',rid=rid,lane=lane,index=self.probe_n,
                   per_lane=self.probe_per_lane)
        return lane

    def _probe_finish(self):
        """Both lanes have their rows: report per-lane throughput over the probe."""
        rows=list(self.probe_rows)
        self.probe_done=True
        rate={}
        for lane in ('stream','sync'):
            sel=[r for r in rows if r[0]==lane]
            tok=sum(r[1] for r in sel)
            sec=sum(r[2] for r in sel)
            rate[lane]=(tok/sec) if sec>0 else None
        st,sy=rate.get('stream'),rate.get('sync')
        ratio=(sy/st) if (st and sy) else None
        self.event('probe_report',rows=len(rows),per_lane=self.probe_per_lane,
                   stream_tps=None if st is None else round(st,4),
                   sync_tps=None if sy is None else round(sy,4),
                   ratio=None if ratio is None else round(ratio,4),
                   samples=[(r[0],r[1],round(r[2],3)) for r in rows])

    def _admission_decision(self,rid,req):
        """Choose the transport once, before any stream exists.

        The gap and acceptance rate are not yet available, so compare the
        traffic class's stream rate with its sync rate at the expected
        occupancy. No stream has opened, so the switch cost B is zero.
        """
        if self.sync_learn is not None:
            self.learn_start.setdefault(rid,time.perf_counter())
            self.learn_cell.setdefault(rid,self._learn_cell(rid))
        probe=self._probe_lane(rid)
        if probe is not None:
            return probe
        # A forced-sync arm must actually be forced. The learner below answers as soon
        # as its cell is ready, and its answer wins the early return -- so a row in the
        # `sync` arm could be admitted to the stream lane while the arm's name says
        # otherwise, which would silently turn the fixed arm into a mixed one.
        if self.admit_sync_on and self.admit_sync_force:
            return 'sync'
        if self.observe_only:
            # Still ask the learner and log its answer: the point of this mode is to
            # OBSERVE g and P on a pure-stream window, so the trajectory must be
            # emitted even though the lane is forced (an early return here produced a
            # window with zero admission events).
            k0=self.round_k.get(rid) or getattr(self.growth,'current',(None,))[0]
            if k0 not in CHOICES:k0=min(CHOICES)
            if self.sync_learn is not None and self.sync_learn.ready(k0):
                choice,why=self.sync_learn.decide_k(k0,self.sync_c_ms)
                trace=dict(getattr(self.sync_learn,'last_price',{}) or {})
                self.event('admission_learn',rid=rid,k=k0,choice='stream',reason='observe_only',
                           would=choice,would_why=why,
                           price_a=trace.get('a'),price_b=trace.get('b'),
                           price_n=trace.get('n_model'),price_need=trace.get('need'),
                           price_d_hi=trace.get('d_hi'),price_p_lcb=trace.get('p_lcb'),
                           price_trials=trace.get('trials'),price_rate=trace.get('rate'),
                           **self.sync_learn.aggregate(k0))
            return 'stream'
        if self.sync_learn is not None and getattr(self.sync_learn,'lane_rule',False):
            # The rule always starts a row on the stream: the first `lane_warm`
            # rounds ARE its measurement, so asking the learner here would decide
            # before the row has produced the g the rule reads.
            self.event('admission_learn',rid=rid,choice='stream',reason='rule_start_stream',
                       k=self.round_k.get(rid) or getattr(self.growth,'current',(None,))[0])
            return 'stream'
        if self.sync_learn is not None:
            # No gap yet, so ask the width-aggregated view rather than the (k, gap)
            # cell this request will only join after its first frame.
            k0=self.round_k.get(rid) or getattr(self.growth,'current',(None,))[0]
            if k0 not in CHOICES:k0=min(CHOICES)
            if self.sync_learn.ready(k0):
                choice,why=self.sync_learn.decide_k(k0,self.sync_c_ms)
                if choice!='sync' and why=='explore_request_stream':
                    self.learn_pin[rid]='stream'
                trace=dict(getattr(self.sync_learn,'last_price',{}) or {})
                self.event('admission_learn',rid=rid,k=k0,choice=choice,reason=why,
                           pinned=self.learn_pin.get(rid),
                           price_a=trace.get('a'),price_b=trace.get('b'),
                           price_n=trace.get('n_model'),price_need=trace.get('need'),
                           price_d_hi=trace.get('d_hi'),price_sd=trace.get('d_sd'),
                           price_p_lcb=trace.get('p_lcb'),price_trials=trace.get('trials'),
                           price_rate=trace.get('rate'),
                           **self.sync_learn.aggregate(k0))
                return 'sync' if choice=='sync' else 'stream'
        if not self.admit_sync_on:
            return 'stream'
        if self.admit_sync_force:
            return 'sync'
        k=self.round_k.get(rid) or getattr(self.growth,'current',(None,))[0]
        if k not in CHOICES:
            k=min(CHOICES)
        stream_rate,n_stream=self._median_rate(self.stream_rates.get(self._batch_key()) or [])
        sync_rate,n_sync=self._median_rate(
            (self.sync_rate.get(self._lane_cell(k,self.hedge_lane_floor)) or {}).get('rates') or [])
        remaining=max(0,req.sampling_params.max_new_tokens-len(req.output_ids))
        detail=dict(k=k,stream_rate=None if stream_rate is None else round(stream_rate,6),
                    n_stream=n_stream,
                    sync_rate=None if sync_rate is None else round(sync_rate,6),n_sync=n_sync,
                    remaining_tokens=remaining,lane_floor=self.hedge_lane_floor)
        if n_stream<self.sync_rate_min or n_sync<self.sync_rate_min:
            self.event('admission_transport',rid=rid,chosen='stream',reason='prior_cold',**detail)
            return 'stream'
        if remaining<self.admit_min_tokens:
            self.event('admission_transport',rid=rid,chosen='stream',reason='request_too_short',**detail)
            return 'stream'
        if sync_rate<=self.admit_margin*stream_rate:
            self.event('admission_transport',rid=rid,chosen='stream',reason='stream_rate_higher',**detail)
            return 'stream'
        self.event('admission_transport',rid=rid,chosen='sync',reason='sync_rate_beats_margin',**detail)
        return 'sync'

    def _lane_cell(self,k,bucket):
        return (self._batch_key(),int(k),int(bucket))

    def _delta_cell(self,k,prefix):
        """Lane cell: the transport is not prefix-dependent.

        Lane timing depends on traffic class and draft width rather than the
        changing prefix length. Rescue posteriors retain their own prefix
        buckets; transport measurements share evidence across those buckets.
        """
        return (self._batch_key(),int(k))

    def _batch_key(self):
        """Stable traffic class for the posteriors.

        The operator supplies TRIAGE_HEDGE_CLASS, which defaults to one class.
        Keep it independent of transient cohort size so observations continue
        to accumulate as requests arrive and finish.
        """
        return getattr(self,'hedge_class','0')

    def _beta_cell(self,k,prefix_length):
        return self.beta.get((self._batch_key(),int(prefix_length)//128,int(k)))

    @staticmethod
    def beta_lower(cell,z=1.96):
        """Wilson lower bound of the rescue rate for a Beta(a,b) cell."""
        if cell is None:
            return 0.0
        from .fork_selector import wilson_lower
        a,b=cell[0],cell[1]
        n=a+b-2
        return wilson_lower(a-1,n,z) if n>0 else 0.0

    def _note_gap_sample(self,rid,gap,k,prefix_length):
        """One sample per newly opened path: a repeated prefix is the same path."""
        seen=self.gap_samples.setdefault(rid,[])
        if len(seen)>=self.hedge_gap_samples:
            return
        if seen and seen[-1][2]==prefix_length:
            return
        seen.append((gap,k,prefix_length))

    def _learn_cell(self,rid,k=None):
        """The cell this request belongs to, from things it carries itself.

        Use draft width, observed root gap and delivery state without dataset
        identity. LaneLearner.cell owns the final bucket definition.
        """
        rid=str(rid)
        if k is None:
            k=self.round_k.get(rid) or getattr(self.growth,'current',(None,))[0]
        if k not in CHOICES:k=min(CHOICES)
        meta=self.gap_seen.get(rid)
        gap=None if meta is None else meta.get('gap')
        seen=self.round_delivered.get(rid) or []
        stamp=(k,None if meta is None else meta.get('gap'),len(seen))
        cached=self.learn_cell_cache.get(rid)
        if cached is not None and cached[0]==stamp:
            return cached[1]
        delivery=None
        if seen:
            ordered=sorted(seen)
            delivery=ordered[len(ordered)//2]
        cell=LaneLearner.cell(k,gap,delivery)
        self.learn_cell_cache[rid]=(stamp,cell)
        return cell

    def _observe_delivery(self,rid,req):
        """Per-round committed tokens for this request (the acceptance criterion).

        `submit_native` runs every tick, so a non-zero jump in the committed count
        is exactly one round's delivery; the first one is what the gate freezes on.
        """
        count=len(req.output_ids)
        previous=self.req_tokens.get(rid)
        self.req_tokens[rid]=count
        if (getattr(self,'plan_timing',False) and rid in self.sync_requests
                and previous is not None and count>previous):
            t1=self.plan_t1.get(rid); t2=self.plan_take.get(rid); t3=time.perf_counter()
            if t1 is not None and t2 is not None:
                self.event('round_timing',rid=rid,wait_ms=round((t2-t1)*1000.0,2),
                           verify_ms=round((t3-t2)*1000.0,2),total_ms=round((t3-t1)*1000.0,2))
        pending=self.hedge_pending.pop(rid,None)
        if pending is not None and count>pending['index']:
            root=req.output_ids[pending['index']]
            rescued=root==pending['alternate']
            primary=root==pending['primary']
            k=pending['k'] if pending['k'] in CHOICES else min(CHOICES)
            if self.sync_learn is not None:
                self.sync_learn.record_rescue(self._learn_cell(rid,k),rescued)
            self.observe_hedge_label(rid,gap=pending['gap'],rescued=rescued,
                                     primary_matched=primary,alternate_matched=rescued,
                                     k=k,prefix_length=pending['index'])
        if previous is None or count<=previous:
            return
        if self.ar_round.pop(rid,False):
            return
        delivered=count-previous
        self.round_count[rid]=self.round_count.get(rid,0)+1
        state=self.sync_decisions.get(rid) or {}
        # Round timing for the online delta: consecutive delivered rounds of this
        # request, split by the lane they ran on. Only a request that has run on
        # both lanes yields a delta, and only that delta enters its bucket.
        now_ts=time.perf_counter()
        last_ts=self.round_ts.get(rid)
        self.round_ts[rid]=now_ts
        if last_ts is not None:
            # Attribute the observation to the lane that actually ran.
            # The legacy gate's hedge flag does not record learner decisions.
            lane='sync' if self.round_transport.get(rid)=='sync' else 'stream'
            round_ms=(now_ts-last_ts)*1000.0
            if lane=='stream' and self.lane_on:
                # The rule's own measurement. Counted here, before any filtering: a thin
                # round is the signal, not noise (see __init__). The pool closes as soon as
                # it holds lane_min_rounds steady rounds, and only rounds that arrived
                # BEFORE that (i.e. before any row was judged) ever enter it.
                own=self.own_stream.setdefault(rid,[0,0.0])
                own[0]+=1; own[1]+=float(delivered)
                if not self.lane_pool_done and own[0]>self.lane_warm:
                    self.win_stream.append(float(delivered))
                    if len(self.win_stream)>=self.lane_min_rounds:
                        self.lane_pool_done=True
            if self.sync_learn is not None:
                cell=self._learn_cell(rid)
                if lane=='sync':
                    base=self.sync_learn.stream_mean_all(cell)
                    self.sync_learn.record_sync(cell,delivered,round_ms)
                    # The online price model learns `dCost = a + b*g` from rounds that
                    # really were hedged: the extra ms against the stream lane's own
                    # ms per round, at the g measured right before the hedge.
                    if base is not None and base[1] > 0 and round_ms > 0:
                        k_obs=self.round_k.get(rid) or (cell[0] if isinstance(cell,tuple) else None)
                        self.sync_learn.record_price(k_obs, base[0], round_ms-base[1])
                    # Label the rescue from the round itself: a sync round that
                    # beats the stream lane's own tokens-per-ms is exactly what a
                    # second root is for. (The exact hedge label, when the delegate
                    # reports it, still lands through observe_hedge_label.)
                    if base is not None and base[1] > 0 and round_ms > 0:
                        self.sync_learn.record_rescue(
                            cell, (delivered/round_ms) > (base[0]/base[1]))
                else:
                    self.sync_learn.record_stream(cell,delivered,round_ms)
                    # Counterfactual acceptance label: the draft for this prefix was
                    # dispatched anyway (the growth plan lane runs on the stream too),
                    # so when it is in hand we can ask whether its token matched the
                    # one the stream actually committed. That is the same question a
                    # hedge would have answered, at no extra cost, and it is the only
                    # way P accumulates while the policy keeps choosing stream.
                    if getattr(self,'cf_labels',False):
                        e=self.entries.get(rid)
                        try:
                            at=len(e.prefix) if e is not None and e.future.done() else None
                            ready=at is not None and len(req.output_ids)>at
                        except Exception:
                            ready=False; at=None
                        if ready and self.cf_done.get(rid)!=at:
                            try:
                                draft_tokens=e.future.result()
                                alt=draft_tokens[0][0] if draft_tokens and draft_tokens[0] else None
                            except Exception:
                                alt=None
                            if alt is not None and tuple(e.prefix)==tuple(req.output_ids[:at]):
                                k_cf=self.round_k.get(rid) or (cell[0] if isinstance(cell,tuple) else None)
                                self.sync_learn.record_accept(
                                    k_cf, req.output_ids[at]==alt, counterfactual=True)
                                self.cf_done[rid]=at      # one label per prefix, many per row
            per_rid=self.lane_round_ms.setdefault(rid,{'stream':[],'sync':[]})
            if lane=='stream':
                series=per_rid['stream']
                series.append(round_ms)
                if len(series)>32:
                    del series[0]
                if round_ms>0 and delivered>=2:
                    own=self.stream_rates.setdefault(self._batch_key(),[])
                    own.append(delivered/round_ms)
                    if len(own)>64:
                        del own[0]
            elif per_rid['stream']:
                # Every hedged round is priced against THIS request's own stream
                # baseline, so one switched request yields a sample per round
                # (waiting for a whole-request mean gave one sample per request,
                # and delta_min=3 then needed three of them).
                baseline=sum(per_rid['stream'])/len(per_rid['stream'])
                delta=round_ms-baseline
                key=state.get('delta_cell')
                key=tuple(key) if key else self._delta_cell(
                    self.round_k.get(rid) or state.get('k') or min(CHOICES), req.output_ids)
                bucket=self.delta_samples.setdefault(key,[])
                bucket.append(delta)
                if len(bucket)>64:
                    del bucket[0]
                if len(bucket)%4==1 or len(bucket)==self.delta_min:
                    self.event('hedge_delta',rid=str(rid),delta_ms=round(delta,2),
                               samples=len(bucket),bucket=list(key),
                               stream_round_ms=round(baseline,2),hedged_round_ms=round(round_ms,2))
            if lane=='sync':
                cell=self.sync_rate.setdefault(
                    self._lane_cell(self.round_k.get(rid) or state.get('k') or min(CHOICES),
                                    self.lane_bucket()),
                    dict(rates=[]))
                if round_ms>0:
                    cell['rates'].append(delivered/round_ms)
                    if len(cell['rates'])>64:
                        del cell['rates'][0]
                if len(cell['rates'])%4==0:
                    self._save_prior()
        # The arm-level rate must come from stream rounds: hedged rounds are the
        # treatment, not the baseline that decides whether to treat.
        if state.get('hedge') is not True:
            self.arm_rate=float(delivered) if self.arm_rate is None else 0.8*self.arm_rate+0.2*delivered
        seen=self.round_delivered.setdefault(rid,[])
        if len(seen)<32:
            seen.append(delivered)

    def _learn_transport(self,rid,prefix,k,action):
        """The learner's answer for this committed prefix, computed once.

        Reuse the verdict until the committed prefix changes instead of
        repeating the same bookkeeping on every scheduler tick.
        """
        key=(rid,tuple(prefix))
        memo=self.learn_verdict.get(rid)
        if memo is not None and memo[0]==key:
            return memo[1],memo[2],memo[3]
        pin=self.learn_pin.get(rid)
        if pin=='stream' and not getattr(self.sync_learn,'lane_rule',False):
            self.learn_verdict[rid]=(key,'stream','explore_request_stream',True)
            self.learn_cell[rid]=self._learn_cell(rid,k)
            self.learn_start.setdefault(rid,time.perf_counter())
            return 'stream','explore_request_stream',True
        cell=self._learn_cell(rid,k)
        self.learn_cell[rid]=cell
        self.learn_start.setdefault(rid,time.perf_counter())
        locked=self.learn_lock.get(rid)
        if getattr(self.sync_learn,'lane_rule',False):
            own=self.own_stream.get(rid)
            rounds=own[0] if own else 0
            if locked is not None:
                choice,why=locked,'rule_locked'
            elif not self.lane_pool_done or rounds<self.lane_warm:
                # Still measuring (the pool freezes on its own), or this row has not
                # produced enough rounds to be worth judging yet: stay on the stream.
                choice,why='stream','rule_warm'
            else:
                g=sum(self.win_stream)/len(self.win_stream)
                threshold=float(getattr(self.sync_learn,'lane_g',3.05))
                prior='sync' if g<threshold else 'stream'
                if self.lane_belief is None:
                    self.lane_belief=prior
                choice=self.lane_belief
                index=len(self.lane_judged)
                explore=(self.lane_hybrid and bool(self.lane_explore_every) and index>0
                         and index%self.lane_explore_every==0)
                if explore:
                    # Sample the other lane every lane_explore_every judged
                    # requests. Charge exploration per request so both lane
                    # beliefs remain testable with whole-request observations.
                    choice='stream' if choice=='sync' else 'sync'
                why='rule_explore_'+choice if explore else ('rule_sync' if choice=='sync' else 'rule_stream')
                self.learn_lock[rid]=choice
                self.lane_judged.append(rid)
                self.event('lane_rule',rid=rid,choice=choice,why=why,g=round(g,4),
                           threshold=threshold,prior=prior,belief=self.lane_belief,
                           explore=bool(explore),row_index=index,
                           own_rounds=rounds,own_tokens=round(own[1],1),
                           window_rounds=len(self.win_stream),warm=self.lane_warm,
                           samples=(len(self.lane_samples['stream']),len(self.lane_samples['sync'])),
                           k=k,cell=list(cell))
            self.learn_verdict[rid]=(key,choice,why,True)
            return choice,why,True
        learn_applied=True
        streak=self.learn_streak.get(rid,0)+1
        self.learn_streak[rid]=streak
        refresh=int(getattr(self,'learn_refresh',8) or 0)
        if locked=='sync' and refresh and streak%refresh==0:
            choice,why='stream','refresh_baseline'
        elif locked is not None:
            choice,why=locked,'locked'
        else:
            choice,why=self.sync_learn.decide(cell,self.sync_c_ms+action.cost_ms())
            if choice=='fallback':
                learn_applied=False
            measured_sync=self.sync_learn.cells.get(cell,{}).get('s_n',0)>=1
            if (choice in ('sync','stream') and measured_sync
                    and why not in ('baseline_cold','explore_rescue',
                                    'explore_request_stream','explore_request_sync',
                                    'explore_hedge','price_uncertain',
                                    'refresh_baseline')):
                self.learn_lock[rid]=choice
        self.learn_verdict[rid]=(key,choice,why,learn_applied)
        calls=self.learn_events.get(rid,0)+1
        self.learn_events[rid]=calls
        if why!=self.learn_reason.get(rid) or calls%64==0:
            self.learn_reason[rid]=why
            self.event('sync_learn',rid=rid,chosen=(learn_applied and choice=='sync'),
                       applied=learn_applied,reason=why,k=k,cell=list(cell),calls=calls,
                       **self.sync_learn.snapshot(cell))
        return choice,why,learn_applied

    def _consider_sync_action(self,rid,req,prefix):
        """Re-decide the transport for this round; B is paid on a change."""
        if (rid in self.stream_plans or rid in self.ar_selected
                or rid in self.sync_requests and rid in self.entries):
            return rid in self.sync_requests
        committed=self.learn_lock.get(rid)
        if committed=='stream':
            self.round_transport[rid]='stream'
            self.sync_requests.discard(rid)
            return False
        if committed=='sync':
            self.sync_requests.add(rid)
            self.round_transport[rid]='sync'
            return True
        previous=self.round_transport.get(rid,'stream')
        if previous=='sync':
            self.sync_requests.discard(rid)
        from .sync_action import SyncDraftAction
        state=self.sync_decisions.get(rid) or {}
        k=state.get('k')
        switch_cost=0.0
        action=None
        base_g,base_c=4.0,120.0
        inputs={}
        if self.sync_rtt_ms <= 0 and not (self.auto_transport and self.second_answer):
            # An unpriced round trip cannot be compared: underpricing RTT is the
            # classic way to get a spurious switch, so stay on the stream.
            self.sync_decisions[rid]=dict(chosen=False,capability=self.second_answer,
                previous=previous,reason='rtt_unpriced')
            self.round_transport[rid]='stream'
            self.event('sync_action',rid=rid,chosen=False,previous=previous,
                       reason='rtt_unpriced',capability=self.second_answer)
            return False
        try:
            # Same K on both arms: price the sync action at the K this request is
            # actually running, and read the baseline only from single-root
            # samples of that K. `growth.current` is never set on this path (the
            # stream selector binds K itself), so the consumed Plan is the
            # authority.
            k_source='decision'
            if k not in CHOICES:
                k_source='last_round'
                k=self.round_k.get(rid)
            if k not in CHOICES:
                k_source='growth_current'
                k=getattr(self.growth,'current',(None,))[0]
            if k not in CHOICES:
                k_source='minimum_choice'
                k=min(CHOICES)
            rows=[v for key,v in self.growth.stats.items()
                  if len(key)>=4 and key[2]==k and key[3]==1 and v.get('n')]
            measured=bool(rows)
            if rows:
                n=sum(v['n'] for v in rows)
                delivered=sum(v['tokens'] for v in rows)/n
                seconds=sum(v['seconds'] for v in rows)/n
                if delivered>0 and seconds>0:
                    base_g,base_c=delivered,seconds*1000.0
            # Remaining *rounds*, not remaining tokens: B is amortised over rounds.
            remaining_tokens=req.sampling_params.max_new_tokens-len(req.output_ids)
            rounds_per_token=max(1.0,base_g)
            remaining_rounds=max(1,int(-(-remaining_tokens//rounds_per_token)))
            switch_cost=self.sync_switch_ms if previous=='stream' else 0.0
            if previous=='sync':
                # Leaving sync must drain the in-flight RPC and reopen a stream.
                switch_cost+=self.sync_switch_ms
            dual_root=self.second_answer or not self.auto_transport
            action=SyncDraftAction(k,rtt_ms=self.sync_rtt_ms,edge_ms_per_token=self.sync_edge_ms,
                c_ms=self.sync_c_ms,c_node_ms=0.03 if dual_root else 0.,switch_ms=switch_cost,
                remaining_rounds=remaining_rounds,
                rescue_gain=self.sync_rescue_gain if dual_root else 0.)
            # Runtime capability: the request needs a second answer only when the
            # root gap the edge measured on its first round makes hedging worth
            # its cost. Decided once per request (a later round's gap describes a
            # speculative position), then treated as a forcing term.
            gap_meta=self.gap_seen.get(rid)
            gap=None if gap_meta is None or not dual_root else float(gap_meta['gap'])
            delivered=self.round_delivered.get(rid)   # list of the first rounds' deliveries
            if self.hedge_arm_rate>0 and gap is not None and 'hedge' not in state:
                # Start on the stream and only collect: up to `hedge_gap_samples`
                # gaps (one per newly opened path) or `hedge_round_budget` rounds,
                # whichever lands first. Nothing is bought before the trigger.
                self._note_gap_sample(rid,gap,k,len(prefix))
            if self.hedge_arm_rate>0 and 'hedge' not in state:
                gaps_seen=self.gap_samples.get(rid,[])
                rounds_seen=self.round_count.get(rid,0)
                if len(gaps_seen)>=self.hedge_gap_samples or rounds_seen>=self.hedge_round_budget:
                    worth,why,detail=self._decide_hedge(rid,k,prefix,base_g,base_c)
                    if worth is None:
                        # Not enough evidence to price this request yet: record
                        # nothing, so the trigger fires again on the next round.
                        self.event('sync_hedge_defer',rid=rid,k=k,prefix_length=len(prefix),reason=why,**detail)
                    else:
                        state.update(hedge=worth,gap_reason=why,**detail)
                        self.event('sync_hedge_gate',rid=rid,hedge=worth,k=k,base_g=base_g,
                                   base_c_ms=base_c,prefix_length=len(prefix),reason=why,**detail)
            if self.hedge_accept_min>0:
                # Experiment variant: hedge a request only while its first
                # observed round delivered fewer tokens than the threshold. Frozen
                # at that first observation, like the gap gate.
                if 'hedge' not in state and delivered and len(delivered)>=self.hedge_accept_rounds:
                    mean=sum(delivered)/len(delivered)
                    worth=mean<self.hedge_accept_min
                    why='accept_below_threshold' if worth else 'accept_above_threshold'
                    state.update(hedge=worth,delivered=mean,rounds=list(delivered),gap_reason=why)
                    self.event('sync_hedge_gate',rid=rid,hedge=worth,delivered=mean,
                               rounds=list(delivered),accept_min=self.hedge_accept_min,
                               accept_rounds=self.hedge_accept_rounds,k=k,base_g=base_g,
                               base_c_ms=base_c,measured_baseline=measured,reason=why)
            elif self.hedge_arm_rate<=0 and 'hedge' not in state and gap is not None:
                sync_round_ms=base_c+self.sync_round_delta_ms
                worth,why=self.gap_curve.worth(gap,base_g,base_c,action,sync_round_ms=sync_round_ms)
                state.update(hedge=worth,gap=gap,gap_reason=why,
                             gap_open_length=gap_meta.get('open_length'))
                self.event('sync_hedge_gate',rid=rid,hedge=worth,gap=gap,
                           p_rescue=self.gap_curve.p_rescue(gap),eta=self.gap_curve.eta(gap),
                           k=k,base_g=base_g,base_c_ms=base_c,measured_baseline=measured,
                           sync_round_ms=sync_round_ms,cost_ms=action.cost_ms(),
                           price_ratio=(sync_round_ms/base_c if base_c>0 else None),
                           price_cap=self.gap_curve.price_cap,p_min=self.gap_curve.p_min,
                           break_even_eta=self.gap_curve.break_even_eta(base_g,base_c,sync_round_ms),
                           reason=why,open_length=gap_meta.get('open_length'))
            hedge=state.get('hedge')
            if self.sync_learn is not None:
                choice,why,learn_applied=self._learn_transport(rid,prefix,k,action)
                if learn_applied:
                    transport='sync' if choice=='sync' else 'stream'
                    reason='learn_%s'%why
                    cell=self._learn_cell(rid,k)
                    inputs=dict(cell=list(cell),**self.sync_learn.snapshot(cell))
            # Capability forces sync; otherwise the same cost score decides with
            # the switch cost B already inside the action, and a sync request is
            # only left when the score clearly favours the stream.
            from .sync_action import decide_transport
            if self.sync_learn is not None and learn_applied:
                pass   # the learner's per-round decision stands; skip the one-shot gate
            elif self.hedge_once and (self.hedge_forced or hedge is not None):
                # One decision per request: capability forces sync, absence of it
                # means the stream, and neither is revisited by the cost score.
                transport='sync' if bool(self.hedge_forced or hedge) else 'stream'
                reason='capability' if transport=='sync' else 'no_capability'
                inputs={}
            elif self.sync_rtt_ms <= 0:
                # Without an RTT estimate, require positive root-gap evidence.
                transport='sync' if hedge else 'stream'
                reason='capability' if hedge else 'pending'
                inputs={}
            else:
                transport,reason,inputs=decide_transport(
                    action,base_g,base_c,previous=previous,
                    capability_required=bool(self.hedge_forced or hedge),margin=1.10)
            if self.hedge_forced:
                reason='capability_forced'
            elif hedge is not None and not str(reason).startswith('learn_'):
                # Keep whatever the criterion recorded (lcb_gain_beats_switch_cost,
                # explore_label_collection, accept_below_threshold, root_gap_hedge...).
                # The online learner's own reason wins: it is the decision that ran.
                reason=state.get('gap_reason') or 'capability'
            elif hedge is None and reason=='stream_score_higher':
                reason=state.get('gap_reason') or 'pending'
            inputs.update(gap=gap,delivered=(sum(delivered)/len(delivered) if delivered else None),
                          eta=self.gap_curve.eta(gap) if gap is not None else 0.0,hedge=hedge)
            chosen=transport=='sync'
            if chosen and self.auto_transport and (remaining_tokens <= max(CHOICES)
                    or len(prefix)+max(CHOICES)+1 > min(self.context, CONTEXT)):
                chosen=False;reason='sync_round_budget'
        except Exception:
            chosen=False;reason='action_error_fallback_stream';inputs={}
            self.round_transport[rid]='stream'
            self.sync_requests.discard(rid)
        self.sync_decisions[rid]=dict(state,chosen=chosen,capability=self.second_answer,
            previous=previous,switch_charged_ms=switch_cost,k=k,k_source=k_source,
            remaining_rounds=remaining_rounds,reason=reason)
        if chosen:
            self.sync_requests.add(rid)
            if previous!='sync':
                self.cache_offers.pop(rid,None)
                self.selected_offers.pop(rid,None)
                self.batch_order.pop(rid,None)
                self.suggested_actions.pop(rid,None)
                self.wait_rollout_cache.pop(rid,None)
                with self.stream.lock:
                    self.stream._pause(rid)
                    self.entries.pop(rid,None)
                    self.stream.root_top2.pop(rid,None)
                self.growth.forget(rid)
        elif previous=='sync':
            self.growth.forget(rid)
            self.admit_sync_rows.discard(rid)
        self.round_transport[rid]='sync' if chosen else 'stream'
        # This runs on every token step, so the event and the in-memory log keep
        # only changes plus a periodic sample instead of one row per call.
        calls=self.sync_calls.get(rid,0)+1
        self.sync_calls[rid]=calls
        if chosen==state.get('chosen') and reason==state.get('reason') and calls%8:
            return chosen
        self.sync_action_log.append(dict(rid=rid,chosen=chosen,base_g=base_g,base_c_ms=base_c,
            k=k,k_source=k_source,measured_baseline=measured,
            prefix_hash=fingerprint(prefix),gap=(None if inputs.get('gap') is None else inputs['gap']),
            hedge=inputs.get('hedge'),cost_ms=action.cost_ms() if action is not None else None,
            action=None if action is None else action.snapshot(),
            stream_score=None if action is None else action.stream_score(base_g,base_c),
            sync_score=None if action is None else action.score(base_g,base_c)))
        if len(self.sync_action_log)>256:
            del self.sync_action_log[:128]
        self.event('sync_action',rid=rid,chosen=chosen,previous=previous,calls=calls,
                   hedged=chosen,prefix_hash=fingerprint(prefix),committed=len(prefix),
                   k=k,k_source=k_source,measured_baseline=measured,
                   switch_charged_ms=switch_cost,base_g=base_g,base_c_ms=base_c,
                   gap=inputs.get('gap'),eta=inputs.get('eta'),hedge=inputs.get('hedge'),
                   cost_ms=action.cost_ms() if action is not None else None,
                   w_sync_ms=None if action is None else action.w_sync_ms(),
                   node_ms=None if action is None else action.node_ms(),switch_ms=switch_cost,
                   capability=self.second_answer,reason=reason,
                   stream_score=inputs.get('stream_score'),sync_score=inputs.get('sync_score'))
        return chosen

    def submit_native(self,reqs,stride=WIDTH,phase='commit',mode=None):
        if stride!=WIDTH or self.closed:raise ValueError('STREAM_SELECTOR_STATE')
        reqs=list(reqs)
        finished=[str(r.rid) for r in reqs if str(r.rid).startswith(self.ec_prefix) and r.finished()]
        self.release(finished)
        eligible=[r for r in reqs if str(r.rid).startswith(self.ec_prefix) and not r.finished() and r.output_ids]
        with self.lock:
            for req in eligible:
                rid=str(req.rid);prefix=tuple(req.origin_input_ids)+tuple(req.output_ids)
                if getattr(req,'inflight_middle_chunks',0):raise ValueError('GROWTH_UNCOMMITTED_NATIVE')
                held=self.stream_plans.get(rid)
                if held is not None and held.prefix_hash!=fingerprint(prefix):
                    raise ValueError('STREAM_SELECTOR_UNCONSUMED_PLAN')
                self.requests_by_rid[rid]=req
                self.similar_cache.observe_committed(req)
                cache_start=time.perf_counter_ns()
                cache_cpu_start=time.thread_time_ns()
                held_offers=self.cache_offers.get(rid)
                if held_offers is None or held_offers[0]!=prefix:
                    remaining=req.sampling_params.max_new_tokens-len(req.output_ids)
                    offers=tuple(offer for k in CHOICES
                        if k<=remaining and len(prefix)+k<=self.context
                        for offer in [self.similar_cache.preview_committed(req,want=k)] if offer is not None)
                    self.cache_offers[rid]=prefix,offers
                self.growth.cache_agreement.record(req,self.cache_offers[rid][1])
                self.cache_cpu_ns+=time.thread_time_ns()-cache_cpu_start
                self.cache_wall_ns+=time.perf_counter_ns()-cache_start
                transport=self.admit_decided.get(rid)
                if transport is None:
                    transport=self._admission_decision(rid,req) if self.sync_enabled else 'stream'
                    self.admit_decided[rid]=transport
                    if transport=='sync':
                        # Owned by the plan lane from round zero: the stream is never
                        # opened for this row, so no drain, no trim, no re-open.
                        self.sync_requests.add(rid)
                        self.admit_sync_rows.add(rid)
                        self.round_transport[rid]='sync'
                        self.sync_decisions[rid]=dict(hedge=True,chosen=True,
                            reason='admission_sync',k=self.round_k.get(rid) or min(CHOICES),
                            capability=self.second_answer,admission=True)
                        self.event('sync_action',rid=rid,chosen=True,previous='stream',
                                   reason='admission_sync',k=self.round_k.get(rid) or min(CHOICES))
                if transport!='sync':
                    self.streamed_rids.add(rid)
                if (self.hedge_accept_min>0 or self.hedge_arm_rate>0) and self.sync_enabled:
                    self._observe_delivery(rid,req)
                if self.sync_enabled and transport!='sync':
                    self._consider_sync_action(rid,req,prefix)
            if len(self.requests_by_rid)>MAX_REQUESTS:raise ValueError('GROWTH_LIVE_LIMIT')
            # A row that just moved to the sync lane had its preview dropped by
            # _consider_sync_action, so this lookup must tolerate the absence.
            deferred={str(req.rid) for req in eligible if self.cache_offers.get(str(req.rid),(None,()))[1]}
            streamed=[req for req in eligible if str(req.rid) not in self.sync_requests]
            # Only an enabled gate can own a request on the sync lane; building
            # this list costs a pass over the cohort on every token step.
            synced=[req for req in eligible if str(req.rid) in self.sync_requests] if self.sync_enabled else []
            # Safety net for a plan whose owner left the lane: retire it so this
            # round can be planned instead of colliding with the orphan.
            for req in synced:
                orphan=(str(req.rid),fingerprint(tuple(req.prefix)))
                if orphan in getattr(self.growth,'issued',{}) and str(req.rid) not in self.entries:
                    self.growth.forget(str(req.rid))
                    self.event('hedge_plan_retired',rid=str(req.rid),prefix_hash=orphan[1])
        if streamed:self.stream.submit_committed(streamed,stride,phase,mode=mode,defer_producer=deferred)
        # A request that chose the sync lane is owned by the inherited plan-RPC
        # path (one round trip per round); it never opens or feeds a stream.
        if synced:GrowthPrefetch.submit_native(self,synced,stride,phase,mode=mode)

    def observe_committed(self,request):
        with self.lock:
            start=time.perf_counter_ns();cpu_start=time.thread_time_ns()
            self.growth.cache_agreement.record(request,(),register=False)
            self.cache_cpu_ns+=time.thread_time_ns()-cpu_start
            self.cache_wall_ns+=time.perf_counter_ns()-start
            return super().observe_committed(request)

    def ready(self,rid,prefix):
        if str(rid) in self.sync_requests:return super().ready(rid,prefix)
        return self.stream.ready(rid,prefix)

    def _batch_mode(self):
        # A cohort owns its current prefixes until consumption/cancel/free, even
        # after its resident count shrinks to one. New prefixes can resume C1 H2.
        return len(self.requests_by_rid)>1 or bool(self.batch_order)

    def _model_choice(self,rid,prefix,req):
        if self._batch_mode():return None
        remaining=req.sampling_params.max_new_tokens-len(req.output_ids)
        required=FORCED[0] if FORCED is not None else min(CHOICES)
        if remaining<required:return AR,'stream_tail_budget',0.
        if FORCED is not None or not self.growth.joint_ar:return None
        if self.growth.ar_seen<16:return AR,'stream_ar_bootstrap',0.
        warm=self.stream.warm_futures.get(rid)
        if warm is not None and not warm.done():return None
        self.stream._observe_alignment(rid,tuple(prefix))
        path=self.stream.paths.get(rid)
        extra=self.stream._extra(rid,tuple(prefix))
        if path is not None and path.paused and extra is not None and len(extra)<self.stream.ahead:
            # Resume the producer now, preserving buffered tokens, rather than
            # predicting overlap that would only start after this native call.
            self.stream._open(rid,list(prefix)+extra,'continue',self.stream.budgets[rid],confirmed=len(prefix))
        else:self.stream._maintain(rid,tuple(prefix),self.stream.budgets[rid])
        path=self.stream.paths.get(rid)
        if path is None:return None
        now=self.clock()
        span=(self.probe.deadline_ns() if self.probe is not None else self.grace_ns)/1e6
        held=self.waits.get(rid)
        wait=max(0.,(held[1]-now)/1e6) if held is not None and held[0]==tuple(prefix) else span
        stamp=(path.epoch,path.streamed,path.done,path.paused,len(path.tokens),path.alignment_failed)
        cached=self.wait_rollout_cache.get(rid)
        if (cached is not None and cached[0]==tuple(prefix) and cached[1]==stamp
                and now<cached[3] and wait>0):
            return cached[2]
        result=self.growth.rollout(rid,prefix,path=path,remaining=remaining,wait_ms=wait,
            deadline_ms=span,now_ns=now,ahead=self.stream.ahead)
        if result is not None and result[0]==WAIT:
            # Re-evaluate at a real arrival, a residual-distribution breakpoint,
            # or the hard deadline, not on every 1ms controller poll. A 2ms
            # decision quantum bounds hot-path CPU work; it never extends a cap.
            expiry=max(now+int(result[2]*1e6),self.clock()+2_000_000)
            if held is not None and held[0]==tuple(prefix):expiry=min(expiry,held[1])
            self.wait_rollout_cache[rid]=(tuple(prefix),stamp,result,expiry)
        else:self.wait_rollout_cache.pop(rid,None)
        return result

    def _batch_preview(self,rid,prefix):
        """Read-only supply/reward preview: no Plan, reservation or cache LRU move."""
        entry=self.stream._entry(rid,prefix)
        self.stream._check_warm()
        if entry.source in ('ar','cancelled'):return {}
        req=self.requests_by_rid[rid]
        remaining=req.sampling_params.max_new_tokens-len(req.output_ids)
        held=self.cache_offers.get(rid,(prefix,()))
        if held[0]!=prefix:raise ValueError('STREAM_BATCH_CACHE_PREFIX')
        cache={len(o.tokens):o for o in held[1]}
        if entry.source=='stream':
            if not cache or rid in self.stream.paths:
                self.stream._maintain(rid,prefix,self.stream.budgets[rid])
            tokens=self.stream._extra(rid,prefix) or ()
            source='edge'
        else:
            if not entry.future.done():return {}
            tokens=entry.future.result()[entry.index]
            source='suffix' if entry.source=='similar_context' else entry.source
        result={}
        for k in CHOICES:
            if k>remaining or len(prefix)+k>min(self.context,CONTEXT):continue
            alternatives=[]
            if k<=len(tokens):
                profile=self.growth.profile(rid,source)
                reward=profile.mean(k,remaining) if profile.supported(k,remaining) else None
                alternatives.append(Offer(rid,prefix,k,source,reward))
            offer=cache.get(k)
            if offer is not None:
                alpha,n=self.growth.cache_agreement.quality(offer)
                reward=self.growth.expected_progress(alpha,k,remaining) if n>=8 else None
                alternatives.append(Offer(rid,prefix,k,offer.source,reward,offer))
            if alternatives:
                result[k]=max(alternatives,key=lambda o:(o.reward if o.reward is not None else -1.,
                                                         o.cache is not None))
        return result

    def _select_batch(self,requests,decision):
        start=time.perf_counter_ns();cpu_start=time.thread_time_ns()
        try:
            with self.lock,self.stream.lock:
                return self._select_batch_locked(requests,decision)
        finally:
            self.selector_cpu_ns+=time.thread_time_ns()-cpu_start
            self.selector_wall_ns+=time.perf_counter_ns()-start

    def _select_batch_locked(self,requests,decision):
        selected=set(decision.selected)
        live={r.rid:r for r in requests if r.rid in selected and r.rid.startswith(self.ec_prefix)}
        if not live:return decision
        for rid,req in live.items():
            self.batch_order.setdefault(rid,req.prefix)
            if self.batch_order[rid]!=req.prefix:raise ValueError('STREAM_BATCH_UNCONSUMED_PREFIX')
        ordered=[rid for rid in self.batch_order if rid in live]
        # Already signed work survives prefill deferral, new arrivals and cost updates.
        held=sorted((p for rid,p in self.stream_plans.items() if rid in live),key=lambda p:p.round_id)
        frozen_ar=[rid for rid in ordered if rid in self.ar_selected]
        if held:
            k=held[0].k
            ids=tuple(p.rid for p in held if p.k==k)[:8]
            for rid in ids:self.plan_for(rid,live[rid].prefix)
            return replace(decision,selected=ids,held_ready=tuple(r for r in ordered if r not in ids),
                           pending=(),reason='stream_batch_frozen')
        if frozen_ar:
            for rid in frozen_ar:
                if self.ar_selected[rid][0]!=live[rid].prefix:raise ValueError('STREAM_SELECTOR_AR_PREFIX')
            return replace(decision,selected=tuple(frozen_ar[:8]),reason='stream_batch_ar_frozen')
        previews={rid:self._batch_preview(rid,live[rid].prefix) for rid in ordered}
        expired=[];tail=[]
        for rid in ordered:
            req=live[rid]
            old=self.waits.get(rid)
            if old is None or old[0]!=req.prefix:
                old=(req.prefix,decision.now_ns+self._wait_budget_ns(),decision.now_ns)
                self.waits[rid]=old
                self.stats['adaptive_waits']+=1
            remaining=req.maximum-len(req.generated)
            if remaining<min(CHOICES) or self.entries[rid].source=='ar':tail.append(rid)
            elif decision.now_ns>=old[1]:expired.append(rid)
        urgent=[rid for rid in ordered if rid in tail or rid in expired]
        anchor=urgent[0] if urgent else next((rid for rid in ordered if previews[rid]),None)
        choice=None
        if anchor is not None:
            priority=[anchor]+[rid for rid in ordered if rid!=anchor]
            ar_rows=tuple(Offer(rid,live[rid].prefix,0,'ar',1.) for rid in priority)
            if anchor in tail or not previews[anchor]:
                ids=urgent[:8]
                choice=Batch('ar',tuple(Offer(rid,live[rid].prefix,0,'ar',1.) for rid in ids),
                             'stream_batch_tail' if anchor in tail else 'stream_batch_deadline')
            else:
                length=max(len(req.prefix) for req in self.requests_by_rid.values())
                choice=self.growth.batch_rate.choose([previews[rid] for rid in priority],CHOICES,anchor,
                    length=length,ar_rows=ar_rows,allow_ar=self.growth.joint_ar,
                    elapsed_ms=self.growth.selector_elapsed_ms,probe_ms=self.growth.selector_probe_ms,
                    forced=FORCED[0] if FORCED is not None else None)
                if choice is None and urgent:
                    choice=Batch('ar',(ar_rows[0],),'stream_batch_deadline')
        if choice is None:
            return replace(decision,selected=(),held_ready=tuple(rid for rid in ordered if previews[rid]),
                pending=tuple(rid for rid in ordered if not previews[rid]),
                deadline_ns=min(self.waits[rid][1] for rid in ordered),reason='stream_batch_wait')
        # Validate the whole batch before issuing any immutable Plan or claiming data.
        for offer in choice.offers:
            if live[offer.rid].prefix!=offer.prefix:raise ValueError('STREAM_BATCH_PREFIX_BINDING')
            if choice.mode=='spec':
                actual=previews[offer.rid].get(offer.k)
                if actual!=offer:raise ValueError('STREAM_BATCH_OFFER_BINDING')
                if offer.cache is not None:self.stream._pause(offer.rid)
                elif not self.stream.has_draft(offer.rid,offer.prefix,count=offer.k):
                    raise ValueError('STREAM_BATCH_REAL_DEPTH')
        for offer in choice.offers:
            rid=offer.rid
            if choice.mode=='ar':
                self.ar_selected[rid]=(offer.prefix,AR,choice.reason,0.)
            else:
                plan=self.growth.bind_ready(rid,offer.prefix,k=(offer.k,choice.reason),
                                            source=offer.source,batch=len(choice.offers))
                self.stream_plans[rid]=plan
                if offer.cache is not None:self.selected_offers[rid]=offer.cache
                self._observe_arrival(rid,decision.now_ns)
        self.growth.batch_rate.choices+=1
        self.event('stream_batch_decision',ids=list(choice.ids),mode=choice.mode,k=choice.k,
            reason=choice.reason,predicted_native_ms=choice.predicted_ms,expected_progress=choice.progress,
            candidates=self.growth.batch_rate.last_candidates,physical_geometry=self.growth.batch_rate.geometry,
            objective='FIFO_CONSTRAINED_BATCH_RATE_NOT_JOINT_BELLMAN')
        unselected=[rid for rid in ordered if rid not in choice.ids]
        return replace(decision,selected=choice.ids,reason=choice.reason,
            held_ready=tuple(rid for rid in unselected if previews[rid]),
            pending=tuple(rid for rid in unselected if not previews[rid]),
            deadline_ns=min((self.waits[rid][1] for rid in unselected),default=None))

    def _wait_budget_ns(self):
        eta=max((self.plan_eta_ms(rid) for rid in self.sync_requests),default=0.)
        if eta>0.:return int(eta*1_000_000)
        if not self.growth.joint_ar:return super()._wait_budget_ns()
        # The event model replaces the old median-rate gate; its hard bound is
        # still exactly the original/probed, non-extendable prefix deadline.
        return self.probe.deadline_ns() if self.probe is not None else self.grace_ns

    def select(self,requests,decision):
        if self._batch_mode():
            sync_ids=set(decision.selected)&self.sync_requests
            if not sync_ids:
                return self._select_batch(requests,decision)
            sync=super().select(requests,replace(decision,
                selected=tuple(rid for rid in decision.selected if rid in sync_ids)))
            stream=self._select_batch(requests,replace(decision,
                selected=tuple(rid for rid in decision.selected if rid not in sync_ids)))
            for rid in decision.selected:
                if rid in sync.selected:
                    return sync
                if rid in stream.selected:
                    return stream
            deadlines=[d for d in (sync.deadline_ns,stream.deadline_ns) if d is not None]
            return replace(stream,held_ready=stream.held_ready+sync.held_ready,
                           pending=stream.pending+sync.pending,
                           deadline_ns=min(deadlines,default=None))
        selected=set(decision.selected)
        for req in requests:
            if req.rid in selected and req.rid.startswith(self.ec_prefix):
                self.has_draft(req.rid,req.prefix)
        due=[]
        for req in requests:
            suggestion=self.suggested_actions.get(req.rid)
            if req.rid in selected and suggestion is not None and suggestion[0]==req.prefix and suggestion[1]==AR:
                self.ar_selected[req.rid]=suggestion
                due.append(req.rid)
        if due:
            ready=set(self.stream_plans)
            return replace(decision,selected=tuple(rid for rid in decision.selected if rid in due),
                reason='stream_model_ar',deadline_ns=None,
                held_ready=tuple(rid for rid in decision.selected if rid in ready and rid not in due),
                pending=tuple(rid for rid in decision.selected if rid not in ready and rid not in due))
        result=super().select(requests,decision)
        wakes=[decision.now_ns+int(suggestion[3]*1e6) for rid,suggestion in self.suggested_actions.items()
               if rid in selected and suggestion[1]==WAIT and suggestion[3]>0]
        if wakes:
            next_wake=min(wakes)
            result=replace(result,deadline_ns=min(result.deadline_ns,next_wake) if result.deadline_ns is not None else next_wake)
        return result

    def has_draft(self,rid,prefix):
        rid=str(rid)
        if rid in self.sync_requests:
            # The inherited has_draft would re-enter this method through the
            # draft_ready alias; ask the plan-lane entry directly instead.
            return GrowthPrefetch.plan_lane_ready(self,rid,prefix)
        start=time.perf_counter_ns();cpu_start=time.thread_time_ns()
        try:
            with self.lock,self.stream.lock:
                entry=self.stream._entry(rid,prefix)
                self.stream._check_warm()
                if entry.source in ('ar','cancelled'):
                    self.suggested_actions.pop(rid,None)
                    return False
                frozen_ar=self.ar_selected.get(rid)
                if frozen_ar is not None:
                    if frozen_ar[0]!=tuple(prefix):raise ValueError('STREAM_SELECTOR_AR_PREFIX')
                    return False
                if rid in self.stream_plans:
                    plan=self.stream_plans[rid]
                    if plan.prefix_hash!=fingerprint(prefix):raise ValueError('GROWTH_STREAM_PREFIX')
                    if rid in self.selected_offers:
                        self.stream._pause(rid)  # a late reader error must not hide behind a cache hit
                        return len(self.selected_offers[rid].tokens)>=plan.k
                    return self.stream.has_draft(rid,prefix,count=plan.k)
                if self._batch_mode():
                    return bool(self._batch_preview(rid,tuple(prefix)))
                source=entry.source
                if source=='stream':
                    offered=len(self.stream._extra(rid,tuple(prefix)) or ())
                    source='edge'
                else:
                    if not entry.future.done():
                        return False
                    offered=len(entry.future.result()[entry.index])
                    if source=='similar_context':source='suffix'
                req=self.requests_by_rid[rid]
                batch=min(8,len(self.requests_by_rid))
                remaining=req.sampling_params.max_new_tokens-len(req.output_ids)
                offers=self.cache_offers.get(rid,(None,()))[1]
                candidate=self.growth.cache_candidate(rid,prefix,offers,remaining=remaining,batch=batch)
                if candidate is not None:
                    offer,reason,cache_rate=candidate
                    edge_rates=[rate for k in CHOICES if k<=offered
                                for rate in [self.growth.ready_rate(rid,k,remaining,batch)] if rate is not None]
                    edge_rate=max(edge_rates,default=0.)
                    ar_rate=(1./self.growth.median(self.growth.ar_ms)
                             if batch==1 and self.growth.joint_ar and self.growth.ar_seen>=16 and len(self.growth.ar_ms)>=8 else 0.)
                    if cache_rate is None or cache_rate>=max(edge_rate,ar_rate):
                        self.stream._pause(rid)
                        plan=self.growth.bind_ready(rid,prefix,k=(len(offer.tokens),reason),source=offer.source,batch=batch)
                        self.stream_plans[rid]=plan
                        self.selected_offers[rid]=offer
                        self.suggested_actions.pop(rid,None)
                        self.wait_rollout_cache.pop(rid,None)
                        return True
                    if ar_rate>edge_rate:
                        self.suggested_actions[rid]=(tuple(prefix),AR,'stream_cache_loses_to_ar',0.)
                        return False
                    # A local offer lost. Start/repair real edge supply before
                    # pricing any overlap, rather than assuming it was running.
                    self.stream._maintain(rid,tuple(prefix),self.stream.budgets[rid])
                    offered=len(self.stream._extra(rid,tuple(prefix)) or ())
                choice=self.growth.choose_ready(rid,prefix,offered=offered,
                    remaining=req.sampling_params.max_new_tokens-len(req.output_ids),source=source,batch=batch)
                model=self._model_choice(rid,prefix,req) if source=='edge' else None
                if source=='edge':offered=len(self.stream._extra(rid,tuple(prefix)) or ())
                if model is not None and not (choice is not None and choice[1]=='stream_ready_probe' and model[0]>0):
                    if model[0] in (AR,WAIT):
                        self.suggested_actions[rid]=(tuple(prefix),*model)
                        return False
                    choice=(model[0],model[1])
                self.suggested_actions.pop(rid,None)
                if choice is None:return False
                if choice[0]>offered:
                    usable=max((k for k in CHOICES if k<=offered),default=0)
                    self.stats['stream_depth_short']=self.stats.get('stream_depth_short',0)+1
                    self.event('stream_depth_short',rid=rid,offered=offered,
                               choice=choice[0],usable=usable)
                    if not usable:
                        self.suggested_actions.pop(rid,None)
                        return False
                    choice=(usable,choice[1] if len(choice)>1 else 'stream_depth_short')
                plan=self.growth.bind_ready(rid,prefix,k=choice,source=source,batch=batch)
                self.stream_plans[rid]=plan
                return True
        finally:
            self.selector_cpu_ns+=time.thread_time_ns()-cpu_start
            self.selector_wall_ns+=time.perf_counter_ns()-start

    draft_ready=has_draft

    def plan_for(self,rid,prefix):
        if str(rid) in self.sync_requests:return super().plan_for(rid,prefix)
        plan=self.stream_plans.get(str(rid))
        if plan is None or plan.prefix_hash!=fingerprint(prefix):raise ValueError('GROWTH_PLAN_LOOKUP')
        return plan

    def shadow_snapshot(self,rid,prefix):
        if str(rid) in self.sync_requests:return super().shadow_snapshot(rid,prefix)
        self.plan_for(rid,prefix)
        offer=self.selected_offers.get(str(rid))
        if offer is not None:
            self.stream._check_warm();self.stream._pause(str(rid))
            return dict(source=offer.source,top2=None)
        entry=self.stream._entry(str(rid),prefix)
        source='edge' if entry.source=='stream' else 'suffix' if entry.source=='similar_context' else entry.source
        if source not in ('edge','suffix','exact_prefix'):raise ValueError('GROWTH_STREAM_SOURCE')
        return dict(source=source,top2=None)

    def take(self,rid,prefix,stride=WIDTH):
        rid=str(rid)
        if rid in self.sync_requests:
            if getattr(self,'plan_timing',False):
                self.plan_take[rid]=time.perf_counter()
            # Remember this round's root candidates so the committed token can be
            # attributed: the target accepting `alternate` at the root is a rescue,
            # and `primary` a normal hit. Both labels come for free from the next
            # commit, so the posterior needs no shadow and no extra RPC.
            meta=self.root_top2.get(rid)
            if self.hedge_arm_rate>0 and meta is not None:
                digest,top2=meta
                if digest==fingerprint(prefix):
                    self.hedge_pending[rid]=dict(primary=top2['primary'],alternate=top2['alternate'],
                                                 gap=top2['gap'],k=self.sync_decisions.get(rid,{}).get('k'),
                                                 index=len(prefix))
            return super().take(rid,prefix,stride)
        plan=self.plan_for(rid,prefix)
        offer=self.selected_offers.get(rid)
        if offer is None:
            tokens=self.stream.take(rid,prefix,stride,count=plan.k)
        else:
            if stride!=WIDTH:raise ValueError('STREAM_SELECTOR_STATE')
            self.stream._entry(rid,prefix);self.stream._check_warm();self.stream._pause(rid)
            tokens=self.similar_cache.claim_offer(self.requests_by_rid[rid],offer,count=plan.k)
            self.stream.entries.pop(rid)
            self.stream.waits.pop(rid,None)
            self.selected_offers.pop(rid)
            self.stats[offer.source+'_hit']=self.stats.get(offer.source+'_hit',0)+1
            self.stats['similar_context_hit']+=1
            self.event('cache_offer_dispatch',rid=rid,source=offer.source,tokens=tokens,
                       prefix_hash=plan.prefix_hash,**offer.detail)
        if len(tokens)!=plan.k:raise ValueError('GROWTH_STREAM_LENGTH')
        if getattr(self,'cf_labels',False):
            meta=self.root_top2.get(rid)
            if meta is not None:
                digest,top2=meta
                if digest==fingerprint(prefix):
                    self.hedge_pending[rid]=dict(primary=top2['primary'],
                                                 alternate=top2['alternate'],
                                                 gap=top2.get('gap'),k=plan.k,
                                                 index=len(prefix))
        if self.sync_enabled:self.round_k[rid]=plan.k
        self.stream_plans.pop(rid)
        self.batch_order.pop(rid,None)
        self.waits.pop(rid,None)
        self.suggested_actions.pop(rid,None)
        self.wait_rollout_cache.pop(rid,None)
        self.growth.consumed(plan)
        self.stats['consumed']+=1
        return tokens

    def take_ar(self,rid,prefix):
        rid=str(rid)
        if self.hedge_accept_min>0:self.ar_round[rid]=True
        if rid in self.sync_requests:
            self.sync_requests.discard(rid)
            return super().take_ar(rid,prefix)
        self.batch_order.pop(rid,None)
        plan=self.stream_plans.pop(rid,None)
        if plan is not None:
            self.growth.issued.pop((rid,plan.prefix_hash),None)
            self.growth.attempt_batches.pop(plan,None)
        self.selected_offers.pop(rid,None)
        selected=self.ar_selected.pop(rid,None)
        self.suggested_actions.pop(rid,None)
        self.wait_rollout_cache.pop(rid,None)
        self.stream.take_ar(rid,prefix)
        self.waits.pop(rid,None)
        if selected is not None:self.event('stream_ar_dispatch',rid=rid,reason=selected[2])
        self.stats['ar_bypass']+=1

    def cancel(self,rid,prefix):
        rid=str(rid)
        self.sync_requests.discard(rid);self.sync_decisions.pop(rid,None)
        self.sync_calls.pop(rid,None);self.round_transport.pop(rid,None)
        self.round_k.pop(rid,None)
        self.req_tokens.pop(rid,None);self.round_delivered.pop(rid,None);self.learn_lock.pop(rid,None);self.learn_pin.pop(rid,None);self.learn_streak.pop(rid,None)
        self.learn_events.pop(rid,None);self.learn_reason.pop(rid,None)
        self.round_ts.pop(rid,None);self.lane_round_ms.pop(rid,None)
        getattr(self,'sync_rewound',set()).discard(rid)
        self.admit_decided.pop(rid,None)
        self.admit_sync_rows.discard(rid)
        self.ar_round.pop(rid,None)
        self.gap_samples.pop(rid,None);self.round_count.pop(rid,None)
        self.hedge_pending.pop(rid,None)
        self.batch_order.pop(rid,None)
        self.selected_offers.pop(rid,None)
        self.cache_offers.pop(rid,None)
        plan=self.stream_plans.pop(rid,None)
        if plan is not None:
            self.growth.issued.pop((rid,plan.prefix_hash),None)
            self.growth.attempt_batches.pop(plan,None)
        self.ar_selected.pop(rid,None)
        self.suggested_actions.pop(rid,None)
        self.wait_rollout_cache.pop(rid,None)
        self.stream.cancel(rid,prefix)

    def release(self,ids,retry_failed=False):
        ids=list(ids)
        for rid in ids:
            rid=str(rid)
            if self.sync_learn is not None:
                cell=self.learn_cell.pop(rid,None)
                start=self.learn_start.pop(rid,None)
                if rid in self.probe_tag and not self.probe_done:
                    st=self.learn_start.get(rid)
                    if st is not None:
                        self.probe_rows.append((self.probe_tag[rid], self.req_tokens.get(rid,0),
                                                max(1e-6, time.perf_counter()-st), rid))
                        if len(self.probe_rows) >= 2*self.probe_per_lane:
                            self._probe_finish()
                if cell is not None and start is not None:
                    tokens=self.req_tokens.get(rid,0)
                    lane=self.round_transport.get(rid,'stream')
                    seconds=max(1e-6, time.perf_counter()-start)
                    self.sync_learn.record_request(cell,lane,tokens,seconds)
                    # The hybrid's evidence: this row's own end-to-end throughput on the lane
                    # it was given. Fed before the learner's state dump so a flip lands in it.
                    self._lane_observe(lane,tokens/seconds,rid)
                    self.event('learn_sample',rid=rid,lane=lane,tokens=tokens,
                               pinned=self.learn_pin.get(rid),
                               st_tps=self.sync_learn.request_rates(cell)[0],
                               sy_tps=self.sync_learn.request_rates(cell)[1])
                    self._dump_learn_state()
                self.learn_pin.pop(rid,None)
            self.cf_done.pop(rid,None)
        for rid in ids:
            self.batch_order.pop(str(rid),None)
            self.cache_offers.pop(str(rid),None)
            self.selected_offers.pop(str(rid),None)
            self.requests_by_rid.pop(str(rid),None)
            self.suggested_actions.pop(str(rid),None)
            self.ar_selected.pop(str(rid),None)
            self.wait_rollout_cache.pop(str(rid),None)
            self.sync_requests.discard(str(rid));self.sync_decisions.pop(str(rid),None)
            self.sync_calls.pop(str(rid),None);self.round_transport.pop(str(rid),None)
            self.round_k.pop(str(rid),None)
            self.req_tokens.pop(str(rid),None);self.round_delivered.pop(str(rid),None)
            self.round_ts.pop(str(rid),None);self.lane_round_ms.pop(str(rid),None)
            getattr(self,'sync_rewound',set()).discard(str(rid))
            self.admit_decided.pop(str(rid),None)
            self.admit_sync_rows.discard(str(rid))
            self.ar_round.pop(str(rid),None)
            self.gap_samples.pop(str(rid),None);self.round_count.pop(str(rid),None)
            self.hedge_pending.pop(str(rid),None)
        return super().release(ids,retry_failed)

    def snapshot(self):
        sync=dict(enabled=self.sync_enabled,dual_root=self.second_answer,rows=len(self.sync_requests),
                  decided=len(self.sync_decisions),last=self.sync_action_log[-2:])
        return dict(super().snapshot(),sync_action=sync,selector_cpu_ns=self.selector_cpu_ns+self.cache_cpu_ns,
                    selector_wall_ns=self.selector_wall_ns+self.cache_wall_ns,
                    selector_cpu_clock='THREAD_TIME_NS_NOT_ELAPSED',
                    ready_selector_cpu_ns=self.selector_cpu_ns,cache_selector_cpu_ns=self.cache_cpu_ns,
                    cache_offer_rows=len(self.cache_offers),selected_cache_rows=len(self.selected_offers),
                    selector='STREAM_EVENT_ROLLOUT_V2_CACHE_RATE_GATE_NOT_GLOBAL_OPTIMUM')

    def _dump_learn_state(self):
        if self.sync_learn is None or not self.learn_state_path:
            return
        try:
            import json as _json
            with open(self.learn_state_path,'w') as fh:
                _json.dump(self.sync_learn.to_state(),fh)
        except Exception as exc:
            self.event('sync_learn_save_failed',path=self.learn_state_path,error=repr(exc))

    def close(self):
        self._dump_learn_state()
        if self.closed:return
        try:
            for future in self.release(list(self.live),retry_failed=True):future.result()
            self.stream.close()
        finally:
            self._save_prior()
            self.pool.shutdown(wait=True,cancel_futures=False)
            self.closed=True

    def _prior_path(self):
        return os.environ.get('TRIAGE_HEDGE_STATE','') or ''

    def _load_prior(self):
        """Persist what a traffic class learned about the two lanes.

        Store lane rates and timing differences by traffic class so later
        requests can reuse observations. Persistence avoids discarding the
        evidence collected by bounded exploration at every service restart.
        """
        path=self._prior_path()
        if not path:
            return
        try:
            with open(path) as handle:
                saved=json.load(handle)
        except FileNotFoundError:
            return
        except Exception as exc:
            self.event('hedge_prior_load_failed',path=path,error=type(exc).__name__)
            return
        cls=str(saved.get('class'))
        if cls!=str(self.hedge_class):
            self.event('hedge_prior_class_mismatch',file=cls,current=str(self.hedge_class))
            return
        for key,cell in (saved.get('sync_rate') or {}).items():
            parts=[int(x) for x in json.loads(key)]
            lane=(self._batch_key(),parts[0],parts[1] if len(parts)>1 else self.hedge_lane_floor)
            self.sync_rate[lane]=dict(rates=[float(x) for x in cell.get('rates',[])][-64:])
        self.stream_rates[str(self.hedge_class)]=[float(x) for x in (saved.get('stream_rates') or [])][-64:]
        for key,samples in (saved.get('delta') or {}).items():
            k=int(json.loads(key)[0])
            self.delta_samples[(self._batch_key(),k)]=[float(x) for x in samples][-64:]
        self.event('hedge_prior_loaded',path=path,**{'class':cls},
                   cells=len(saved.get('sync_rate') or {}),
                   rounds=sum(len(c.get('rates',[])) for c in (saved.get('sync_rate') or {}).values()))

    def _save_prior(self):
        path=self._prior_path()
        if not path or not (self.sync_rate or self.delta_samples):
            return
        try:
            payload=dict({'class':str(self.hedge_class)},
                stream_rates=[round(x,6) for x in self.stream_rates.get(str(self.hedge_class),[])][-32:],
                sync_rate={
                json.dumps([int(x) for x in key[1:]]):dict(
                    rates=[round(x,6) for x in cell.get('rates',[])][-32:])
                for key,cell in self.sync_rate.items()
                if isinstance(key,tuple) and len(key)>=2},
                delta={json.dumps([int(key[1])]):[round(x,3) for x in samples][-32:]
                    for key,samples in self.delta_samples.items()
                    if isinstance(key,tuple) and len(key)==2})
            tmp=path+'.tmp'
            with open(tmp,'w') as handle:
                json.dump(payload,handle,sort_keys=True)
            os.replace(tmp,path)
            self.event('hedge_prior_saved',path=path,cells=len(payload['sync_rate']))
        except Exception as exc:
            self.event('hedge_prior_save_failed',path=path,error=type(exc).__name__)

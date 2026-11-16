import math
import time
from .growth_cloud_shape import ShapeBudgetPolicy
from .growth_cloud_budget import BASELINE
from .growth_cloud_cost import fit_cloud

SELECTOR='cloud_source_conditioned_cost_v13_calibrated_frontier'
SOURCES=('suffix','exact_prefix','edge')

def group(source):
    if type(source)is not str or source not in SOURCES:raise ValueError('SOURCE_COST_DRAFT_SOURCE')
    return 'edge'if source=='edge'else'cache'

def source_hint(prefetch,prefix,generated):
    """Peek existing committed-cache state without observe/propose/LRU writes.

    Ordinary per-request cache observation still happens later, unchanged. This
    is an as-of-planning forecast, not a guaranteed actual proposal source.
    """
    from .exact_cache import ExactPrefixCache
    from .growth_prefetch import GrowthSuffixCache
    from .similar_cache import UnifiedContextCache
    exact=prefetch.exact_cache;cache=prefetch.cache
    if type(cache) is UnifiedContextCache:
        detail={}
        hint=cache.peek_source(prefix, generated, want=7, detail=detail)
        if hint=='edge':
            return 'edge'
        prompt=tuple(prefix)[:len(prefix)-len(tuple(generated))] if len(generated) else tuple(prefix)
        # Price the complete plan: a scope whose
        # measured rate already loses is planned as an edge round.
        return hint if prefetch.forecast_allows(prompt, hint, detail) else 'edge'
    if type(cache)is not GrowthSuffixCache or exact is not None and type(exact)is not ExactPrefixCache:
        raise ValueError('SOURCE_COST_CACHE_CAPABILITY')
    if exact is not None:
        value=exact.entries.get(tuple(prefix))
        if value is not None and len(value)>=7:return 'exact_prefix'
    for width in range(min(cache.max_suffix,len(generated)),cache.min_suffix-1,-1):
        value=cache.entries.get(tuple(generated[-width:]))
        if value is not None and len(value)>=7:return 'suffix'
    return 'edge'

class SourceCostPolicy(ShapeBudgetPolicy):
    source_mode=True
    selector_name=SELECTOR

    def __init__(self,event,clock=time.monotonic_ns):
        def named(name,**fields):
            if 'selector'in fields:fields['selector']=SELECTOR
            event(name,**fields)
        super().__init__(named,clock)
        self.event('growth_source_cost_bound',selector=SELECTOR,source_conditioned=True,cache_preview_no_LRU_write=True,no_future_target_or_historical_seed=True,original_exploration_budget_unchanged=True,mature_winner_exploit_first=True,required_source_components_age_bound=True,coherent_factual_cost_delivery=True,tail_pressure_decision_score=True)

    def init_budget_state(self):
        super().init_budget_state();self.source_scope=None;self._source_feedback=None
        from .growth_attempts import AttemptLedger
        self.attempt_mode=True
        self.attempt_ledger=AttemptLedger(self.COST_WINDOW,16)
        self.attempt_batches={}

    def choose(self,rid,prefix,batch):
        plan=super().choose(rid,prefix,batch)
        self.attempt_batches[plan]=batch
        return plan

    def consumed(self,plan):
        super().consumed(plan)
        self.attempt_batches.pop(plan,None)

    def _forget_plan_batch(self,rid):
        for plan in list(self.attempt_batches):
            if plan.rid==str(rid):self.attempt_batches.pop(plan)

    def forget(self,rid):
        self.attempt_ledger.discard(rid)
        self._forget_plan_batch(rid)
        return super().forget(rid)

    def retire_fallback_plan(self,rid):
        # Stream fallback retires native-plan ownership, not its billing intent.
        self._forget_plan_batch(rid)
        return super().forget(rid)

    def note_fallback_attempt(self,plan,prefix,*,source,continuation=False):
        if not continuation:
            if plan.reason=='preflight_baseline' or self.attempt_batches.get(plan)!=1:
                self.attempt_batches.pop(plan,None)
                return False
            from .growth import Plan
            Plan.bind(plan.wire(),plan.rid,prefix)
            if self.issued.get((plan.rid,plan.prefix_hash))!=plan:
                raise ValueError('GROWTH_ATTEMPT_NOT_ISSUED')
        recorded=self.attempt_ledger.note(plan,prefix,group(source),initial=not continuation)
        if recorded:self.attempt_batches.pop(plan,None)
        return recorded

    def finish_fallback_attempt(self,rid,plan):
        self.attempt_ledger.finish(rid,plan)

    def set_source_scope(self,ids,sources):
        if (type(ids)not in(list,tuple)or not 1<=len(ids)<=8
                or any(type(r)is not str or not r for r in ids)or len(set(ids))!=len(ids)
                or type(sources)is not dict or any(type(r)is not str for r in sources)
                or set(sources)!=set(ids)):
            raise ValueError('SOURCE_COST_COHORT_BINDING')
        kinds=tuple(group(sources[rid])for rid in ids)
        self.source_scope=dict(ids=tuple(ids),groups=kinds,counts={k:kinds.count(k)for k in('cache','edge')},
                               sources=tuple(sources[rid] for rid in ids),
                               replay_only=all(sources[rid]=='exact_prefix' for rid in ids))
        self.prediction_cache.clear()

    def prepare_cohort(self,fresh,all_prefixes,*,mode='spec',max_total_tokens=None,remaining_by_rid=None,source_by_rid=None):
        if mode=='spec'and fresh:
            self.set_source_scope([rid for rid,prefix in all_prefixes],source_by_rid)
            self.event('growth_source_scope',selector=SELECTOR,ids=list(self.source_scope['ids']),sources=[source_by_rid[r]for r in self.source_scope['ids']],source_counts=dict(self.source_scope['counts']),asof='COMMITTED_CACHE_BEFORE_ORDINARY_FRESH_OBSERVATION_NOT_ORACLE')
        super().prepare_cohort(fresh,all_prefixes,mode=mode,max_total_tokens=max_total_tokens,remaining_by_rid=remaining_by_rid)

    @staticmethod
    def validate_source_row(row):
        counts,delivered=row['source_rows'],row['source_delivered'];B=row['actual_batch'];k=row['k']
        if (type(counts)is not dict or type(delivered)is not dict
                or set(counts)!={'cache','edge'}or set(delivered)!={'cache','edge'}
                or any(type(x)is not int or x<0 for x in[*counts.values(),*delivered.values()])
                or sum(counts.values())!=B or sum(delivered.values())!=row['delivered']
                or any(delivered[g]>counts[g]*(k+1)or counts[g]==0 and delivered[g]!=0 for g in counts)):
            raise ValueError('SOURCE_COST_FACTUAL_BINDING')

    def score_prediction(self,row):
        return row['decision_ms_per_tail_equivalent_progress']

    def coherent_pair(self,key,cost_rows,fit,query):
        """Worst source-projected unit cost from intact prior feedback rows.

        Empirical heuristic only. It is not a confidence bound or an outcome
        of running the alternative under the current request/KV history.
        """
        scope=self.source_scope;required=[g for g,n in scope['counts'].items()if n]
        pairs=[r for r in cost_rows if all(r['source_rows'][g]for g in required)]
        if len(pairs)<self.MIN_FACTUAL:return None
        floor=fit['fixed_effective_ms']+fit['marginal_ms_per_query']*query;scored=[]
        for row in pairs:
            rates={g:row['source_delivered'][g]/row['source_rows'][g]for g in required}
            delivery=sum(min(rates[g],left)for g,left in zip(scope['groups'],self.budget_scope['remaining']))
            cloud=max(row['cloud_rpc_ms'],floor);other=row['unhidden_noncloud_ms']
            if delivery<=0 or not math.isfinite(cloud+other+delivery)or min(cloud,other)<0:raise ValueError('SOURCE_COST_PREDICTION')
            scored.append(dict(unit=(cloud+other)/delivery,row=row,rates=rates,delivery=delivery,cloud=cloud,other=other))
        result=max(scored,key=lambda r:r['unit'])
        result.update(rows=pairs,age=self.budget_tick[key]-max(self.sample_tick[r['generation'],r['cycle_id']]for r in pairs))
        return result

    def c1_native_rows(self,key,action,source):
        now=self.budget_tick[key]
        # Source is part of the sample condition, not a post-truncation filter.
        # Never fill a missing source class with another class's rewards.
        return [r for r in self.cost_rows.get(key,())
                if (r['k'],r['branches'])==action
                and now-self.sample_tick.get((r['generation'],r['cycle_id']),-1000)<self.RECENT_CYCLES
                and r.get('source_rows',{}).get(source)==1
                and (action[1]==1 or self.actual_side(r)>0)][-self.ACTION_RECENT:]

    def validate_c1_native(self,row):
        self.validate_source_row(row)
        hits=row.get('side_accepts')
        # A double-root tree can commit the alternate root followed by its chain.
        # validate_source_row already bounds delivery by source_rows * (k+1).
        if (type(hits)is not int or not 0<=hits<=self.actual_side(row)
                or hits and row['source_rows']['edge']!=1):
            raise ValueError('SOURCE_C1_SIDE_FEEDBACK')

    def primary_profile(self,key,source,layer=None):
        from .growth_survival import PrefixSurvival
        rid=self.source_scope['ids'][0];now=self.budget_tick[key]
        # Reward follows the current request; hardware cost can reuse genuine
        # same-bucket/source measurements. Never carry another prompt's reward.
        # layer narrows cache rows to exact_prefix or suffix; the historical
        # group filter stays cache/edge so existing callers are unchanged.
        rows=[r for r in self.cost_rows.get(key,())
              if tuple(r.get('ids',()))==(rid,)
              and now-self.sample_tick.get((r['generation'],r['cycle_id']),-1000)<self.RECENT_CYCLES
              and r.get('source_rows',{}).get(source)==1
              and (layer is None or r.get('sources')==[layer])]
        for row in rows:self.validate_c1_native(row)
        return PrefixSurvival(rows,self.MIN_FACTUAL)

    def predictions(self,key):
        # This experiment changes only C1 pricing. No C1 reward is transferred
        # into batch forecasts; the prior source/shape policy is intact there.
        if key[0]!=1:return self._legacy_predictions(key)
        if key in self.prediction_cache:return self.prediction_cache[key]
        scope=self.source_scope
        if scope is None or len(scope['ids'])!=1:return [],None
        history=list(self.cost_rows.get(key,()))
        fit=fit_cloud(history[-self.CLOUD_FIT_WINDOW:]);table=[]
        source=scope['groups'][0];remaining=self.budget_scope['remaining'][0]
        profile=self.primary_profile(key,source)
        if fit:
            for k,roots in self.ACTIONS:
                if roots==2 and source!='edge':continue
                rows=self.c1_native_rows(key,(k,roots),source)
                if not rows and not self.attempt_ledger.fresh(key,(k,roots),source,self.budget_tick[key],self.RECENT_CYCLES):continue
                for row in rows:self.validate_c1_native(row)
                now=self.budget_tick[key]
                failures=self.attempt_ledger.fresh(key,(k,roots),source,now,self.RECENT_CYCLES)
                samples=[dict(r,attempt_tick=self.sample_tick[r['generation'],r['cycle_id']],
                              sample_kind='NATIVE_VERIFY')for r in rows]
                samples.extend(dict(r,attempt_tick=r['tick'])for r in failures)
                samples=sorted(samples,key=lambda r:(r['attempt_tick'],r['cycle_id']))[-self.ACTION_RECENT:]
                n=len(samples)
                # Success and AR-fallback episodes share the same serving-time
                # objective, but fallback never becomes a native-verify reward.
                factual_delivery=sum(min(r['delivered'],remaining)for r in samples)/n
                verified=sum(r['sample_kind']=='NATIVE_VERIFY'for r in samples)
                primary=profile.mean(k,remaining)
                side=(profile.side_rate if roots==2 and remaining>=2 else 0.)
                reward_ready=(not verified or profile.supported(k,remaining)
                              and(roots==1 or remaining<2 or profile.side_supported))
                if primary is not None and side is not None:
                    failed_delivery=sum(min(r['delivered'],remaining)for r in samples if r['sample_kind']!='NATIVE_VERIFY')
                    delivery=(verified*(primary+side)+failed_delivery)/n
                else:delivery=factual_delivery
                cycle=sum(r['cycle_ms']for r in samples)/n
                unknown_rpc=any(r['cloud_rpc_ms'] is None for r in samples)
                cloud=sum(r['cloud_rpc_ms'] if r['cloud_rpc_ms'] is not None else
                          min(r['cycle_ms'],fit['fixed_effective_ms']+fit['marginal_ms_per_query']*r['query_nodes'])
                          for r in samples)/n
                other=max(0.,cycle-cloud)
                if not math.isfinite(cycle)or cycle<=0 or delivery<=0:
                    raise ValueError('SOURCE_C1_CYCLE')
                age=now-max(r['attempt_tick']for r in samples)
                native_costs=[r['cycle_ms']for r in samples if r['sample_kind']=='NATIVE_VERIFY']
                upper=profile.optimistic_depth(k,remaining)
                if upper is not None and side is not None:upper+=side
                else:upper=None
                table.append(dict(k=k,branches=roots,factual_samples=n,mature=n>=self.MIN_FACTUAL and reward_ready,
                    reward_supported=reward_ready,reward_model='CURRENT_REQUEST_CENSORED_PRIMARY_SURVIVAL',
                    reward_model_samples=profile.n,primary_survival=list(profile.survival),
                    survival_offered=list(profile.offered),survival_risk=list(profile.risk),
                    side_model_samples=profile.side_offers,side_model_rate=profile.side_rate,
                    optimistic_native_delivery=upper,min_native_cycle_ms=min(native_costs)if native_costs else None,
                    native_verified_samples=verified,fallback_attempt_samples=n-verified,
                    sample_age=age,expected_query_nodes=sum(r['query_nodes']for r in samples)/n,
                    expected_delivered=delivery,mean_factual_delivered=sum(r['delivered']for r in samples)/n,
                    predicted_cloud_rpc_ms=cloud,predicted_unhidden_noncloud_ms=other,
                    predicted_cycle_ms=cycle,predicted_ms_per_delivered_token=cycle/delivery,
                    cost_source_sample_age=age,delivered_source_sample_ages={source:age},
                    forecast_source_counts=dict(scope['counts']),
                    delivered_per_source_row={source:delivery},
                    actual_side_samples=sum(self.actual_side(r)>0 for r in samples if r['sample_kind']=='NATIVE_VERIFY'),
                    cost_delivery_pairing='FACTUAL_ATTEMPT_COST_WITH_LOCAL_SURVIVAL_REWARD_MODEL',
                    rpc_component_kind='AR_RPC_MODEL_PROXY_WHEN_TIMER_UNAVAILABLE'if unknown_rpc else'MEASURED_RPC_COMPONENTS',
                    estimate_kind='LOCAL_SURVIVAL_REWARD_PER_FACTUAL_ATTEMPT_MS_NOT_CONFIDENCE',
                    decision_ms_per_tail_equivalent_progress=cycle/delivery,
                    decision_score_kind='C1_EMPIRICAL_CYCLE_MS_PER_TAIL_CLIPPED_DELIVERY',
                    tail_equivalent_progress=delivery,cohort_completion_pressure_cycles=remaining/delivery,
                    admissible=True,side_hit_rate=sum(r['side_accepts']for r in samples)/n,
                    primary_delivery_without_leaf=sum(min(r['delivered']-r['side_accepts'],remaining)for r in samples)/n,
                    marginal_leaf_delivery=sum(min(r['delivered'],remaining)-min(r['delivered']-r['side_accepts'],remaining)for r in samples)/n,
                    factual_cycles=[dict(generation=r['generation'],cycle_id=r['cycle_id'],sample_kind=r['sample_kind'])for r in samples]))
        parents={(r['k'],r['branches']):r for r in table}
        for row in table:
            if row['branches']!=2:continue
            parent=parents.get((row['k'],1))
            ready=(parent is not None and parent['mature'] and row['mature']
                   and row['actual_side_samples']>=self.MIN_FACTUAL)
            delta=max(0.,row['predicted_cycle_ms']-parent['predicted_cycle_ms'])if ready else None
            gain=(row['marginal_leaf_delivery']*parent['predicted_cycle_ms']
                  -row['primary_delivery_without_leaf']*delta)if ready else None
            row.update(admissible=bool(ready and row['marginal_leaf_delivery']>0 and gain>0),
                marginal_parent_mature=ready,marginal_extra_ms=delta,marginal_gain_token_ms=gain,
                marginal_gate='DELTA_DELIVERY_TIMES_PARENT_MS_GT_PRIMARY_DELIVERY_TIMES_DELTA_MS')
        for row in table:
            if row['k']==3:continue
            parent=parents.get((3,row['branches']))
            upper=row['optimistic_native_delivery'];cost=row['min_native_cycle_ms']
            # Even an optimistic empirical tail cannot pay a measured depth
            # increment. Not a confidence bound; unknown costs are never pruned.
            dominated=(parent is not None and parent['mature'] and upper is not None and upper>0
                       and cost is not None and row['native_verified_samples']>=self.MIN_FACTUAL
                       and row['fallback_attempt_samples']==0
                       and cost/upper*self.SWITCH_MARGIN>=self.score_prediction(parent))
            row['depth_probe_dominated']=bool(dominated)
        result=table,fit;self.prediction_cache[key]=result;return result

    def _legacy_predictions(self,key):
        if key in self.prediction_cache:return self.prediction_cache[key]
        scope=self.source_scope
        if scope is None or len(scope['ids'])!=key[0]:return [],None
        history=list(self.cost_rows.get(key,()));fit=fit_cloud(history[-self.CLOUD_FIT_WINDOW:]);table=[]
        if fit:
            for k,roots in self.ACTIONS:
                # Cached proposals carry no top2 alternate. Do not price a
                # fully cached forecast as a potential real-side root2 action.
                if roots==2 and not scope['counts']['edge']:continue
                rows=[r for r in self.fresh_rows(key,(k,roots))if 'source_rows'in r and 'source_delivered'in r]
                if not rows:continue
                for row in rows:self.validate_source_row(row)
                cost_rows=[r for r in rows if bool(r['source_rows']['edge'])==bool(scope['counts']['edge'])]
                if not cost_rows:continue
                cost_age=self.budget_tick[key]-max(self.sample_tick[r['generation'],r['cycle_id']]for r in cost_rows)
                rates={};source_ages={};samples=[];baseline=(k,roots)==BASELINE
                for g,count in scope['counts'].items():
                    if not count:continue
                    factual=[r for r in rows if r['source_rows'][g]]
                    if not factual:break
                    rates[g]=(sum(r['source_delivered'][g]for r in factual)/sum(r['source_rows'][g]for r in factual)if baseline else min(r['source_delivered'][g]/r['source_rows'][g]for r in factual))
                    samples.append(len(factual))
                    source_ages[g]=self.budget_tick[key]-max(self.sample_tick[r['generation'],r['cycle_id']]for r in factual)
                else:
                    remaining=self.budget_scope['remaining']
                    if len(remaining)!=len(scope['groups']):raise ValueError('SOURCE_COST_REMAINING_BINDING')
                    delivery=sum(min(rates[g],left)for g,left in zip(scope['groups'],remaining));n=min([len(cost_rows),*samples])
                    query=(sum(r['query_nodes']for r in cost_rows)/len(cost_rows)if baseline else key[0]*(k+1)+(scope['counts']['edge']if roots==2 else 0))
                    cloud=(sum(r['cloud_rpc_ms']for r in cost_rows)/len(cost_rows)if baseline else max(max(r['cloud_rpc_ms']for r in cost_rows),fit['fixed_effective_ms']+fit['marginal_ms_per_query']*query))
                    other=(sum(r['unhidden_noncloud_ms']for r in cost_rows)/len(cost_rows)if baseline else max(r['unhidden_noncloud_ms']for r in cost_rows))
                    joint=None if baseline else self.coherent_pair(key,cost_rows,fit,query)
                    used=rows;pairing='baseline_source_mean'if baseline else'independent_extrema_fallback'
                    if joint is not None:
                        cloud,other,delivery,rates=joint['cloud'],joint['other'],joint['delivery'],joint['rates']
                        used=joint['rows'];n=len(used);cost_age=joint['age'];source_ages={g:cost_age for g in rates};pairing='max_coherent_pair_unit_cost'
                    if delivery<=0 or not math.isfinite(cloud+other+delivery)or min(cloud,other)<0:raise ValueError('SOURCE_COST_PREDICTION')
                    pair_ids=[dict(generation=r['generation'],cycle_id=r['cycle_id'])for r in used]if joint else[]
                    selected=dict(generation=joint['row']['generation'],cycle_id=joint['row']['cycle_id'])if joint else None
                    table.append(dict(k=k,branches=roots,factual_samples=n,mature=n>=self.MIN_FACTUAL,sample_age=max([cost_age,*source_ages.values()]),expected_query_nodes=query,expected_delivered=delivery,mean_factual_delivered=sum(r['delivered']for r in used)/len(used),predicted_cloud_rpc_ms=cloud,predicted_unhidden_noncloud_ms=other,predicted_cycle_ms=cloud+other,predicted_ms_per_delivered_token=(cloud+other)/delivery,cost_source_sample_age=cost_age,delivered_source_sample_ages=dict(source_ages),sample_age_kind='MAX_REQUIRED_SOURCE_COMPONENT_AGE_HEURISTIC_NOT_CONFIDENCE',forecast_source_counts=dict(scope['counts']),delivered_per_source_row=rates,actual_side_samples=sum(self.actual_side(r)>0 for r in used),cost_delivery_pairing=pairing,coherent_pair_factuals=pair_ids,selected_coherent_pair=selected,estimate_kind='MAX_COHERENT_SOURCE_PROJECTED_UNIT_HEURISTIC_NOT_CONFIDENCE_OR_COUNTERFACTUAL'if joint else'SOURCE_CONDITIONED_FACTUAL_HEURISTIC_NOT_CONFIDENCE_OR_COUNTERFACTUAL'))
        remaining=self.budget_scope['remaining']
        for row in table:
            progress=[min(row['delivered_per_source_row'][g],left)for g,left in zip(scope['groups'],remaining)]
            if not progress or min(progress)<=0:raise ValueError('SOURCE_TAIL_PROGRESS')
            pressure=max(left/d for left,d in zip(remaining,progress))
            equivalent=min(row['expected_delivered'],key[0]*max(remaining)/pressure)
            score=row['predicted_cycle_ms']/equivalent
            if not math.isfinite(score)or score<=0:raise ValueError('SOURCE_TAIL_SCORE')
            row.update(cohort_completion_pressure_cycles=pressure,tail_equivalent_progress=equivalent,decision_ms_per_tail_equivalent_progress=score,decision_score_kind='COHORT_WORK_PRESSURE_HEURISTIC_NOT_ACTUAL_TOKENS_OR_MAKESPAN_BOUND')
        self.prediction_cache[key]=(table,fit);return table,fit

    def probe_candidates(self,key,known):
        choices=super().probe_candidates(key,known)
        if self.source_scope is None or len(self.source_scope['ids'])!=key[0]:raise ValueError('SOURCE_COST_SCOPE_REQUIRED')
        available=[a for a in choices if (a[1]==1 or self.source_scope['counts']['edge'])
                   and not known.get(a,{}).get('depth_probe_dominated',False)]
        replay_only=self.source_scope.get('replay_only',False)
        if replay_only:
            # Complete committed-prefix replay can supply any K from cache. Among
            # not-yet-mature actions prefer the longer length, so a first real K7
            # sample is not delayed behind a K5 confirmation detour. This is only
            # candidate ordering: the v13 optimism frontier and every outer budget,
            # stability, remaining-token, factual-minimum and Plan guard still run.
            ordered=sorted(available,key=lambda action:(0 if action not in known or not known[action]['mature'] else 1,
                                                        -action[0],self.PROBE_ORDER.index(action)))
            if ordered!=available:
                self.event('growth_replay_longest_probe_offer',selector=SELECTOR,bucket=list(key),
                    candidates=[list(a)for a in ordered],offered_not_forced=True,
                    unchanged_factual_minimum_and_outer_guards=True,
                    kind='V13_FRONTIER_ORDER_ONLY_NOT_FORCED_ACTION')
            available=ordered
        baseline=known.get(BASELINE)
        if available and baseline and baseline['mature'] and baseline['sample_age']<self.BASELINE_MAX_AGE:
            winners={a:r for a,r in known.items()if a!=BASELINE and r['mature'] and r.get('admissible',True)
                     and self.score_prediction(r)*self.SWITCH_MARGIN<self.score_prediction(baseline)}
            if winners:
                # A mature winner is not proof that an unobserved tail cannot
                # improve it. Reopen only a potentially better C1 probe; the
                # original budget/gap/stability/tail gates still decide dispatch.
                best=min([baseline,*winners.values()],key=self.score_prediction)
                if key[0]==1:
                    _,fit=self.predictions(key)
                    if fit:
                        profile=self.primary_profile(key,self.source_scope['groups'][0])
                        remaining=self.budget_scope['remaining'][0]
                        frontier=[]
                        for action in available:
                            k,roots=action
                            upper=profile.optimistic_depth(k,remaining)
                            if upper is None:upper=float(min(k+1,remaining))
                            if roots==2 and remaining>=2:
                                # Depth changes do not create a second unknown
                                # alternate-root parameter. Reuse calibrated
                                # common side reward; optimize only unknown tails.
                                if profile.side_supported and profile.side_rate is not None:
                                    upper+=profile.side_rate
                                else:
                                    head=profile.survival[0]
                                    upper+=1.-head if head is not None else 1.
                            upper=min(float(remaining),float(k+1),upper)
                            estimated=self.estimate_probe_ms(key,action,known,baseline,fit)
                            optimistic=estimated;measured=known.get(action)
                            if measured and not measured.get('fallback_attempt_samples',0):
                                if measured.get('native_verified_samples',0)<self.MIN_FACTUAL:
                                    optimistic=min(estimated,self.cold_probe_ms(key,action,baseline,fit))
                                elif measured.get('min_native_cycle_ms')is not None:
                                    optimistic=min(estimated,measured['min_native_cycle_ms'])
                            score=optimistic/upper
                            if score*self.SWITCH_MARGIN<self.score_prediction(best):
                                frontier.append(dict(action=action,optimistic_unit_ms=score,
                                    reward_ceiling=upper,estimated_cycle_ms=estimated,
                                    optimistic_probe_ms=optimistic,has_factual_cost=measured is not None))
                        if frontier:
                            if replay_only:
                                # Only frontier members that already passed v13's
                                # optimistic-value screen are reordered here.
                                frontier.sort(key=lambda r:(0 if r['action'] not in known or not known[r['action']]['mature'] else 1,
                                                            -r['action'][0],r['optimistic_unit_ms'],
                                                            self.PROBE_ORDER.index(r['action'])))
                                self.event('growth_replay_longest_probe_offer',selector=SELECTOR,bucket=list(key),
                                    candidates=[list(r['action'])for r in frontier],offered_not_forced=True,
                                    unchanged_factual_minimum_and_outer_guards=True,
                                    kind='V13_FRONTIER_SURVIVORS_ONLY')
                            else:
                                frontier.sort(key=lambda r:(r['optimistic_unit_ms'],
                                    0 if r['action']in known else 1,self.PROBE_ORDER.index(r['action'])))
                            self.event('growth_probe_frontier',selector=SELECTOR,bucket=list(key),
                                current_best=[best['k'],best['branches']],best_unit_ms=self.score_prediction(best),
                                candidates=frontier,kind='EMPIRICAL_OPTIMISM_NOT_CONFIDENCE_OR_NATIVE_REWARD')
                            return [r['action']for r in frontier]
                if replay_only:
                    # Complete cache replay has no draft-model length cost. When
                    # v13's optimism screen defers everything, still offer only
                    # longer immature lengths, either to confirm recent evidence
                    # before it expires or to revisit a stale/never-seen tail.
                    best_action=(best['k'],best['branches'])
                    longer=[a for a in available if a[0]>best_action[0]]
                    confirm_pending=any(a in known and not known[a]['mature']
                                        and known[a]['sample_age']<self.RECENT_CYCLES for a in longer)
                    stale=self.budget_tick[key]-self.last_probe[key]>=self.RECENT_CYCLES
                    if longer and (confirm_pending or stale):
                        self.event('growth_replay_probe_available',selector=SELECTOR,bucket=list(key),
                            candidates=[list(a)for a in longer],offered_not_forced=True,
                            confirming_recent_sample=confirm_pending,stale_evidence=stale,
                            unchanged_time_budget=True,kind='LONGER_IMMATURE_REPLAY_ONLY_NOT_FORCED')
                        return longer
                self.event('growth_source_probe_deferred',selector=SELECTOR,bucket=list(key),
                    deferred_probes=[list(a)for a in available],
                    predicted_winners=[dict(action=list(a),unit_ms=r['predicted_ms_per_delivered_token'],decision_score=self.score_prediction(r),decision_score_kind=r['decision_score_kind'])for a,r in winners.items()],
                    reason='FRESH_MATURE_MODEL_WINNER_EXPLOIT_FIRST_NOT_GUARANTEED_REWARD')
                return []
        return available

    def commit_batch(self,rows,*,rpc_ms=0.):
        if self._source_feedback is not None:raise RuntimeError('SOURCE_COST_FEEDBACK_REENTRY')
        self.attempt_ledger.validate(rows)
        speculative=bool(rows)and all(r['plan']is not None for r in rows)
        if speculative:self._source_feedback=tuple((r['rid'],group(r.get('source')),r['delivered'])for r in rows)
        try:
            result=super().commit_batch(rows,rpc_ms=rpc_ms)
            cycle=self.observations[-1]
            if cycle['k'] is None:
                learn=(not cycle['terminal_censored'] and self.control.get('learn',True)
                       and getattr(self,'preflight_rid',None)not in cycle['ids'])
                failed=self.attempt_ledger.commit(rows,cycle,rpc_ms,bucket=self.bucket,
                    account=self.account_cycle_time,learn=learn)
                if failed:self.event('growth_fallback_cycle',selector=SELECTOR,**failed)
            return result
        finally:self._source_feedback=None

    def observe_budget_cost(self,row):
        feedback=self._source_feedback
        if feedback is None or tuple(row['ids'])!=tuple(x[0]for x in feedback):raise ValueError('SOURCE_COST_FEEDBACK_PLAN_BINDING')
        row['source_rows']={g:sum(kind==g for rid,kind,n in feedback)for g in('cache','edge')}
        row['source_delivered']={g:sum(n for rid,kind,n in feedback if kind==g)for g in('cache','edge')}
        cycle=self.observations[-1]
        if (cycle['generation'],cycle['cycle_id'])!=(row['generation'],row['cycle_id']):
            raise ValueError('SOURCE_COST_SIDE_CYCLE_BINDING')
        # Offered alternates and accepted alternates are different quantities.
        # Cost rows retain the actual accepted-leaf count from the same cycle.
        row['side_accepts']=cycle['side_accepts']
        self.validate_source_row(row)
        super().observe_budget_cost(row)
        self.event('growth_source_cost_cycle',selector=SELECTOR,generation=row['generation'],cycle_id=row['cycle_id'],ids=row['ids'],k=row['k'],branches=row['branches'],source_rows=dict(row['source_rows']),source_delivered=dict(row['source_delivered']),learning=row['learning'],terminal_censored=row['terminal_censored'])

    def snapshot(self):
        return dict(super().snapshot(),selector=SELECTOR,source_cost_v13_c1=True,
            fallback_pending=len(self.attempt_ledger.active),fallback_cycles=self.attempt_ledger.completed_cycles,
            c1_objective='LOCAL_PREFIX_SURVIVAL_REWARD_PER_ALL_ATTEMPT_COST',
            batch_objective='UNCHANGED_V7_SOURCE_TAIL_HEURISTIC',
            replay_probe_confirmation=True,replay_probe_longer_only=True,
            replay_probe_longest_first=True,replay_probe_replay_only=True,
            replay_probe_order='V13_FRONTIER_THEN_LONGER_IMMATURE_NOT_FORCED',
            source_forecast='AS_OF_COMMITTED_CACHE_NOT_FUTURE_TARGET_OR_ORACLE')

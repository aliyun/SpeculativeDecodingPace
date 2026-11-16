from collections import defaultdict
import math
import os
import time

from .growth import KS
from .growth_cloud_budget import BudgetCostPolicy, BASELINE, enabled as budget_enabled
from .growth_cloud_cost import fit_cloud

SELECTOR = 'cloud_actual_shape_budget_v6'


def enabled():
    value=os.environ.get('TRIAGE_SHAPE_BUDGET_GROWTH','0')
    if value not in ('0','1'):
        raise ValueError('SHAPE_BUDGET_FLAG')
    return value=='1'


class ShapeBudgetPolicy(BudgetCostPolicy):
    selector_name = SELECTOR

    def __init__(self,event,clock=time.monotonic_ns):
        if not getattr(self,'live_runtime',False) and (not enabled() or not budget_enabled()):
            raise ValueError('SHAPE_BUDGET_REQUIRES_OPT_IN_AND_BUDGET')
        def named(name,**fields):
            if 'selector' in fields:fields['selector']=SELECTOR
            event(name,**fields)
        super().__init__(named,clock)
        self.event('growth_shape_bound',selector=SELECTOR,actual_side_required_for_root2_reward=True,
                   alternative_cost_floor='RECENT_MAX_FACTUAL_CLOUD_AND_NONCLOUD_NOT_CONFIDENCE_BOUND',
                   actual_zero_side_reward_not_copied_to_baseline=True,
                   compact_immutable_plan_records=True,no_native_GPU_change=True)

    def init_budget_state(self):
        super().init_budget_state()
        self.shape_last_attempt=defaultdict(lambda:-1)
        self.zero_side_cycles=0

    @staticmethod
    def actual_side(row):
        B,k,roots=row['actual_batch'],row['k'],row['branches']
        queries=row['query_nodes']
        if (type(B) is not int or not 1<=B<=8 or type(queries) is not int or
                type(k) is not int or k not in KS or type(roots) is not int or roots not in (1,2)):
            raise ValueError('SHAPE_NATIVE_QUERY_BINDING')
        side=queries-B*(k+1)
        if not 0<=side<=B or roots==1 and side:
            raise ValueError('SHAPE_NATIVE_QUERY_BINDING')
        return side

    def fresh_rows(self,key,action):
        now=self.budget_tick[key]
        return [r for r in self.cost_rows.get(key,())
                if (r['k'],r['branches'])==action
                and now-self.sample_tick.get((r['generation'],r['cycle_id']),-1000)<self.RECENT_CYCLES
                and (action[1]==1 or self.actual_side(r)>0)][-self.ACTION_RECENT:]

    def predictions(self,key):
        if key in self.prediction_cache:return self.prediction_cache[key]
        history=list(self.cost_rows.get(key,()))
        fit=fit_cloud(history[-self.CLOUD_FIT_WINDOW:])
        table=[]
        if fit:
            for k,roots in self.ACTIONS:
                rows=self.fresh_rows(key,(k,roots))
                if not rows:continue
                n=len(rows)
                baseline=(k,roots)==BASELINE
                mean_delivery=sum(r['delivered'] for r in rows)/n
                delivery=mean_delivery if baseline else min(r['delivered'] for r in rows)
                scope=self.budget_scope
                if scope and scope['B']==key[0]:
                    delivery=sum(min(delivery/key[0],left) for left in scope['remaining'])
                # The next root2 proposal source/side mask is not yet known.
                # Do not price a true-side proposal using a prior chain-only Q.
                query=(sum(r['query_nodes'] for r in rows)/n if baseline else
                       key[0]*(k+roots) if roots==2 else max(r['query_nodes'] for r in rows))
                factual_cloud=[r['cloud_rpc_ms'] for r in rows]
                factual_other=[r['unhidden_noncloud_ms'] for r in rows]
                if baseline:
                    cloud=sum(factual_cloud)/n;other=sum(factual_other)/n
                else:
                    cloud=max(max(factual_cloud),fit['fixed_effective_ms']+fit['marginal_ms_per_query']*query)
                    other=max(factual_other)
                if delivery<=0 or not math.isfinite(cloud+other) or min(cloud,other)<0:
                    raise ValueError('SHAPE_COST_PREDICTION')
                last=max(self.sample_tick[r['generation'],r['cycle_id']] for r in rows)
                table.append(dict(k=k,branches=roots,factual_samples=n,
                    expected_query_nodes=query,expected_delivered=delivery,
                    mean_factual_delivered=mean_delivery,sample_age=self.budget_tick[key]-last,
                    mature=n>=self.MIN_FACTUAL,predicted_cloud_rpc_ms=cloud,
                    predicted_unhidden_noncloud_ms=other,predicted_cycle_ms=cloud+other,
                    predicted_ms_per_delivered_token=(cloud+other)/delivery,
                    actual_side_samples=sum(self.actual_side(r)>0 for r in rows),
                    actual_side_range=[min(self.actual_side(r) for r in rows),max(self.actual_side(r) for r in rows)],
                    estimate_kind='ACTUAL_SIDE_FACTUAL_COST_FLOOR_NOT_COUNTERFACTUAL_OR_CONFIDENCE_BOUND'))
        self.prediction_cache[key]=(table,fit)
        return table,fit

    def probe_candidates(self,key,known):
        pending=[a for a in self.PROBE_ORDER if a not in known or not known[a]['mature']]
        # Reconfirm one genuine sample first. Zero-side attempts do not mature
        # root2 and must not permanently starve the other supported actions.
        return sorted(pending,key=lambda a:(0 if a in known else 1,
            self.shape_last_attempt[key,a],self.PROBE_ORDER.index(a)))

    def observe_budget_cost(self,row):
        super().observe_budget_cost(row)
        if all(p['reason']=='preflight_baseline' for p in row['plans']):return
        key=row['actual_batch'],row['prefix_bucket']
        self.actual_side(row)
        if row['branches']==2 and self.actual_side(row)==0:self.zero_side_cycles+=1
        if any(p['reason']=='budget_probe' for p in row['plans']):
            self.shape_last_attempt[key,(row['k'],row['branches'])]=self.budget_tick[key]

    def record_plan(self,p,frame,batch):
        # Full immutable request/prefix/plan identity retained. Scores already
        # occur once in growth_budget_decision; never rebuild them per request.
        self.event('growth_plan',plan=p.wire(),produce_batch=batch,bucket=frame['key'],
                   selector=SELECTOR,bucket_choice=frame['choice'])

    def snapshot(self):
        result=super().snapshot()
        result.update(selector=SELECTOR,shape_budget_v6=True,
                      requested_root2_actual_zero_side_cycles=self.zero_side_cycles,
                      delivery_estimate='ACTUAL_SIDE_QUALIFIED_RECENT_MIN_ALTERNATIVES',
                      cost_estimate='RECENT_FACTUAL_MAX_ALTERNATIVES_NO_SHAPE_ZERO_CREDIT')
        return result

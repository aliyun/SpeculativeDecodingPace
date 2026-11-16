from collections import defaultdict, deque
import math
import os
import time

from .growth_batch_policy import BatchPolicy

SELECTOR = 'cloud_fixed_marginal_cost_v4'

def enabled():
    value = os.environ.get('TRIAGE_CLOUD_COST_GROWTH', '0')
    if value not in ('0', '1'):
        raise ValueError('CLOUD_COST_FLAG')
    return value == '1'


def fit_cloud(rows):
    """Two-parameter nonnegative least-squares, no external solver or retry."""
    if not rows:
        return None
    pairs = [(r['query_nodes'], r['cloud_rpc_ms']) for r in rows]
    if any(not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0 for x, y in pairs):
        raise ValueError('CLOUD_COST_FIT_INPUT')
    n = len(pairs)
    sx = sum(x for x, y in pairs)
    sy = sum(y for x, y in pairs)
    sxx = sum(x*x for x, y in pairs)
    sxy = sum(x*y for x, y in pairs)
    candidates = [(sy/n, 0.), (0., sxy/sxx)]
    determinant = n*sxx-sx*sx
    identifiable = determinant > 1e-9
    if identifiable:
        a = (sy*sxx-sx*sxy)/determinant
        slope = (n*sxy-sx*sy)/determinant
        if a >= 0 and slope >= 0:
            candidates.append((a, slope))
    def error(ab):
        a, b = ab
        return sum((y-a-b*x)**2 for x, y in pairs)
    a, b = min(candidates, key=error)
    return dict(fixed_effective_ms=a, marginal_ms_per_query=b,
        rms_ms=math.sqrt(error((a, b))/n), samples=n,
        query_range=[min(x for x, y in pairs), max(x for x, y in pairs)],
        fixed_marginal_identifiable=identifiable,
        scope='VERIFY_RPC_EFFECTIVE_OVERHEAD_NOT_SEPARATE_UVA_COPY_TIME')


class CloudCostPolicy(BatchPolicy):
    cost_mode = True
    COST_WINDOW = 128
    CLOUD_FIT_WINDOW = 32
    ACTION_WINDOW = 8
    REFRESH_PERIOD = 8
    SWITCH_MARGIN = 1.02

    def __init__(self, event, clock=time.monotonic_ns):
        if not getattr(self,'live_runtime',False) and not enabled():
            raise ValueError('CLOUD_COST_REQUIRES_OPT_IN')
        super().__init__(event, clock)
        def named_event(name, **fields):
            if 'selector' in fields:
                fields['selector'] = SELECTOR
            event(name, **fields)
        self.event = named_event
        self.cost_rows = defaultdict(lambda: deque(maxlen=self.COST_WINDOW))
        self.probe_issues = defaultdict(int)
        self.cost_observations = []
        self.last_predictions = {}
        self.event('growth_cloud_cost_bound', selector=SELECTOR,
            flag='TRIAGE_CLOUD_COST_GROWTH=1', minimum_factual_samples=1,
            cold_coverage_actions=[list(a) for a in self.ACTIONS],
            native_GPU_math_changed=False, isolated_weight_copy_time_measured=False)

    def load_seed(self, rows):
        if rows:
            raise ValueError('CLOUD_COST_REQUIRES_FRESH_TIMED_FEEDBACK_NO_LEGACY_PRIOR')
        super().load_seed(rows)

    def predictions(self, key):
        rows = list(self.cost_rows.get(key, ()))
        fit = fit_cloud(rows[-self.CLOUD_FIT_WINDOW:])
        result = []
        if fit is None:
            return result, fit
        for k, roots in self.ACTIONS:
            chosen = [r for r in rows if (r['k'], r['branches']) == (k, roots)][-self.ACTION_WINDOW:]
            if not chosen:
                continue
            n = len(chosen)
            query = sum(r['query_nodes'] for r in chosen)/n
            delivered = sum(r['delivered'] for r in chosen)/n
            noncloud = sum(r['unhidden_noncloud_ms'] for r in chosen)/n
            cloud = fit['fixed_effective_ms'] + fit['marginal_ms_per_query']*query
            total = cloud + noncloud
            if not math.isfinite(total) or total <= 0 or delivered <= 0:
                raise ValueError('CLOUD_COST_PREDICTION')
            result.append(dict(k=k, branches=roots, factual_samples=n,
                expected_query_nodes=query, expected_delivered=delivered,
                predicted_cloud_rpc_ms=cloud, predicted_unhidden_noncloud_ms=noncloud,
                predicted_cycle_ms=total, predicted_ms_per_delivered_token=total/delivered,
                estimate_kind='MODEL_PREDICTION_NOT_COUNTERFACTUAL_FACT'))
        return result, fit

    def score(self, bucket, action):
        table, _ = self.predictions(bucket)
        row = next((r for r in table if (r['k'], r['branches']) == action), None)
        return 1000/row['predicted_ms_per_delivered_token'] if row else None

    def _adaptive(self, key, st):
        table, fit = self.predictions(key)
        known = {(r['k'], r['branches']): r for r in table}
        missing = [a for a in self.ACTIONS if a not in known]
        old = st['current']
        if missing:
            # Six eligible cohort decisions can cover all six actions. Pending
            # cohorts still pin K/root and never count as new independent trials.
            action = min(missing, key=lambda a: (self.probe_issues[(key, a)], st['tried'][a], a))
            self.probe_issues[(key, action)] += 1
            reason = 'cost_cold_coverage'
        elif st['choices'] % self.REFRESH_PERIOD == 0:
            action = min((a for a in self.ACTIONS if a != old), key=lambda a: (st['tried'][a], a))
            reason = 'cost_refresh_probe'
        else:
            best = min(known, key=lambda a: (known[a]['predicted_ms_per_delivered_token'], a))
            if old not in known or known[best]['predicted_ms_per_delivered_token']*self.SWITCH_MARGIN < known[old]['predicted_ms_per_delivered_token']:
                action = best
            else:
                action = old
            st.update(current=action, target=None)
            reason = 'cost_model_switch' if action != old else 'cost_model_hold'
        record = dict(selector=SELECTOR, bucket=list(key), bucket_choice=st['choices'],
            current_before=list(old), selected=list(action), reason=reason,
            all_six_have_factual_sample=not missing, missing_actions=[list(a) for a in missing],
            cloud_fit=fit, predictions=table, no_min32_gate=True,
            fixed_time_not_pure_weight_copy=True, no_unmeasured_B1_copy=True)
        self.last_predictions[key] = record
        self.event('growth_cost_decision', **record)
        return action, reason

    def commit_batch(self, rows, *, rpc_ms=0.):
        super().commit_batch(rows, rpc_ms=rpc_ms)
        cycle = self.observations[-1]
        if cycle['k'] is None:
            return
        k, roots = cycle['k'], cycle['branches']
        counts = [r.get('query_nodes') for r in rows]
        if any(type(q) is not int or q not in (k+1, k+2) or roots == 1 and q != k+1 for q in counts):
            raise ValueError('CLOUD_COST_NATIVE_QUERY_BINDING')
        if not math.isfinite(rpc_ms) or rpc_ms <= 0:
            raise ValueError('CLOUD_COST_VERIFY_TIMER')
        elapsed = cycle['cycle_ms']
        if elapsed is not None and (not math.isfinite(elapsed) or rpc_ms > elapsed + 1e-6):
            raise ValueError('CLOUD_COST_VERIFY_EXCEEDS_SERIALIZED_CYCLE')
        key = self.bucket(cycle['actual_batch'], max(cycle['prefix_lengths']))
        row = dict(selector=SELECTOR, unit='serialized_controller_delivery_cycle_v3',
            generation=cycle['generation'], cycle_id=cycle['cycle_id'], phase_key=cycle['phase_key'],
            ids=cycle['ids'], plans=cycle['plans'], actual_batch=cycle['actual_batch'],
            sources=cycle.get('sources'),
            prefix_bucket=key[1], k=k, branches=roots, query_counts=counts, query_nodes=sum(counts),
            delivered=cycle['delivered'], cloud_rpc_ms=rpc_ms, cycle_ms=elapsed,
            unhidden_noncloud_ms=max(0., elapsed-rpc_ms) if elapsed is not None else None,
            learning=cycle['learning'], terminal_censored=cycle['terminal_censored'],
            cloud_timer_scope='TARGET_VERIFY_RPC_WALL_INCLUDES_NATIVE_UVA_ACCESS_NOT_ISOLATED_COPY',
            noncloud_scope='SERIALIZED_CYCLE_MINUS_VERIFY_RPC_NOT_SUM_OF_OVERLAPPED_EDGE_WORK')
        if len(self.cost_observations) >= 20000:
            raise RuntimeError('CLOUD_COST_OBSERVATION_LIMIT')
        if row['learning']:
            self.cost_rows[key].append(row)
        self.cost_observations.append(row)
        self.event('growth_cost_cycle', **row)

    def snapshot(self):
        result = super().snapshot()
        result.update(selector=SELECTOR, cost_model=True, cost_observations=len(self.cost_observations),
            cold_coverage='ALL_SIX_FACTUAL_SAME_B_BEFORE_EXPLOIT_NO_MIN32',
            cost_refresh_period=self.REFRESH_PERIOD,
            cost_buckets=[dict(batch=key[0], prefix_bucket=key[1], samples=len(rows),
                cloud_fit=self.predictions(key)[1], predictions=self.predictions(key)[0]) for key, rows in self.cost_rows.items()])
        return result

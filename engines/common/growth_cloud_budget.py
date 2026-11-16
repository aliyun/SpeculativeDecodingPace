from collections import defaultdict
import math
import os
import time

from .growth_cloud_cost import CloudCostPolicy, fit_cloud

SELECTOR = 'cloud_cost_budget_v5'
BASELINE = (3, 1)


def enabled():
    value = os.environ.get('TRIAGE_CLOUD_BUDGET_GROWTH', '0')
    if value not in ('0', '1'):
        raise ValueError('CLOUD_BUDGET_FLAG')
    return value == '1'


class BudgetCostPolicy(CloudCostPolicy):
    selector_name = SELECTOR
    budget_mode = True
    MIN_FACTUAL = 2
    RECENT_CYCLES = 12
    ACTION_RECENT = 4
    BASELINE_MAX_AGE = 3
    PROBE_FRACTION = .20
    BOOTSTRAP_CREDIT = 1.
    PROBE_GAP = 2
    PROBE_MIN_REMAINING = 16
    SHRUNK_STABLE = 3
    PROBE_ORDER = ((5, 1), (7, 1), (3, 2), (5, 2), (7, 2))

    def __init__(self, event, clock=time.monotonic_ns):
        if not getattr(self,'live_runtime',False) and not enabled():
            raise ValueError('CLOUD_BUDGET_REQUIRES_OPT_IN')
        def renamed(name, **fields):
            if 'selector' in fields:
                fields['selector'] = SELECTOR
            event(name, **fields)
        super().__init__(renamed, clock)
        self.init_budget_state()
        self.event('growth_budget_bound', selector=SELECTOR,
            minimum_factual_samples=self.MIN_FACTUAL, recent_sameB_cycles=self.RECENT_CYCLES,
            no_forced_six_action_coverage=True, probe_fraction=self.PROBE_FRACTION,
            baseline_max_age=self.BASELINE_MAX_AGE, no_native_GPU_change=True,
            delivery_heuristic='MIN_RECENT_FACTUAL_DELIVERY_NOT_STATISTICAL_CONFIDENCE')

    def init_budget_state(self):
        self.budget_tick = defaultdict(int)
        self.sample_tick = {}
        self.elapsed_ms = defaultdict(float)
        self.probe_ms = defaultdict(float)
        self.last_probe = defaultdict(lambda: -100)
        self.peak_B = 0
        self.last_B = None
        self.stable_B = 0
        self.budget_scope = None
        self.prediction_cache = {}
        self.bootstrap_credit_ms = None

    def set_budget_scope(self, B, remaining):
        if (type(B) is not int or B < 1 or len(remaining) != B or
                any(type(n) is not int or n <= 0 for n in remaining)):
            raise ValueError('CLOUD_BUDGET_REMAINING_BINDING')
        self.stable_B = self.stable_B+1 if self.last_B == B else 1
        self.last_B = B
        self.peak_B = max(self.peak_B, B)
        self.budget_scope = dict(B=B, remaining=list(remaining), peak_B=self.peak_B,
                                 stable_B=self.stable_B)
        self.prediction_cache.clear()
        return dict(self.budget_scope)

    def prepare_cohort(self, fresh, all_prefixes, *, mode='spec', max_total_tokens=None,
                       remaining_by_rid=None):
        preflight = bool(fresh) and all(rid == getattr(self, 'preflight_rid', None) for rid, _ in fresh)
        if mode == 'spec' and fresh and not preflight:
            ids = [rid for rid, _ in all_prefixes]
            if not isinstance(remaining_by_rid, dict) or set(remaining_by_rid) != set(ids):
                raise ValueError('CLOUD_BUDGET_REMAINING_BINDING')
            scope = self.set_budget_scope(len(ids), [remaining_by_rid[rid] for rid in ids])
            self.event('growth_budget_scope', selector=SELECTOR, ids=ids,
                       prefix_lengths=[len(prefix) for _, prefix in all_prefixes], **scope)
        super().prepare_cohort(fresh, all_prefixes, mode=mode, max_total_tokens=max_total_tokens)

    def fresh_rows(self, key, action):
        now = self.budget_tick[key]
        return [r for r in self.cost_rows.get(key, ())
                if (r['k'], r['branches']) == action and
                now-self.sample_tick.get((r['generation'], r['cycle_id']), -1000) < self.RECENT_CYCLES][-self.ACTION_RECENT:]

    def predictions(self, key):
        if key in self.prediction_cache:
            return self.prediction_cache[key]
        history = list(self.cost_rows.get(key, ()))
        fit = fit_cloud(history[-self.CLOUD_FIT_WINDOW:])
        table = []
        if fit:
            for k, roots in self.ACTIONS:
                rows = self.fresh_rows(key, (k, roots))
                if not rows:
                    continue
                n = len(rows)
                query = sum(r['query_nodes'] for r in rows)/n
                mean_delivery = sum(r['delivered'] for r in rows)/n
                delivery = mean_delivery if (k, roots) == BASELINE else min(r['delivered'] for r in rows)
                scope = self.budget_scope
                if scope and scope['B'] == key[0]:
                    delivery = sum(min(delivery/key[0], left) for left in scope['remaining'])
                cloud = fit['fixed_effective_ms']+fit['marginal_ms_per_query']*query
                other = sum(r['unhidden_noncloud_ms'] for r in rows)/n
                if delivery <= 0 or not math.isfinite(cloud+other):
                    raise ValueError('CLOUD_BUDGET_PREDICTION')
                last = max(self.sample_tick[r['generation'], r['cycle_id']] for r in rows)
                table.append(dict(k=k, branches=roots, factual_samples=n,
                    expected_query_nodes=query, expected_delivered=delivery,
                    mean_factual_delivered=mean_delivery, sample_age=self.budget_tick[key]-last,
                    mature=n >= self.MIN_FACTUAL, predicted_cloud_rpc_ms=cloud,
                    predicted_unhidden_noncloud_ms=other, predicted_cycle_ms=cloud+other,
                    predicted_ms_per_delivered_token=(cloud+other)/delivery,
                    estimate_kind='CONSERVATIVE_MODEL_PREDICTION_NOT_COUNTERFACTUAL_OR_CONFIDENCE_BOUND'))
        result = table, fit
        self.prediction_cache[key] = result
        return result

    def probe_candidates(self, key, known):
        pending = [a for a in self.PROBE_ORDER if a not in known or not known[a]['mature']]
        return sorted(pending, key=lambda a:(0 if a in known else 1, self.PROBE_ORDER.index(a)))

    def score_prediction(self, row):
        """Default physical unit cost; specialized objectives must be explicit."""
        return row['predicted_ms_per_delivered_token']

    def cold_probe_ms(self,key,action,baseline,fit):
        k,roots=action
        queries=key[0]*(k+roots)
        return max(baseline['predicted_cycle_ms'],
            fit['fixed_effective_ms']+fit['marginal_ms_per_query']*queries+
            baseline['predicted_unhidden_noncloud_ms']*k/3)

    def estimate_probe_ms(self,key,action,known,baseline,fit):
        measured=known.get(action)
        return (measured['predicted_cycle_ms']if measured else
                self.cold_probe_ms(key,action,baseline,fit))

    def _adaptive(self, key, st):
        scope = self.budget_scope
        if not scope or scope['B'] != key[0]:
            raise ValueError('CLOUD_BUDGET_SCOPE_REQUIRED')
        table, fit = self.predictions(key)
        known = {(r['k'], r['branches']): r for r in table}
        baseline = known.get(BASELINE)
        old = st['current']
        action, reason = BASELINE, 'budget_baseline_evidence'
        credit = allowed = spent = estimated = 0.
        candidate = None
        if max(scope['remaining']) <= 1:
            reason = 'budget_terminal_floor'
        elif baseline and baseline['mature']:
            if self.bootstrap_credit_ms is None:
                self.bootstrap_credit_ms = self.BOOTSTRAP_CREDIT*baseline['predicted_cycle_ms']
            credit = self.bootstrap_credit_ms
            allowed = self.PROBE_FRACTION*sum(self.elapsed_ms.values())+credit
            spent = sum(self.probe_ms.values())
            if baseline['sample_age'] >= self.BASELINE_MAX_AGE:
                reason = 'budget_baseline_anchor'
            else:
                eligible = {a:r for a,r in known.items() if r['mature'] and a != BASELINE
                            and r.get('admissible', True)
                            and self.score_prediction(r)*self.SWITCH_MARGIN < self.score_prediction(baseline)}
                if eligible:
                    action = min(eligible, key=lambda a:(self.score_prediction(eligible[a]),a))
                    reason = 'budget_model_hold' if action == old else 'budget_model_switch'
                else:
                    reason = 'budget_baseline_best'
                probe_ok = (sum(scope['remaining'])/key[0] >= self.PROBE_MIN_REMAINING and
                    (key[0] == self.peak_B or self.stable_B >= self.SHRUNK_STABLE) and
                    self.budget_tick[key]-self.last_probe[key] >= self.PROBE_GAP)
                # Confirm a partially observed action before opening yet another one.
                pending = self.probe_candidates(key, known)
                if probe_ok and pending:
                    candidate = pending[0]
                    estimated = self.estimate_probe_ms(key,candidate,known,baseline,fit)
                    if spent+estimated <= allowed:
                        action, reason = candidate, 'budget_probe'
                        self.last_probe[key] = self.budget_tick[key]
                    elif not eligible:
                        reason = 'budget_probe_time_guard'
                elif not probe_ok and not eligible:
                    reason = 'budget_probe_remaining_or_stability_guard'
        st.update(current=action, target=None)
        record = dict(selector=self.selector_name, bucket=list(key), bucket_choice=st['choices'],
            current_before=list(old), selected=list(action), reason=reason,
            scope=dict(scope), cloud_fit=fit, predictions=table,
            probe_candidate=list(candidate) if candidate else None,
            probe_estimated_ms=estimated, probe_allowed_ms=allowed, probe_spent_ms=spent,
            bootstrap_credit_ms=credit, actual_sameB_tick=self.budget_tick[key],
            no_forced_six_coverage=True, no_B1_or_historical_reward_copy=True)
        self.last_predictions[key] = record
        self.event('growth_budget_decision', **record)
        return action, reason

    def account_cycle_time(self, key, elapsed_ms, *, probe):
        """Debit one serialized delivery interval, including failed attempts."""
        self.budget_tick[key] += 1
        if elapsed_ms is not None:
            self.elapsed_ms[key] += elapsed_ms
            if probe:
                self.probe_ms[key] += elapsed_ms
        self.prediction_cache.clear()
        return self.budget_tick[key]

    def observe_budget_cost(self, row):
        # Preflight contributes neither budget credits nor reward evidence.
        if all(p['reason'] == 'preflight_baseline' for p in row['plans']):
            return
        key = row['actual_batch'], row['prefix_bucket']
        self.account_cycle_time(key, row['cycle_ms'],
                                probe=any(p['reason'] == 'budget_probe' for p in row['plans']))
        if row['learning']:
            self.sample_tick[row['generation'], row['cycle_id']] = self.budget_tick[key]
        self.prediction_cache.clear()

    def commit_batch(self, rows, *, rpc_ms=0.):
        super().commit_batch(rows, rpc_ms=rpc_ms)
        if self.observations[-1]['k'] is not None:
            self.observe_budget_cost(self.cost_observations[-1])

    def snapshot(self):
        result = super().snapshot()
        result.update(selector=SELECTOR, budget_v5=True,
            cold_coverage='BOUNDED_PROBES_NOT_FORCE_ALL_SIX',
            delivery_estimate='RECENT_MIN_FOR_ALTERNATIVES_MEAN_FOR_K3_NOT_CONFIDENCE_BOUND',
            probe_budget=[dict(bucket=list(k),elapsed_ms=self.elapsed_ms[k],probe_ms=self.probe_ms[k])
                          for k in self.elapsed_ms])
        return result

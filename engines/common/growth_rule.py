from collections import defaultdict, deque
import math
import os
import time

from .growth import KS
from .growth_cloud_budget import BASELINE
from .growth_cloud_cost import fit_cloud
from .growth_source_cost import SourceCostPolicy, group
from .growth_survival import PrefixSurvival

SELECTOR = 'cloud_branch_rule_v2'
STREAM = (3, 1)

# Precomputed forced-fork calibration: cumulative side-hit probability for a
# root gap at or below the bound. This is a workload/calibration table, not a
# model constant; TRIAGE_GROWTH_FORK_TOPK_MS and _RPC_MS supply the cost side.
FORK_GAP_HIT = ((0.5, 0.0172), (1.0, 0.0174), (1.5, 0.0250),
                (2.0, 0.0212), (3.0, 0.0200))
FORK_GAP_TAIL = 0.006


def enabled():
    value = os.environ.get('TRIAGE_GROWTH_RULE', '0')
    if value not in ('0', '1'):
        raise ValueError('GROWTH_RULE_MODE')
    return value == '1'


def fork_enabled():
    value = os.environ.get('TRIAGE_GROWTH_RULE_FORK', '0')
    if value not in ('0', '1'):
        raise ValueError('GROWTH_RULE_FORK_MODE')
    return value == '1'


class RulePolicy(SourceCostPolicy):
    """Pooled-parameter measured U; no six-arm cold coverage."""

    selector_name = SELECTOR
    live_runtime = True
    RULE_BASELINE_AGE = 32
    RULE_PROBE_EVERY = 16
    RULE_PROBE_MARGIN = 1.05
    RULE_UNMEASURED_PENALTY = 1.05
    RULE_ACTION_WINDOW = 8
    RULE_POOL_WINDOW = 32
    RULE_BETA = 0.20
    RULE_PROBE_FRACTION = 0.10
    RULE_PROBE_MIN_RATIO = 0.80
    RULE_PROBE_BOOTSTRAP = 2.0
    RULE_FAILURE_AGE = 64
    RULE_WIDE_FAILURE_LIMIT = 4
    RULE_WIDE_FAILURE_RATIO = 2
    # Precomputed fork cost. The target marginal row is added at runtime; the
    # top-2/edge column and (for a streamed K3) the plan RPC are explicit.
    FORK_TOPK_MS = float(os.environ.get('TRIAGE_GROWTH_FORK_TOPK_MS', '1.0'))
    _rpc_env = os.environ.get('TRIAGE_GROWTH_FORK_RPC_MS')
    FORK_RPC_MS = float(_rpc_env) if _rpc_env not in (None, '') else None
    FORK_GAP_HIT = FORK_GAP_HIT

    def __init__(self, event, clock=time.monotonic_ns):
        super().__init__(event, clock)
        self.observations = deque(maxlen=self.WINDOW)
        self.cost_observations = deque(maxlen=self.COST_WINDOW)
        self.fork_mode = fork_enabled()
        probe_all = os.environ.get('TRIAGE_GROWTH_RULE_PROBE_ALL', '0')
        if probe_all not in ('0', '1'):
            raise ValueError('GROWTH_RULE_PROBE_ALL_MODE')
        self.probe_all = probe_all == '1'
        # Source-layer evidence survives prefix-bucket churn. Keys are the
        # native source taxonomy (exact_prefix, suffix, edge), not the coarse
        # cache/edge pricing group.
        self.pooled = defaultdict(lambda: deque(maxlen=512))
        # Exploration is a bounded budget, not an unbounded 1-in-N schedule:
        # a wide probe at high RTT costs a full plan round trip and an AR
        # fallback, so the rule must pay for it out of measured elapsed time.
        self.rule_elapsed = defaultdict(float)
        self.rule_probes = defaultdict(float)
        self.rule_bootstrap = None
        self.rule_wide_native = defaultdict(int)
        self.rule_fork_probed = False
        self.fork_gaps = {}
        self.fork_topk_ms = self.FORK_TOPK_MS
        self.fork_rpc_ms = self.FORK_RPC_MS

        def rule_event(name, **fields):
            fields['selector'] = SELECTOR
            event(name, **fields)

        self.event = rule_event
        self.event('growth_rule_bound', selector=SELECTOR,
                   rule='pooled_survival_measured_D_over_T', fork=self.fork_mode,
                   baseline_age=self.RULE_BASELINE_AGE,
                   probe_every=self.RULE_PROBE_EVERY,
                   probe_margin=self.RULE_PROBE_MARGIN,
                   unmeasured_penalty=self.RULE_UNMEASURED_PENALTY,
                   pool_window=self.RULE_POOL_WINDOW, beta=self.RULE_BETA,
                   probe_fraction=self.RULE_PROBE_FRACTION,
                   probe_min_ratio=self.RULE_PROBE_MIN_RATIO,
                   probe_bootstrap_cycles=self.RULE_PROBE_BOOTSTRAP,
                   failure_age=self.RULE_FAILURE_AGE,
                   wide_failure_limit=self.RULE_WIDE_FAILURE_LIMIT,
                   wide_failure_ratio=self.RULE_WIDE_FAILURE_RATIO,
                   no_wire_change=True, no_native_gpu_change=True)

    def refresh(self):
        if self.control is None:
            self.epoch = 0
            self.control = dict(epoch=0, mode='adaptive', learn=True)
            self.event('growth_rule_live', selector=SELECTOR,
                       generation=self.generation)

    def commit_batch(self, rows, *, rpc_ms=0.):
        super().commit_batch(rows, rpc_ms=rpc_ms)
        if self.cycle_id % self.WINDOW == 0:
            retained = {(r['generation'], r['cycle_id'])
                        for values in self.cost_rows.values() for r in values}
            self.sample_tick = {key: tick for key, tick in self.sample_tick.items()
                                if key in retained}
        if not self.cost_observations:
            return
        cycle = self.cost_observations[-1]
        if cycle.get('k') is None or not cycle.get('learning'):
            return
        for source in set(cycle.get('sources') or []):
            if isinstance(source, str) and source:
                self.pooled[source].append(cycle)
        if (cycle.get('k') is not None and cycle.get('learning')
                and (cycle['k'] > 3 or cycle['branches'] == 2)):
            for source in set(cycle.get('sources') or []):
                if isinstance(source, str) and source:
                    self.rule_wide_native[group(source)] += 1
        if cycle.get('cycle_ms') is not None:
            key = (cycle['actual_batch'], cycle['prefix_bucket'])
            self.rule_elapsed[key] += cycle['cycle_ms']
            reasons = {p.get('reason') for p in (cycle.get('plans') or []) if p}
            if reasons & {'rule_calibration_probe', 'rule_switch_probe'}:
                self.rule_probes[key] += cycle['cycle_ms']

    # -- evidence -----------------------------------------------------

    @staticmethod
    def _clamp(rows, remaining):
        return sum(min(r['delivered'], remaining) for r in rows) / len(rows)

    def _bucket_rows(self, key, action):
        rows = [r for r in self.cost_rows.get(key, ())
                if (r['k'], r['branches']) == action and r.get('learning')]
        if key[0] == 1 and self.source_scope is not None:
            groups = set(self.source_scope['groups'])
            if len(groups) == 1:
                only = next(iter(groups))
                rows = [r for r in rows if r.get('source_rows', {}).get(only) == 1]
        return rows[-self.RULE_ACTION_WINDOW:]

    def _pooled_rows(self, source, action=None):
        rows = [r for r in self.pooled.get(source, ())
                if r.get('actual_batch') == 1 and r.get('learning')]
        if action is not None:
            rows = [r for r in rows if (r['k'], r['branches']) == action]
        return rows[-self.RULE_POOL_WINDOW:]

    def _pooled_observed(self, source, action, remaining):
        rows = self._pooled_rows(source, action)
        if len(rows) < self.MIN_FACTUAL:
            return None
        return dict(n=len(rows), D=self._clamp(rows, remaining),
                    T=sum(r['cycle_ms'] for r in rows) / len(rows),
                    noncloud=sum(r['unhidden_noncloud_ms'] for r in rows) / len(rows),
                    kind='pooled', mature=True)

    def _pooled_profile(self, source):
        rows = [r for r in self._pooled_rows(source) if r.get('branches') == 1]
        return PrefixSurvival(rows, self.MIN_FACTUAL) if rows else None

    def _observed(self, key, action):
        rows = list(self._bucket_rows(key, action))
        # A wide plan that is served by AR is not a cheap success. The attempt
        # ledger records those episodes separately from native verification;
        # charging them here prevents a few lucky native rows from making a
        # sticky wide action look good at high RTT.
        if key[0] == 1 and self.source_scope is not None:
            groups = set(self.source_scope['groups'])
            if len(groups) == 1:
                only = next(iter(groups))
                rows += self.attempt_ledger.fresh(
                    key, action, only, self.budget_tick[key], self.RECENT_CYCLES)
        if len(rows) < self.MIN_FACTUAL:
            return None
        remaining = self.budget_scope['remaining'][0]
        ticks = [self.sample_tick.get((r.get('generation'), r.get('cycle_id')), r.get('tick', -1))
                 for r in rows]
        return dict(n=len(rows), D=self._clamp(rows, remaining),
                    T=sum(r['cycle_ms'] for r in rows) / len(rows),
                    noncloud=sum(r.get('unhidden_noncloud_ms') or 0. for r in rows) / len(rows),
                    last_tick=max(ticks) if ticks else -1,
                    kind='measured', mature=True)

    def _fork_option(self, key, source, action, remaining, bucket_base, pooled_base):
        """Expected value of the gap-gated root alternate.

        branches=2 requests top2; the pack gate appends the alternate only on
        small gaps. The option value is the single-root reward plus the
        observed fork probability times the conditional side gain, and the
        single-root cycle plus that probability times the conditional row cost.
        """
        k, _ = action
        rows = self._bucket_rows(key, action)
        single = self._bucket_rows(key, (k, 1))
        if len(rows) < self.MIN_FACTUAL or len(single) < self.MIN_FACTUAL:
            rows = self._pooled_rows(source, action)
            single = self._pooled_rows(source, (k, 1))
        if len(rows) < self.MIN_FACTUAL or len(single) < self.MIN_FACTUAL:
            return None
        forked = [r for r in rows if r['query_nodes'] == k + 2]
        if not forked:
            return None
        p = len(forked) / len(rows)
        d_single = self._clamp(single, remaining)
        t_single = sum(r['cycle_ms'] for r in single) / len(single)
        d_fork = self._clamp(forked, remaining)
        t_fork = sum(r['cycle_ms'] for r in forked) / len(forked)
        gain = d_fork - d_single
        delta = t_fork - t_single
        if gain <= 0 or delta < 0:
            return None
        delivered = d_single + p * gain
        cycle = t_single + p * delta
        return dict(action=action, k=k, roots=2, D=delivered, T=cycle,
                    U=delivered / cycle, kind='fork_option', n=len(rows),
                    mature=len(rows) >= self.MIN_FACTUAL)

    # -- selection ----------------------------------------------------

    def _probe_budget(self, baseline):
        if self.rule_bootstrap is None:
            self.rule_bootstrap = self.RULE_PROBE_BOOTSTRAP * baseline['T']
        return (self.RULE_PROBE_FRACTION * sum(self.rule_elapsed.values())
                + self.rule_bootstrap - sum(self.rule_probes.values()))

    def _wide_failures_recent(self, source_group):
        return sum(1 for r in self.attempt_ledger.observations
                   if (r.get('k', 0) > 3 or r.get('branches') == 2)
                   and r.get('source_group') == source_group
                   and r.get('learning')
                   and self.cycle_id - r.get('last_cycle_id', r.get('cycle_id', 0))
                   < self.RULE_FAILURE_AGE)

    def _fork_rpc_ms(self, source):
        """Exposed plan RPC paid when a streamed K3 is turned into a fork plan."""
        if self.fork_rpc_ms is not None:
            return self.fork_rpc_ms
        rows = self._pooled_rows(source)
        wide = [r for r in rows if r['k'] > 3]
        stream = [r for r in rows if r['k'] == 3]
        if wide and stream:
            import statistics
            return max(0., statistics.median(r['unhidden_noncloud_ms'] for r in wide)
                       - statistics.median(r['unhidden_noncloud_ms'] for r in stream))
        return math.inf  # no measured plan exchange: do not fork a streamed K3

    def _fork_gap_limit(self, k, theta, source, fit):
        """Largest gap whose expected rescue pays for this fork's cost.

        p_side(gap) / Theta is the saved time per token; row + top2 (+ plan RPC
        for a streamed K3) is the additional time. No eligible gap means no
        fork, instead of a fixed logit threshold.
        """
        if theta <= 0 or type(self.fork_topk_ms) not in (int, float):
            return None
        row = float(fit.get('marginal_ms_per_query', 0.)) + float(self.fork_topk_ms)
        rpc = self._fork_rpc_ms(source) if k == 3 else 0.
        delta = row + rpc
        if not math.isfinite(delta):
            return None
        rates = [p for _, p in self.FORK_GAP_HIT]
        if max(rates) / theta <= delta:
            return None
        eligible = [gap for gap, p in self.FORK_GAP_HIT if p / theta > delta]
        return max(eligible) if eligible else None

    def fork_gap_for(self, k):
        return self.fork_gaps.get(k)

    def _rule_candidates(self, key, fit):
        scope = self.budget_scope
        actions = [(3, 1)] + [(k, 1) for k in KS if k != 3]
        if key[0] > 1:
            # Batch verification keeps the complete B1..8 catalogue for 3/5/7.
            actions = [a for a in actions if a[0] in (3, 5, 7)]
        if self.fork_mode and self.source_scope['counts']['edge']:
            actions += [(3, 2)] + [(k, 2) for k in KS if k != 3]
        # A wide plan that has never completed natively but has already
        # produced AR episodes is a transport failure, not exploration. Keep
        # the streamed action available and stop paying plan RTTs for it.
        if key[0] == 1 and self.source_scope is not None:
            groups = set(self.source_scope['groups'])
            if len(groups) == 1:
                only = next(iter(groups))
                failures = self._wide_failures_recent(only)
                native = self.rule_wide_native.get(only, 0)
                if (failures >= self.RULE_WIDE_FAILURE_LIMIT
                        and failures > self.RULE_WIDE_FAILURE_RATIO * native + 2):
                    actions = [a for a in actions if a == STREAM]
        remaining = scope['remaining'][0] if key[0] == 1 else min(scope['remaining'])
        source = self.source_scope['sources'][0] if key[0] == 1 else None
        bucket_base = self._observed(key, BASELINE)
        pooled_base = (self._pooled_observed(source, BASELINE, remaining)
                       if source is not None else None)
        base_evidence = bucket_base or pooled_base or {}
        baseline_row = dict(predicted_cycle_ms=base_evidence.get('T', 0.),
                            predicted_unhidden_noncloud_ms=base_evidence.get('noncloud', 0.))
        if baseline_row['predicted_cycle_ms'] <= 0:
            return []
        self.fork_gaps.clear()
        theta = 0.
        if base_evidence.get('T', 0.) > 0:
            theta = base_evidence.get('D', 0.) / base_evidence['T']
        out = []
        for action in actions:
            k, roots = action
            if any(rem < k + 1 for rem in scope['remaining']):
                continue
            observed = self._observed(key, action)
            if observed is None and source is not None:
                observed = self._pooled_observed(source, action, remaining)
                if observed is not None and bucket_base is not None and pooled_base is not None:
                    # Keep the current bucket's base latency; pooling only
                    # supplies the action delta.
                    observed = dict(observed, kind='pooled',
                                    T=max(1e-6, bucket_base['T'] + observed['T'] - pooled_base['T']))
            if observed is not None:
                delivered, cycle = observed['D'], observed['T']
                kind, mature, n = observed['kind'], observed['mature'], observed['n']
            else:
                if roots == 2:
                    gap = self._fork_gap_limit(k, theta, source, fit)
                    if gap is None:
                        continue
                    self.fork_gaps[k] = gap
                    option = self._fork_option(key, source, action, remaining,
                                               bucket_base, pooled_base)
                    if option is not None:
                        out.append(self._upper(option))
                        continue
                profile = (self._pooled_profile(source) if source is not None else None)
                if profile is None:
                    if key[0] != 1:
                        continue
                    profile = self.primary_profile(key, group(source), layer=source)
                delivered = profile.mean(k, remaining)
                if delivered is None:
                    delivered = profile.optimistic_depth(k, remaining)
                if delivered is None:
                    continue
                if roots == 2:
                    if profile.survival[0] is None:
                        continue
                    # No side evidence yet: use the maximum possible rescue
                    # (1 - primary root acceptance) as a cold-start prior. The
                    # first fork probe supplies the real side hit rate, and
                    # _fork_option takes over from there.
                    if profile.side_supported and profile.side_rate is not None:
                        prior = profile.side_rate
                    else:
                        prior = max(0.0, 1.0 - profile.survival[0])
                    if prior <= 0.:
                        continue
                    delivered += prior
                cycle = float(self.cold_probe_ms(key, action, baseline_row, fit))
                cycle *= self.RULE_UNMEASURED_PENALTY
                kind, mature, n = 'optimistic', False, 0
            if key[0] == 1:
                delivered = min(delivered, float(scope['remaining'][0]))
            if (not math.isfinite(delivered) or not math.isfinite(cycle)
                    or delivered <= 0 or cycle <= 0):
                continue
            out.append(self._upper(dict(action=action, k=k, roots=roots, D=delivered,
                                        T=cycle, U=delivered / cycle, kind=kind,
                                        mature=mature, n=n)))
        return out

    def _upper(self, row):
        bonus = 1. + self.RULE_BETA / math.sqrt(max(row.get('n', 0), 1))
        row['U_upper'] = row['U'] * bonus
        return row

    def _adaptive(self, key, st):
        scope = self.budget_scope
        if (not scope or scope['B'] != key[0] or self.source_scope is None
                or len(self.source_scope['ids']) != key[0]):
            return super()._adaptive(key, st)
        rows = list(self.cost_rows.get(key, ()))
        fit = fit_cloud(rows[-self.CLOUD_FIT_WINDOW:])
        baseline = self._observed(key, BASELINE)
        if fit is None or baseline is None:
            return super()._adaptive(key, st)
        candidates = self._rule_candidates(key, fit)
        if not candidates:
            return super()._adaptive(key, st)
        baseline_tick = baseline.get('last_tick', -1)
        baseline_age = (self.budget_tick[key] - baseline_tick
                        if baseline_tick >= 0 else 0)
        budget = self._probe_budget(baseline)

        current = st['current']
        by_action = {r['action']: r for r in candidates}
        cur = by_action.get(current)
        if self.probe_all and st['choices'] % self.RULE_PROBE_EVERY == 0:
            def probe_ok(r):
                if not r['mature'] and r['action'] == current:
                    return False
                if budget < r['T']:
                    return False
                if r['action'][1] == 2 and not self.rule_fork_probed:
                    return True  # one cold-start fork probe, then real evidence
                return cur is None or r['U_upper'] >= cur['U_upper'] * self.RULE_PROBE_MIN_RATIO
            immature = [r for r in candidates if not r['mature'] and probe_ok(r)]
            if immature:
                probe = min(immature, key=lambda r: r['k'])
                action, reason = probe['action'], 'rule_calibration_probe'
                if action[1] == 2:
                    self.rule_fork_probed = True
                st.update(current=action, target=None)
                self._record_rule(key, st, current, action, reason, candidates,
                                  fit, baseline, baseline_age=0)
                return action, reason
        best = max(candidates, key=lambda r: (r['U_upper'], -r['k']))
        if cur is not None and (best['action'] == current
                                or cur['U_upper'] * self.SWITCH_MARGIN >= best['U_upper']):
            action, reason = cur['action'], 'rule_hold'
        else:
            action, reason = best['action'], 'rule_switch'

        chosen = by_action[action]
        if not chosen['mature'] and chosen.get('n', 0) == 0:
            # A purely modelled tier is an experiment, not a default. It must
            # fit the remaining exploration budget and clear the value screen.
            allowed = (st['choices'] % self.RULE_PROBE_EVERY == 0
                       and budget >= chosen['T']
                       and (cur is None or chosen['U_upper'] >= cur['U_upper'] * self.RULE_PROBE_MIN_RATIO))
            gain = (cur is None or chosen['U_upper'] > cur['U_upper'] * self.RULE_PROBE_MARGIN)
            if not (allowed and gain):
                action = cur['action'] if cur is not None else BASELINE
                reason = 'rule_hold_immature'
            else:
                reason = 'rule_switch_probe'

        if action == BASELINE and baseline_age >= self.RULE_BASELINE_AGE:
            reason = 'rule_baseline_refresh'

        if action[1] == 2:
            self.rule_fork_probed = True
        st.update(current=action, target=None)
        self._record_rule(key, st, current, action, reason, candidates,
                          fit, baseline, baseline_age)
        return action, reason

    def _record_rule(self, key, st, current, action, reason, candidates,
                     fit, baseline, baseline_age):
        scope = self.budget_scope
        record = dict(selector=SELECTOR, bucket=list(key), bucket_choice=st['choices'],
                      current_before=list(current), selected=list(action), reason=reason,
                      scope=dict(scope), cloud_fit=fit,
                      candidates=[dict(action=list(r['action']), D=r['D'], T=r['T'],
                                       U=r['U'], U_upper=r.get('U_upper'),
                                       n=r.get('n'), kind=r['kind'],
                                       mature=r['mature'])
                                  for r in candidates],
                      branch='stream' if action == STREAM else 'wide',
                      fork=self.fork_mode, baseline_age=baseline_age,
                      fork_gaps=dict(self.fork_gaps), fork_topk_ms=self.fork_topk_ms,
                      fork_rpc_ms=self.fork_rpc_ms,
                      probe_all=self.probe_all,
                      rule=dict(baseline_age=self.RULE_BASELINE_AGE,
                                probe_every=self.RULE_PROBE_EVERY,
                                probe_margin=self.RULE_PROBE_MARGIN,
                                unmeasured_penalty=self.RULE_UNMEASURED_PENALTY,
                                pool_window=self.RULE_POOL_WINDOW,
                                beta=self.RULE_BETA,
                                probe_fraction=self.RULE_PROBE_FRACTION,
                                probe_min_ratio=self.RULE_PROBE_MIN_RATIO,
                                probe_bootstrap_cycles=self.RULE_PROBE_BOOTSTRAP,
                                failure_age=self.RULE_FAILURE_AGE,
                                wide_failure_limit=self.RULE_WIDE_FAILURE_LIMIT,
                                wide_failure_ratio=self.RULE_WIDE_FAILURE_RATIO),
                      no_wire_change=True, no_native_gpu_change=True)
        self.last_predictions[key] = record
        self.event('growth_rule_decision', **record)

    def snapshot(self):
        result = super().snapshot()
        result.update(selector=SELECTOR, rule_v2=True, rule_fork=self.fork_mode,
                      rule_probe_all=self.probe_all,
                      rule_fork_probed=self.rule_fork_probed,
                      rule_baseline_age=self.RULE_BASELINE_AGE,
                      rule_probe_every=self.RULE_PROBE_EVERY,
                      rule_probe_margin=self.RULE_PROBE_MARGIN,
                      rule_unmeasured_penalty=self.RULE_UNMEASURED_PENALTY,
                      rule_pool_window=self.RULE_POOL_WINDOW,
                      rule_beta=self.RULE_BETA,
                      rule_probe_fraction=self.RULE_PROBE_FRACTION,
                      rule_probe_min_ratio=self.RULE_PROBE_MIN_RATIO,
                      rule_wide_native={k: v for k, v in self.rule_wide_native.items()},
                      rule_wide_failure_limit=self.RULE_WIDE_FAILURE_LIMIT,
                      fork_gaps=dict(self.fork_gaps), fork_topk_ms=self.fork_topk_ms,
                      fork_rpc_ms=self.fork_rpc_ms,
                      rule_wide_failure_ratio=self.RULE_WIDE_FAILURE_RATIO,
                      rule_fork_gaps=dict(self.fork_gaps),
                      rule_branch='POOLED_SURVIVAL_AND_MEASURED_CYCLE',
                      no_wire_change=True)
        return result

from collections import defaultdict, deque
import math
import time

from . import growth as core
from . import growth_batch_profile as profile
from .policy import CONTEXT, fingerprint

BasePolicy = getattr(core, '_SelectorV2Policy', core.Policy)
SELECTOR = 'batch_delivery_cycle_v3'
UNIT = 'serialized_controller_delivery_cycle_v3'


class BatchPolicy(BasePolicy):
    def __init__(self, event, clock=time.monotonic_ns):
        if not getattr(self,'live_runtime',False) and not profile.enabled():
            raise ValueError('BATCH_POLICY_REQUIRES_OPT_IN')
        def named_event(name, **fields):
            if 'selector' in fields:
                fields['selector'] = SELECTOR
            event(name, **fields)
        super().__init__(named_event, clock)
        self.stats = defaultdict(lambda: dict(n=0, tokens=0, seconds=0.,
            window_n=0, window_tokens=0., window_seconds=0.))
        self.frame = None
        self.delivery_end_ns = None
        self.cycle_id = 0
        self.seed_rows = 0

    def prepare_cohort(self, fresh, all_prefixes, *, mode='spec', max_total_tokens=None):
        self.refresh()
        self.frame = None
        if mode != 'spec' or not fresh:
            return
        if (not 1 <= len(all_prefixes) <= profile.BATCH_MAX
                or len({rid for rid, _ in all_prefixes}) != len(all_prefixes)
                or len({rid for rid, _ in fresh}) != len(fresh)
                or not set(fresh).issubset(set(all_prefixes))):
            raise ValueError('BATCH_GROWTH_PLANNING_SCOPE')
        digests = {rid: fingerprint(prefix) for rid, prefix in fresh}
        for rid, prefix in fresh:
            if (rid, digests[rid]) in self.issued:
                raise ValueError('GROWTH_PLAN_ALREADY_PENDING')
        key = self.bucket(len(all_prefixes), max(len(p) for _, p in all_prefixes))
        # Historical experiments used one global drain barrier. In production
        # continuously refilled cohorts must be able to explore while other
        # immutable plans remain outstanding. Admission splits actual actions.
        pending = (set() if getattr(self, 'live_runtime', False) else
                   {(p.k, p.branches) for p in [*self.issued.values(), *self.last_used.values()]})
        if len(pending) > 1:
            raise ValueError('BATCH_GROWTH_MIXED_INFLIGHT_ACTIONS')
        preflight = all(rid == getattr(self, 'preflight_rid', None) for rid, _ in fresh)
        st = self._bucket(key)
        if pending:
            action = next(iter(pending))
            reason = 'pending_cohort_hold'
        elif preflight:
            action, reason = (3, 1), 'preflight_baseline'
        else:
            self.choices += 1
            st['choices'] += 1
            low_kv = (self.state['capacity'] and self.state['free_blocks'] is not None
                      and self.state['free_blocks'] < self.state['capacity'] // 10)
            if low_kv:
                action, reason = (3, 1), 'kv_reservation_pressure'
                st.update(current=action, target=None, last_switch=st['choices'])
            elif self.control['mode'] == 'static':
                action = self.control['k'], self.control['branches']
                reason = 'static_calibration'
            else:
                action, reason = self._adaptive(key, st)
            st['tried'][action] = st['choices']
        budget=max(len(p) for _,p in all_prefixes) if max_total_tokens is None else max_total_tokens
        if type(budget) is not int or budget<max(len(p) for _,p in all_prefixes):
            raise ValueError('BATCH_GROWTH_REMAINING_CONTEXT_BINDING')
        if budget + action[0] + 1 > CONTEXT:
            if pending:
                raise ValueError('BATCH_GROWTH_CONTEXT_WITH_PENDING_COHORT')
            action, reason = (3, 1), 'context_floor'
            if budget + 4 > CONTEXT:
                raise ValueError('BATCH_GROWTH_CONTEXT_REQUIRES_NATIVE_AR')
            st.update(current=action, target=None, last_switch=st['choices'])
        if not pending and core.FORCED is not None and budget + core.FORCED[0] + 1 <= CONTEXT:
            # The isolated-arm override has to land here too: this cohort action
            # is decided in prepare_cohort, not in Policy.choose, so without it a
            # "fixed k" arm silently kept the adaptive heuristic.
            action = core.FORCED
            reason = 'forced_action_over_%s' % reason
            st.update(current=action, target=None, last_switch=st['choices'])
        self.current = action
        # Keep only exact immutable integer tuples. This is per-cohort, bounded
        # by BATCH_MAX, and never an identity cache for mutable token containers.
        immutable = {rid: prefix for rid, prefix in fresh
                     if type(prefix) is tuple and all(type(t) is int and t >= 0 for t in prefix)}
        self.frame = dict(remaining=digests, immutable_prefixes=immutable,
                          batch=len(all_prefixes), key=key, action=action,
                          reason=reason, choice=st['choices'])

    def choose(self, rid, prefix, batch):
        frame = self.frame
        same = (type(prefix) is tuple and frame is not None
                and frame.get('immutable_prefixes', {}).get(rid) is prefix)
        digest = frame['remaining'].get(rid) if same else fingerprint(prefix)
        if (frame is None or frame['batch'] != batch
                or frame['remaining'].get(rid) != digest):
            raise ValueError('BATCH_GROWTH_COHORT_REQUIRED')
        if (rid, digest) in self.issued:
            raise ValueError('GROWTH_PLAN_ALREADY_PENDING')
        self.round_id += 1
        p = core.Plan(rid, digest, *frame['action'], self.round_id,
                      self.generation, frame['reason'])
        self.issued[(rid, digest)] = p
        frame['remaining'].pop(rid)
        frame.get('immutable_prefixes', {}).pop(rid, None)
        self.record_plan(p, frame, batch)
        return p

    def record_plan(self, p, frame, batch):
        self.event('growth_plan', plan=p.wire(), state=dict(self.state),
            produce_batch=batch, bucket=frame['key'], selector=SELECTOR,
            bucket_choice=frame['choice'], bucket_current=list(frame['action']),
            scores=[dict(k=k, branches=b, score=self.score(frame['key'], (k, b)))
                    for k, b in self.ACTIONS])

    def _learn_cycle(self, row):
        key = (*self.bucket(row['actual_batch'], max(row['prefix_lengths'])),
               row['k'], row['branches'])
        seconds = (row['end_ns'] - row['start_ns']) / 1e9
        s = self.stats[key]
        s['n'] += 1
        s['tokens'] += row['delivered']
        s['seconds'] += seconds
        window = self.windows.setdefault(key, deque())
        if len(window) == self.WINDOW:
            delivered, elapsed = window.popleft()
            s['window_tokens'] -= delivered
            s['window_seconds'] -= elapsed
        window.append((row['delivered'], seconds))
        s['window_n'] = len(window)
        s['window_tokens'] += row['delivered']
        s['window_seconds'] += seconds

    def commit_batch(self, rows, *, rpc_ms=0.):
        """Called once after the whole native batch has been emitted, before free."""
        now = self.clock()
        before = self.delivery_end_ns
        if before is not None and now <= before:
            raise ValueError('GROWTH_CLOCK')
        if not rows or len({r['rid'] for r in rows}) != len(rows):
            raise ValueError('BATCH_GROWTH_DELIVERY_BINDING')
        plans = [r['plan'] for r in rows]
        speculative = all(p is not None for p in plans)
        if not speculative and any(p is not None for p in plans):
            raise ValueError('BATCH_GROWTH_MIXED_AR_AND_SPEC')
        if speculative:
            if (len(rows) > profile.BATCH_MAX or len({(p.k, p.branches) for p in plans}) != 1
                    or any(p.rid != r['rid'] or self.last_used.get(p.rid) != p
                           for p, r in zip(plans, rows))):
                raise ValueError('BATCH_GROWTH_COMMIT_PLAN')
        for r in rows:
            if (type(r['delivered']) is not int or type(r['raw_emitted']) is not int
                    or not 0 <= r['delivered'] <= r['raw_emitted']
                    or r['remaining'] <= 0 or r['prefix_length'] < 1):
                raise ValueError('BATCH_GROWTH_DELIVERY_COUNTS')
        if (not speculative and self.control.get('skip_first_request', False)
                and getattr(self, 'preflight_rid', None) is None):
            if len(rows) != 1:
                raise ValueError('BATCH_GROWTH_PREFLIGHT_MUST_BE_SINGLE')
            self.preflight_rid = rows[0]['rid']
            self.event('growth_preflight_excluded', rid=self.preflight_rid,
                       phase_key=self.control.get('phase_key'))
        censored = any(r['remaining'] <= r['raw_emitted'] or r['delivered'] < r['raw_emitted'] for r in rows)
        learning = bool(speculative and before is not None and not censored
                        and self.control.get('learn', True)
                        and getattr(self, 'preflight_rid', None) not in [r['rid'] for r in rows])
        self.cycle_id += 1
        row = dict(unit=UNIT, selector=SELECTOR, generation=self.generation,
            cycle_id=self.cycle_id, phase_key=self.control.get('phase_key'),
            ids=[r['rid'] for r in rows], plans=[p.wire() if p else None for p in plans],
            actual_batch=len(rows), prefix_lengths=[r['prefix_length'] for r in rows],
            k=plans[0].k if speculative else None,
            branches=plans[0].branches if speculative else None,
            delivered=sum(r['delivered'] for r in rows),
            raw_emitted=sum(r['raw_emitted'] for r in rows),
            sources=[r.get('source') for r in rows],
            side_accepts=sum(bool(r.get('side', False)) for r in rows),
            terminal_censored=censored, learning=learning,
            start_ns=before, end_ns=now,
            cycle_ms=(now-before)/1e6 if before is not None else None,
            target_rpc_host_ms=rpc_ms,
            boundary='previous_controller_delivery_to_this_whole_batch_emit')
        if len(self.observations) >= 20000:
            raise RuntimeError('GROWTH_OBSERVATION_LIMIT')
        if learning:
            self._learn_cycle(row)
        for p, r in zip(plans, rows):
            self.last_commit[r['rid']] = now
            if p is not None:
                self.last_used.pop(p.rid)
        self.delivery_end_ns = now
        self.observations.append(row)
        self.event('growth_batch_cycle', **row)

    def load_seed(self, rows):
        if self.issued or self.last_used or self.last_commit or self.stats:
            raise ValueError('BATCH_GROWTH_SEED_REQUIRES_FRESH_POLICY')
        seen = set()
        last_end = {}
        for row in rows:
            identity = row['generation'], row['cycle_id']
            plans = row['plans']
            if (row.get('unit') != UNIT or row.get('selector') != SELECTOR
                    or not row.get('learning') or row.get('terminal_censored')
                    or identity in seen or not 1 <= row['actual_batch'] <= profile.BATCH_MAX
                    or len(row['ids']) != row['actual_batch'] or len(set(row['ids'])) != len(row['ids'])
                    or len(row['prefix_lengths']) != row['actual_batch']
                    or len(plans) != row['actual_batch'] or row['start_ns'] is None
                    or type(row['start_ns']) is not int or type(row['end_ns']) is not int
                    or row['end_ns'] <= row['start_ns']
                    or type(row['delivered']) is not int or type(row['raw_emitted']) is not int
                    or row['delivered'] <= 0 or row['raw_emitted'] != row['delivered']
                    or row['delivered'] > row['actual_batch']*(row['k']+1)
                    or len(row['sources']) != row['actual_batch']
                    or any(s not in ('edge','suffix','exact_prefix') for s in row['sources'])
                    or row['start_ns'] < last_end.get(row['generation'],row['start_ns'])
                    or not math.isfinite(row['cycle_ms'])
                    or abs(row['cycle_ms']-(row['end_ns']-row['start_ns'])/1e6) > 1e-6):
                raise ValueError('BATCH_GROWTH_PRIOR_NOT_FACTUAL_UNCENSORED')
            bound = [core.Plan(**p) for p in plans]
            if ([(p.k, p.branches) for p in bound] != [(row['k'], row['branches'])]*len(bound)
                    or [p.rid for p in bound] != row['ids']
                    or any(p.generation != row['generation'] for p in bound)
                    or len({p.round_id for p in bound}) != len(bound)
                    or any(type(n) is not int or n < 1 for n in row['prefix_lengths'])):
                raise ValueError('BATCH_GROWTH_SEED_PLAN_BINDING')
            seen.add(identity)
            last_end[row['generation']] = row['end_ns']
            self._learn_cycle(row)
        self.seed_rows = len(rows)

    def snapshot(self):
        result = super().snapshot()
        result.update(selector=SELECTOR, sample_unit=UNIT, seed_rows=self.seed_rows,
            cycle_id=self.cycle_id, waiting_does_not_force_K3=True,
            complete_batches=list(range(1, profile.BATCH_MAX+1)),
            limits=dict(target_graphs=profile.TARGET_GRAPHS, edge_graphs=profile.EDGE_GRAPHS,
                        max_graphs=profile.GRAPH_LIMIT, default_max_graphs=64))
        return result

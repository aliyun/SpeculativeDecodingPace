from collections import defaultdict, deque
import math


class AttemptLedger:
    def __init__(self, window, limit):
        self.active = {}
        self.history = defaultdict(lambda: deque(maxlen=window))
        self.observations = deque(maxlen=window)
        self.limit = limit
        self.completed_cycles = 0

    def note(self, plan, prefix, source, *, initial):
        prefix = tuple(prefix)
        entry = self.active.get(plan.rid)
        if entry is None:
            if not initial:
                return False  # A non-C1 dispatch may have retired this intent.
            if len(self.active) >= self.limit:
                raise RuntimeError('GROWTH_ATTEMPT_LIMIT')
            entry = dict(plan=plan, origin=prefix, source=source, pending=None,
                         last_length=None, last_delivered=0, retiring=False, record=None)
            self.active[plan.rid] = entry
        elif initial or entry['plan'] != plan:
            raise ValueError('GROWTH_ATTEMPT_IDENTITY')
        if entry['pending'] is not None:
            raise ValueError('GROWTH_ATTEMPT_ALREADY_PENDING')
        if prefix[:len(entry['origin'])] != entry['origin']:
            raise ValueError('GROWTH_ATTEMPT_PREFIX')
        if entry['last_length'] is not None and len(prefix) != entry['last_length']+entry['last_delivered']:
            raise ValueError('GROWTH_ATTEMPT_PROGRESS')
        entry['pending'] = prefix
        return True

    def validate(self, rows):
        for row in rows:
            entry = self.active.get(row['rid'])
            if entry is not None and entry['pending'] is not None:
                if row['plan'] is not None or row['prefix_length'] != len(entry['pending']):
                    raise ValueError('GROWTH_ATTEMPT_FEEDBACK')

    def finish(self, rid, plan):
        entry = self.active.get(str(rid))
        if entry is not None:
            if entry['plan'] != plan:
                raise ValueError('GROWTH_ATTEMPT_RETIRE_IDENTITY')
            if entry['pending'] is None:
                if entry['record'] is not None:entry['record']['retiring']=True
                self.active.pop(str(rid))
            else:
                entry['retiring'] = True

    def discard(self, rid):
        self.active.pop(str(rid), None)

    def commit(self, rows, cycle, rpc_ms, *, bucket, account, learn):
        if len(rows) != 1:
            for row in rows:
                self.discard(row['rid'])  # Never transfer batch cost into C1.
            return None
        row = rows[0]
        entry = self.active.get(row['rid'])
        if entry is None or entry['pending'] is None:
            return None  # Initial/pure AR with no issued speculative attempt.
        elapsed = cycle['cycle_ms']
        if elapsed is None:
            # No prior delivery boundary: do not invent a wait duration.
            self.discard(row['rid'])
            return None
        if (not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(rpc_ms)
                or rpc_ms < 0 or rpc_ms > elapsed+1e-6):
            raise ValueError('GROWTH_ATTEMPT_TIMER')
        plan = entry['plan']
        key = bucket(1, len(entry['origin']))
        tick = account(key, elapsed, probe=plan.reason == 'budget_probe')
        record = entry['record']
        if record is None:
            record = dict(generation=cycle['generation'], cycle_id=cycle['cycle_id'],
                          attempt_plan=plan.wire(), k=plan.k, branches=plan.branches,
                          prefix_bucket=key[1], source_group=entry['source'],
                          cycle_ms=0., cloud_rpc_ms=0., delivered=0, side_accepts=0,
                          query_nodes=0, learning=True, parts=[],
                          sample_kind='ISSUED_PLAN_SERVED_BY_AR_NOT_NATIVE_VERIFY')
            entry['record'] = record
            self.history[key].append(record)
        record['cycle_ms'] += elapsed
        record['delivered'] += row['delivered']
        record['query_nodes'] += 1  # Actual AR query, not the intended k+roots.
        record['cloud_rpc_ms'] = (record['cloud_rpc_ms']+rpc_ms
                                  if record['cloud_rpc_ms'] is not None and rpc_ms > 0 else None)
        record['unhidden_noncloud_ms'] = (record['cycle_ms']-record['cloud_rpc_ms']
                                         if record['cloud_rpc_ms'] is not None else None)
        record['learning'] = record['learning'] and learn
        record['tick'] = tick
        record['parts'].append(cycle['cycle_id'])
        record['last_cycle_id'] = cycle['cycle_id']
        record['retiring'] = entry['retiring']
        entry['last_length'] = len(entry['pending'])
        entry['last_delivered'] = row['delivered']
        entry['pending'] = None
        snapshot = dict(record, parts=list(record['parts']), charged_cycle_ms=elapsed,
                        actual_cycle_id=cycle['cycle_id'])
        self.observations.append(snapshot)
        self.completed_cycles += 1
        if entry['retiring']:
            self.active.pop(row['rid'])
        return snapshot

    def fresh(self, key, action, source, now, age):
        return [r for r in self.history.get(key, ())
                if (r['k'], r['branches']) == action and r['source_group'] == source
                and r['learning'] and now-r['tick'] < age]
